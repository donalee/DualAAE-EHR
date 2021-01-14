import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var

class Encoder(nn.Module):

    def __init__(self, embedding_weight, rnn, rnn_type, hidden_size, hidden_factor, latent_size, num_layers=1, bidirectional=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        
        self.embedding_weight = embedding_weight
        self.vocab_size, self.embedding_size = embedding_weight.shape
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.hidden_factor = hidden_factor
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.encoder_rnn = rnn(self.embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.hidden2latent = nn.Linear(hidden_size * self.hidden_factor, latent_size)

    def forward(self, input_sequence, length):
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        prob_sequence = torch.zeros(input_sequence.size(0), input_sequence.size(1), self.vocab_size, out=self.tensor()).scatter_(2, input_sequence, 1)
        prob_sequence[:,:,0] = 0

        # encoder input
        input_embedding = torch.mm(prob_sequence.view(-1, self.vocab_size), self.embedding_weight).view(batch_size, -1, self.embedding_size)
        input_embedding = torch.tanh(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        if self.rnn_type != 'lstm':
            _, hidden = self.encoder_rnn(packed_input)
        else:
            _, (hidden, _) = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        else:
            hidden = hidden.squeeze()

        z = self.hidden2latent(hidden)
        z = torch.tanh(z)

        return z

class Decoder(nn.Module):

    def __init__(self, embedding_weight, rnn, rnn_type, hidden_size, hidden_factor, latent_size, entity_dropout, embedding_dropout,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, max_visit_length, num_layers=1, bidirectional=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.entity_dropout_rate = entity_dropout

        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.max_sequence_length = max_sequence_length
        self.max_visit_length = max_visit_length

        self.embedding_weight = embedding_weight
        self.vocab_size, self.embedding_size = embedding_weight.shape
        self.embedding_dropout = embedding_dropout
        self.decoder_rnn = rnn(self.embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.latent2hidden = nn.Linear(latent_size, hidden_size)
        self.outputs2vocab = nn.Linear(hidden_size, self.vocab_size)

        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

    def forward(self, z, input_sequence, length):
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]
        
        if self.entity_dropout_rate > 0:
            # randomly discard entities from decoder inputs
            entity_dropout_mask = torch.rand(input_sequence.size()) > self.entity_dropout_rate
            entity_dropout_mask[:,0,:] = 1
            input_sequence = torch.mul(input_sequence, entity_dropout_mask.long().cuda())

        prob_sequence = torch.zeros(input_sequence.size(0), input_sequence.size(1), self.vocab_size, out=self.tensor()).scatter_(2, input_sequence, 1)
        prob_sequence[:,:,0] = 0
    
        hidden = self.latent2hidden(z)
        hidden = hidden.unsqueeze(0)

        # decoder input
        input_embedding = torch.mm(prob_sequence.view(-1, self.vocab_size), self.embedding_weight).view(batch_size, -1, self.embedding_size)
        input_embedding = torch.tanh(input_embedding)
        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        if self.rnn_type != 'lstm':
            outputs, _ = self.decoder_rnn(packed_input, hidden)
        else:
            outputs, _ = self.decoder_rnn(packed_input, (hidden, torch.zeros_like(hidden)))

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        m_output = torch.sum(torch.abs(padded_outputs), dim=2) > 0
        b,s,_ = padded_outputs.size()

        # project outputs to vocab
        logits = self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2)))
        s_output = self._sample(logits).view(b, s, -1)

        p_output = torch.sigmoid(logits)
        p_output = p_output.view(b, s, self.vocab_size)

        return p_output, s_output, m_output

    def inference(self, n=4, z=None):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)
        hidden = hidden.unsqueeze(0)
        input_hidden = hidden

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
        sequence_length = torch.zeros(batch_size, out=self.tensor()).long()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_sequence_length + 1, self.max_visit_length).fill_(self.pad_idx).long()
        generations[:, 0, 0] = self.sos_idx

        t=0
        while(t<self.max_sequence_length and len(running_seqs)>0):

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())
                input_sequence = input_sequence.unsqueeze(dim=1)
            
            input_sequence = input_sequence.unsqueeze(dim=1)
            
            prob_sequence = torch.zeros(input_sequence.size(0), input_sequence.size(1), self.vocab_size, out=self.tensor()).scatter_(2, input_sequence, 1)
            prob_sequence = prob_sequence/torch.sum(prob_sequence, dim=2).unsqueeze(dim=2)

            input_embedding = torch.mm(prob_sequence.view(-1, self.vocab_size), self.embedding_weight).view(input_sequence.size(0), -1, self.embedding_size)
            input_embedding = torch.tanh(input_embedding)
                
            if self.rnn_type != 'lstm':
                output, hidden = self.decoder_rnn(input_embedding, hidden)
            else:
                output, (hidden, _) = self.decoder_rnn(input_embedding, (hidden, torch.zeros_like(hidden)))

            logits = self.outputs2vocab(output.view(-1, output.size(2)))

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t+1)

            # update gloabl running sequence
            sequence_length[sequence_running] += 1
            sequence_mask[sequence_running] = ((input_sequence == self.eos_idx).long().sum(dim=1) == 0).data
            sequence_running = sequence_idx.masked_select(sequence_mask)
            
            # update local running sequences
            running_mask = ((input_sequence == self.eos_idx).long().sum(dim=1) == 0).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        s_gen = generations

        sorted_lengths, sorted_idx = torch.sort(sequence_length, descending=True)
        s_gen = s_gen[sorted_idx]
        s_gen_input, s_gen_target = s_gen[:,:-1,:], s_gen[:,1:,:]

        # build a prob tensor
        prob_sequence = torch.zeros(s_gen_input.size(0), s_gen_input.size(1), self.vocab_size, out=self.tensor()).scatter_(2, s_gen_input, 1)
        prob_sequence[:,:,0] = 0

        input_embedding = torch.mm(prob_sequence.view(-1, self.vocab_size), self.embedding_weight).view(batch_size, -1, self.embedding_size)
        input_embedding = torch.tanh(input_embedding)

        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        if self.rnn_type != 'lstm':
            outputs, _ = self.decoder_rnn(packed_input, input_hidden)
        else:
            outputs, _ = self.decoder_rnn(packed_input, (input_hidden, torch.zeros_like(input_hidden)))

        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        m_gen = torch.sum(torch.abs(padded_outputs), dim=2) > 0.0
        b,s,_ = padded_outputs.size()

        logits = self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2)))
        p_gen = torch.sigmoid(logits)
        p_gen = p_gen.view(b, s, self.vocab_size)

        return p_gen, s_gen_target, m_gen

    def _sample(self, prob, theta=0.5):
        prob, sample = torch.topk(prob[:, 3:], self.max_visit_length, dim=-1)
        sample = sample + 3

        prob, sample = prob.view(-1), sample.view(-1)

        running_idx = torch.arange(0, prob.size(0), out=self.tensor()).long()
        multihot_mask = torch.sigmoid(prob).le(theta)
        termi_idx = running_idx.masked_select(multihot_mask)
        sample[termi_idx] = self.pad_idx
        sample = sample.view(-1, self.max_visit_length)

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t,:] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to

class Seq2seq_Autoencoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, latent_size, entity_dropout, embedding_dropout,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, max_visit_length, num_layers=1, bidirectional=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.max_sequence_length = max_sequence_length
        self.max_visit_length = max_visit_length

        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.vocab_size = vocab_size

        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.embedding_size = embedding_size
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.entity_dropout_rate = entity_dropout

        if rnn_type == 'rnn':
            self.rnn = nn.RNN
        elif rnn_type == 'gru':
            self.rnn = nn.GRU
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM
        else:
            raise ValueError()

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.encoder = Encoder(
            embedding_weight=self.embedding.weight,
            rnn=self.rnn,
            rnn_type=self.rnn_type,
            hidden_size=self.hidden_size,
            hidden_factor=self.hidden_factor,
            latent_size=self.latent_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional
            )

        self.decoder = Decoder(
            embedding_weight=self.embedding.weight,
            rnn=self.rnn,
            rnn_type=self.rnn_type,
            hidden_size=self.hidden_size,
            hidden_factor=self.hidden_factor,
            latent_size=self.latent_size,
            entity_dropout=self.entity_dropout_rate,
            embedding_dropout=self.embedding_dropout,
            sos_idx=self.sos_idx,
            eos_idx=self.eos_idx,
            pad_idx=self.pad_idx,
            unk_idx=self.unk_idx,
            max_sequence_length=self.max_sequence_length,
            max_visit_length=self.max_visit_length,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional
            )

    def forward(self, input_sequence, target_sequence, length):
        z = self.encoder(input_sequence, length)
        if len(z.shape) == 1: z = z.unsqueeze(0)
        p_output, s_output, m_output = self.decoder(z, input_sequence, length)

        # build a target prob tensor
        p_input = torch.zeros(target_sequence.size(0), target_sequence.size(1), self.vocab_size, out=self.tensor()).scatter_(2, target_sequence, 1)
        p_input[:,:,0] = 0
        
        return z, p_input, p_output, s_output, m_output

class CNN_Discriminator(nn.Module):

    def __init__(self, embedding_weight, embedding_dropout, feature_dropout, filter_size, window_sizes):
        super().__init__()
        self.vocab_size, self.embedding_size = embedding_weight.shape
        self.embedding_weight = nn.Parameter(embedding_weight.data, requires_grad=False)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.feature_dropout = nn.Dropout(p=feature_dropout)

        self.filter_size = filter_size
        self.window_sizes = window_sizes

        self.convs = nn.ModuleList([torch.nn.Conv2d(1, filter_size, (window_size, self.embedding_size), padding=(1, 0)) for window_size in window_sizes])
        self.feature2binary = nn.Linear(filter_size * len(window_sizes), 1)

    def forward(self, prob_sequence, mask=None):
        batch_size = prob_sequence.size(0)

        input_embedding = torch.mm(prob_sequence.view(-1, self.vocab_size), self.embedding_weight).view(batch_size, -1, self.embedding_size)
        input_embedding = torch.tanh(input_embedding)
        input_embedding = self.embedding_dropout(input_embedding)

        if mask is not None:
            mask = mask.unsqueeze(dim=2).repeat([1, 1, self.embedding_size])
            input_embedding = torch.mul(input_embedding, mask.float())

        input_embedding = input_embedding.unsqueeze(1)

        feature_maps = [nn.functional.relu(conv(input_embedding)).squeeze(3) for conv in self.convs]
        feature_maps = [nn.functional.max_pool1d(feature_map, feature_map.size(2)).squeeze(2) for feature_map in feature_maps]

        feature_vecs = torch.cat(feature_maps, 1)
        feature_vecs = self.feature_dropout(feature_vecs)
        scores = self.feature2binary(feature_vecs).squeeze(1)
        return scores

    def cal_gradient_penalty(self, real_data, fake_data, mask, gp_lambda=10):
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1)
        epsilon = epsilon.expand(real_data.size())

        if torch.cuda.is_available():
            epsilon = epsilon.cuda()

        interpolates = epsilon * real_data + (1 - epsilon) * fake_data
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        D_interpolates = self.forward(interpolates, mask)

        gradients = torch.autograd.grad(outputs=D_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(D_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(D_interpolates.size()),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
        return gradient_penalty

class MLP_Generator(nn.Module):
    def __init__(self, input_size, output_size, archs, activation=nn.ReLU()):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        layer_sizes = [input_size] + archs
        layers = []

        fc_layers = []
        bn_layers = []
        ac_layers = []

        for i in range(len(layer_sizes)-1):
            fc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False))
            bn_layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            ac_layers.append(activation)

        fc_layers.append(nn.Linear(layer_sizes[-1], output_size, bias=False))
        bn_layers.append(nn.BatchNorm1d(output_size))
        ac_layers.append(nn.Tanh())

        self.fc_layers = nn.ModuleList(fc_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.ac_layers = nn.ModuleList(ac_layers)

    def forward(self, x=None, batch_size=None):
        if x is None:
            x = to_var(torch.randn([batch_size, self.input_size]))
        
        for i in range(len(self.fc_layers)):
            x1 = self.fc_layers[i](x)
            x2 = self.bn_layers[i](x1)
            x3 = self.ac_layers[i](x2)
            if i != len(self.fc_layers)-1:
                x = x + x3
            else:
                x = x3
        
        return x

class MLP_Discriminator(nn.Module):
    def __init__(self, input_size, output_size, archs, activation=nn.LeakyReLU(0.2)):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        layer_sizes = [input_size] + archs
        layers = []

        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(activation)

        layers.append(nn.Linear(layer_sizes[-1], output_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.cat((x, x.mean(dim=0).repeat([batch_size, 1])), dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x.squeeze(1)

    def cal_gradient_penalty(self, real_data, fake_data, gp_lambda=10):
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1)
        epsilon = epsilon.expand(real_data.size())

        if torch.cuda.is_available():
            epsilon = epsilon.cuda()

        interpolates = epsilon * real_data + (1 - epsilon) * fake_data
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        D_interpolates = self.forward(interpolates)

        gradients = torch.autograd.grad(outputs=D_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(D_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(D_interpolates.size()),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
        return gradient_penalty

