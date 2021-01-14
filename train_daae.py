import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from ehr import EHR
from utils import to_var, idx2entity
from model import Seq2seq_Autoencoder, CNN_Discriminator, MLP_Generator, MLP_Discriminator

def main(args):

    torch.cuda.set_device(args.gpu_devidx)    
    splits = ['train', 'valid']

    datasets = OrderedDict()
    for split in splits:
        datasets[split] = EHR(
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ
        )

    AE = Seq2seq_Autoencoder(
        vocab_size=datasets['train'].vocab_size,
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        unk_idx=datasets['train'].unk_idx,
        max_sequence_length=args.max_sequence_length,
        max_visit_length=args.max_visit_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        entity_dropout=args.entity_dropout,
        embedding_dropout=args.embedding_dropout,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )

    Dx = CNN_Discriminator(
        embedding_weight=AE.embedding.weight,
        embedding_dropout=args.embedding_dropout,
        feature_dropout=args.feature_dropout,
        filter_size=args.filter_size,
        window_sizes=args.window_sizes,
        )

    G = MLP_Generator(
        input_size=args.noise_size,
        output_size=args.latent_size,
        archs=args.gmlp_archs
        )

    Dz = MLP_Discriminator(
        input_size=args.latent_size*2,
        output_size=1,
        archs=args.dmlp_archs
        ) 

    if torch.cuda.is_available():
        AE = AE.cuda()
        Dx = Dx.cuda()
        G = G.cuda()
        Dz = Dz.cuda()

    def compute_NLL_loss(prob, target, length):
        target = target[:, :prob.size(1), :]
        labels = torch.zeros_like(prob).scatter_(2, target, 1)
        mask = target.sum(dim=2) > 0
        
        NLL_loss = torch.mul(labels[:,:,1:], torch.log(prob[:,:,1:] + 1e-12))
        NLL_loss += torch.mul(1 - labels[:,:,1:], torch.log(1 - prob[:,:,1:] + 1e-12))
        #NLL_loss = -torch.mul(NLL_loss.sum(dim=2), mask.float()).sum(dim=1).mean(dim=0)
        NLL_loss = -torch.mul(NLL_loss.sum(dim=2), mask.float()).sum()
 
        return NLL_loss

    opt_enc = torch.optim.Adam(AE.encoder.parameters(), lr=args.learning_rate)
    opt_dec = torch.optim.Adam(AE.decoder.parameters(), lr=args.learning_rate)
    opt_dix = torch.optim.Adam(Dx.parameters(), lr=args.learning_rate)
    opt_diz = torch.optim.Adam(Dz.parameters(), lr=args.learning_rate)
    opt_gen = torch.optim.Adam(G.parameters(), lr=args.learning_rate)

    if args.dp_sgd == True:
        import pyvacy
        opt_dec = pyvacy.optim.DPAdam(params=AE.decoder.parameters(), lr=args.learning_rate, batch_size=args.batch_size,
                                    l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
        opt_gen = pyvacy.optim.DPAdam(params=G.parameters(), lr=args.learning_rate, batch_size=args.batch_size,
                                    l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.noise_multiplier)
        epsilon = pyvacy.analysis.moments_accountant(len(datasets['train'].data), args.batch_size, args.noise_multiplier, args.epochs, args.delta)

        print('Training procedure satisfies (%f, %f)-DP' % (2*epsilon, args.delta))

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    for epoch in range(args.epochs):

        print("Epoch\t%02d/%i"%(epoch, args.epochs))
        for split in splits:
            
            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split=='train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )
            
            n_data = len(datasets[split].data)
            n_visit = sum([d['length'] for k, d in datasets[split].data.items()])
       
            NLL_total_loss = 0.0
            xCritic_total_loss, zCritic_total_loss = 0.0, 0.0
        
            if split == 'train':
                AE.entity_dropout_rate = args.entity_dropout
                AE.decoder.entity_dropout_rate = args.entity_dropout
                AE.train()
                Dx.train()
                G.train()
                Dz.train()
            else:
                AE.entity_dropout_rate = 0.0
                AE.decoder.entity_dropout_rate = 0.0
                AE.eval()
                Dx.eval()
                G.eval()
                Dz.eval()

            for iteration, batch in enumerate(data_loader):
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                one = torch.tensor(1, dtype=torch.float)
                mone = one * -1

                if torch.cuda.is_available():
                    one = one.cuda()
                    mone = mone.cuda()

                # Step 0: Evaluate current loss
                z, Pinput, Poutput, Soutput, Moutput = AE(batch['input'], batch['target'], batch['length'])
                NLL_loss = compute_NLL_loss(Poutput, batch['target'], batch['length'])

                zgen = G(batch_size=z.size(0))
                Pgen, Sgen, Mgen = AE.decoder.inference(z=zgen)

                Dx.embedding_weight.data = AE.embedding.weight.data
                Dinput, Doutput, Dgen = Dx(Pinput).mean(), Dx(Poutput, Moutput).mean(), Dx(Pgen, Mgen).mean()
                Dreal, Dfake = Dz(z).mean(), Dz(zgen).mean()

                xCritic_loss = - Dinput + 0.5 * (Doutput + Dgen)
                zCritic_loss = - Dreal + Dfake
                
                if split == 'train':
                    # Step 1: Update the Critic_x
                    opt_dix.zero_grad()
                    Dinput, Doutput = Dx(Pinput).mean(), Dx(Poutput, Moutput).mean()
                    Dinput.backward(mone, retain_graph=True)
                    Doutput.backward(one, retain_graph=True)
                    Dx.cal_gradient_penalty(Pinput[:, :Poutput.size(1), :], Poutput, Moutput).backward()
                    opt_dix.step()

                    opt_dix.zero_grad()
                    Dinput, Dgen = Dx(Pinput).mean(), Dx(Pgen, Mgen).mean()
                    Dinput.backward(mone, retain_graph=True)
                    Dgen.backward(one, retain_graph=True)
                    Dx.cal_gradient_penalty(Pinput[:, :Pgen.size(1), :], Pgen, Mgen).backward()
                    opt_dix.step()
                        
                    # Step 2: Update the Critic_z
                    opt_diz.zero_grad()
                    Dreal, Dfake = Dz(z).mean(), Dz(zgen).mean()
                    Dreal.backward(mone, retain_graph=True)
                    Dfake.backward(one, retain_graph=True)
                    Dz.cal_gradient_penalty(z, zgen).backward()
                    opt_diz.step()

                    # Step 3, 4: Update the Decoder and the Encoder
                    opt_dec.zero_grad()
                    Doutput, Dgen = Dx(Poutput, Moutput).mean(), Dx(Pgen, Mgen).mean()
                    Doutput.backward(mone, retain_graph=True)
                    Dgen.backward(mone, retain_graph=True)
                    
                    opt_enc.zero_grad()
                    Dreal = Dz(z).mean()
                    Dreal.backward(one, retain_graph=True)

                    NLL_loss.backward(retain_graph=True)
                    opt_dec.step()
                    opt_enc.step()
 
                    # Step 5: Update the Generator
                    opt_gen.zero_grad()
                    Dfake = Dz(zgen).mean()
                    Dfake.backward(mone, retain_graph=True)
                    opt_gen.step()
                
                NLL_total_loss += NLL_loss.data
                xCritic_total_loss += xCritic_loss.data
                zCritic_total_loss += zCritic_loss.data

            # print the losses for each epoch
            print("\t\t%s\tNLL_loss\t%9.4f\txCritic_loss\t%9.4f\tzCritic_loss\t%9.4f"%(split.upper(), NLL_total_loss/n_visit, xCritic_total_loss/n_data, zCritic_total_loss/n_data))
               
    # Generate the synthetic sequences as many as you want 
    AE.eval()
    G.eval()

    gen_zs, gen_xs = [], []
    for i in range(args.gendata_size//args.batch_size):
        zgen = G(batch_size=args.batch_size)
        Pgen, Sgen, Mgen = AE.decoder.inference(z=zgen)
        
        gen_zs.append(zgen)
        gen_xs += idx2entity(Sgen, i2w=datasets['train'].get_i2w(), pad_idx=datasets['train'].pad_idx, eos_idx=datasets['train'].eos_idx)

    gen_zlist = torch.cat(gen_zs).cpu().detach().numpy()
    gen_xdict = {p : gen_xs[p] for p in range(len(gen_xs))}
    
    np.save('daae_generated_codes.npy', gen_zlist)
    np.save('daae_generated_patients.npy', gen_xdict) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=10)
    parser.add_argument('--max_visit_length', type=int, default=40)
    parser.add_argument('--min_occ', type=int, default=0)
    
    parser.add_argument('-ep', '--epochs', type=int, default=500)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-gs','--gendata_size', type=int, default=100000)
    parser.add_argument('-gd', '--gpu_devidx', type=int, default=0)

    parser.add_argument('-eb', '--embedding_size', type=int, default=128)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=128)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ns', '--noise_size', type=int, default=128)
    parser.add_argument('-ls', '--latent_size', type=int, default=128)
    parser.add_argument('-fs', '--filter_size', type=int, default=16)
    parser.add_argument('-ws', '--window_sizes', nargs='+', type=int, default=[2, 3])
    parser.add_argument('-wd', '--entity_dropout', type=float, default=0.05)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-fd', '--feature_dropout', type=float, default=0.5)
    parser.add_argument('-ga', '--gmlp_archs', nargs='+', type=int, default=[128, 128])
    parser.add_argument('-da', '--dmlp_archs', nargs='+', type=int, default=[256, 128])

    parser.add_argument('--dp_sgd', type=bool, default=False)
    parser.add_argument('--noise_multiplier', type=float, default=1)
    parser.add_argument('--l2_norm_clip', type=float, default=0.5)
    parser.add_argument('--delta', type=float, default=1e-3)

    args = parser.parse_args()
    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.entity_dropout <= 1

    main(args)
