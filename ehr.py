import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

from utils import OrderedCounter

class EHR(Dataset):

    def __init__(self, data_dir, split, create_data, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 20)
        self.max_visit_length = kwargs.get('max_visit_length', 40)
        self.min_occ = kwargs.get('min_occ', 3)

        self.raw_data_path = os.path.join(data_dir, 'ehr.'+split+'.npy')
        self.data_file = 'ehr.'+split+'.json'
        self.vocab_file = 'ehr.vocab.json'

        if create_data:
            print("Creating new %s ehr data."%split.upper())
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("%s preprocessed file not found at %s. Creating new."%(split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)
        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'length': self.data[idx]['length'],
            'pid': self.data[idx]['pid']
        }

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w


    def _load_data(self, vocab=True):

        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']
            
    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):
        if self.split == 'train':
            self._create_vocab()
        else:
            self._load_vocab()

        data = defaultdict(dict)
        patients = np.load(self.raw_data_path, allow_pickle=True)[()]
        clean_patients = np.load(os.path.join(self.data_dir, 'ehr.valid.npy'), allow_pickle=True)[()]

        for patient in patients.keys():
            visits = patients[patient]                

            input = [['<sos>']] + visits
            input = input[:self.max_sequence_length]

            target = visits[:self.max_sequence_length-1]
            target = target + [['<eos>']]

            if self.split != 'train' and self.split != 'valid':
                clean_visits = clean_patients[patient]
                target = clean_visits[:self.max_sequence_length-1]
                target = target + [['<eos>']]

            assert len(input) == len(target), "%i, %i"%(len(input), len(target))
            length = len(input)

            for i in range(len(input)): input[i] = (input[i] + ['<pad>'] * self.max_visit_length)[:self.max_visit_length]
            for i in range(len(target)): target[i] = (target[i] + ['<pad>'] * self.max_visit_length)[:self.max_visit_length]

            input.extend([['<pad>'] * self.max_visit_length] * (self.max_sequence_length-length))
            target.extend([['<pad>'] * self.max_visit_length] * (self.max_sequence_length-length))

            #input.extend([['<pad>']] * (self.max_sequence_length-length))
            #target.extend([['<pad>']] * (self.max_sequence_length-length))

            for i in range(len(input)): input[i] = [self.w2i.get(e, self.w2i['<unk>']) for e in input[i]]
            for i in range(len(target)): target[i] = [self.w2i.get(e, self.w2i['<unk>']) for e in target[i]]

            id = len(data)
            data[id]['input'] = input
            data[id]['target'] = target
            data[id]['length'] = length
            data[id]['pid'] = int(patient)

        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocablurary can only be created for training file."

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        patients = np.load(self.raw_data_path, allow_pickle=True)[()]
        
        for patient in patients.keys():
            for visit in patients[patient]:
                w2c.update(visit)

        for w, c in w2c.items():
            if c > self.min_occ and w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocablurary of %i keys created." %len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()
