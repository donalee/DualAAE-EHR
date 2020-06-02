import torch
import numpy as np
from torch.autograd import Variable
from collections import defaultdict, Counter, OrderedDict

class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def idx2entity(idx, i2w, pad_idx, eos_idx):
    sequences = []

    for i, patient in enumerate(idx):
        sequence = []
        for visit in patient:
            if visit[0].item() == eos_idx:
                break

            entities = []
            for entity_id in visit:
                if entity_id.item() == pad_idx or entity_id.item() == eos_idx:
                    continue
                entities.append(i2w[str(entity_id.item())])
            sequence.append(entities)

            if eos_idx in visit:
                break
        sequences.append(sequence)
    return sequences

