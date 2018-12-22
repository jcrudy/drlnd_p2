from multipledispatch.dispatcher import Dispatcher
from torch.tensor import Tensor
from numpy import ndarray
import torch
import numpy as np
import os

torchify = Dispatcher('torchify')

@torchify.register(Tensor)
def torchify_tensor(tens):
    return tens

@torchify.register(ndarray)
def torchify_numpy(arr):
    return torch.from_numpy(arr)

def torchify32(x):
    # TODO: This could obviously be more efficient.  It 
    # also might be good for it to handle integers differently.
    return torchify(numpify(x).astype(np.float32))

numpify = Dispatcher('numpify')

@numpify.register(Tensor)
def numpify_tensor(tens):
    return tens.detach().numpy()

@numpify.register(ndarray)
def numpify_numpy(arr):
    return arr

def rolling_mean(y, window):
    result = (
              np.convolve(y, np.ones(shape=window), mode='full') / 
              np.convolve(np.ones_like(y), np.ones(shape=window), mode='full')
              )
    return result[:(1-window)]

def weighted_choice(sample_size, weights):
    accum = np.cumsum(weights)
    choice_values = np.random.uniform(high=accum[-1], size=sample_size)
    return np.searchsorted(accum, choice_values)

def constant(value):
    def _constant(*args, **kwargs):
        return value
    return _constant

def split_path(path):
    path, tail = os.path.split(path)
    if not path:
        if not tail:
            return []
        else:
            return [tail]
    return split_path(path) + ([tail] if tail else [])
        

