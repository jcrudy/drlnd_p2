import torch.nn as nn
from six.moves import reduce  # @UnresolvedImport
import torch
import numpy as np
from operator import methodcaller
from functools import partial
from itertools import repeat
from deeprl.base import torchify, torchify32
class Network(nn.Module):
    def __init__(self, layers, activations):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.activations = tuple(activations)
    
    @staticmethod
    def _step(state, layer_and_activation):
        layer, activation = layer_and_activation
        return activation(layer(state))
    
    def forward(self, state):
        '''
        Compute the outputs for the given tensor of states for all actions.  
        Return a torch tensor.
        '''
        return reduce(self._step, zip(self.layers, self.activations), state)

def assoc(index, value, tup):
    return tup[:index] + (value,)
    

class ModuleStack(nn.Module):
    def __init__(self, modules, input_dim=0, output_dim=0):
        super().__init__()
        self.parts = nn.ModuleList(modules)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, data):
        results = []
        if len(data.shape) < self.input_dim:
            raise ValueError()
        if data.shape[self.input_dim] != len(self.parts):
            raise ValueError()
        for i, module in enumerate(self.parts):
            slice_tup = (tuple(repeat(slice(None, None, None), self.input_dim)) +
                         (i,) + 
                         tuple(repeat(slice(None, None, None), len(data.shape) - self.input_dim - 1)))
            results.append(module(data[slice_tup]))
        return torch.stack(results, dim=self.output_dim)
    
if __name__ == '__main__':
    def make_net():
        return nn.Sequential(
                     nn.Linear(10, 1),
#                      nn.ReLU(),
#                      nn.Linear(20, 1),
                     )
    
    nets = [make_net() for _ in range(4)]
    
    stack = ModuleStack(nets)
    
    
    
    X = np.random.normal(size=(10000, 10))
    beta = np.random.normal(size=10)
    y = np.random.normal(np.dot(X, beta))
    
    XX = np.stack([np.random.normal(size=(10000, 10)) for _ in range(4)])
    print(nets[0](torchify32(np.random.normal(size=(1,10,)))).shape)
    
    
    print(nets[0](torchify32(XX)))
    print(stack(torchify32(XX)))
#     print(stack(torchify32(XX)))
    
#     print(nets(X[:4, :]))
    
    
    
    
    
    
    