import torch.nn as nn
from six.moves import reduce  # @UnresolvedImport

class Network(nn.Module):
    def __init__(self, state_size, n_actions, layers, activations):
        super().__init__()
        self.state_size = state_size
        self.n_actions = n_actions
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
    