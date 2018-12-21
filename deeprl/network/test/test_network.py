from deepq.network.base import Network
import torch.nn as nn
import torch.nn.functional as F
from toolz import identity
import numpy as np
import torch
from nose.tools import assert_equal

def test_network():
    layers = (nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 5))
    activations = (F.relu, F.relu, identity)
    net = Network(10, 5, layers, activations)
    state = np.random.normal(size=(1,10)).astype(np.float32)
    q_values = net.forward(torch.from_numpy(state))
    assert_equal(q_values.shape, (1,5))

if __name__ == '__main__':
    # Run the tests in this file.
    import sys
    import nose
    module_name = sys.modules[__name__].__file__
 
    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s','-v'])