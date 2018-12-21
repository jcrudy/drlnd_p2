import torch.nn.functional as F
import torch.nn as nn
from toolz import identity
from deepq.network.base import Network
import numpy as np
from deepq.model.double_dqn import DoubleDQNModel

def test_double_dqn():
    # TODO: Make test data more meaningful and add some assertions.
    
    np.random.seed(0)
    
    layers = (nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 5))
    activations = (F.relu, F.relu, identity)
    net = Network(10, 5, layers, activations)
    model = DoubleDQNModel(net)
    state = np.random.normal(size=(100, 10))
    action = np.random.randint(5, size=100)
    reward = np.random.normal(size=100)
    next_state = np.random.normal(size=(100, 10))
    weight = np.ones(shape=100)
    done = np.random.binomial(1, .1, size=100)
    model.learn(state, action, reward, next_state, done, weight)
    
if __name__ == '__main__':
    # Run the tests in this file.
    import sys
    import nose
    module_name = sys.modules[__name__].__file__
 
    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s','-v'])