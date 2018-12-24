from deeprl.buffer.prioritized_replay import PrioritizedReplayBuffer
from deeprl.environment.base import Environment, CartesianProduct,\
    IntInterval, FloatInterval
import numpy as np
from deeprl.agent.base import Experience
from nose.tools import assert_equal  # @UnresolvedImport
from nose.tools import assert_greater, assert_almost_equal
from deeprl.base import weighted_choice

def test_weighted_choice():
    np.random.seed(0)
    weights = np.array([1., 1., 2.])
    sample = weighted_choice(100000, weights)
    assert_almost_equal(np.sum(sample == 2) / sample.shape[0], .5, places=2)

def test_uniform_sampling_replay_buffer():
    np.random.seed(0)
    class FauxEnvironment(Environment):
        def reset(self):
            return np.random.normal(size=10)
        
        def step(self, action):
            return np.random.normal(size=10), np.random.normal(size=1)[0], False if np.random.random() < .9 else True
        
        def close(self):
            pass
        
        @property
        def action_space(self):
            return CartesianProduct([IntInterval(0, 9)])
        
        @property
        def state_space(self):
            return CartesianProduct([FloatInterval()] * 10)
    
    np.random.seed(0)
    buffer_size = 10
    buffer = PrioritizedReplayBuffer(buffer_size, b_start=1.)
    env = FauxEnvironment()
    for _ in range(10):
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.random()
            next_state, reward, done = env.step(action)
            experience = Experience(state, action, reward, next_state, done)
            buffer.append(experience)
    buffer.report_errors([0], [1000.])
    indices, probs = buffer.sample_indices(100)
    assert_equal(len(indices), 100)
    assert_equal(len(probs), 100)
    experience = buffer[indices]
    assert_equal(len(experience), 100)
    assert_greater(np.sum(indices==0), 50)

if __name__ == '__main__':
    # Run the tests in this file.
    import sys
    import nose
    module_name = sys.modules[__name__].__file__
 
    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s','-v'])