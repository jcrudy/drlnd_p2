from deepq.buffer.uniform_sampling import UniformSamplingReplayBuffer
from deepq.environment.base import Environment
import numpy as np
from deepq.agent import Experience
from nose.tools import assert_equal  # @UnresolvedImport

def test_uniform_sampling_replay_buffer():
    
    class FauxEnvironment(Environment):
        _state_size = 10
        _n_actions = 10
        
        def reset(self):
            return np.random.normal(size=10)
        
        def step(self, action):
            return np.random.normal(size=10), np.random.normal(size=1)[0], False if np.random.random() < .9 else True
        
        def close(self):
            pass
    
    np.random.seed(0)
    buffer_size = 10
    buffer = UniformSamplingReplayBuffer(buffer_size)
    env = FauxEnvironment()
    for _ in range(10):
        state = env.reset()
        done = False
        while not done:
            action = np.random.randint(env.n_actions)
            next_state, reward, done = env.step(action)
            experience = Experience(state, action, reward, next_state, done)
            buffer.append(experience)
    indices, probs = buffer.sample_indices(10)
    assert_equal(len(indices), 10)
    assert_equal(len(probs), 10)
    experience = buffer[indices]
    assert_equal(len(experience), 10)

if __name__ == '__main__':
    # Run the tests in this file.
    import sys
    import nose
    module_name = sys.modules[__name__].__file__
 
    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s','-v'])