from deepq.environment.unity_adapter import BananaEnvironment
import numpy as np
from nose.tools import assert_equal, assert_is_instance  # @UnresolvedImport

def test_banana_environment():
    env = BananaEnvironment()
    state = env.reset(True)
    action = np.random.choice(env.n_actions)
    next_state, reward, done = env.step(action)
    assert_equal(state.shape, next_state.shape)
    assert_is_instance(reward, float)
    assert_is_instance(done, bool)
    env.close()
    env.close()
    
if __name__ == '__main__':
    # Run the tests in this file.
    import sys
    import nose
    module_name = sys.modules[__name__].__file__
 
    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s','-v'])