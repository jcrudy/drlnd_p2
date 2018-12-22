from deeprl.policy.epsilon_greedy import EpsilonGreedyPolicy
import numpy as np
from numpy.testing.nose_tools.utils import assert_approx_equal

def test_epsilon_greedy_policy():
    np.random.seed(0)
    q_values = np.random.normal(size=10)
    policy = EpsilonGreedyPolicy(1., .9, .1)
    for i in range(20):
        policy.choose(q_values)
        assert_approx_equal(policy.epsilon, max(.9 ** (i+1), .1))
    
    best_choices = 0
    best_choice = np.argmax(q_values)
    n_trials = 100000
    policy = EpsilonGreedyPolicy(.9, 1., .9)
    for i in range(n_trials):
        best_choices += int(policy.choose(q_values) == best_choice)
    assert_approx_equal(best_choices / float(n_trials), .19, significant=2)

if __name__ == '__main__':
    # Run the tests in this file.
    import sys
    import nose
    module_name = sys.modules[__name__].__file__
 
    result = nose.run(argv=[sys.argv[0],
                            module_name,
                            '-s','-v'])