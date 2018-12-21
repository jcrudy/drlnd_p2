from abc import abstractmethod, ABCMeta
from six import with_metaclass

class Policy(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def choose(self, values):
        '''
        Choose an action based on the given values.
        '''
    
    @abstractmethod
    def reset(self):
        '''
        Reset any internal state of the policy, such as epsilon value.
        '''
