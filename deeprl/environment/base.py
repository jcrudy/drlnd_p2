from abc import abstractmethod, ABCMeta
from six import with_metaclass

class ClosedEnvironmentError(Exception):
    pass

class Environment(with_metaclass(ABCMeta, object)):
    @property
    def state_size(self):
        return self._state_size
    
    @property
    def n_actions(self):
        return self._n_actions
    
    @abstractmethod
    def reset(self, train):
        '''
        Start a new episode and return the initial state.
        '''
    
    @abstractmethod
    def step(self, action):
        '''
        Take action and return (state, reward, done) tuple.
        '''
    
    @abstractmethod
    def close(self):
        '''
        Close the environment.
        '''
    
#     def __del__(self):
#         self.close()
