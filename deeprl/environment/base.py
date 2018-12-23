from abc import abstractmethod, ABCMeta, abstractproperty
from six import with_metaclass, class_types, integer_types
import numpy as np
from itertools import product
from toolz.functoolz import compose

class ClosedEnvironmentError(Exception):
    pass

class ActionValueError(Exception):
    '''
    Raised if an action is not a valid value in the relevant action space.
    '''



class ActionSpace(object):
    @abstractmethod
    def validate(self, action):
        '''
        Raise ActionValueError if action is not valid in this action space.
        '''
    
    @abstractproperty
    def shape(self):
        '''
        The shape of the required actions.
        '''
    
    def __contains__(self, action):
        try:
            self.validate(action)
        except ActionValueError:
            return False
        return True

class Interval(ActionSpace):
    def __init__(self, lower, upper, lower_closed=True, upper_closed=True):
        self.lower = lower
        self.upper = upper
        self.lower_closed = lower_closed
        self.upper_closed = upper_closed
        # Require element_type to be defined to instantiate
        type(self).element_type  # @UndefinedVariable
    
    @property
    def shape(self):
        return tuple()
    
    def __str__(self):
        left = '[' if self.lower_closed else '('
        right = ']' if self.upper_closed else ')'
        return left + repr(self.lower) + ',' + repr(self.upper) + right
    
    def validate(self, action):
        if not isinstance(action, type(self).element_type):
            raise ActionValueError('Action {} is not an integer.'.format(action))
        if (action < self.lower or action > self.upper or 
            ((not self.lower_closed) and action == self.lower) or
            ((not self.upper_closed) and action == self.upper)):
            raise ActionValueError('Action {} is not in the {} {}.'.format(action, type(self).__name__,
                                                                                str(self)))

class IntInterval(Interval):
    element_type = tuple(integer_types) + (np.number,)

class FloatInterval(Interval):
    element_type = (float, np.floating)

class ObjectInterval(Interval):
    element_type = class_types
    
    
#     def validate(self, action):
#         if not isinstance(action, integer_types):
#             raise ActionValueError('Action {} is not an integer.'.format(action))
#         if action < self.lower or action > self.upper:
#             raise ActionValueError('Action {} is not in the closed integer interval [{}, {}].'.format(
#                                                                                 self.lower, self.upper))



#     @abstractmethod
#     def example(self):
#         '''
#         Return an example action in this action space.
#         '''
#     
#     @abstractmethod
#     def random(self):
#         '''
#         Return a random action in this action space.
#         '''

class CartesianProductActionSpace(ActionSpace):
    def __init__(self, set_array):
        self.set_array = np.asarray(set_array)
    
    @property
    def shape(self):
        return self.set_array.shape
    
    def validate(self, action):
        action = np.asarray(action)
        if action.shape != self.shape:
            raise ActionValueError('Action shape {} does not match required shape {}.'.format(action.shape, self.set_array.shape))
        
        for coord in product(*map(compose(tuple, range), self.shape)):
            print(coord)
            action_element = action[coord]
            set_element = self.set_array[coord]
            set_element.validate(action_element)
#             if not action_element in set_element:
#                 raise ActionValueError('Action element {} not in set {}.'.format(action_element, set_element))
        
#         for action_element, set_element in np.nditer([action, self.set_array], ['refs_ok']):
#             print(action_element, set_element)
#             if not action_element in set_element:
#                 raise ActionValueError('Action element {} not in set {}.'.format(action_element, set_element))
    
        
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
