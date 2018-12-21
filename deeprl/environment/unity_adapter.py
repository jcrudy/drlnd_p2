from .base import Environment, ClosedEnvironmentError
from unityagents import UnityEnvironment
from abc import abstractclassmethod
from . import resources
import os

class UnityBasedEnvironment(Environment):
    @abstractclassmethod
    def path(self):
        '''
        Subclasses should set this to be the path to the 
        desired Unity environment.
        '''
    
    # It's necessary for UnityEnvironments to have unique worker_ids.  We can 
    # ensure no two environments in a process share the same worker id by keeping
    # a count.
    worker_count = 0
    
    def __del__(self):
        UnityBasedEnvironment.worker_count -= 1
    
    def __init__(self, graphics=False):
        # Attempt to make worker_id differ across processes.  Obviously not guaranteed,
        # especially if each process has multiple workers.
        worker_id = (os.getpid() % 100) + UnityBasedEnvironment.worker_count
        self.env = UnityEnvironment(file_name=self.path, no_graphics=(not graphics),
                                    worker_id=worker_id)
        UnityBasedEnvironment.worker_count += 1
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        example_state = env_info.vector_observations[0]
        self._state_size = len(example_state)
        self._n_actions = self.brain.vector_action_space_size
        self.closed = False
    
    def reset(self, train):
        if self.closed:
            raise ClosedEnvironmentError('Environment is already closed.')
        env_info = self.env.reset(train_mode=train)[self.brain_name]
        return env_info.vector_observations[0]
    
    def step(self, action):
        if self.closed:
            raise ClosedEnvironmentError('Environment is already closed.')
        env_info = self.env.step(action)[self.brain_name]
        state = env_info.vector_observations[0]
        reward = env_info.rewards[0] 
        done = env_info.local_done[0] 
        return state, reward, done
    
    def close(self):
        if self.closed:
            return
        self.env.close()
        self.closed = True
    
class BananaEnvironment(UnityBasedEnvironment):
    path = resources.banana
    