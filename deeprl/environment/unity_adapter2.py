from .base import Environment, ClosedEnvironmentError
from unityagents import UnityEnvironment
from abc import abstractclassmethod
from . import resources
import os
from multiprocessing import Process, Pipe

def unity_worker(kwargs, connection):
    '''
    Because of a bug in unityagents, it's not possible to run a UnityEnvironment in a 
    python process that has previously closed a UnityEnvironment.  As a workaround,
    we can run the UnityEnvironment in a separate process.
    '''
    print('kwargs = {}'.format(kwargs))
    print(os.getcwd())
    env = UnityEnvironment(**kwargs)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    print('env created')
    while True:
        print('waiting for message')
        method, data = connection.recv()
        if method == 'reset':
            print('reset')
            connection.send(env.reset(train_mode=data).vector_observations[0])
        elif method == 'step':
            print('step')
            env_info = env.step(data)[brain_name]
            connection.send((env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]))
        elif method == 'vector_action_space_size':
            print('vector_action_space_size')
            connection.send(brain.vector_action_space_size) 
        elif method == 'close':
            print('close')
            env.close()
            connection.send(None)
            break
        else:
            raise ValueError('Unknown message.')
    

class UnityBasedEnvironment(Environment):
    @abstractclassmethod
    def path(self):
        '''
        Subclasses should set this to be the path to the 
        desired Unity environment.
        '''
    
#     # It's necessary for UnityEnvironments to have unique worker_ids.  We can 
#     # ensure no two environments in a process share the same worker id by keeping
#     # a count.
#     worker_count = 0
    
#     def __del__(self):
#         UnityBasedEnvironment.worker_count -= 1
    
    def __init__(self, graphics=False, worker_id=0,
                 base_port=5005, seed=0):
        # Attempt to make worker_id differ across processes.  Obviously not guaranteed,
        # especially if each process has multiple workers.
#         print('worker_count = {}'.format(self.worker_count))
#         worker_id = (hash(os.getpid()) % 100) + UnityBasedEnvironment.worker_count
        conn1, conn2 = Pipe()
        self.connection = conn1
        self.process = Process(target=unity_worker, 
                               args=(dict(file_name=self.path, no_graphics=(not graphics),
                                          worker_id=worker_id, base_port=base_port, 
                                          seed=seed), conn2))
        self.process.start()
        self.closed = False
#         self.env = UnityEnvironment(file_name=self.path, no_graphics=(not graphics),
#                                     worker_id=worker_id, base_port=base_port, seed=seed)
#         UnityBasedEnvironment.worker_count += 1
#         self.brain_name = self.env.brain_names[0]
#         self.brain = self.env.brains[self.brain_name]
#         env_info = self.env.reset(train_mode=True)[self.brain_name]
#         example_state = env_info.vector_observations[0]
        example_state = self.reset(True)
        self._state_size = len(example_state)
        self._n_actions = self.get_worker_result('vector_action_space_size')
#          self.brain.vector_action_space_size
        
    
    def get_worker_result(self, method, data=None):
        print('sending {}'.format(method))
        self.connection.send((method, data))
        print('waiting for response...')
        return self.connection.recv()
    
    def reset(self, train):
        if self.closed:
            raise ClosedEnvironmentError('Environment is already closed.')
        return self.get_worker_result('reset', train)
#         self.env.reset(train_mode=train_banana)[self.brain_name]
#         return env_info.vector_observations[0]
    
    def step(self, action):
        if self.closed:
            raise ClosedEnvironmentError('Environment is already closed.')
#         env_info = self.env.step(action)[self.brain_name]
        return self.get_worker_result('step', action)
#         state = env_info.vector_observations[0]
#         reward = env_info.rewards[0] 
#         done = env_info.local_done[0] 
#         return state, reward, done
    
    def close(self):
        if self.closed:
            return
        self.get_worker_result('close')
#         self.env.close()
        self.process.join()
        self.closed = True
    
class BananaEnvironment(UnityBasedEnvironment):
    path = resources.banana

class ReacherV1Environment(UnityBasedEnvironment):
    path = resources.reacher_v1
    
class ReacherV2Environment(UnityBasedEnvironment):
    path = resources.reacher_v2



    