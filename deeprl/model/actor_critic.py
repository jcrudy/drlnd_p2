from .base import Model
from ..base import constant, torchify
from torch import optim
from functools import partial


class ActorCriticModel(Model):
    def __init__(self, actor_network, critic_network, actor_optimizerer=partial(optim.Adam, lr=5e-4), 
                 critic_optimizerer=partial(optim.Adam, lr=5e-4), 
                 schedulerer=partial(optim.lr_scheduler.LambdaLR, lr_lambda=constant(1.)), 
                 gamma=.9, tau=1., window=100):
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.actor_optimizerer = actor_optimizerer(self.actor_network.parameters())
        self.critic_optimizerer = critic_optimizerer(self.critic_network.parameters())
        self.scheduler = schedulerer(self.optimizer)
        self.gamma = gamma
        self.tau = tau
        self.window = window
    
    def evaluate(self, state):
        '''
        Returned value is an action.
        '''
        return self.actor_network.forward(torchify(state))
    
    def learn(self, state, action, reward, next_state, done, weight):
        pass
    
    def register_progress(self, agent):
        pass
    
    def save_weights(self, filename):
        pass


