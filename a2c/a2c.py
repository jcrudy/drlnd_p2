from copy import deepcopy
from torch import optim
from functools import partial
from .util import constant
from infinity import inf
import pickle
from matplotlib import pyplot as plt
import numpy as np
from a2c.util import rolling_mean, ModuleStack, numpify, torchify
from itertools import repeat
import torch
from .util import torchify32
from toolz.functoolz import compose, curry

class A2CAgent(object):
    def __init__(self, actor_network, critic_network, actor_optimizerer=partial(optim.Adam, lr=5e-4), 
                 critic_optimizerer=partial(optim.Adam, lr=5e-4), 
                 actor_schedulerer=partial(optim.lr_scheduler.LambdaLR, lr_lambda=constant(1.)), 
                 critic_schedulerer=partial(optim.lr_scheduler.LambdaLR, lr_lambda=constant(1.)), 
                 gamma=.9, tau=1., window=100):
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.actor_optimizerer = actor_optimizerer
        self.critic_optimizerer = critic_optimizerer
        self.actor_schedulerer = critic_schedulerer#(self.actor_optimizer)
        self.critic_schedulerer = critic_schedulerer#(self.critic_optimizer)
        self.gamma = gamma
        self.tau = tau
        self.window = window
        self.epochs_trained = 0
        self.train_scores = []
    
    def to_pickle(self, filename):
        with open(filename, 'wb') as outfile:
            pickle.dump(self, outfile)
    
    @classmethod
    def from_pickle(cls, filename):
        with open(filename, 'rb') as outfile:
            result = pickle.load(outfile)
        if not isinstance(result, cls):
            raise TypeError('Unpickled object is not correct type.')
        return result
    
    def train(self, environment, num_epochs=inf, save_every=inf, save_path=None, 
              early_stopper=constant(False), plot=False, plot_window=100):
        '''
        The environment is assumed to be a replicated environment, with the first axis representing 
        replications.
        '''
        self.n_environment_copies = environment.action_space.shape[0]
        if (not hasattr(self, 'actor_stack')) or len(self.actor_stack) != self.n_environment_copies:
            self.actor_stack = ModuleStack.repeat(self.actor_network, self.n_environment_copies)
            self.critic_stack = ModuleStack.repeat(self.critic_network, self.n_environment_copies)
            self.actor_optimizer = self.actor_optimizerer(self.actor_stack.parameters())
            self.critic_optimizer = self.critic_optimizerer(self.critic_stack.parameters())
            self.actor_scheduler = self.actor_schedulerer(self.actor_optimizer)
            self.critic_scheduler = self.critic_schedulerer(self.critic_optimizer)
#         if len(self.actor_network_copies) != self.n_environment_copies:
#             self.actor_network_copies = tuple(map(deepcopy, repeat(self.actor_network, self.n_environment_copies)))
#             self.actor_optimizers = tuple(map(self.optimizerer, self.actor_network_copies))
#             self.critic_network_copies = tuple(map(deepcopy, repeat(self.critic_network, self.n_environment_copies)))
#             self.critic_optimizers = tuple(map(self.optimizerer, self.critic_network_copies))
            
        epoch = 0
        scores = []
        if plot:
            plt.ion()
            graph = plt.plot([0,1], [0,1])[0]
            meangraph = plt.plot([0,1], [0,1])[0]
        while epoch < num_epochs:
            
            score = self.train_epoch(environment)
            
            self.train_scores.append(score)
            scores.append(score)
            epoch += 1
            self.epochs_trained += 1
            
            # Save progress if appropriate
            if (
                (save_every is not None) and
                (epoch % save_every == 0)
                ):
                path = save_path.format(num_epochs)
                self.to_pickle(path)
            
            # Update live plot
            if plot and epoch >= 2:
                xplot = np.arange(len(scores))
                graph.set_xdata(xplot)
                graph.set_ydata(scores)
                meangraph.set_xdata(xplot)
                meangraph.set_ydata(rolling_mean(scores, plot_window))
                plt.xlim(0, num_epochs)
                lower = min(scores)
                upper = max(scores)
                over = (upper - lower) * .1
                plt.ylim(lower - over, upper + over)
                plt.draw()
                plt.pause(0.01)
    
    def collect_epoch(self, environment, training):
        next_states = environment.reset(train=True)
        dones = [False]
        results = []
        values = None
        while not all(dones):
            states = next_states
            actions = self.actor_stack(torchify32(states))
            if training:
                values = self.critic_stack(torchify32(states))
            next_states, rewards, dones = environment.step(numpify(actions))
#             next_values = self.critic_stack(next_states)
            
            results.append((states, actions, rewards, dones) + ((values,) if training else tuple()))
        return results
    
    
    def accumulate_episode(self, environment):
        total_score = 0.
        epoch_data = self.collect_epoch(environment, training=True)
        states, actions, rewards, dones, values = zip(*epoch_data)
        states = torchify32(np.array(states))
        actions = torch.stack(actions, dim=0)
        rewards = torchify32(np.array(rewards))
        dones = torchify(np.array(dones, dtype=np.uint8))
        values = torch.stack(values, dim=0)
        
        td_values = rewards + self.gamma * torch.cat([values[1:,:,:], torch.zeros(1, values.shape[1], values.shape[2])], dim=0)
        
        critic_loss = torch.sum((values - td_values) ** 2)
#         self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
#         advantage = 
        
        self.critic_optimizer.step()
        
        
        
        1+1
        
        
        
#         episode_score = 0.
#         states = environment.reset(train=True)
#         
#         dones = [False]
#         actions = np.empty(shape=environment.action_space.shape)
#         while not all(dones):
#             actions = torch.stack(self.actor_network_copies)(state)
#             predicted_value_before = torch.stack(self.critic_network_copies)(state)
#             
#             next_state, reward, done = environment.step(actions)
#             
#             predicted_value_after = torch.stack(self.critic_network_copies)(next_state)
#             
#             advantage = (reward + self.gamma * predicted_value_after) - predicted_value_before
#             loss = -torch.sum(advantage, dim=0)
#             
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
            
            
            
    
# class A2CWorker(object):
#     def __init__(self, model, gamma):
#         self.model = model
#         self.gamma = gamma
#     
#     def __call__(self, state, action, reward, next_state, done, weight):
        