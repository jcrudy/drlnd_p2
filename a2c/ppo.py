from torch import nn, optim
from torch.nn import functional as F
from a2c.util import torchify32, numpify, constant, rolling_mean, discount
from functools import partial
from torch.distributions.normal import Normal
from copy import deepcopy
from itertools import repeat
import torch
from six import with_metaclass
from abc import ABCMeta, abstractmethod
from infinity import inf
from matplotlib import pyplot as plt
import numpy as np

class PolicyModel(with_metaclass(ABCMeta, nn.Module)):
    @abstractmethod
    def rv(self, state):
        pass
    
    def sample(self, state, *args, **kwargs):
        return self.rv(state).sample(*args, **kwargs)
    
    def log_prob(self, state, sample):
        return torch.sum(self.rv(state).log_prob(sample), dim=-1)
    
    def prob(self, state, sample):
        return torch.exp(self.log_prob(state, sample))# self.rv(state).prob(sample)
    
class MuSigmaLayer(nn.Module):
    def __init__(self, input_size, output_size):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.mu_layer = nn.Linear(input_size, output_size)
        self.sigma_layer = nn.Linear(input_size, output_size)
    
    def forward(self, input_data):
        mu = self.mu_layer(input_data)
        sigma = F.softplus(self.sigma_layer(input_data))
        return torch.stack([mu, sigma], dim=-1)
    
class NormalPolicy(PolicyModel):
    def __init__(self, network):
        '''
        network (nn.Module): Must end with a MuSigmaLayer or similar.
        '''
        nn.Module.__init__(self)
        self.network = network
    
    def rv(self, state):
        mu_sigma = self.network(state)
        slicer = (slice(None, None, None),) * (len(mu_sigma.shape) - 1)
        mu = mu_sigma[slicer + (0,)]
        sigma = mu_sigma[slicer + (1,)]
        return Normal(mu, sigma)
    
    

class PPOAgent(object):
    def __init__(self, policy_model, optimizerer=partial(optim.Adam, lr=5e-4), 
                 schedulerer=partial(optim.lr_scheduler.LambdaLR, lr_lambda=constant(1.)),
                 gamma=.9, n_trajectories_per_batch=10, n_updates_per_batch=4):
        self.policy_model = policy_model
        self.optimizer = optimizerer(self.policy_model.parameters())
        self.scheduler = schedulerer(self.optimizer)
        self.gamma = gamma
        self.n_trajectories_per_batch = n_trajectories_per_batch
        self.n_updates_per_batch = n_updates_per_batch
        self.train_scores = []
        self.epochs_trained = 0
    
    def collect_trajectory(self, environment):
        next_state = environment.reset(train=True)
        done = False
        trajectory = []
        while not done:
            state = next_state
            torch_state = torchify32(state)
            action = self.policy_model.sample(torch_state)
            prob = self.policy_model.prob(torch_state, action)
            numpy_action = numpify(action)
            next_state, reward, done = environment.step(numpy_action)
            
            trajectory.append((state, numpy_action, numpify(prob), reward, np.array([1 if done else 0])))
        return list(map(partial(np.concatenate, axis=0), zip(*trajectory)))
    
    def surrogate(self, states, actions, probs, rewards, dones):
        new_probs = self.policy_model.prob(torchify32(states), torchify32(actions))
#         weight = new_probs / torchify32(probs
#         discount = self.gamma ** np.arange(rewards.shape[1])
        
        future_discounted_rewards = discount(self.gamma, rewards)
#         (np.sum(rewards, axis=1, keepdims=True) - np.cumsum(discount * rewards, axis=1)) / discount
        
        # Normalize
        sd = np.std(future_discounted_rewards, axis=0, keepdims=True)
        sd = np.where(sd > 1e-6, sd, 1.)
#         sd[np.where(sd < 1e-3)] = 1.
        normalized_future_discounted_rewards = (future_discounted_rewards - np.mean(future_discounted_rewards, axis=0)) / sd
        
        # TODO: Clipping
        return torch.sum(new_probs * torchify32(normalized_future_discounted_rewards / probs))
    
    def train_batch(self, environment):
        trajectories = list(map(self.collect_trajectory, repeat(environment, self.n_trajectories_per_batch)))
        states, actions, probs, rewards, dones = map(partial(np.stack, axis=0), zip(*trajectories))
        for _ in range(self.n_updates_per_batch):
            loss = -self.surrogate(states, actions, probs, rewards, dones)
            loss.backward()
            self.optimizer.step()
            
        return np.sum(rewards)
        
    def train(self, environment, num_epochs=1000, save_every=None, save_path=None, 
              early_stopper=constant(False), plot=False, plot_window=100):
        epoch = 0
        scores = []
        if plot:
            plt.ion()
            graph = plt.plot([0,1], [0,1])[0]
            meangraph = plt.plot([0,1], [0,1])[0]
        while epoch < num_epochs:
            score = self.train_batch(environment)
            
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
    
    
        
        
        
        