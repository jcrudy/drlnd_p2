from torch import nn, optim
from torch.nn import functional as F
from a2c.util import torchify32, numpify, constant, rolling_mean, discount,\
    td_target, gae_from_td_target, gae, torchify
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
    def __init__(self, policy_model, value_model, policy_optimizerer=partial(optim.Adam, lr=5e-4), 
                 value_optimizerer=partial(optim.Adam, lr=5e-4), 
                 value_schedulerer=partial(optim.lr_scheduler.LambdaLR, lr_lambda=constant(1.)),
                 policy_schedulerer=partial(optim.lr_scheduler.LambdaLR, lr_lambda=constant(1.)),
                 gamma=.9, lambda_=.1, n_trajectories_per_batch=5, n_updates_per_batch=3, epsilon=.1, 
                 expected_minibatch_size=10000, policy_clip=1., value_clip=None):
        self.policy_model = policy_model
        self.value_model = value_model
        self.policy_optimizer = policy_optimizerer(self.policy_model.parameters())
        self.policy_scheduler = policy_schedulerer(self.policy_optimizer)
        self.value_optimizer = value_optimizerer(self.value_model.parameters())
        self.value_scheduler = value_schedulerer(self.value_optimizer)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.n_trajectories_per_batch = n_trajectories_per_batch
        self.n_updates_per_batch = n_updates_per_batch
        self.train_scores = []
        self.epochs_trained = 0
        self.epsilon = epsilon
        self.expected_minibatch_size = expected_minibatch_size
        self.policy_clip = policy_clip
        self.value_clip = value_clip
    
    def collect_trajectory(self, environment):
        next_state = environment.reset(train=True)
        done = False
        trajectory = []
        while not done:
            state = next_state
            torch_state = torchify32(state)
            action = self.policy_model.sample(torch_state)
            prob = self.policy_model.prob(torch_state, action)
#             value = self.value_model(torch_state)
            numpy_action = numpify(action)
            next_state, reward, done = environment.step(numpy_action)
            
            trajectory.append((state, numpy_action, numpify(prob), reward, np.array([1 if done else 0])))
        return list(map(partial(np.concatenate, axis=0), zip(*trajectory)))
    
    def collect_multitrajectory(self, environment):
        next_state = environment.reset(train=True)
        done = [False]
        trajectory = []
        while not np.all(done):
            state = next_state
            torch_state = torchify32(state)
            action = self.policy_model.sample(torch_state)
            prob = self.policy_model.prob(torch_state, action)
#             value = self.value_model(torch_state)
            numpy_action = numpify(action)
            next_state, reward, done = environment.step(numpy_action)
            
            trajectory.append((state, numpy_action, numpify(prob), reward, done))
        return list(map(partial(np.stack, axis=1), zip(*trajectory)))
    
    def value_targets(self, states, actions, probs, rewards, dones, values):
#         values = self.value_model(torchify32(states)).squeeze(-1)
#         numpy_values = numpify(values)
        targets = td_target(self.gamma, rewards, values)
        return targets
    
#     def value_loss(self, states, actions, probs, rewards, dones, values):
# #         values = self.value_model(torchify32(states)).squeeze(-1)
#         targets = self.value_target(states, actions, probs, rewards, dones, values)
#         value_loss = torch.sum((torchify32(targets) - values) ** 2)
#         return value_loss
#     
    def advantages(self, states, actions, probs, rewards, dones, values):
        advantages = gae(self.gamma, self.lambda_, rewards, values)
        return advantages
    
#     def policy_loss(self, states, actions, probs, rewards, dones, advantages):
#         new_probs = self.policy_model.prob(torchify32(states), torchify32(actions))
# #         values = self.value_model(torchify32(states)).squeeze(-1)
# # #         weight = new_probs / torchify32(probs
# # #         discount = self.gamma ** np.arange(rewards.shape[1])
# # #         numpy_values = numpify(values)
# # #         targets = td_target(self.gamma, rewards, numpy_values)
# #         advantages = gae(self.gamma, self.lambda_, rewards, values)
# #         gae_from_td_target(self.gamma, self.lambda_, targets, numpy_values)
#         
#         
# #         future_discounted_rewards = discount(self.gamma, rewards)
# #         (np.sum(rewards, axis=1, keepdims=True) - np.cumsum(discount * rewards, axis=1)) / discount
#         
#         # Normalize
# #         sd = np.std(future_discounted_rewards, axis=0, keepdims=True)
# #         sd = np.where(sd > 1e-6, sd, 1.)
# # #         sd[np.where(sd < 1e-3)] = 1.
# # #         normalized_future_discounted_rewards = (future_discounted_rewards - np.mean(future_discounted_rewards, axis=0)) / sd
# #         advantages = future_discounted_rewards - numpify(values.squeeze(dim=2))
# #         sd = np.std(future_discounted_rewards, axis=0, keepdims=True)
# #         sd = np.where(sd > 1e-6, sd, 1.)
# #         normalized_advantages = (advantages - np.mean(advantages, axis=0)) / sd
#         prob_ratio = new_probs / torchify32(probs)
#         clipped_prob_ratio = torch.min(torchify32(1+self.epsilon), torch.max(torchify32(1-self.epsilon), prob_ratio)[0])[0]
# #         normalized_future_discounted_rewards = torchify32(normalized_advantages)
# #         normalized_advantages =  torchify32(normalized_advantages)
#         unclipped_policy_loss = torch.sum(prob_ratio * torchify32(advantages))
#         
#         clipped_policy_loss = torch.sum(clipped_prob_ratio * torchify32(advantages))
#         policy_loss = torch.min(clipped_policy_loss, unclipped_policy_loss)[0]
#         return policy_loss
#         
    
    
    def train_batch(self, environment=None, environment=None):
        if environment is not None:
            trajectories = list(map(self.collect_trajectory, repeat(environment, self.n_trajectories_per_batch)))
            states, actions, probs, rewards, dones = map(partial(np.stack, axis=0), zip(*trajectories))
        else:
            trajectories = list(map(self.collect_multitrajectory, repeat(environment, self.n_trajectories_per_batch)))
            states, actions, probs, rewards, dones = map(partial(np.concatenate, axis=0), zip(*trajectories))
        
        for _ in range(self.n_updates_per_batch):
            
            # Compute value targets.
            values = self.value_model(torchify32(states)).squeeze(-1)
            numpy_values = numpify(values)
            value_targets = self.value_targets(states, actions, probs, rewards, dones, numpy_values)
            
            # Select the subset for this minibatch.
            selection = np.random.binomial(1, self.expected_minibatch_size / float(np.prod(rewards.shape)),
                                           size=rewards.shape)
            torch_selection = torchify32(selection)
            
            # Update value function.
            self.value_optimizer.zero_grad()
            value_loss = torch.sum(torch_selection * ((values - torchify32(value_targets)) ** 2))
            value_loss.backward()
            if self.value_clip is not None:
                nn.utils.clip_grad_norm(self.policy_model.parameters(), self.value_clip)
            self.value_optimizer.step()
            
            # Calculate advantages.
            values = self.value_model(torchify32(states)).squeeze(-1)
            numpy_values = numpify(values)
            advantages = self.advantages(states, actions, probs, rewards, dones, numpy_values)
            torch_advantages = torchify32(advantages)
            
            # Calculate probability ratios.
            new_probs = self.policy_model.prob(torchify32(states), torchify32(actions))
            prob_ratios = new_probs / torchify32(probs)
            clipped_prob_ratios = torch.min(torchify32(1+self.epsilon), torch.max(torchify32(1-self.epsilon), prob_ratios))
            
            # Update policy.
#             unclipped_policy_loss = prob_ratios[selection] * torch_advantages[selection]
#             clipped_policy_loss = clipped_prob_ratios[selection] * torch_advantages[selection]
            policy_loss = - torch.sum(
                            torch_selection * 
                            (torch.min(
                                    prob_ratios * torch_advantages,
                                    clipped_prob_ratios * torch_advantages,
                                    )
                             )
                            )
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            if self.policy_clip is not None:
                nn.utils.clip_grad_norm(self.policy_model.parameters(), self.policy_clip)
            self.policy_optimizer.step()
            
            
#             loss = -self.surrogate(states, actions, probs, rewards, dones)
#             loss.backward()
#             nn.utils.clip_grad_norm(self.policy_model.parameters(), 1.)
#             self.optimizer.step()
            
        return np.mean(np.sum(rewards, axis=1))
        
    def train(self, environment=None, environment=None, num_epochs=1000, save_every=None, save_path=None, 
              early_stopper=constant(False), plot=False, plot_window=100, scheduler_window=100):
        if environment is None and environment is None:
            raise ValueError()
        if environment is not None and environment is not None:
            raise ValueError()
        
        epoch = 0
        scores = []
        if plot:
            plt.ion()
            graph = plt.plot([0,1], [0,1])[0]
            meangraph = plt.plot([0,1], [0,1])[0]
        while epoch < num_epochs:
            score = self.train_batch(environment=environment, environment=environment)
            
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
            
            self.policy_scheduler.step(rolling_mean(scores, scheduler_window))
            self.value_scheduler.step(rolling_mean(scores, scheduler_window))
            
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
    
    
        
        
        
        