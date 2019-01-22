from torch import optim, nn
from functools import partial
from a2c.util import constant, torchify32, numpify, torchify, gae, td_target,\
    rolling_mean
import numpy as np
from abc import abstractmethod, ABCMeta
import torch
from six import with_metaclass
from torch.distributions.normal import Normal
from torch.nn import functional as F
from multipledispatch.dispatcher import Dispatcher
from typing import Iterable
from toolz.functoolz import curry
from matplotlib import pyplot as plt
from tqdm import tqdm

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

tupify = Dispatcher('tupify')

@tupify.register((np.ndarray, object))
def tupify_ndarray(arr):
    return (arr,)

@tupify.register(Iterable)
def tupify_iterable(itr):
    return tuple(itr)

@curry
def select_first_dims(selection, arr):
    return arr[tupify(selection) + (slice(None, None, None),) * (len(arr.shape) - len(selection))]

@curry
def select_batch(expected_batch_size, *args):
    p = expected_batch_size / float(np.prod(args[0].shape[:2]))
    selection = np.where(np.random.binomial(1, p, size=args[0].shape[:2]) > 0)
    return tuple(map(select_first_dims(selection), args))
    
class Agent(object):
    def __init__(self, policy_model, value_model, policy_optimizerer=partial(optim.Adam, lr=3e-4), 
                 value_optimizerer=partial(optim.Adam, lr=3e-4), 
                 value_schedulerer=partial(optim.lr_scheduler.LambdaLR, lr_lambda=constant(1.)),
                 policy_schedulerer=partial(optim.lr_scheduler.LambdaLR, lr_lambda=constant(1.)),
                 gamma=.9, lambda_=0., n_updates_per_episode=10, epsilon=.1, 
                 expected_minibatch_size=15000, policy_clip=None, value_clip=None):
        self.policy_model = policy_model
        self.value_model = value_model
        self.policy_optimizer = policy_optimizerer(self.policy_model.parameters())
        self.policy_scheduler = policy_schedulerer(self.policy_optimizer)
        self.value_optimizer = value_optimizerer(self.value_model.parameters())
        self.value_scheduler = value_schedulerer(self.value_optimizer)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.n_updates_per_episode = n_updates_per_episode
        self.train_episodes = []
        self.train_scores = []
        self.epochs_trained = 0
        self.episodes_trained = 0
        self.epsilon = epsilon
        self.expected_minibatch_size = expected_minibatch_size
        self.policy_clip = policy_clip
        self.value_clip = value_clip
    
#     def collect_trajectory(self, environment):
#         next_state = environment.reset(train=True)
#         done = [False]
#         trajectory = []
#         while not np.all(done):
#             state = torchify32(next_state)
#             action = self.policy_model.sample(state)
#             prob = self.policy_model.prob(state, action)
#             value = self.value_model(state)
# #             value = self.value_model(torch_state)
# #             numpy_action = numpify(action)
#             next_state, reward, done = environment.step(numpify(action))
#             
#             
#             trajectory.append((state, action, prob.detach(), torchify32(reward), torchify(done), ))
#         return list(map(partial(np.stack, axis=1), zip(*trajectory)))
    def collect_trajectory(self, environment):
        next_state = environment.reset(train=True)
        done = [False]
        trajectory = []
        while not np.all(done):
            state = next_state
            torch_state = torchify32(state)
            action = self.policy_model.sample(torch_state)
            prob = self.policy_model.prob(torch_state, action)
            value = self.value_model(torch_state).squeeze(-1)
#             value = self.value_model(torch_state)
            numpy_action = numpify(action)
            next_state, reward, done = environment.step(np.tanh(numpy_action))
            
            trajectory.append((state, numpy_action, numpify(prob), reward, done, numpify(value)))
        return list(map(partial(np.stack, axis=1), zip(*trajectory)))
    
    def plot(self, epochs=None, moving_window=100):
        x = np.array(self.train_episodes)
        y = np.array(self.train_scores)
        y_mean = rolling_mean(y, moving_window)
        if epochs is not None:
            x = x[-epochs:]
            y = y[-epochs:]
            y_mean = y_mean[-epochs:]
        plt.plot(x, y)
        plt.plot(x, y_mean)
        
    
    def train(self, environment, num_epochs=1000):
        for _ in tqdm(range(num_epochs)):
            self.train_step(environment)
#             self.policy_scheduler.step(self.train_scores[-1])
#             self.value_scheduler.step(self.train_scores[-1])
    
    def train_step(self, environment):
        '''
        Assume environment output has shape (n_agents, n_time_steps, ...)
        '''
        # Collect an episode of data for all agents.
        states, actions, old_probs, rewards, dones, old_values = self.collect_trajectory(environment)
        
        # Collect stats.
        n_episodes = states.shape[0]
        total_rewards = np.sum(rewards)
        average_rewards = total_rewards / float(n_episodes)
        
        # Compute TD target and gae.
        td_target_values = td_target(self.gamma, rewards, old_values)
        advantages = gae(self.gamma, self.lambda_, rewards, old_values)
        advantage_means = np.mean(advantages, axis=1, keepdims=True)
        advantage_sds = np.std(advantages, axis=1, keepdims=True)
        normalized_advantages = (advantages - advantage_means) / np.where(advantage_sds > 1e-6, advantage_sds, 1.)
        
        for _ in range(self.n_updates_per_episode):
            # Sample minibatch.
            (batch_states, batch_actions, batch_old_probs, 
             batch_td_targets, batch_advantages) = \
                select_batch(self.expected_minibatch_size, states, actions, old_probs, 
                                  td_target_values, normalized_advantages)
            
            batch_torch_states = torchify32(batch_states)
            batch_torch_td_targets = torchify32(batch_td_targets)
            batch_torch_actions = torchify32(batch_actions)
            
            # Update value model.
            value_loss = torch.mean((self.value_model(batch_torch_states).squeeze(-1) - batch_torch_td_targets) ** 2)
            print('value loss before:', value_loss)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            if self.value_clip is not None:
                nn.utils.clip_grad_norm(self.policy_model.parameters(), self.value_clip)
            self.value_optimizer.step()
            print('value loss after:', torch.mean((self.value_model(batch_torch_states).squeeze(-1) - batch_torch_td_targets) ** 2))
            
            # Update the policy model.
            self.policy_optimizer.zero_grad()
            
            batch_probs = self.policy_model.prob(batch_torch_states, batch_torch_actions)
            batch_ratio = batch_probs / torchify32(batch_old_probs)
            batch_clipped_ratio = batch_ratio.clamp(min = 1. - self.epsilon, max = 1 + self.epsilon)
            batch_torch_advantages = torchify32(batch_advantages)
            policy_loss = -torch.mean(torch.min(batch_ratio * batch_torch_advantages, 
                                          batch_clipped_ratio * batch_torch_advantages))
            print('policy loss before:', policy_loss)
            policy_loss.backward()
            if self.policy_clip is not None:
                nn.utils.clip_grad_norm(self.policy_model.parameters(), self.policy_clip)
            self.policy_optimizer.step()
            batch_probs = self.policy_model.prob(batch_torch_states, batch_torch_actions)
            batch_ratio = batch_probs / torchify32(batch_old_probs)
            batch_clipped_ratio = batch_ratio.clamp(min = 1. - self.epsilon, max = 1 + self.epsilon)
            batch_torch_advantages = torchify32(batch_advantages)
            policy_loss = -torch.mean(torch.min(batch_ratio * batch_torch_advantages, 
                                          batch_clipped_ratio * batch_torch_advantages))
            print('policy loss after:', policy_loss)
        
        self.episodes_trained += n_episodes
        self.epochs_trained += 1
        self.train_scores.append(average_rewards)
        self.train_episodes.append(self.episodes_trained)
        
        
if __name__ == '__main__':
    from a2c.environment.unity_adapter import ReacherV2Environment
    environment = ReacherV2Environment()
    
    hidden_size = 512
    state_size = environment.state_space.shape[1]
    action_size = environment.action_space.shape[1]
    shared_network = nn.Sequential(
                                  nn.Linear(state_size, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, hidden_size),
                                  )
    actor_network = nn.Sequential(
                                  nn.Linear(state_size, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, hidden_size),
                                  nn.ReLU(),
                                  MuSigmaLayer(hidden_size, action_size),
                                  )
    critic_network = nn.Sequential(
                                   nn.Linear(state_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, 1),
                                   )
#     critic_network = nn.Sequential(
#                                   nn.Linear(environment.state_space.shape[1], hidden_size),
#                                   nn.ReLU(),
#                                   nn.Linear(hidden_size, hidden_size),
#                                   nn.ReLU(),
#                                   nn.Linear(hidden_size, 1)
#                                   )
    
    actor_model = NormalPolicy(actor_network)
    
    agent = Agent(policy_model=actor_model, value_model=critic_network)
    agent.train(environment, 1000)
    agent.plot()
    plt.show()
    
    