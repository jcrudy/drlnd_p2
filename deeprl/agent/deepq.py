from .base import Agent
from ..policy.epsilon_greedy import EpsilonGreedyPolicy
from ..base import numpify
from .base import Experience
import numpy as np

class DeepQAgent(Agent):
    '''
    Parameters
    ==========
    
    model (deeprl.model.base.Model): The model used to learn the state-action value 
        function.
    
    replay_buffer (deeprl.buffer.base.ReplayBuffer): The buffer that will be used to 
        store and sample experience during training.
        
    training_policy (deeprl.policy.base.Policy): The policy used to choose actions
        during training.
    
    learn_every (int, >0): The number of actions to perform between learning steps.
    
    batch_size (int, >0): The number of experiences to sample during each learning
        step.
    
    '''
    def __init__(self, model, replay_buffer, training_policy, 
                 testing_policy=EpsilonGreedyPolicy(0, 0, 0), 
                 learn_every=4, batch_size=64):
        Agent.__init__(self)
        self.model = model
        self.replay_buffer = replay_buffer
        self.training_policy = training_policy
        self.testing_policy = testing_policy
        self.learn_every = learn_every
        self.batch_size = batch_size
        self.train_episode_lengths = []
        self.test_episode_lengths = []
        
    def save_weights(self, filename):
        self.model.save_weights(filename)
    
    def learn(self):
        '''
        Sample from the replay buffer and call the model's learn method with the 
        resulting data.
        '''
        # Sample from the replay buffer.
        sample_indices, weights = self.replay_buffer.sample_indices(self.batch_size)
        sample = self.replay_buffer[sample_indices]
        
        # Convert the sample to the form required by the model.
        state, action, reward, next_state, done = map(np.array, zip(*sample))
        
        # Learn from the sample.
        error = self.model.learn(state, action, reward, next_state, done, weights)
        
        # Inform the buffer of the errors for the sample.  Necessary for prioritized 
        # sampling.
        self.replay_buffer.report_errors(sample_indices, error)
        
    
    
    def train_epoch(self, environment):
        '''
        Execute an epoch of training and return the total reward.
        '''
        # Initialize the environment and epoch variables.
        episode_score = 0.
        state = environment.reset(train=True)
        done = False
        action_count = 0
        while not done:
            # Compute the action values for the current state.
            values = self.model.evaluate(state)
            
            # Choose and take an action.
            action = self.training_policy.choose(numpify(values))
            next_state, reward, done = environment.step(action)
            episode_score += reward
            self.t += 1
            
            # Store experience for later learning.
            experience = Experience(state, action, reward, next_state, done)
            self.replay_buffer.append(experience)
            
            # Update state for next iteration
            state = next_state
            
            # Learn from recorded experiences.
            if self.t % self.learn_every == 0 and len(self.replay_buffer) >= self.batch_size:
                self.learn()
                
            action_count += 1
        
        self.train_episode_lengths.append(action_count)
        self.train_scores.append((self.epochs_trained, episode_score))
        self.epochs_trained += 1
        return episode_score
    
    def test_epoch(self, environment):
        '''
        Execute an episode under the testing policy.
        '''
        # Initialize the environment and episode variables.
        episode_score = 0.
        state = environment.reset(train=True)
        done = False
        action_count = 0
        while not done:
            # Compute the action values for the current state.
            values = self.model.evaluate(state)
            
            # Choose and take an action.
            action = self.testing_policy.choose(numpify(values))
            next_state, reward, done = environment.step(action)
            episode_score += reward
            
            # Update state for next iteration
            state = next_state
            
            action_count += 1
        
        self.test_episode_lengths.append(action_count)
        self.test_scores.append((self.epochs_trained, episode_score))
        return episode_score
