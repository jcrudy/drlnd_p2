from collections import namedtuple
import numpy as np
from tqdm import tqdm
from infinity import inf
from matplotlib import pyplot as plt
import pickle
import pandas
from six import with_metaclass
from abc import ABCMeta, abstractmethod
from toolz import last
from ..base import rolling_mean, numpify
from ..policy.epsilon_greedy import EpsilonGreedyPolicy



Experience = namedtuple("Experience", 
                        field_names=["state", "action", "reward", 
                                     "next_state", "done"])

class EarlyStopper(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def __call__(self, agent):
        '''
        Given an agent, return True if the agent should stop early.
        '''

    def __str__(self):
        '''
        Pleasant default printing for EarlyStoppers.
        '''
        return '{}({})'.format(type(self).__name__, ', '.join(map(lambda x: '{} = {}'.format(*x), self.__dict__.items())))

class NeverStopEarly(EarlyStopper):
    def __call__(self, agent):
        return False
    
    
class AverageReturnThreshold(EarlyStopper):
    def __init__(self, threshold, episodes):
        self.threshold = threshold
        self.episodes = episodes
    
    def __call__(self, agent):
        if len(agent.train_scores) >= self.episodes:
            if np.mean(list(map(last, agent.train_scores[-self.episodes:]))) >= self.threshold:
                print(np.mean(agent.train_scores[-self.episodes:]))
                return True
        return False



class Agent(object):
    def plot_train_scores(self, episodes=inf, window=100):
        '''
        Plot the training scores for the last episodes, including a 
        plot of the average total reward.
        
        Parameters
        ==========
        
        episodes (int >0 or inf): Number of episodes to include.  If inf, all episodes 
            are included.
            
        window (int, >0): Number of episodes to average for the average reward plot.
        
        '''
        x, y = map(np.array, zip(*self.train_scores))
        rolling_y = rolling_mean(y, window)
        idx = x > (self.episodes_trained - episodes)
        x = x[idx]
        y = y[idx]
        rolling_y = rolling_y[idx]
        plt.plot(x, y, label='Training Episode Scores')
        plt.plot(x, rolling_y, label='Training Rolling Average Scores ({})'.format(window))
    
    def plot_test_scores(self, episodes=inf):
        '''
        Plot average rewards for testing simulations against number of training episodes.  
        If no testing simulations have been performed, nothing is done.  Error bars are 
        two standard deviations of the mean.
        
        Parameters
        ==========
        
        episodes (int >0 or inf): Number of episodes to include.  If inf, all episodes 
            are included.
        
        '''
        if not self.test_scores:
            return
        x, y = map(np.array, zip(*self.test_scores))
        idx = x > (self.episodes_trained - episodes)
        x = x[idx]
        y = y[idx]
        df = pandas.DataFrame(dict(x=x, y=y))
        df = df.groupby('x').aggregate([np.mean, np.std, len])
        plt.errorbar(df.index, df[('y', 'mean')], yerr=2 * df[('y', 'std')] / np.sqrt(df[('y', 'len')]), 
                     fmt='r.', ecolor='r', label='Test Scores', zorder=10)

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
    
    def train(self, environment, num_episodes=inf, validate_every=None, validation_size=10,
              save_every=None, save_path=None, early_stopper=NeverStopEarly(), plot=False,
              plot_window=100):
        '''
        Train for num_episodes episodes and return the episode scores.
        
        Parameters
        ==========
        
        environment (deeprl.environment.base.Environment): The environment on which to train_banana.
        
        num_episodes (int >0 or inf): The maximum number of episodes for which to train_banana.
        
        validate_every (int >0 or None): The number of episodes between validations using 
            the testing policy.
        
        validation_size (int >0): The number of episodes per validation.
        
        save_every (int >0 or None): The number of episodes between saving the agent.
        
        save_path (int >0 or None): The path to which to save progress.  Will be formatted
            with number of episodes trained.
        
        early_stopper (callable): A callable that accepts the agent as argument and returns
            True if the agent should stop training early and False otherwise.
        
        plot (bool): If True, plot progress during training using pyplot interactive mode.
        
        plot_window (int >0): If plotting, window size for the average reward plot.
        
        '''
        scores = []
        if plot:
            plt.ion()
            graph = plt.plot([0,1], [0,1])[0]
            meangraph = plt.plot([0,1], [0,1])[0]
        
        with tqdm(total=num_episodes) as t:
            episode = 0
            while episode < num_episodes:
                episode += 1
                
                # Run validation if appropriate
                if (
                    (validate_every is not None) and 
                    (episode % validate_every == 0)
                    ):
                    self.test(environment, validation_size)
                
                # Train for one episode
                episode_score = self.train_episode(environment)
                
                # Inform the model and buffer of our progress
                self.model.register_progress(self)
                self.replay_buffer.register_progress(self)
                
                # Record score from training episode
                scores.append(episode_score)
                
                # Update the progress bar
                t.update(1)
                t.set_description('Last Episode Reward: {}'.format(episode_score))
                
                # Save progress if appropriate
                if (
                    (save_every is not None) and
                    (episode % save_every == 0)
                    ):
                    path = save_path.format(num_episodes)
                    self.to_pickle(path)
                
                # Update live plot
                if plot and episode >= 2:
                    xplot = np.arange(len(scores))
                    graph.set_xdata(xplot)
                    graph.set_ydata(scores)
                    meangraph.set_xdata(xplot)
                    meangraph.set_ydata(rolling_mean(scores, plot_window))
                    plt.xlim(0, num_episodes)
                    lower = min(scores)
                    upper = max(scores)
                    over = (upper - lower) * .1
                    plt.ylim(lower - over, upper + over)
                    plt.draw()
                    plt.pause(0.01)
                
                # Check early stopping conditions
                if early_stopper(self):
                    print('Stopping early after {} episodes due to {}.'.format(self.episodes_trained, 
                                                                               early_stopper))
                    break
        
        if plot:
            plt.ioff()
        return scores
    
    def test(self, environment, num_episodes):
        '''
        Run for num_episodes episodes under the testing policy and return 
        the episode scores.
        
        Parameters
        ==========
        
        environment (deeprl.environment.base.Environment): The environment on which to train_banana.
        
        num_episodes (int >0 or inf): The maximum number of episodes for which to test.
        
        '''
        scores = []
        with tqdm(total=num_episodes) as t:
            for _ in range(num_episodes):
                scores.append(self.test_episode(environment))
                t.update(1)
        return scores
    

