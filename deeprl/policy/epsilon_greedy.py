from .base import Policy
import numpy as np

class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon_start, epsilon_decay, epsilon_min):
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.reset()
    
    def reset(self):
        self.epsilon = self.epsilon_start
    
    def choose(self, values):
        if np.random.random() < self.epsilon:
            result = np.random.choice(len(values))
        else:
            result = np.random.choice(np.where(np.max(values) == values)[0])
        
        # Update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        return result
