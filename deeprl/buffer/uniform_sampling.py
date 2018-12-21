from .base import ReplayBuffer
import numpy as np
from collections import deque

class UniformSamplingReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size):
        super().__init__(buffer_size)
        self.buffer = deque(maxlen=self.buffer_size)
    
    def register_progress(self, agent):
        pass
    
    def sample_indices(self, sample_size):
        return np.random.choice(range(len(self)), size=sample_size), np.ones(shape=sample_size) / float(sample_size)
    
    def report_errors(self, indices, errors):
        pass
    
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        self.buffer.append(experience)

    def __getitem__(self, indices):
        return tuple(self.buffer[idx] for idx in indices)
    