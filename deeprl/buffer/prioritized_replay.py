from .base import ReplayBuffer
import numpy as np
from toolz import first
from ..base import weighted_choice

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, e=1., a=1., b_start=.5, log_b_decay=.99):
        super().__init__(buffer_size)
        self.e = e
        self.a = a
        self.b_start = b_start
        if self.b_start < 0 or self.b_start > 1.:
            raise ValueError('b_start must be in [0,1].')
        self.log_b_decay = log_b_decay
        if self.log_b_decay < 0 or self.log_b_decay > 1.:
            raise ValueError('log_b_decay must be in [0,1].')
        self.logb = np.log(self.b_start)
        self.max_weight = self.e ** self.a
        self.total_weight = 0.
    
    def register_progress(self, agent):
        self.decay_b()
    
    @property
    def b(self):
        return np.exp(self.logb)
    
    def decay_b(self):
        self.logb *= self.log_b_decay
    
    def weights(self):
        return np.array(list(map(first, self.buffer)))

    def sample_indices(self, sample_size):
        weights = self.weights()
        sample = weighted_choice(sample_size, weights)
        coefs = (1. / weights[sample]) ** (self.b / 2.)
        coefs /= np.max(coefs)
        return sample, coefs
    
    def report_errors(self, indices, errors):
        for i, index in enumerate(indices):
            old_weight = self.buffer[index][0]
            new_weight =  (np.abs(errors[i]) + self.e) ** self.a
            self.buffer[index][0] = new_weight
            self.total_weight += new_weight - old_weight
            if new_weight > self.max_weight:
                self.max_weight = new_weight
    
    def append(self, experience):
        if len(self.buffer) == self.buffer_size:
            # In this case, the oldest experience will be dropped,
            # so total_weight and possibly max_weight must be updated.
            drop_weight = self.buffer[0][0]
            self.total_weight -= drop_weight
            if drop_weight == self.max_weight:
                # The weight being dropped is equal to the max_weight,
                # so it's necessary to recalculate max_weight.
                self.max_weight = np.max(self.weights()[1:])
        self.total_weight += self.max_weight
        self.buffer.append([self.max_weight, experience])

    def __getitem__(self, indices):
        return tuple(self.buffer[idx][1] for idx in indices)
    