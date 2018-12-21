from .base import DoubleNetworkModel
from ..base import torchify32

class FixedQTargetModel(DoubleNetworkModel):
    def target(self, reward, next_state, done):
        q_target_output = (torchify32(~done).unsqueeze(1) * 
            self.q_target.forward(torchify32(next_state)).detach().max(1)[0].unsqueeze(1))
        return torchify32(reward).unsqueeze(1) + self.gamma * q_target_output
    
