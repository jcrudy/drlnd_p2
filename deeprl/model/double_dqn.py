from .base import DoubleNetworkModel
from ..base import torchify32

class DoubleDQNModel(DoubleNetworkModel):
    def target(self, reward, next_state, done):
        next_state_ = torchify32(next_state)
        action_selections = self.q_local.forward(next_state_).detach().max(1)[1].unsqueeze(1)
        q_target_output = (torchify32(~done).unsqueeze(1) * 
            self.q_target.forward(next_state_).detach().gather(1, action_selections))
        return torchify32(reward).unsqueeze(1) + self.gamma * q_target_output
    
