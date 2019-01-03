from a2c.environment.unity_adapter import ReacherV2Environment,\
    ReacherV1Environment
from torch import nn
from a2c.a2c import A2CAgent
from a2c.ppo import PPOAgent, MuSigmaLayer, NormalPolicy




if __name__ == '__main__':
    environment = ReacherV1Environment()
    hidden_size = 100
    actor_network = nn.Sequential(
                                  nn.Linear(environment.state_space.shape[0], hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, hidden_size),
                                  nn.ReLU(),
                                  MuSigmaLayer(hidden_size, environment.action_space.shape[0])
                                  )
#     critic_network = nn.Sequential(
#                                   nn.Linear(environment.state_space.shape[1], hidden_size),
#                                   nn.ReLU(),
#                                   nn.Linear(hidden_size, hidden_size),
#                                   nn.ReLU(),
#                                   nn.Linear(hidden_size, 1)
#                                   )
    
    actor_model = NormalPolicy(actor_network)
    
    agent = PPOAgent(actor_model)
    
    agent.train(environment, num_epochs=10, plot=True)