from a2c.environment.unity_adapter import ReacherV2Environment
from torch import nn
from a2c.ppo import PPOAgent, MuSigmaLayer, NormalPolicy




if __name__ == '__main__':
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
    
    agent = PPOAgent(policy_model=actor_model, value_model=critic_network)
    
    agent.train(environment=environment, num_epochs=1000, plot=True)