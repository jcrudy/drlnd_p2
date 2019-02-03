from ppo.agent import MuSigmaLayer, NormalPolicy, Agent
from matplotlib import pyplot as plt
import os

if __name__ == '__main__':
    from ppo.environment.unity_adapter import ReacherV2Environment
    from torch import nn
    
    weights_path = os.path.join('checkpoint.pth')
    plot_path = os.path.join('plot.png')
    
    environment = ReacherV2Environment()
    
    hidden_size = 400
    state_size = environment.state_space.shape[1]
    action_size = environment.action_space.shape[1]
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
    
    actor_model = NormalPolicy(actor_network)
    
    agent = Agent(policy_model=actor_model, value_model=critic_network)
    agent.train(environment, 1000)
    agent.to_pickle(weights_path)
    agent.plot()
    plt.savefig(plot_path)
    plt.show()
    