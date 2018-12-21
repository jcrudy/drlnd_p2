from deepq.agent import Agent, AverageReturnThreshold
from deepq.environment.unity_adapter import BananaEnvironment
from collections import defaultdict

def create_basic_agent():
    pass

def create_double_dqn_agent():
    pass

def create_prioritized_replay_agent():
    pass

def create_prioritized_replay_double_dqn_agent():
    pass

trials_per_agent = 3
agents = [
          ('basic', create_basic_agent),
          ('double_dqn', create_double_dqn_agent),
          ('prioritized_replay', create_prioritized_replay_agent),
          ('prioritized_replay_double_dqn', create_prioritized_replay_double_dqn_agent),
          ]

early_stopper = AverageReturnThreshold()

if __name__ == '__main__':
    
    # Create the training environment
    environment = BananaEnvironment()
    
    
    results = defaultdict(list)
    for name, agent_factory in agents:
        for _ in range(trials_per_agent):
            agent = agent_factory()
            agent.train(environment, 2000,
                        early_stopper=AverageReturnThreshold(threshold=13., episodes=100))
            results[name].append(agent.episodes_trained)