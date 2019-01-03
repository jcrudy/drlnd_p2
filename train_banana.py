from deeprl.environment.unity_adapter import BananaEnvironment
from deeprl.model.fixed_q_target import FixedQTargetModel
from deeprl.model.double_dqn import DoubleDQNModel
from deeprl.agent.base import Agent, AverageReturnThreshold
from deeprl.agent.deepq import DeepQAgent
from deeprl.buffer.uniform_sampling import UniformSamplingReplayBuffer
from deeprl.buffer.prioritized_replay import PrioritizedReplayBuffer
from deeprl.policy.epsilon_greedy import EpsilonGreedyPolicy
from deeprl.network.base import Network
import torch.nn as nn
import torch.nn.functional as F
from toolz import identity
from toolz.functoolz import compose
from torch import optim
from functools import partial

def main(args):
    # Get command line arguments
    num_episodes = args.n
    model_path = args.m
    validate_every = args.v
    validation_episodes = args.e
    save_every = args.s
    save_filename = args.f
    window_size = args.w
    threshold = args.t
    weights_path = args.u
    
    # Create the training environment
    environment = BananaEnvironment()
    
    # Attempt to load the agent.  Create a new agent if loading fails.
    try:
        agent = Agent.from_pickle(model_path)
    except FileNotFoundError:
        hidden_size = 75
        buffer_size = 10000
        network = Network(state_size=environment.state_space.shape[0], 
                          n_actions=environment.action_space.size,
                          layers=(nn.Linear(environment.state_space.shape[0], hidden_size),
                                  nn.Linear(hidden_size, hidden_size),
                                  nn.Linear(hidden_size, environment.action_space.size)),
                          activations=(compose(nn.Dropout(.2), F.relu), 
                                       compose(nn.Dropout(.2), F.relu), 
                                       identity))
        model = DoubleDQNModel(network, 
                               optimizerer=optim.Adam,
                               schedulerer=partial(optim.lr_scheduler.ReduceLROnPlateau, 
                                                   mode='max',
                                                   patience=20,
                                                   verbose=True,
                                                   ),
                               tau=1.)
        
#         buffer = UniformSamplingReplayBuffer(buffer_size)
        buffer = PrioritizedReplayBuffer(buffer_size)
        training_policy = EpsilonGreedyPolicy(1., .95, .005)
        agent = DeepQAgent(model=model, replay_buffer=buffer, 
                  training_policy=training_policy,
                  batch_size=128)
    
    # Train the agent
    agent.train(environment, num_episodes, validate_every=validate_every,
                validation_size=validation_episodes, save_every=save_every,
                save_path=save_filename, 
                early_stopper=AverageReturnThreshold(threshold=threshold, epochs=window_size),
                plot=True, plot_window=window_size)
    
    # Save trained agent to disk
    agent.to_pickle(model_path)
    
    if weights_path is not None:
        agent.save_weights(weights_path)
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a deep Q network')
    parser.add_argument('-m', metavar='<model_path>', 
                        help='The path of the model file.', 
                        default='banana_agent.pkl')
    parser.add_argument('-n', metavar='<num_episodes>', type=int,
                        help='The number of episodes for which to train.',
                        default=1000)
    parser.add_argument('-v', metavar='<validate_every>', 
                        help='The number of episodes between validations.',
                        default=None, type=int)
    parser.add_argument('-e', metavar='<validation_episodes>',
                        help='The number of episodes for which to validate during each validation',
                        default=100, type=int)
    parser.add_argument('-s', metavar='<save_every>',
                        help='The number of episodes between saves.', 
                        default=None, type=int)
    parser.add_argument('-f', metavar='<save_filename>',
                        help='Filename for saving intermediate results.  Will be formatted with number of episodes before saving.',
                        default=None)
    parser.add_argument('-t', metavar='<stopping_threshold>',
                        help='Stop training after an average reward of this much per episode is achieved.',
                        type=int, default=13)
    parser.add_argument('-w', metavar='<stopping_window>',
                        help='Window size to use for reward averaging for early stopping and plotting..',
                        type=int, default=100)
    parser.add_argument('-u', metavar='<weights_path>',
                        help='Path to save weights after training.',
                        default=None)
    
    args = parser.parse_args()
    
    main(args)