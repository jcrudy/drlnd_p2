from deeprl.agent.base import Agent
from abc import abstractmethod, abstractproperty
from infinity import inf


class ActorCriticBase(Agent):
    pass


class ActorCriticDataCollector(object):
    @abstractproperty
    def n_envs(self):
        '''
        The number of environments that will be run simultaneously by the 
        data collector.
        '''
    
    @abstractmethod
    def collect(self, n_steps=inf):
        '''
        Run all environments for one episode or `n_steps` actions, whichever comes first. 
        '''
        

class ActorCriticMultiEnvironmentDataCollector(ActorCriticDataCollector):
    pass

class ActorCriticParallelDataCollector(ActorCriticDataCollector):
    pass


class A2CAgent(ActorCriticBase):
    '''
    
    '''
    
    def test_epoch(self, environment):
        # Initialize the environment and episode variables.
        episode_score = 0.
        state = environment.reset(train=True)
        done = False
        action_count = 0
        while not done:
            
            action = self.actor_network
            
            # Compute the action values for the current state.
            values = self.model.evaluate(state)
            
            # Choose and take an action.
            action = self.testing_policy.choose(numpify(values))
            next_state, reward, done = environment.step(action)
            episode_score += reward
            
            # Update state for next iteration
            state = next_state
            
            action_count += 1
        
        self.test_episode_lengths.append(action_count)
        self.test_scores.append((self.epochs_trained, episode_score))
        return episode_score
    
    

