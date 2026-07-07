import numpy as np 

"""Agent wrappers for CARLTON network managers.

Paper mapping (arXiv:2402.17773):
- Section III-B: selected action is a channel index.
- Section III-D: each network manager acts as an independent learner.
"""

class Agent(object):
    
    def __init__(self, model, experience_replay_buffer, sensing_window = None, i_d = None,
                 verbose = False):
        """Create an agent that owns model + replay memory.

        Paper reference:
        - Section III-D: each network manager maintains local replay memory and
          learns through value-function updates.
        """
        
        # model == Agent brain
        assert sensing_window != None ,"Please Enter a Valid window sensing size"
        self.sensing_window = sensing_window
        self.model = model
        self.experience_replay_buffer = experience_replay_buffer
        self.current_channel = None 
        self.old_channel = None
        self.verbose = verbose
        self.change_channel_counter = -1
        self.i_d = i_d
        self.net_location = None
        
        
    def update_current_channel(self, channel):
        """Track spectrum mobility statistics for this agent.

        This supports CARLTON's convergence/channel-switch analysis metrics.
        """
        if self.current_channel is not None:
            if self.old_channel != channel: 
                self.old_channel = int(self.current_channel)
                self.change_channel_counter += 1
        self.current_channel = channel
        
    def sample_action(self, state, eps,training):
        """Select a channel action given current state and exploration level.

        Paper reference:
        - Section III-B (action space), Eq. (14)-(15) implemented by model.
        """
        action = self.model.sample_action(state, eps, training)
        self.action = action
        return action
    
    def learn(self, global_ref=None, fedprox_mu=0.0): # experience_replay_buffer
        """Perform one replay-based model update for this agent."""
        cost = self.model.learn(
            self.experience_replay_buffer,
            self.experience_replay_buffer.batch_size,
            global_ref=global_ref,
            fedprox_mu=fedprox_mu,
        )

        return cost
    
    def save_weights(self):
        self.model.save_weights()

        
        
    def __load_weights(self):
        self.model.load_weights()
        
    
    def load_given_weights(self, w):
        self.model.load_given_weights(w)
        if self.verbose: 
            print("Given weights were loaded!")
      
    def get_model_weights(self):
        return self.model.get_state_dict()

    def load_model_weights(self, state_dict):
        self.model.load_state_dict(state_dict)

    def state_dict(self):
        return self.model.get_state_dict()

    def load_state_dict(self, weights):
        self.model.load_state_dict(weights)
            
                
    
class RandomAgent(object):
    def __init__(self, action_space):
        
        self.action_space = action_space
        
    def sample_action(self, state, eps):
        action = np.random.choice(self.action_space)
        return action
    
    def learn(self, data): # experience_replay_buffer
        
        return 0