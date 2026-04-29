
import numpy as np 
import random 

MAX_FRAMES = 10000000   # Total number of frames the agent sees , it was 50 million
MAX_EPISODE_LENGTH = 18000       # Equivalent of 5 minutes of gameplay at 60 frames per second
MAX_EXPERIENCES = 1000000 #500000#  in the paper it is 1 million
MIN_EXPERIENCES =  50000 #5000
IM_SIZE = 84
# K =  #env.action_space.n = 6, we set to 4 since there are only 4 meaningfull actions 
UPDATE_FREQ = 4# it was 4 


class ReplayMemory(object):
    
    def __init__(self, capacity, number_of_channels ,
                 agent_history_length = 4, batch_size = 32):
        self.batch_size = batch_size
        self.size = capacity
        self.current = 0
        self.count = 0
        self.width = 1 # the width of the vector 
        self.agent_history_length = agent_history_length
        self.number_of_channels = number_of_channels
        # Pre-allocate memory 
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype = np.float32)
        self.observations = np.empty(shape  = (self.size, self.number_of_channels,self.width), dtype = np.float32)
        self.terminal_flags =  np.empty(self.size, dtype = np.bool_)
        self.original_decision = np.empty(shape = self.size, dtype = np.int32)   
        
        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty(shape = (self.batch_size, self.agent_history_length, self.number_of_channels, self.width), dtype = np.float32)
        self.new_states = np.empty(shape = (self.batch_size, self.agent_history_length, self.number_of_channels, self.width), dtype= np.float32)
        self.indices = np.empty(self.batch_size, dtype = np.int32)
        
        self.time = np.empty(self.batch_size, dtype = np.float32)
        
        self.asrdot = [None] * 6 # multiply by 5 to inslude original decision 
    
    def define_new_batch_size(self, batch_size):
        self.batch_size = batch_size
        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty(shape = (self.batch_size, self.agent_history_length, self.number_of_channels, self.width), dtype = np.float32)
        self.new_states = np.empty(shape = (self.batch_size, self.agent_history_length, self.number_of_channels, self.width), dtype= np.float32)
        self.indices = np.empty(self.batch_size, dtype = np.int32)
        
        
    def add_experience(self, action, observation, reward, terminal):
        
        if observation.shape != (self.number_of_channels,1):
            print("Observation shape:", observation.shape)
            raise ValueError('Dimension of observation is wrong!')
            
        
        self.actions[self.current] = action 
        self.observations[self.current,...] = observation
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current+ 1)
        self.current = (self.current + 1) % self.size
        
    
    def apply_asrdot(self, value, index):
        "asrdot -> action, state, reward, done, original decision by Amos, time"
        self.asrdot[index] = value
        #print("asrdo:", self.asrdo)
        
        pushing = True
        for i in self.asrdot:
            if i is None:
                pushing = False
                break
                
        # if None not in self.asrdo:
        if pushing:   
            action, observation, reward, terminal,_, _ = self.asrdot
            
            self.original_decision[self.current] = self.asrdot[4]
            self.time[self.current] = self.asrdot[5]
            self.add_experience(action, observation, reward, terminal)

            self.asrdot = [None] * 6
            
            
        
        
    def _get_state(self, index):
        index = int(index)
        if self.count == 0:
            raise ValueError("The replay memory is empty!")
            
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min %s" % str(self.agent_history_length - 1))
        # print(index)
        s = self.observations[index - self.agent_history_length + 1: index + 1, ...]
        return s
    
    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True :
                index = random.randint(self.agent_history_length, self.count - 1)
                # print(index, self.batch_size)
                if index < self.agent_history_length:
                    continue 
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue #  # it is a circular so this tuple is not a valid state
                    
                if self.terminal_flags[index - self.agent_history_length: index].any():
                    continue # we check here that we are not at the boundary of an episode
                    
                break
            # print("hallo:", index)
            self.indices[i] = index
            
    def get_minibatch(self):
        
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')
            
        self._get_valid_indices()
        
        for i, idx in enumerate(self.indices):
         
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)
       
        return np.transpose(self.states, axes = (0,2,3,1)), self.actions[self.indices], \
                self.rewards[self.indices], \
                    np.transpose(self.new_states, axes = (0,2,3,1)),\
                    self.terminal_flags[self.indices],\
                        self.original_decision[self.indices]