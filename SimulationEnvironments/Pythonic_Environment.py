"""Wireless environment used by CARLTON training/evaluation.

This module implements the wireless simulation described in:
`SINR-Aware Deep Reinforcement Learning for Distributed Dynamic Channel Allocation`
(arXiv:2402.17773).

Paper mapping:
- Section II-A: system and interference model over non-orthogonal channels.
- Eq. (2): SINR definition.
- Eq. (3)-(4): received power and Egli path-loss components.
- Eq. (5)-(6): interference aggregation across other networks/channels.
- Eq. (7): thermal noise contribution.
- Section III-A, Eq. (11)-(12): BSINR thresholding and quality-vector construction.

Implementation note:
- The code keeps the same original behavior and units handling used in this repository.
  Comments below explain the intended paper correspondence at each step.
"""

import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import numpy as np 
import sys 
from SimulationEnvironments.Egli import get_path_loss, get_attenuation
from SimulationEnvironments.Env_Utiles import watts_to_dbm, dbm_to_watts, db_to_watts
import random

THRESHOLD = -100 #dBm
class User(object):
    """Represents one wireless user inside a network.

    Attributes such as antenna gain/height and transmit power are used by the
    Egli-based link budget and interference computations in `Net`.
    """
    def __init__(self, location, identity, power, num_channels):
        self.location = location
        self.user_id = identity                           # tuple: (network_id, user_id)
        self.power = power # This is in dB
        self.num_channels = num_channels
        self.interference = np.zeros(num_channels)
        self.H = 1                                        # Antenna height[m]
        self.G = 1                                        # Antenna gain [m]
        
        
      
class Net():
    """A local network whose manager chooses one operating channel.

    Paper reference:
    - Section II-A: each network operates on a single channel per slot.
    - Eq. (3)-(4): link-budget terms (power minus path loss).
    """
    def __init__(self, number_of_users, location, net_id, 
                 power = 2,optional_channels  = 30, # was 30 
                 initial_channel  = None,
                 set_users_locations = None,
                 add_noise = False):
        
        # power in dB , 2db == 32 dBm
        
        
        self.number_of_users = number_of_users

        
        # self.users_locations1 = np.random.normal(loc = mean_x, scale = std_x, size = (number_of_users,2))
        if set_users_locations is None:
            # print(location,number_of_users)
            self.mean_x, self.mean_y, std_x, std_y = location
            location_x = np.random.normal(loc = self.mean_x, scale = std_x, size = (number_of_users,1))
            location_y = np.random.normal(loc = self.mean_y, scale = std_y, size = (number_of_users,1))
            self.users_locations = np.concatenate((location_x, location_y), axis = 1)
           # self.users_locations = np.random.multivariate_normal(mean = [mean_x, mean_y],
            #                                                   cov = [[std_x**2, 0],[0, std_y**2]],
             #                                                  size = number_of_users)
        else:
            # should be a matrix of size (numebr_of_users x 2)
            # print("set_users_locations:", set_users_locations)
            assert (len(set_users_locations) == self.number_of_users and len(set_users_locations[0]) == 2) , "Make sure that the size of users_locations = (number_of_user, 2)"
            self.users_locations = set_users_locations
            self.mean_x, self.mean_y = np.mean(set_users_locations, axis= 0 )
            
        # print(np.shape(self.users_locations1), np.shape(self.users_locations))
        self.optional_channels = optional_channels 
        # print("optional_channels: Net: ", optional_channels)
        self.add_noise = add_noise
        ### choose the master 
        self.net_id = net_id
        
        if initial_channel == None:
            self.channel =  np.random.choice(self.optional_channels)
            
        else:
            self.channel = initial_channel 
            
        ## create users 
        self.users = []
        
        for i in range(number_of_users):
            if type(power) == int:
                p = power
            elif type(power) == list:
                p = power[i]
            else:
                assert False, " The power variable can be int or list with length of number of users only!"

            user = User(location = self.users_locations[i,:],
                        identity = (net_id,i),
                        power = p,
                        num_channels = self.optional_channels)
            
            self.users.append(user)
            
        
        ## Define the master of the net
        # Section II-B: network-manager concept. The selected manager is the user
        # with minimum total Euclidean distance to other users in the same network.
        self.master =None 
        min_val = sys.maxsize
        self.intensity_radius = 1.2
        for i in range(number_of_users):
            val  = 0
            p1 = self.users[i].location
            for j in range(number_of_users):
                if i == j:
                    continue 
                p2 = self.users[j].location 
                
                val += self.calculate_distance(p1,p2)
                
            if val < min_val:
                self.master = i
                min_val = val
        
        ## calculate the minimal PR
        # Eq. (3): PR = PT - PL. We keep the minimal in-network received power and
        # use it as a conservative signal term for SINR sensing in create_sensed_vector.
        self.pr_min = sys.maxsize
        for  i in range(number_of_users):
            user_1 = self.users[i]
            for j in range(i+1, number_of_users):
                user_2 = self.users[j]
                
                lp = get_path_loss(user_1, user_2, self.channel, 'Egli') # dB
                
                self.pr_min =  min(user_2.power - lp, self.pr_min) # dB

        if self.pr_min == sys.maxsize and number_of_users > 0:
            # Single-user network: no intra-net path for pr_min (Table II allows M=1).
            self.pr_min = float(self.users[0].power)
                
    def calculate_distance(self,p1,p2):
        """Euclidean distance helper used for manager/neighbor selection logic."""
        return np.sqrt((p1[0] - p2[0])**2 + (p2[1] - p1[1])**2)
    
    
    def define_channel(self, channel):
        """Apply the selected channel action for this network.

        Paper reference:
        - Section III-B (Action Space): an action is a channel index in K.
        """
        self.channel = channel
      
    
    
    def create_noise_matrix(self, networks, threshold = None, channel_model = 'Egli'):
        """Compute per-user/per-channel interference matrix in dBm.

        Paper reference:
        - Eq. (5): aggregate interference from all other networks.
        - Eq. (6): per-link interference with spectral attenuation.
        - Eq. (3)-(4): received interference term uses PT - PL (Egli model).
        - Table I: attenuation as function of spectral distance.

        Returns:
            np.ndarray: shape (num_users, num_channels), interference in dBm.
        """
        self.noise_matrix = np.zeros(shape = (self.number_of_users, self.optional_channels), dtype= np.float64)
        for i, user in enumerate(self.users):
            user.interference = np.zeros(self.optional_channels)
            for channel in range(self.optional_channels):
                for inter_net in networks:
                  
                    if inter_net.net_id != self.net_id:    #interference is only with respect to different networks
                        for inter_user in inter_net.users:
                            # Here we calculate interference from inter_user to user
                            # at candidate channel `channel` (Eq. (5)-(6)).
                            LP = get_path_loss(user, inter_user, channel, channel_model)# IN dB
                            PR = inter_user.power - LP ## That is ttrue iff inter_user.power is in dB
                            attunation = get_attenuation(channel, inter_net.channel) # This is a number from a table this is in dBm
                            # print("attunation:", attunation)
                            Attunation = attunation - 30 # 10*np.log10(dbm_to_watts(attunation)) # dB
                            # print("Attunation:", Attunation)
                            # Eq. (6): phi = PR - T(k, k_tilde)
                            intereference_dB = PR - Attunation #dB # THIS IS CURRECT ! But we get the attunation in dBm
                            # print("intereference_dB:",intereference_dB)
                            interference_W = db_to_watts(intereference_dB)
                            # print("interference_W:", interference_W)
                            user.interference[channel] += interference_W
                  
                # print("user.interference[channel]:", user.interference[channel])
                # Sum in Watts for physical correctness, then convert to dBm.
                user.interference[channel] = watts_to_dbm(user.interference[channel])
                # print("user.interference[channel]:", user.interference[channel])
                self.noise_matrix[i,channel] =  user.interference[channel]
        
        return self.noise_matrix #in dbm
    
    def mask_the_vec(self, vec):
        pass

                
        
    def create_sensed_vector(self,networks, threshold = None, channel_model = 'Egli'):
        """Create the network's low-dimensional sensed quality vector.

        Paper reference:
        - Eq. (2): SINR = PR / (I_T + I).
        - Eq. (7): thermal noise term I_T.
        - Eq. (11): BSINR thresholding at target SINR*.
        - Eq. (12): QV_k = average(BSINR over users for channel k).
        - Section III-A: observation-space reduction to a quality vector.

        Implementation note:
        - The code uses a fixed threshold of 4 (dB-like scale) as SINR* surrogate,
          matching the repository's training setup.
        """
        noise_matrix = self.create_noise_matrix(networks, threshold = None, channel_model = 'Egli') #dBm
        # Eq. (7): additive thermal noise contribution.
        thermal_noise = -134.9 # [dB]
       
        noise_matrix_in_watts = dbm_to_watts(noise_matrix) + db_to_watts(thermal_noise) # [W]
        
        # print(noise_matrix_in_watts)
        
        # Eq. (2): SINR-like ratio in linear scale (W/W).
        snir_matrix = db_to_watts(self.pr_min) / noise_matrix_in_watts #[W]

        watts_to_dbm_vectorized = np.vectorize(watts_to_dbm)
        snir_matrix = watts_to_dbm_vectorized(snir_matrix) # dbm
        snir_matrix -= 30 # dB 
        snir_matrix += 30 #dBm # size(number of users x number of channels)
        # snir_matrix = 10*np.log10(snir_matrix) # db
        ## Old option ####################################
        # noise_matrix_binary  = noise_matrix < THRESHOLD 
        # 
       ##################################################### 
        ## new Version 
        # print(np.max(snir_matrix), np.min(snir_matrix))
        # Eq. (11): BSINR = 1(SINR > SINR*). Here SINR* is implemented as 4.
        noise_matrix_binary = snir_matrix > 4 # maybe we should leave it with the SNIR
        ############################################3
        
        # Eq. (12): quality vector QV = average BSINR over users per channel.
        non_interference_vec = np.mean(noise_matrix_binary, axis = 0)
        
        if self.add_noise:
            for i in range(len(non_interference_vec)):
                if non_interference_vec[i] != 0:
                    non_interference_vec += np.random.normal(loc = 0 , scale = 1/(self.number_of_users*3))
            
            non_interference_vec = np.clip(non_interference_vec, 0., 1.)
            
        # masked_non_interference_vec = self.mask_the_vec(non_interference_vec)
        return non_interference_vec
   
class python_env(object):
    """Turn-based multi-network environment for CARLTON.

    Paper reference:
    - Section III-D, Algorithm 4: sequential decisions across networks and
      scenario horizon that scales with number of networks.

    Implementation note:
    - This environment returns observations only. Reward is computed externally
      in `BuildingBlocks/Coordinator.py` to implement CARLTON's two-part reward.
    """
    def __init__(self, number_of_nets: np.int32,
                 number_of_users_in_each_net: np.array,
                 net_center_location_and_std: np.array,
                 add_noise: bool,
                 possible_channels = 30, # was 30 
                 power = 2,
                 users_locations_per_net = None,
                 training = False):
        
        self.add_noise = add_noise # noise to the sensed vector
        self.number_of_nets = number_of_nets # int
        self.number_of_users_in_each_net = number_of_users_in_each_net # vector 
        self.net_center_location_and_std = net_center_location_and_std # vec of touple [(mean, std),...]
        
        self.training = training
        if training:
            self.max_length = number_of_nets * 20 + number_of_nets # max(100, number_of_nets * 20)
            
        else:
            self.max_length = number_of_nets * 20 + number_of_nets
            
        self.possible_channels = possible_channels
        
        
        self.users_locations_per_net = users_locations_per_net
        
        
        self.power = power
        
        if self.training :
            self.used_channels  = []  # to avoid from two nets to wakeup at the same cahnnel 
            
        
    def reset(self):
        """Start a new scenario and return initial observation/info.

        Returns:
            obs: quality vector for the first scheduled network manager.
            info: metadata including manager id, current channel, time, location.
        """
        self.counter_game_length = 0
        self.nets = []
        self.ids_of_all_nets = []        
        assert (self.users_locations_per_net is not None) or (self.net_center_location_and_std is not None), \
            "net_center_location_and_std and users_locations_per_net cannot both be None, you should define one of them properly."
        for j in range(self.number_of_nets):
            number_of_users = self.number_of_users_in_each_net[j]

            
            if self.users_locations_per_net is  None:
                mean_x, mean_y , std_x, std_y = self.net_center_location_and_std[j]

                # print("define net:", j, "location:", mean_x, mean_y, std_x, std_y, "number_of_nets:",
                      # self.number_of_nets)
                net = self.create_net(number_of_users = number_of_users,
                                      location = (mean_x, mean_y, std_x, std_y),
                                      net_id= j)

            else:
               
               set_users_locations = self.users_locations_per_net[j]
             
               net = self.create_net(number_of_users = number_of_users,
                                      location = None,
                                      net_id= j,
                                      set_users_locations = set_users_locations)


            self.nets.append(net)
            self.ids_of_all_nets.append(j)
            # print("bomm")
        
        # Section III-D: decentralized execution with per-network turn order.
        random.shuffle(self.ids_of_all_nets)
        # print(self.ids_of_all_nets)
        self.turn_idx = 0
        idx = self.ids_of_all_nets[self.turn_idx] #np.random.choice(self.number_of_nets)
        net = self.nets[idx]
        
        # for i in range(self.number_of_nets):
        #     print("net:", i, "Non-interference:",np.round(self.nets[i].create_sensed_vector(self.nets),2))
            
        obs = net.create_sensed_vector(self.nets)
        info = {'Master_Id': net.net_id,
                'primaryChannel': net.channel,
                'Time': self.counter_game_length, 
                'Net_location_x_y': np.array([net.mean_x, net.mean_y])}
        
        # self.plot_locations_per_nets()
        
        return obs, info  
    
    def step(self, action , agent_id):
        """Apply one channel-selection action and move to next network turn.

        Args:
            action: selected channel index (Section III-B action space).
            agent_id: id of the acting network manager.

        Returns:
            next observation, placeholder reward (`None`), done flag, and info.
        """
        self.counter_game_length += 1
        
        if self.counter_game_length >= self.max_length:
            done = True 
        else:
            done = False
            
        net = self.nets[agent_id]
        #print("ddd", net.channel)
        # Dynamic channel selection is executed here.
        net.define_channel(action)
        #print(action, agent_id,net.channel)
        ### call another agent
        self.turn_idx = (1 + self.turn_idx) % self.number_of_nets 
        next_net_idx=  self.ids_of_all_nets[self.turn_idx]   #(agent_id + 1) % self.number_of_nets
        # print("next_net_idx:", next_net_idx)
        next_net = self.nets[next_net_idx]
        
        info = {'Master_Id': next_net.net_id,
                'primaryChannel': next_net.channel,
                'Time': self.counter_game_length,
                'Net_location_x_y': np.array([next_net.mean_x, next_net.mean_y])}
        
        obs = next_net.create_sensed_vector(self.nets) 
        
        # Reward is computed in Coordinator (Algorithm 2/3 and Eq. (13)).
        r = None 
        return obs, r, done, info  
        
        
    def create_net(self, number_of_users, location, net_id, set_users_locations = None):
        """Factory for one network instance with randomized initial channel."""
        initial_channel = np.random.choice(self.possible_channels) # only 24 posiible 
        
        if self.training:
            self.used_channels.append(initial_channel)
        #     internal_counter = 0
        #     while (initial_channel in self.used_channels) and (internal_counter < self.possible_channels + 1) :
        #         initial_channel = np.random.choice(self.possible_channels)
        #         internal_counter += 1
                
        #     self.used_channels.append(initial_channel)
        
        # print("joko:", self.used_channels)
        net = Net(number_of_users = number_of_users,
                  location = location, net_id= net_id,
                  power = self.power,
                  initial_channel= initial_channel,# net_id % self.possible_channels,
                  add_noise = self.add_noise,
                  set_users_locations = set_users_locations,
                  optional_channels = self.possible_channels)
        
        return net 

    
    def plot_locations_per_nets(self):
        
        
        try:
            
            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.size"] = 14
            colors = cm.rainbow(np.linspace(0, 1, self.number_of_nets))
            figure, ax = plt.subplots(1)
            for j in range(self.number_of_nets):#self.nets:
                net = self.nets[j]
                x = net.users_locations[:,0]
                y = net.users_locations[:,1]
                name = net.net_id
                c = colors[j]
                ax.scatter(x = x , y = y, color = c, label = str(name))
                ax.text(np.mean(x), np.mean(y), str(name), fontsize=14)
                for i in range(net.number_of_users):
                    x = net.users_locations[i,0]
                    y = net.users_locations[i,1]
                    r = net.intensity_radius
                    circle = plt.Circle((x,y), r , color=c,  alpha=0.1)
                    circle_line =  ax.add_patch(circle)
            
            # plt.legend(frameon = False)
        except:
            print("Make sure to reset the game first by: X.reset()")
        
        # plt.savefig("env.png")
        # plt.close()
    def create_all_sensed_matrixes(self):
        """Return sensed matrices for all networks at current simulation state."""
        # This matrix give you all the sensed matrix of all nets at t = t, same time step
        sensed_all = []
        for net in self.nets:
            sensed_matrix = net.create_noise_matrix(self.nets)  
            sensed_all.append(sensed_matrix)
            
        return sensed_all
            



# # nets = env.nets 

# net0 = nets[0]

# matrix = net0.create_noise_matrix([nets[1]])



# sensed_all = env.create_all_sensed_matrixes()

# obs, info= env.reset()
# env.plot_locations_per_nets()
# agent_id = info['Master_Id']
# channel = info['primaryChannel'] 
# done = False

# while not done:
#     action = np.random.choice(23)
    
#     obs, _, done, info = env.step(action, agent_id)
    
    
#     agent_id = info['Master_Id']
#     channel = info['primaryChannel'] 
    
#     print(obs, agent_id)
    
        
    
    
    