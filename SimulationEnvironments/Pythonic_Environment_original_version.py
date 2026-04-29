import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import numpy as np 
import sys 

class Net():
    def __init__(self, number_of_users, location, net_id, initial_channel  = None, add_noise = False):
        mean, std = location
        self.number_of_users = number_of_users
        self.users = np.random.normal(loc = mean, scale = std, size = (number_of_users,2))
        self.optional_channels = 30 
        
        self.add_noise = add_noise
        ### choose the master 
        self.net_id = net_id
        
        if initial_channel == None:
            self.channel =  np.random.choice(self.optional_channels)
            
        else:
            self.channel = initial_channel 
            
            
        self.master =None 
        min_val = sys.maxsize
        self.intensity_radius = 1.2
        for i in range(number_of_users):
            val  = 0
            p1 = self.users[i,:]
            for j in range(number_of_users):
                if i == j:
                    continue 
                p2 = self.users[j,:]
                
                val += self.calculate_distance(p1,p2)
                
            if val < min_val:
                self.master = i
                min_val = val
        
    def calculate_distance(self,p1,p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p2[1] - p1[1])**2)
    
    
    def define_channel(self, channel):
        self.channel = channel
      
    def create_noise_matrix(self,nets):
        sensed_matrix = np.zeros(shape = (self.number_of_users, self.optional_channels)) - 1
        for net in nets:
            channel = net.channel
            
            if net.net_id == self.net_id:
                continue 
            
            for j in range(len(net.users)):
                uj = net.users[j,:]
                for i in range(self.number_of_users):
                    ui = self.users[i,:]
                    
                    if self.calculate_distance(ui, uj) <= self.intensity_radius:
                        sensed_matrix[i,channel] += 1
                        # print("j:", j)
                        continue 
                    
        return sensed_matrix
    
    def create_sensed_vector(self, nets):
        # This function return the percentage of free noeds at each channel 
        sensed_matrix = self.create_noise_matrix(nets)
        mask = sensed_matrix == -1
        sensed_vec = np.mean(mask , axis = 0)

        if self.add_noise:
            noise = np.random.normal(loc = 0.0, scale = 0.02 , size = sensed_vec.shape)
            # note that the nosie will range between P[mean - 2*scale <nosie< mean + 2*scale] = 95.4%
            # P[mean - 3*scale <nosie< mean + 3*scale] = 99.9%
            # P[-0.04 < noise < 1.04] = 95.4% 
            #  P[-0.06 < noise < 1.06] = 99.9%
            
            unique_values = np.unique(sensed_vec)
            
            for value in unique_values:  
                noise = np.random.normal(loc = 0.0, scale = 0.02)
                sensed_vec[sensed_vec == value] += noise # Now the values can be higher than 1 or lower than 0
            
            ## Perform cliping 
            sensed_vec = np.clip(sensed_vec, 0., 1.)
        
        
        sensed_vec[24:] = -1
        # print("sensed_vec:", sensed_vec)
        return sensed_vec
        
    
    
class python_env(object):
    def __init__(self, number_of_nets: np.int32, number_of_users_in_each_net: np.array,
                 net_center_location_and_std: np.array,
                 add_noise: np.bool,
                 possible_channels = 30,
                 training = False):
        
        self.add_noise = add_noise # noise to the sensed vector
        self.number_of_nets = number_of_nets # int
        self.number_of_users_in_each_net = number_of_users_in_each_net # vector 
        self.net_center_location_and_std = net_center_location_and_std # vec of touple [(mean, std),...]
        
        self.max_length = number_of_nets * 20 + number_of_nets # max(100, number_of_nets * 20)
        self.possible_channels = possible_channels
        self.training = training
        
        if self.training :
            self.used_channels  = []  # to avoid from two nets to wakeup at the same cahnnel 
            
        
    def reset(self):
        self.counter_game_length = 0
        self.nets = []
        for j in range(self.number_of_nets):
            number_of_users = self.number_of_users_in_each_net[j]
            mean, std = self.net_center_location_and_std[j]
            
            net = self.create_net(number_of_users = number_of_users,
                                  location = (mean, std), net_id= j)
            
            self.nets.append(net)
            
        
        idx = np.random.choice(self.number_of_nets)
        net = self.nets[idx]
        obs = net.create_sensed_vector(self.nets)
        info = {'Master_Id': net.net_id,
                'primaryChannel': net.channel}
        return obs, info  
    
    def step(self, action , agent_id):
        self.counter_game_length += 1
        
        if self.counter_game_length >= self.max_length:
            done = True 
        else:
            done = False
            
        net = self.nets[agent_id]
        #print("ddd", net.channel)
        net.define_channel(action)
        #print(action, agent_id,net.channel)
        ### call another agent 
        next_net_idx= (agent_id + 1) % self.number_of_nets
        
        next_net = self.nets[next_net_idx]
        
        info = {'Master_Id': next_net.net_id,
                'primaryChannel': next_net.channel}
        
        obs = next_net.create_sensed_vector(self.nets) 
        
        r = None 
        return obs, r, done, info  
        
        
    def create_net(self, number_of_users, location, net_id):
        
        initial_channel = np.random.choice(self.possible_channels) # only 24 posiible 
        if self.training:
            while initial_channel in self.used_channels:
                initial_channel = np.random.choice(self.possible_channels) 
                
            self.used_channels.append(initial_channel)
        # print(self.used_channels)
        net = Net(number_of_users = number_of_users,
                  location = location, net_id= net_id,
                  initial_channel= net_id,
                  add_noise = self.add_noise)
        
        return net 

    
    def plot_locations_per_nets(self):
        try:
            
            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.size"] = 14
            colors = cm.rainbow(np.linspace(0, 1, self.number_of_nets))
            figure, ax = plt.subplots(1)
            for j in range(self.number_of_nets):#self.nets:
                net = self.nets[j]
                x = net.users[:,0]
                y = net.users[:,1]
                name = net.net_id
                c = colors[j]
                ax.scatter(x = x , y = y, color = c, label = str(name))
                
                for i in range(net.number_of_users):
                    x = net.users[i,0]
                    y = net.users[i,1]
                    r = net.intensity_radius
                    circle = plt.Circle((x,y), r , color=c,  alpha=0.1)
                    circle_line =  ax.add_patch(circle)
            
            plt.legend(frameon = False)
        except:
            print("Make sure to reset the game first by: X.reset()")
        
        
    def create_all_sensed_matrixes(self):
        # This matrix give you all the sensed matrix of all nets at t = t, same time step
        sensed_all = []
        for net in self.nets:
            sensed_matrix = net.create_noise_matrix(self.nets)  
            sensed_all.append(sensed_matrix)
            
        return sensed_all
            

            
# env = python_env(number_of_nets= 3   , number_of_users_in_each_net= [3,5, 10],
#                  net_center_location_and_std = [(1,0.5), (3,0.5), (4,0.2)])

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
    
        
    
    
    