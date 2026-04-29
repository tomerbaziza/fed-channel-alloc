
import numpy as np 
from DeepMellow_Single_agent.Nets_keras import AnnResNet_tunable, AnnResNet
from DeepMellow_Single_agent.DeepMellow_no_epsilon import DeepMellow
from DeepMellow_Single_agent.ReplayMemory import ReplayMemory 
from BuildingBlocks.Agent import Agent 
import tensorflow as tf 
import os, signal
import pickle 
import matplotlib.pyplot as plt 



def keep_only_bat_files(all_files):
    scenarios = []
    for i in all_files:
        filename, file_extansion = os.path.splitext(i)
        if file_extansion == '.bat' and '(' in filename:
            scenarios.append(i)
            
    return scenarios



def add_memory_to_general_storage(agent):
    rmb = agent.experience_replay_buffer
    
    if not os.path.isdir('Memory_Storage'):
        os.makedirs("Memory_Storage")
        num_of_channels , num_of_bits = 30, 5  
        history_length = 2 
        rmb =  {"actions":[],
                "rewards": [],
                "observation":[],
                "terminal_flags": [],
                "oiginal_decison": [],}
        
                # "actions" = np.empty(self.size, dtype=np.int32)
                # self.rewards = np.empty(self.size, dtype = np.float32)
                # self.observations = np.empty(shape  = (self.size, self.number_of_channels,self.width), dtype = np.float32)
                # self.terminal_flags =  np.empty(self.size, dtype = np.bool_)
                # self.original_decision = np.empty(shape = self.size)   
        
        with open('Memory_Storage/general_replayMemory.pk', 'wb') as file:
            pickle.dump(rmb, file)
     
    else:
        with open('Memory_Storage/general_replayMemory.pk', 'rb') as file:
            rmb = pickle.load(file)
            
            
    rmb["actions"].append(agent.experience_replay_buffer.actions)
    rmb["rewards"].append(agent.experience_replay_buffer.rewards)
    rmb["observations"].append(agent.experience_replay_buffer.observations)
    rmb["terminal_flags"].append(agent.experience_replay_buffer.terminal_flags)
    rmb["original_decision"].apped(agent.experience_replay_buffer.original_decision)
    
    with open('Memory_Storage/general_replayMemory.pk', 'wb') as file:
        pickle.dump(rmb, file)
    
    
    print("Memory was added succesfully!")

def save_all_agents_memory(agents_as_dir):
    # The agents shold be a dictionary 
    for key in agents_as_dir.keys():
        agent = agents_as_dir[key]
        add_memory_to_general_storage(agent)
    

def load_agent():# Here I want to load a tained agent 
    if not os.path.isdir("Saved_Agents"):
        return False
    
    else:
        
        options = os.listdir("Saved_Agents/")
        idx = np.random.choice(len(options))
        agent_name = "Saved_Agents/" + options[idx] 
        
        with open(agent_name,'rb') as file:
            agent = pickle.load(file)
        return agent 
        
def saveAgent(agent, agent_id):
    if not os.path.isdir("Saved_Agents"):
        os.makedirs("Saved_Agents")
    
    name = "Saved_Agents/Aget"+ str(agent_id)+".pk"
    with open(name, 'wb') as file:
        pickle.dump(agent, file)

    print("Agent ", str(agent_id), "was saved succesfully.")
    
def creat_player(number_of_actions = 30,
                 history_length = 1,
                 learning_rate= 0.00025,
                 max_experience = 10000,
                 batch_size = 64, 
                 sensing_window = 5,
                 number_of_layers = 3,
                 number_of_nodes = 128,
                 activation_fucntion = tf.nn.leaky_relu,
                 mellowmax_constant = 0.02,
                 gamma = 0.9,
                 dropout = False,
                 l2_regularization=False,
                 i_d = 9999,
                 i_d_folder = '',
                 verbose = False):
    
    ############Load trained one######3
    """load_agent()"""
    num_of_channels = number_of_actions
    num_of_bits = int(np.floor(np.log2(num_of_channels)) + 1)
    
    # net = AnnResNet(input_shape = (history_length, num_of_channels+num_of_bits),
    #                 K = number_of_actions) <---""" I should chanhe it to be channgable """
    
    net = AnnResNet_tunable(input_shape = (history_length, num_of_channels+num_of_bits),
                 K = number_of_actions,
                 number_of_layers= number_of_layers,
                 number_of_nodes = number_of_nodes,
                 activation = activation_fucntion,
                 dropout = dropout)
    
    model = DeepMellow(net = net, 
                       gamma = gamma,
                       number_of_actions = number_of_actions,
                       lr = learning_rate,
                       optimizer = tf.keras.optimizers.Adam,
                       los_func = tf.keras.losses.Huber(),
                       mellowmax_constant = mellowmax_constant,
                       l2_regularization = l2_regularization)
    
    experience_replay_buffer = ReplayMemory(capacity = max_experience,
                                             number_of_channels = num_of_channels +num_of_bits,
                                             agent_history_length  = history_length,
                                             batch_size = batch_size)
    agent = Agent(model = model,
                  sensing_window = sensing_window,
                  experience_replay_buffer = experience_replay_buffer,
                  i_d = i_d,
                  verbose = verbose)
    
    ## check for weights update 
    weight_folder_name = 'Train_weights_' + str(i_d_folder)
    
    if os.path.isdir(weight_folder_name ):
        # load the trained weights (the last one)
        files = sorted(os.listdir(weight_folder_name + "/"), key = lambda t: os.stat(weight_folder_name + "/" + t).st_mtime)
        latest_weights_name = files[-1]
        path = weight_folder_name + '/' + latest_weights_name
        
        with open(path, 'rb') as file:
            weights = pickle.load(file)
        
        agent.load_given_weights(weights)
        
    return agent


def update_state(state, obs):
    # print(state.shape)
    # print(obs.shape)# something is wrong here 
    return np.append(state[:,:,1:], np.expand_dims(obs, axis =2 ), axis = 2)

def convert_to_base_2(x, num_of_bits = 5):
    y = [np.float32(i) for i in np.binary_repr(x)]
    # print("0:", y, "x:", x)
    y_size = len(y)
    diff = num_of_bits - y_size
    y = [0]*diff + y    
    y = np.asarray(y)

    # print("1: ", y )
    return np.expand_dims(y, axis = 1)

def modify_obs_add_channnel_2b(obs, num_of_bits = 0, channel = None):
    if num_of_bits == 0:
        return obs
    
    master_initial_channel = channel
    channel_as_bin = convert_to_base_2(master_initial_channel, num_of_bits)
    #print(channel_as_bin, channel_as_bin.shape, np.shape(obs))
    obs_next = np.concatenate((channel_as_bin, obs), axis = 0)
    return obs_next


class Choose_same_number_twice():
    def __init__(self, number_of_channels = 5, max_iteration = 1000):
        self.number_of_channels = number_of_channels
        self.max_iteration = max_iteration
        self.info = dict()
        self.message_location = np.random.choice(number_of_channels)
        self.previoused_step =self.message_location
    def reset(self):
        state = [0] * self.number_of_channels
        state[self.message_location] = 1
        
        self.total_steps = 0
        return np.expand_dims(state, axis = 1) 
    
    def step(self, action):
        state = [0] * self.number_of_channels
        if action == self.message_location:
            r = 1
            rand_channel = np.random.choice(self.number_of_channels)
            self.previoused_step = action 
            state[rand_channel] = 1
            self.message_location = rand_channel
        else:
            r = -5 
            self.previoused_step = action
            self.message_location = action
            state[self.message_location] = 1
        
        self.total_steps += 1
        
        if self.total_steps >= self.max_iteration:
            done = True 
        else:
            done  = False
            
        self.info["state"] = state 
        self.info['action'] = action
        self.info["previoused_step"] = self.previoused_step
        
        return np.expand_dims(state, axis = 1) , r, done, self.info
    

def play_one(env, agent, history, num_of_bits = 0,  total_t = 0, train = True, verbose = True):
    
    
    obs = env.reset()
    obs = modify_obs_add_channnel_2b(obs, env, num_of_bits = num_of_bits)
    state = np.stack([obs] * history, axis = 2) # now it is (#channels, 1, History)
   
    done = False
    cumulative_rewards = 0
    update_agent_frequncy = 1
    eps = None
    rewards_list = []
    rewards_list_modified = []
    average_20_modified = []
    while not done:
        # print(np.shape(state), "hallo")
        action = agent.sample_action(state, eps = eps)
        
        obs_next, r, done, info = env.step(action)
        obs_next = modify_obs_add_channnel_2b(obs_next, env,num_of_bits=num_of_bits)
        next_state = update_state(state, obs_next)
        
        state = next_state
        cumulative_rewards += r
        rewards_list.append(r)
        rewards_list_modified.append(max(0.,r))
        # Collect the samples: we collect obs(t+1) replay will give us the obs(t) when training 
        agent.experience_replay_buffer.add_experience(action, obs_next, r, done) #

        if train :
            if total_t % update_agent_frequncy == 0:
                if agent.experience_replay_buffer.count >= agent.experience_replay_buffer.batch_size*3:
                    cost = agent.learn()
                else:
                    cost = -0.0000
        else:
            cost = -9999.

        state = next_state
        total_t += 1
        
        # epsilon = modify_epsilon(epsilon,total_t,  minimum=minimum_epsilon )

        if verbose:
            print("Step_info:", info)
            
    
        val_mean_20 = np.mean(rewards_list_modified[-20:])
        average_20_modified.append(val_mean_20)
        if total_t  % 20 == 0:
            print("Total_t:", total_t, 
                  "Reward Average 20:",val_mean_20,
                  "Cost: %.8f" % cost,
                  "w_mellow:",agent.model.w)

            # print("Time for 10 games:", time.time()- t0)
            # t0 = time.time()
            
        if val_mean_20 >= 0.99:
            train = False
            eps = 0.0
    
        else:
            train = True 
                    
    return cumulative_rewards
        
def kill_simulation(operation = "op_run"):
    output = os.popen('wmic process get description, processid').readlines()
    n = len(output)
    for i in range(n):
        a = output[i].split()
        
        if len(a) > 0 and operation in a[0]:
            pid = int(a[1])
            os.kill(pid , signal.SIGTERM)
            print("A running simulation was found and killed!")
   
    
def save_to_gmb(agents,  address_algo, replay_memory_size = 100000, history_length = 1, batch_size = 64,
                i_d_folder = '', number_of_channels = 30,  verbose = False):
    #address = 'C:\Aladdin\Aladdin\Algorithms\IQL_v2_for_real_simulation'
    os.chdir(address_algo)
    
    global_rb_folder = "Global_RB_Storage_" + str(i_d_folder) 
    if not os.path.isdir(global_rb_folder): #"Global_RB_Storage"):
        if verbose:
            print("Creating new Global Memory Replay Buffer!")
        os.makedirs(global_rb_folder) 
        num_of_bits = int(np.floor(np.log2(number_of_channels)) + 1)     
        experience_replay_buffer = ReplayMemory(capacity =int(replay_memory_size),
                                                  number_of_channels =number_of_channels + num_of_bits ,
                                                  agent_history_length  = history_length,
                                                  batch_size = batch_size)
        
        path_rb = global_rb_folder + '/' + 'global_RB.pk'
        with open(path_rb , 'wb') as file: #"Global_RB_Storage/global_RB.pk"
            pickle.dump(experience_replay_buffer, file)
        
    #print("hallo111")
    path_to_folder = global_rb_folder + '/'
    files = os.listdir(path_to_folder)

    files = sorted(os.listdir(path_to_folder), key = lambda t: os.stat(path_to_folder + t).st_mtime) 
    #print("koko:", files)
    latest_gmb_name = files[-1]
    
    ## Fist let's call the gmb
    path = path_to_folder + latest_gmb_name
    #print("hallo")
    with open(path, "rb") as file:
        global_rmb = pickle.load(file)
        
    for k in agents.keys():
        agent = agents[k]
        rb = agent.experience_replay_buffer
        idx_agent = rb.current
        
        for i in range(idx_agent):
            action = rb.actions[i]
            observation = rb.observations[i]
            reward = rb.rewards[i]
            terminal = rb.terminal_flags[i]
            ## copy state and actions, dones 
            global_rmb.original_decision[global_rmb.current] = rb.original_decision[i]
            global_rmb.add_experience(action, observation, reward, terminal)
            
    # save the new gobal replay memory buffer
    file_name = "global_RB_" +str(global_rmb.current) + '.pk'
    file_path = path_to_folder + file_name # "Global_RB_Storage/" + file_name
    with open(file_path , 'wb') as file:
        pickle.dump(global_rmb, file)
        
    ####### keep only 10 
    if len(files) > 10:
        files_name_to_delete = files[:len(files)-10]
        for name in files_name_to_delete:
            file_path =  path_to_folder + name #  "Global_RB_Storage/"
            os.remove(file_path)
  
    if verbose:
        print("Global Replay memory was updated successfully!")
    
    
    



def create_training_plots(average_accumulated_reward_vec_all, average_changed_channels_vec_all):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    x_original = np.array(average_accumulated_reward_vec_all)
    y = np.mean(x_original, axis = 0)
    std_y = np.std(x_original, axis = 0)
    x = np.arange(len(y))
    plt.figure(1)
    plt.plot(x, y, 'r')
    plt.fill_between(x, y-std_y, y + std_y,
        alpha=0.2, facecolor='r',
        linewidth=4, linestyle='dashdot', antialiased=True)
    
    
    plt.xlabel("#Episode")
    plt.ylabel("Accumulated Reward")
    
    ## Create a statistic plot wiht std 
    plt.figure(2)
    x_original = np.array(average_changed_channels_vec_all)
    y = np.mean(x_original, axis = 0)
    std_y = np.std(x_original, axis = 0)
    x = np.arange(len(y))
    
    plt.plot(x, y, 'r')
    plt.fill_between(x, y-std_y, y + std_y,
        alpha=0.2, facecolor='r',
        linewidth=4, linestyle='dashdot', antialiased=True)
    
    plt.xlabel("#Episode")
    plt.ylabel("The average number of changing a channel per net")
