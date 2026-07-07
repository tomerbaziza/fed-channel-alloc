"""Episode coordinator for CARLTON-style multi-agent channel allocation.

This module runs the decentralized execution loop and computes CARLTON rewards.

Paper mapping (arXiv:2402.17773):
- Section III-B: action space over channel indices.
- Section III-C, Algorithm 2: personal reward.
- Section III-C, Algorithm 3: social welfare reward.
- Eq. (13): total reward as weighted sum of personal + social terms.
- Section III-D, Eq. (17): state = concatenate(CBR, QV).
- Section III-D, Algorithm 4: sequential turns and local replay collection.
"""

import sys 
sys.path.append('SimulationEnvironments/')
sys.path.append('DeepMellow_Single_agent/')
import os 
import numpy as np 
from copy import deepcopy
## Base on Independent learners, all freinds are part of the environment
from Utils.utils import modify_obs_add_channnel_2b, save_to_gmb ,\
                        update_state, creat_player,save_all_agents_memory,saveAgent
# from SimulationEnvironments.Environment import EnvironmentWrapper  
from ReplayMemory import ReplayMemory
from datetime import datetime
import pickle 
import torch
import pandas as pd 

""" set Environment"""
# LEARNING_RATE = 0.00025# was 0.00025
#HISTORY_LENGTH = 1 # MAYBE WE SHOULD WORK WITH 1 
# SENSING_WINDOW = 5 # was 5 this is not important and not effect nothing since we used simple environment 
## here we deals with fully observable :)
# MAX_EXPERIENCES = 5000#5000
# BATCH_SIZE = 32
NUM_OF_CHANNELS = 10
# Table II / Eq. (13): personal reward weight rho
RHO = 0.7

activate_current_algo = False

def calculate_rewards_personal(obs, action, agent, rho=None):
    """Compute CARLTON personal reward component.

    Paper reference:
    - Section III-C, Algorithm 2 (personal reward, r_p).

    Behavior:
    - Uses channel quality (`obs[action]`) and ranking among all channels.
    - Gives desired reward when quality exceeds threshold (~zeta=0.9).
    - Adds a stay bonus when the channel does not change.
    - Scales by rho; social term is added later in `calculate_rewards_sw`.
    """
    rho = RHO if rho is None else float(rho)
    obs = np.squeeze(obs)
    current_channel = action
    value = obs[current_channel]
    
    if value >= 0.9:
        r = 4
    
    else:
        n = len(obs)
        
        obs = sorted(obs)
        
        i = 0
        
        while i <= n-1 and value >= obs[i]:
            i += 1
     
        
        r = (i/n  - 0.5) * 2
    
    # print("fast rewards:", r)
    # Algorithm 2 line 11-12 style: encourage channel stability.
    if current_channel == agent.old_channel and r > 0.0: # 'median up approach'
        # it was agent.current channel before updating
        r += 0.1*r 

    return rho * r

def calculate_rewards_sw(obs,agent, current_channel, agents, time, rho=None):
    """Compute social welfare reward from nearby agents.

    Paper reference:
    - Section III-C, Algorithm 3 (social welfare reward, r_sw).
    - Eq. (13): total reward combines personal and social terms.

    Implementation note:
    - Neighbors are selected by Euclidean distance <= 500m (Gamma in paper).
    - This function returns the social contribution scaled by (1 - rho).
    """
    rho = RHO if rho is None else float(rho)
    if rho >= 1.0:
        return 0.0
    ## fix the -1 
    # print("current:", current_channel)
    
    idx = agent.experience_replay_buffer.current - 1
    
    # if idx <= 0:
    #     time = 0
    # else:
        # time = agent.experience_replay_buffer.time[idx]
    # time = time #agent.experience_replay_buffer.asrdot[5] +  
    
    if idx + 1  <=0:
        previuse_time = 0
    else:
        previuse_time =  agent.experience_replay_buffer.time[idx]
    
    # print("time:", time)
    # print("Pre_time:", previuse_time)
    counter = 0
    sw_r = 0
    for k in agents.keys():
        agent_i = agents[k]
        if agent.i_d == agent_i.i_d:
            continue
        
        dis_vec = agent_i.net_location- agent.net_location ##take care only over close nets 
        radius = np.sqrt(np.matmul(dis_vec, dis_vec.T))
        
        
        # print( agent_i.i_d ,":",agent_i.net_location)
        # print( agent.i_d ,":",agent.net_location)
        # print("R:", radius)
        # Algorithm 3: neighbors are within distance threshold Gamma.
        if  radius<= 500:
            """I need to fic it to be better !!!"""
            agent_max_indx = agent_i.experience_replay_buffer.current - 1
            # print("agent_max_indx:", agent_max_indx)
            # for j in range(agent_max_indx, -1, -1):
            agent_time = agent_i.experience_replay_buffer.asrdot[5]
            # print("agent_time _freinds:", agent_time)
            # print("Neighbor of net %i, is %i" %(agent.i_d, agent_i.i_d) )
            r_s = agent_i.experience_replay_buffer.asrdot[2] #rewards[j]
            # print("r_s:", r_s)
            # Aggregate neighboring personal rewards between two consecutive
            # decision points of the current agent.
            if agent_time > previuse_time and agent_time < time:
                
                sw_r += r_s
                counter += 1
                # You should calculate the channel quality to perform fairness
                # print("time:", time, " previuse_time:", previuse_time)
                # print("agent_time:", agent_time, " r:", r_s)
                # input("pause")
               
    r_sw = sw_r / counter if counter != 0 else 0   
    # print("r_sw:", r_sw)
    obs = np.squeeze(obs)
    value = obs[current_channel]
    
    if value >= 0.9:
        r = 4
    
    else:
        n = len(obs)
        
        obs = sorted(obs)
        
        i = 0
        
        while i <= n-1 and value >= obs[i]:
            i += 1
     
        
        r = (i/n  - 0.5) * 2
    

    if current_channel == agent.old_channel and r > 0.0: # 'median up approach'
        # it was agent.current channel before updating
        r += 0.1*r 
            
    # print("r:", r)
    # print("r_sw:", r_sw) #0.6 * r
    # Eq. (13): r = rho * r_p + (1 - rho) * r_sw.
    return (1.0 - rho) * r_sw
    
def define_new_agent(i, history_length,
                     learning_rate,
                     replay_memory_size,
                     batch_size,
                     number_of_layers,
                     number_of_nodes,
                     activation_fucntion,
                     mellowmax_constant,
                     gamma,
                     dropout,
                     l2_regularization,
                     global_weights = None,
                     i_d_folder = '' ,
                     sensing_window = 5,
                     verbose = False):
    
    """Factory for one independent learner (network manager)."""
    agent = creat_player(number_of_actions = NUM_OF_CHANNELS,
                 history_length = history_length,
                 learning_rate = learning_rate,
                 max_experience = replay_memory_size,
                 batch_size = batch_size, 
                 sensing_window = sensing_window,
                 number_of_layers = number_of_layers,
                 number_of_nodes = number_of_nodes,
                 activation_fucntion = activation_fucntion,
                 mellowmax_constant = mellowmax_constant,
                 gamma = gamma,
                 dropout = dropout,
                 l2_regularization=l2_regularization,
                 i_d = i,
                 i_d_folder = i_d_folder,
                 verbose = verbose )
    if global_weights is not None:
        agent.load_state_dict(global_weights)
                 
    return agent
   

# def calculate_rewards(obs, a, agent):
    
#     if obs[a] >= 0.9:
#         x1 = 0.9
#         x2 = 1.0
        
#         y1 = 2
#         y2 = 5
        
#         r =  (obs[a] - x1)*(y2-y1)/(x2-x1) + y1 # /1.0#obs[a]

            
#         # ### incourage dense
#         # if a >0 and a< NUM_OF_CHANNELS-1:
#         #     if obs[a-1] < cq and obs[a+1]< cq:
#         #         r = 1.5*r
        
#     else:
#         r = obs[a] #- 0.9
    
#     if a == agent.old_channel: # 'median up approach'
#         # it was agent.current channel before updating
#         r = 0.05 * r
        
#     return r 

def coordinator(environment, mutex = None, address_algo = '',training= True,
                history_length= 1,
                replay_memory_size = 10000 ,
                number_of_layers= 3, 
                number_of_nodes = 128,
                learning_rate = 0.00025, 
                activation_fucntion = None,
                mellowmax_constant = 0.02,
                gamma = 0.9,
                batch_size  = 32,
                dropout = False,
                l2_regularization = False,
                i_d_folder = '',
                epsilon = 0,
                save_to_global_rb = False,
                global_weights = None,
                local_train_steps = 20,
                fedprox_mu = 0.0,
                persistent_replay_buffer = None,
                rho = None,
                verbose = False):
    
    """Run one CARLTON episode over a scenario.

    Paper reference:
    - Section III-D, Algorithm 4: each network acts in sequence, stores local
      transitions, and contributes to global replay after episode end.
    - Eq. (17): state input includes CBR + QV.

    Returns:
        average_accumulated_reward_val, average_change_channel_counter,
        agents dictionary, and raw game history rows.
    """
    ## activate scenario 
    
    env = environment # EnvironmentWrapper()
    # Eq. (17): CBR (channel binary representation) length.
    num_of_bits = int(np.floor(np.log2(NUM_OF_CHANNELS)) + 1)
    
    # create the agents
    agents = {}
    
    # Initialize the environment 
    # kill_simulation()
    obs, info = env.reset() 
    """what I am getting from here ??? and Implement the level of certainty""" 
    
    agent_id = info['Master_Id']
    current_master_channel = info['primaryChannel']
    time = info['Time'] ## this is a simulated time 
    net_location = info['Net_location_x_y'] # (x,y)
    
    agents[agent_id] = agents.get(agent_id, define_new_agent(i = agent_id, history_length = history_length,
                                                                learning_rate = learning_rate,
                                                                replay_memory_size = replay_memory_size,
                                                                batch_size = batch_size,
                                                                number_of_layers = number_of_layers,
                                                                number_of_nodes = number_of_nodes,
                                                                activation_fucntion = activation_fucntion,
                                                                mellowmax_constant = mellowmax_constant,
                                                                gamma = gamma,
                                                                dropout = dropout,
                                                                l2_regularization = l2_regularization,
                                                                global_weights = global_weights,
                                                                i_d_folder = i_d_folder,
                                                                verbose = verbose))
    agents[agent_id].current_channel = current_master_channel
    agents[agent_id].net_location = net_location
    # print("obs:", obs.shape)
 
    obs = np.reshape(obs, (NUM_OF_CHANNELS, 1))
    # Eq. (17): s(t) = concatenate(CBR, QV(t)).
    obs = modify_obs_add_channnel_2b(obs,num_of_bits = num_of_bits,
                                    channel = current_master_channel)
    
    
    state = np.stack([obs] * history_length, axis = 2) # now it is (#channels, 1, History)
    
    update_agent_frequncy = 1
    total_t = {agent_id: 0}
    done = False 
    #train_vec = [True] *num_of_agents
    cumulative_rewards = {agent_id: 0} #[0] * num_of_agents
    
    rewards_list = {agent_id: []}
    rewards_list_modified = {agent_id: []}
    
    average_all = []
    info_collection = []
    
    #for k in range(num_of_agents):
    #   rewards_list[k] = []
    #   rewards_list_modified[k] = []
    last_state = {}
    
    train_vec = {}
    train_vec[agent_id] = True 
    last_state[agent_id] = state 
    actions = []    
    
    game_history = []
    
    average_performance_of_all = []
    iteration_num = 0
    current_master_channel = info['primaryChannel']
    # local_dictionary_for_state = {}
    # local_dictionary_for_state[agent_id] = []
    while not done:
        #print("step:", iteration_num, "/", env.max_length )
        # print(iteration_num,"/",env.max_length)
        iteration_num += 1
        agent = agents[agent_id]
        
        agent.update_current_channel(current_master_channel)
        
        
        if training:
            # action = agent.sample_action(state, eps = 1.0) # Applying sampling prom distribution
            # Section III-B: action is channel index in K.
            state_tensor = torch.as_tensor(state, dtype=torch.float32)
            action = agent.sample_action(state_tensor, eps = epsilon, training = training ) ## exploration
            # print("state:\n", state[num_of_bits:,0,0])
            # action = int(input("inser_action:"))
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float32)
            action = agent.sample_action(state_tensor, eps = 0.0, training = training) ## <-- we are going with the max value 
        
        # Section III-C Algorithm 2: personal reward component.
        reward = calculate_rewards_personal(
            obs=state[num_of_bits:, 0, 0], action=action, agent=agent, rho=rho
        )
        reward = float(torch.as_tensor(reward, dtype=torch.float32).item())
                                           
        # reward = calculate_rewards(obs = state[num_of_bits:,0,0],
        #                            agent = agent,
        #                            current_channel = action,
        #                            agents = agents, 
        #                            time = time)
        # if agent_id not in agents.keys():
        #     local_dictionary_for_state[agent_id] = []
            
        # local_dictionary_for_state[agent_id].append(state[num_of_bits:,0,0])
        
        # print("time -->:", time)
        game_history.append([time, agent_id, action]+list(state[num_of_bits:,0,0]))

        agent.experience_replay_buffer.apply_asrdot(value = reward, index = 2)
        agent.experience_replay_buffer.apply_asrdot(value = action, index = 0)
        agent.experience_replay_buffer.apply_asrdot(value = time, index = 5)
        
        ## stupid approach! Now we have a mask over the actions 
        get_protection_on_actions = False
        if get_protection_on_actions:
            assert action <= 23, 'action got value higher than 23 which is impossible!!'
            if action > 23 :
                print("Somthing is not right !!!! ")
                action = np.random.choice(23)
            
            
        
        # Environment transition after selecting dynamic channel action.
        next_obs, _ , done, info = env.step(action, agent_id)  
        
        if done :
            break 
        

        agent.experience_replay_buffer.apply_asrdot(value = done, index = 3)
        
        agent.experience_replay_buffer.apply_asrdot(value = 5, index = 4) #<--- This is a location for expert
        """
        Understand the next observation, it is belong only to specific agent 
        write a function which translate the environment message and 
        prove you with sensed vector and number of agent 
        """
        
        agent_id = info['Master_Id']
        current_master_channel = info['primaryChannel']
        time = info['Time']
        ########## Now you need to check which agent is it
        
        
        if agent_id not in agents.keys():
            ### create new state and agent  
            obs = np.reshape(next_obs, (NUM_OF_CHANNELS, 1))
            obs= modify_obs_add_channnel_2b(obs,num_of_bits = num_of_bits,
                                            channel = current_master_channel)
            state = np.stack([obs] * history_length, axis = 2) 
            
            agents[agent_id] = define_new_agent(i = agent_id, history_length = history_length,
                                                                        learning_rate = learning_rate,
                                                                        replay_memory_size = replay_memory_size,
                                                                        batch_size = batch_size,
                                                                        number_of_layers = number_of_layers,
                                                                        number_of_nodes = number_of_nodes,
                                                                        activation_fucntion = activation_fucntion,
                                                                        mellowmax_constant = mellowmax_constant,
                                                                        gamma = gamma,
                                                                        dropout = dropout,
                                                                        l2_regularization=l2_regularization,
                                                                        global_weights = global_weights,
                                                                        i_d_folder = i_d_folder,
                                                                        verbose = verbose)
            # new = True
            net_location = info['Net_location_x_y'] 
            agents[agent_id].net_location = net_location
            
            last_state[agent_id] = state
            total_t[agent_id] = 0
            cumulative_rewards[agent_id] = 0
            rewards_list[agent_id] = []
            rewards_list_modified[agent_id] = []
            r_and_r_ws_reward = None
            train_vec[agent_id] = True 
            agent = agents[agent_id]
            agent.update_current_channel(current_master_channel)
            
        else:
            
            next_obs = np.reshape(next_obs, (NUM_OF_CHANNELS, 1))
          
            
            # retro_r = calculate_rewards(obs, agents[agent_id], current_master_channel,agents)## we should define it better 
            previouse_obs = last_state[agent_id][num_of_bits:,0,0]
            # print("previouse_obs:", previouse_obs)
            # sdfsdf = input("sdfs")
            # Section III-C Algorithm 3: delayed social reward update.
            r_sw = calculate_rewards_sw(
                obs=previouse_obs,
                agent=agents[agent_id],
                current_channel=current_master_channel,
                agents=agents,
                time=time,
                rho=rho,
            )
            r_sw = float(torch.as_tensor(r_sw, dtype=torch.float32).item())
            # Eq. (13): r = rho * r_p + (1-rho) * r_sw.
            agents[agent_id].experience_replay_buffer.asrdot[2]+= r_sw # (value = retro_r, index = 2)
            r_and_r_ws_reward = agents[agent_id].experience_replay_buffer.asrdot[2]
            # new = False
       ######## You can not add because you dont have the next observation  
            # Eq. (17): build next state as [CBR ; QV].
            next_obs = modify_obs_add_channnel_2b(next_obs, # it was next_obs before
                                             num_of_bits = num_of_bits,
                                             channel = current_master_channel)
            
            agents[agent_id].experience_replay_buffer.apply_asrdot(value = next_obs, index = 1)  
            state = last_state[agent_id]
            next_state = update_state(state,next_obs)
            last_state[agent_id] = next_state
            ###"Shere insert"

            state = next_state
            
            
        info_collection.append(info)
        
        ### <-- should be coded 
        
        if r_and_r_ws_reward is not None: # it was r_retro
            # if cumulative_rewards.get(agent_id,"Empty") == "Empty":
                # cumulative_rewards[agent_id] = [0]
            cumulative_rewards[agent_id] += r_and_r_ws_reward
            rewards_list[agent_id].append(r_and_r_ws_reward)
            rewards_list_modified[agent_id].append(max(0.,r_and_r_ws_reward))
            
        # print("message number:", iteration_num)
    
        
    ## show me performance
    average_accumulated_reward = [] 
    change_channel_counter_vec = []
    for agent_id in agents.keys():
        agent = agents[agent_id]
        r = rewards_list[agent_id] # < -- this is a vector 
        average_rewards = np.mean(r)
        accumulate_rewards = np.sum(r)
        change_channel_counter = agent.change_channel_counter
        change_channel_counter_vec.append(change_channel_counter)
        if verbose:
            print("Agent i_d: %i, Accumulate reward: %.2f, Rewards/Step: %.2f, Change_channel_counter: %i" 
                  % (agent_id, accumulate_rewards, average_rewards, change_channel_counter)
                  )
        average_accumulated_reward.append(accumulate_rewards)
    average_accumulated_reward_val = np.mean(average_accumulated_reward)
    average_change_channel_counter = np.mean(change_channel_counter_vec)
    
    if verbose:
        print('average reward:', average_accumulated_reward_val)
    

    # Once the episode is over we shoud insert all state the done flags !!!!!!!!!!!!!
    for k in agents.keys():
        # first check if you have some data left in the stack of each agent # but we donr care about it !
        agent = agents[k] 
        idx = agent.experience_replay_buffer.current - 1
        agent.experience_replay_buffer.terminal_flags[idx] = done
      
    # In FRL mode each worker keeps only local replay.
    if save_to_global_rb:
        if mutex:
            mutex.acquire()
        if training:
            save_to_gmb(agents = agents,
                        address_algo = address_algo,
                        replay_memory_size = replay_memory_size,
                        history_length= history_length,
                        batch_size = batch_size,
                        i_d_folder = i_d_folder, 
                        number_of_channels = NUM_OF_CHANNELS)
        if mutex:
            mutex.release()

    # Merge episode transitions into persistent local replay (FRL across rounds).
    if training and persistent_replay_buffer is not None:
        for agent in agents.values():
            persistent_replay_buffer.append_from_buffer(agent.experience_replay_buffer)
        for agent in agents.values():
            agent.experience_replay_buffer = persistent_replay_buffer

    # Local private replay training before federated upload.
    if training and local_train_steps > 0:
        global_ref = deepcopy(global_weights) if global_weights is not None and fedprox_mu > 0.0 else None
        for agent in agents.values():
            rb = agent.experience_replay_buffer
            if rb.count >= rb.batch_size:
                for _ in range(local_train_steps):
                    agent.learn(global_ref=global_ref, fedprox_mu=fedprox_mu)

    return average_accumulated_reward_val, average_change_channel_counter, agents, game_history 

