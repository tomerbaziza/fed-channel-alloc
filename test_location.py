import numpy as np 
import matplotlib.pyplot as plt 
import os 
from BuildingBlocks.Worker import worker
import random 
from Utils.utils import keep_only_bat_files, kill_simulation, create_training_plots
from Utils.PerformanceStatistics import statistic_performance #train_model
from BuildingBlocks.TrainBlock import train_model
import time 
from IPython import get_ipython
#import matplotlib.pyplot as plt 
import pickle 
from SimulationEnvironments.Pythonic_Environment import python_env
import shutil
from Utils.CalculateDis import calculate_dis_min
import matplotlib.pyplot as plt
from Utils.get_adress_scen_and_adress_algo import get_adress_scen_and_adress_algo 
import sys 
from Utils.RandomLocationOfNetworks import set_random_location_of_networks
from Utils.InferenceFunctions import Inference_step_on_test_cases
from Utils.SetSpecificEnv import set_specific_env
        
        

             
    
N = 300
"""You should update the reward in such a way of encouraging staying on the same channel"""
average_accumulated_reward_vec = []
average_changed_channels_vec = []
number = [i +2 for i in range(23)] 
# number = number[15:]
lr = 0.00025# 0.0003



number_of_nets= np.random.choice(number)

number_of_users_in_each_net, net_center_location_and_std = set_random_location_of_networks(number_of_nets)

# env = set_specific_env('Amud_Anan', number_of_channels=30)#'Amud_Anan')DenseGridof_4_5_9
# env.reset()

# nets = env.nets
# net1 = nets[0]
# net1.net_id
# net1.channel = 1
# channels = [i for i in range(30)]
# sensed_vectors = []
# for channel in channels:
#     net1.channel = channel
#     for net in nets:
#         sensed_vec = net.create_sensed_vector(nets)
#         sensed_vectors.append(sensed_vec)
counter = 0    
k=1000
for _ in range(k):   
    env = python_env(number_of_nets= 2 ,
                                  number_of_users_in_each_net= [10]*2,
                                  net_center_location_and_std = [(0,0,50,50),
                                                                  (0, 800, 50,50 )],
                                  possible_channels = 10,
                                  add_noise = False,
                                  training = True) #random.choice(scenarios)
          
    
    
    obs , info = env.reset()
    
    for i in obs:
        if i < 1:
            counter += 1
            break
print(counter / k)
env.plot_locations_per_nets()
# print(s, info)

# done = False
# while not done:
#     print(obs, info)
#     action = int(input("action:"))
#     idd = int(input("id:"))
#     obs, r, done, info   = env.step(action, idd)

    
    
    