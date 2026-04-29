import numpy as np 
import matplotlib.pyplot as plt 
import os 
from BuildingBlocks.Worker import worker

from BuildingBlocks.TrainBlock import train_model

#import matplotlib.pyplot as plt 
import pickle 
from SimulationEnvironments.Pythonic_Environment import python_env
import shutil
from Utils.get_adress_scen_and_adress_algo import get_adress_scen_and_adress_algo 
from Utils.InferenceFunctions import Inference_step_on_test_cases
from Utils.RandomLocationOfNetworks import set_random_location_of_networks
from Utils.ScenarioExamination import get_game_performamce
from Utils.save_to_df_csv import wrrape_game_history_do_df
from Utils.dotdict import dotdict
from Utils.createImages import create_images
import random 


def get_numbers_from_str(string):
    
   numbers = [str(i) for i in range(10)] 
   numbers_in_the_string = []
   number = ''
   for i in range(len(string)):
       char = string[i]
       if char in numbers:
           number += char
       else:
           if len(number) > 0:
               numbers_in_the_string.append(number)  
           number = ''
           
   if len(number) > 0:
       numbers_in_the_string.append(number)
       
   return numbers_in_the_string
           
           
    

def get_the_number_of_scenarios_trained(i_d_folder = ''):
    
     weight_folder_name = 'Train_weights_' + str(i_d_folder)
     
     if os.path.isdir(weight_folder_name ):
         # load the trained weights (the last one)
         files = sorted(os.listdir(weight_folder_name + "/"), key = lambda t: os.stat(weight_folder_name + "/" + t).st_mtime)
         latest_weights_name = files[-1]
         
     else:
         return '0'
     
     
     # extract the number
     all_numbers = get_numbers_from_str(latest_weights_name)
     
     return all_numbers[0]
     
             
             
     
_ , address_algo = get_adress_scen_and_adress_algo(script_path = os.getcwd())
address_scen = ''


save_game_history = True


def single_training_run(history_length = int(1)):    
    N = 1000
    """You should update the reward in such a way of encouraging staying on the same channel"""
    # average_accumulated_reward_vec = []
    # average_changed_channels_vec = []
    number_of_possible_nets = 7#15 #[i +2 for i in range(16)] 
    # number = number[15:]
    lr = 0.00025# 0.0003
    number_of_channels = 10 ## You need to change also in 
    epsilon = 0.5
    
    train_info = {'ancc_vec_train': [],
    'ct_vec_train': [],
    'cq_mean_vec_train': [],
    'cq_median': [],
    'cq_max': [],
    'cq_min': [],
    'number_of_users_above_90': [],
    'number_of_users_under_90':[],
    'se_vec_train': [],
    'ancc_score_vec_train': [], 
    'ct_score_vec_train': [],
    'ws_vec_train':[],
    'number_of_used_channels_vec_train': [],
    'score_resued_vec_train': [],
    'average_accumulated_reward_vec': [],
    'average_changed_channels_vec': [],
    'numebr_of_nets': [],
    'game_history': []}
    
    train_info = dotdict(train_info)
    
    for j in range(0,N,1):
        if j % 10 ==0:
            print(j, "/", N)
            
            
        number_of_nets = np.random.randint(2, number_of_possible_nets) ## Thsi is not ok (you should sample from que)
        # number_of_nets = 7
        number_of_users_in_each_net, net_center_location_and_std = set_random_location_of_networks(number_of_nets)

        random_scenario = python_env(number_of_nets= number_of_nets ,
                                     number_of_users_in_each_net= number_of_users_in_each_net,
                                     net_center_location_and_std = net_center_location_and_std,
                                     possible_channels = number_of_channels,
                                     add_noise = False,
                                     training = True) #random.choice(scenarios)
        
        # print("pass", "number of nets:", number_of_nets)
        # max_steps = number_of_nets * 20
        # max_rewards_possible = 
        # print("Running Scenario:", random_scenario)
        
        average_accumulated_reward_val, average_changed_channels, agents, game_history = worker(address_scen = address_scen,
                                                                           scenario = random_scenario,
                                                                           address_algo = address_algo,
                                                                           history_length = history_length,
                                                                           training = True,
                                                                           epsilon = epsilon)
        
    
        ## calculate all parameters of Interest
        
        
        ### Train an agent 
        ## add condition about the size of the global buffer size ! 
        ## if it is big enough, then you can train , lso modify the rewards !!!!
        epsilon = max(0.01, epsilon - (0.5-0)/(N/2) )
        
        if j > 500:
            y1 = 0.02
            x1 = 500
            y2 = 0.2
            x2 = 1200
            
            mellowmax_constant = max(y1 + (j - x1) * (y2-y1)/(x2-x1),2.0) 
            lr = 0.0001
        else:
            mellowmax_constant = 0.02
            
            
        train_model(trainig_iterations =40,
                    action_space= number_of_channels,
                    batch_size = 32,
                    learning_rate = lr,#25,
                    history = 1,
                    mellowmax_constant = mellowmax_constant,
                    optimizer= None,
                    verbose = False)
        
        
        game_history = wrrape_game_history_do_df(game_history, number_of_channels)
        
        (ancc, ct, 
               cq_mean,cq_median,  cq_max, cq_min, number_of_users_above_90, number_of_users_under_90, 
               se, ancc_score ,ct_score ,
               ws, number_of_used_channels, score_resued ) = get_game_performamce(game_history = game_history,
                                              number_of_channels = number_of_channels,
                                              save_file = False)

        train_info.ancc_vec_train.append(ancc)
        train_info.ct_vec_train.append(ct)
        train_info.cq_mean_vec_train.append(cq_mean)
        
        train_info.cq_median.append(cq_median)
        train_info.cq_max.append(cq_max)
        train_info.cq_min.append(cq_min)
        train_info.number_of_users_above_90.append(number_of_users_above_90)
        train_info.number_of_users_under_90.append(number_of_users_under_90)
        
        train_info.se_vec_train.append(se)
        train_info.ancc_score_vec_train.append(ancc_score) 
        train_info.ct_score_vec_train.append(ct_score)
        train_info.ws_vec_train.append(ws)
        train_info.number_of_used_channels_vec_train.append(number_of_used_channels)
        train_info.score_resued_vec_train.append(score_resued)
        train_info.average_accumulated_reward_vec.append(average_accumulated_reward_val)
        train_info.average_changed_channels_vec.append(average_changed_channels)
        train_info.numebr_of_nets.append(number_of_nets)
        train_info.game_history.append(game_history)
        # lr =0.00025# max(lr- (0.0003 - 0.00005)/1000,  0.00005 )
        
        
        if j % 50 == 0:
            print("Episode: %i, number_of_nets: %.1f average_accumulated_reward_val: %.3f, average_changed_channels: %.3f, lr: %.5f"
                    % (j,number_of_nets,average_accumulated_reward_val, average_changed_channels, lr))
            print("Agerage accumulated reward mean 20 games:", np.mean(train_info.average_accumulated_reward_vec[-20:]))
            print("Agerage average_changed_channels_vec 20 games:", np.mean(train_info.average_changed_channels_vec[-20:]))
            print("train_info.cq_mean_vec_train 20 games:", np.mean(train_info.cq_mean_vec_train[-20:]))
            print("train_info.cq_min 20 games:", np.mean(train_info.cq_min[-20:]))
            print("train_info.cq_median 20 games:", np.mean(train_info.cq_median[-20:]))
            print("epsilon:", epsilon)
            print("mellowmax_constant:", mellowmax_constant)
        
            with open("train_info.pk", "wb") as f:
                pickle.dump(train_info, f)
          
        if j % 10 ==0 :     
            create_images(train_info)
            
            
        if j % 50 == 0 and False:
            weight_number = get_the_number_of_scenarios_trained()
            weights_name = "weights_" + weight_number
            _ = Inference_step_on_test_cases(weights_name = weights_name,
                                                        address_scen = address_scen,
                                                        address_algo = address_algo,
                                                        number_of_channels=number_of_channels,
                                                        training = False,
                                                        history_length = 1)
            
            
        
    return train_info, game_history
     

average_accumulated_reward_vec_all = []
average_changed_channels_vec_all = []

n_statistics = 1
for i in range(n_statistics):
    # delete Global MB and weights 
    # average_accumulated_reward_vec, average_changed_channels_vec, agents, game_history =  single_training_run()
    # average_accumulated_reward_val, average_changed_channels, game_history, scenario_name  = Inference_step_on_test_cases(weights_name = 'weights_0',
    #                                                                                                              address_scen = address_scen,
    #                                                                                                              address_algo = address_algo,
    #                                                                                                              history_length = 1)
    
    train_info, game_history= single_training_run()
    if i < n_statistics - 1:
        shutil.rmtree('Global_RB_Storage')
        shutil.rmtree('Train_weights')




# """Write a function which take the agents and creates all the stuff that I need for analysis"""