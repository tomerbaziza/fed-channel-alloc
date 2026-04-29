
import numpy as np 
from Utils.SetSpecificEnv import set_specific_env
from BuildingBlocks.Worker import worker
from Utils.save_to_df_csv import save_game_history_as_df_and_csv
import os
from SimulationEnvironments.Pythonic_Environment import python_env


def Inference_step_on_test_cases(weights_name, address_scen,
                                   address_algo,
                                   number_of_channels,
                                   training = False,
                                   epsilon = 0,
                                   history_length = 1,
                                   save_csv = True):
    
    ## get all scenarios to test 
    items = os.listdir('scenarios_for_test') 
    scenarios_for_test = [item for item in items if os.path.isdir(os.path.join('scenarios_for_test', item))]
    all_output = []
    for scenario in scenarios_for_test:
        output = inference_pythonEnv(scenario_name = scenario,
                             address_scen = address_scen,
                             address_algo = address_algo,
                             history_length = 1,
                             number_of_channels = number_of_channels,
                             add_to_the_file_name = weights_name,
                             epsilon = epsilon,
                             training = False,
                             save_csv = save_csv)
        
        all_output.append(output)
    average_accumulated_reward_val, average_changed_channels, game_history, scenario_name = output 
    print("All scenarios were cheked!")
    return  all_output # average_accumulated_reward_val, average_changed_channels, game_history, scenario_name

def inference_pythonEnv(scenario_name = None, address_algo = '', address_scen = '' ,history_length = 1,
                        number_of_channels = 30,
                        add_to_the_file_name = '',
                        epsilon = 0,
                        training = False, 
                        save_csv = True):
    
    if scenario_name is None:
        """You should update the reward in such a way of encouraging staying on the same channel"""
        number = [i +2 for i in range(23)] # max 23 nets

    
        number_of_nets= np.random.choice(number)
        number_of_users_in_each_net = []
        net_center_location_and_std = []
        for k in range(number_of_nets):
            number_of_users_in_each_net.append(np.random.choice([5,6,7,8,10]))
            mean_x =  (2*np.random.random() - 1 ) * 100 # <-- location
            std_x = mean_x * 0.5
            mean_y =  (2*np.random.random() - 1 ) * 100
            std_y = mean_y * 0.1
            
            net_center_location_and_std.append((mean_x, mean_y , std_x, std_y))
        
    
        scenario_env = python_env(number_of_nets= number_of_nets ,
                                     number_of_users_in_each_net= number_of_users_in_each_net,
                                     net_center_location_and_std = net_center_location_and_std,
                                     possible_channels = number_of_channels,
                                     add_noise = False,
                                     training = False) #random.choice(scenarios)
        
    else:
         ## locad locations  
        scenario_env = set_specific_env(scenario_name = scenario_name, number_of_channels = number_of_channels,
                                        training = False)
        
        number_of_users_in_each_net = None
        net_center_location_and_std = None
        
    average_accumulated_reward_val, average_changed_channels, agents, game_history = worker(address_scen = address_scen,
                                                                           scenario = scenario_env,
                                                                           address_algo = address_algo,
                                                                           history_length = history_length,
                                                                           epsilon = epsilon,
                                                                           training = False)
        
    if save_csv:
        save_game_history_as_df_and_csv(average_accumulated_reward_val = average_accumulated_reward_val, 
                                            average_changed_channels = average_changed_channels,
                                            game_history = game_history,
                                            scenario_name = scenario_name,
                                            number_of_users_in_each_net = number_of_users_in_each_net,
                                            number_of_channels = number_of_channels,
                                            net_center_location_and_std = net_center_location_and_std,
                                            add_to_the_file_name = add_to_the_file_name)
        
        
    return  average_accumulated_reward_val, average_changed_channels, game_history, scenario_name 