import os 
import numpy as np 
import pandas as pd 
from SimulationEnvironments.Pythonic_Environment import python_env

"""Scenario loader for CARLTON evaluation/training.

Builds a `python_env` instance from CSV-defined user locations.
This file provides the fixed-scenario branch of Section II-B setup and feeds
the environment used later in Section III-D execution/training loops.
"""


def set_specific_env(scenario_name, number_of_channels=30, training = False):
    """Instantiate `python_env` from a named scenario on disk.

    Paper reference:
    - Section II-B: scenario geometry (network/user spatial layout).
    - Section III-D: resulting environment is consumed by CARLTON loop.
    """
    print("Activate: ", scenario_name, " for testing!")
    ## Check if there is a folder for checking 
    path = 'scenarios_for_test/' + scenario_name
    assert os.path.isdir(path), "The scenario is missing! Check that the scenario exist!"
    

    file_path2 = path + '/Locations.csv'
    df2 = pd.read_csv(file_path2) # This is actually the locations of each node, but with more colums
    
    # if scenario_name == 'Amud_Anan':
    #     columns_to_include2 = ['X [UTM]', 'Y [UTM]']
    #     filtered_df2 = df2.loc[:, columns_to_include2] # This is the X, Y location
    # else:
    columns_to_include2 = ['X [UTM]', 'Y [UTM]', 'net_id']
    filtered_df2 = df2.loc[:, columns_to_include2]
   
    
    # if scenario_name ==  'Amud_Anan':
    #     file_path = path + '/LocationsUnits.csv'
    #     df = pd.read_csv(file_path) ##
    #     columns_to_include = ['X [UTM]', 'Y [UTM]']
    #     filtered_df = df.loc[:, columns_to_include] ## just a X, Y location of each net  (The center)
    #     locations = [np.array([row[0], row[1]]) for index, row in filtered_df.iterrows()]   #Amud anan
    #     number_of_nets = len(locations)
    # else:
    networks_id = df2['net_id'].unique()
    number_of_nets = len(networks_id)
    number_of_users_in_each_net = []
        
    for i in networks_id:
       number_of_users_in_each_net.append(len(df2[df2['net_id'] == i]))
        
    power1 = 32                   # user's power in dBm
    power = power1 - 30                 # dB
    
    # if scenario_name == 'Amud_Anan':
    #     num_users = 6
    #     # number_of_nets = 20
    #     number_of_users_in_each_net = [num_users]*number_of_nets
    
    # num_channels = number_of_channels
    # sensitivity = -100           # interference should be less then -100dB, otherwise user is inteferred.
    # networks = []

    users_locations_per_net = []
    # if scenario_name == 'Amud_Anan':
    #     for i in range(number_of_nets):
    #         num_users = number_of_users_in_each_net[i]
    #         a = [[row[0], row[1]] for index, row in filtered_df2[i*num_users:(i+1)*num_users][:].iterrows()] 
            # users_locations_per_net.append(np.array(a, dtype = np.float64))
    # else:
    for i in networks_id:
        users_location = filtered_df2[filtered_df2['net_id'] == i].loc[:,['X [UTM]', 'Y [UTM]']].to_numpy()
        users_locations_per_net.append(users_location)
            
        
    env = python_env(number_of_nets= number_of_nets,
                     number_of_users_in_each_net= number_of_users_in_each_net,
                      net_center_location_and_std = None,
                      power = int(power),
                      users_locations_per_net = users_locations_per_net,
                      possible_channels = number_of_channels,
                      add_noise=False,
                      training = training)
    
    return env 




