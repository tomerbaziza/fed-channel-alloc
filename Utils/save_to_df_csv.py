import numpy as np 
import pandas as pd 
import os 
from datetime import datetime, date
from Utils.ScenarioExamination import get_game_performamce

def wrrape_game_history_do_df(game_history, number_of_channels):
    a = ['time', 'agent_id', 'action'] 
    channel_names = ['Ch_' + str(i) for i in range(number_of_channels)]
    
    columns = a + channel_names
    
    df = pd.DataFrame(np.array(game_history),
                      columns = columns)
    
    return df 

def save_game_history_as_df_and_csv(average_accumulated_reward_val, average_changed_channels, game_history, 
                                    scenario_name,
                                    number_of_users_in_each_net = None,
                                    net_center_location_and_std = None,
                                    add_to_the_file_name = '',
                                    number_of_channels = 30,
                                    examine_the_game = True):
    
    if scenario_name is None:
        scenario_name = 'No_Name'
        
    df = wrrape_game_history_do_df(game_history,number_of_channels)    
    a = ['time', 'agent_id', 'action'] 
    # channel_names = ['Ch_' + str(i) for i in range(30)]
    
    # columns = a + channel_names
    
    # df = pd.DataFrame(np.array(game_history),
    #                   columns = columns)
    
    folder_name = 'Inference_examination'
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
        
    
    today = date.today()
    date_str = today.strftime("%b_%d_%Y")
    
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")

    timing = date_str + '_' + current_time
    name = folder_name + '/'+ scenario_name + '_game_history'+ '_' + timing + '_' + add_to_the_file_name
    path = name + '.csv'
    
    df.to_csv(path_or_buf= path,  index = False)
    
    ## Save the details 
    if number_of_users_in_each_net is not None:
        a = np.array(net_center_location_and_std)
        b = np.array(number_of_users_in_each_net, dtype = np.int32)
        c = np.concatenate((a, np.expand_dims(b, axis =1)), axis = 1 )
        
        df2_locations = pd.DataFrame(c,
                                     columns = ['mean_x', 'mean_y' , 'std_x', 'std_y','Num_users'])
        
        path = folder_name+ '/' + scenario_name+'locations' + '_' + timing + add_to_the_file_name + '.csv'
        df2_locations.to_csv(path_or_buf= path,index = False)
        
    
    file_name = name + '_performance.csv'
    
    get_game_performamce(game_history = df,
                         file_name = file_name,
                         w_annc = 0.1,
                         w_ct = 0.4,
                         w_cq = 0.4,
                         w_se= 0.1,
                         number_of_channels = number_of_channels)

    
    
    
    
    