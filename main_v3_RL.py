import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pandas as pd
#import matplotlib.pyplot as plt 
from Utils.get_adress_scen_and_adress_algo import get_adress_scen_and_adress_algo 
from Utils.InferenceFunctions import Inference_step_on_test_cases
from Utils.ScenarioExamination import get_game_performamce
from Utils.dotdict import dotdict
from Utils.save_to_df_csv import wrrape_game_history_do_df
import pickle
             
             
     
_ , address_algo = get_adress_scen_and_adress_algo(script_path = os.getcwd())
address_scen = ''


save_game_history = True
 
number_of_channels = 24 ## You need to change also in 
epsilon = 0.01

dictionary = {}

for i in range(10):
    print(i,"/", 10)
    outputs_vec = Inference_step_on_test_cases("weights_name", address_scen,
                                       address_algo,
                                       number_of_channels,
                                       training = False,
                                       epsilon = epsilon,
                                       history_length = 1,
                                       save_csv = False)
    
    for output in outputs_vec:
        scenario_name = output[-1]
        scenario_game_history = output[-2]
        scenario_vec_history = dictionary.get(scenario_name, 0)
        
        if scenario_vec_history == 0:
            scenario_vec_history = []
            dictionary[scenario_name] = scenario_vec_history
        
        dictionary[scenario_name].append(scenario_game_history)
    
    
with open("dictionary001.pk","wb") as f:
    pickle.dump(dictionary, f)
# outputs_vec[0] = [average_accumulated_reward_val, average_changed_channels, game_history, scenario_name]

# with open("dictionary3.pk","rb") as f:
#     dictionary = pickle.load(f)

all_scenarios_performance = {}
for k in dictionary:
    the_list = dictionary[k]
    for i, game_history in enumerate(the_list):
        
        game_history_df = wrrape_game_history_do_df(game_history, number_of_channels)
        vec = get_game_performamce(game_history_df, file_name = '', w_annc = 0.1, w_ct = 0.4, 
                                 w_cq = 0.4, w_se= 0.1, 
                                 number_of_channels = number_of_channels,
                                 save_file = False)
        
        columns = ['ANCC', 'Convergence_Time', 'Channel_Quality_meanBased','Channel_Quality_median',
                   'Channel_Quality_max', 'Channel_Quality_min', 'CQ_above90', 'CQ_below90',
                   'Spectrum_Efficiency', 
                   'ancc_score', 'Convergence_Time_Score', 'Weighted_Score',
                   'number_of_used_channels', 'score_resued']
        
        
        df = pd.DataFrame([np.array(vec)],
                          columns = columns)

        print(df)

        if i ==0 :
            df_all = df.copy()
        else:
            df_all = pd.concat((df_all, df))
    
    all_scenarios_performance[k] = df_all
   
with open("all_scenarios_performance_agenteps001.pk","wb") as f:
    pickle.dump(all_scenarios_performance, f)
    
    

# game_history1 =  dictionary[k][0]
# game_history2 = dictionary[k][1]
   

# game_history_df = wrrape_game_history_do_df(game_history2, number_of_channels)
# vec = get_game_performamce(game_history_df, file_name = '', w_annc = 0.1, w_ct = 0.4, 
#                           w_cq = 0.4, w_se= 0.1, 
#                           number_of_channels = number_of_channels,
#                           save_file = False)
 
# columns = ['ANCC', 'Convergence_Time', 'Channel_Quality_meanBased','Channel_Quality_median',
#             'Channel_Quality_max', 'Channel_Quality_min',
#             'Spectrum_Efficiency', 
#             'ancc_score', 'Convergence_Time_Score', 'Weighted_Score',
#             'number_of_used_channels', 'score_resued']
 
 
# df2 = pd.DataFrame([np.array(vec)],
#                    columns = columns)