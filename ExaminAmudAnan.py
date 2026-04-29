
from Utils.ScenarioExamination import  get_game_performamce
from Utils.save_to_df_csv import wrrape_game_history_do_df
import pickle 
import matplotlib.pyplot as plt 
from Utils.SetSpecificEnv import set_specific_env
import pandas as pd 



def from_message_to_game_history(message):
    
    game_history = []
    
    for i in range(1,len(message) + 1,1):
        message_vec = message[i]
        
        vec = [message_vec[1],
               message_vec[2],
               message_vec[4]]
        asint = [int(j)/100 for j in message_vec[5:]]
        
        vec += asint
        
        game_history.append(vec)
    
    return game_history
all_game_messages = ['AmudAnan_messages_145.pk'] #'AmudAnan_messages_155.pk'




for message_i in all_game_messages:
    path = 'AmudAnan_games_weights_1000/' + message_i
    
    with open(path, "rb") as file:
        message = pickle.load(file)
        
    
    game_history = from_message_to_game_history(message)
    game_history = wrrape_game_history_do_df(game_history = game_history, number_of_channels=24)
    
    print(len(game_history['agent_id'].unique()))
    
    
    
    get_game_performamce( game_history = game_history, 
                         file_name= message_i,
                         number_of_channels = 24)
    
############ Game movememnt ##########
all_id_agents = game_history['agent_id'].unique()



for agent_id in all_id_agents:
    agent_history_i = game_history[game_history['agent_id'] == agent_id]
    
    counter = 0
    ### Fix the arry before ploting ! 
    actions = list(agent_history_i['action'])
    time = list(agent_history_i['time'])
    
    print("number_ofDP:", len( list(agent_history_i['action'])))
    fixed_actions = []
    fixed_time = []
    
    for k, a in enumerate(actions):
        
        if k ==0 :
            fixed_actions.append(a)
            fixed_time.append(time[k])
            
        else:
            if a != fixed_actions[-1]:
                fixed_actions.append(a)
                fixed_time.append(time[k-1])
                counter += 1
            fixed_actions.append(a)
            fixed_time.append(time[k])
            
          
    print("counter:", counter)
    plt.plot(fixed_time, fixed_actions)
    

env = set_specific_env('Amud_Anan', number_of_channels=30)
env.reset()
env.plot_locations_per_nets()
## locate the Nets Masters!!
all_locations = []
location_file = pd.read_csv('AmudAnan_game_history_148_based_1231Weights/Locations.csv')
for agent_id in all_id_agents:
    agent_location = location_file[location_file['Plat Index'] == agent_id]
    x, y = agent_location.iloc[0]['X [UTM]'], agent_location.iloc[0]['Y [UTM]']
    channel = int(game_history[game_history['agent_id'] == agent_id]['action'].iloc[-2])
    
    plt.text(x, y, str(agent_id) + " c = " + str(channel), fontsize=16)
    
    all_locations.append((x,y,agent_id))