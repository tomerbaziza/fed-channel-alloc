
import os 
import numpy as np 
import pandas as pd 

# game_history = pd.read_csv(filepath_or_buffer = 'Amud_Anan_game_history_Jun_13_2023_09_49_37.csv')
# game_history = pd.read_csv('inference_examination/Amud_Anan_game_history_Jun_13_2023_14_12_38.csv')
def get_game_performamce(game_history, file_name = '', w_annc = 0.1, w_ct = 0.4, 
                         w_cq = 0.4, w_se= 0.1, 
                         number_of_channels = 30,
                         save_file = True):
    ## Here you 
    ancc, ancc_score = clacANCC(game_history)
    ct, max_value_time = convergenceTime(game_history)
    cq_mean, cq_median, cq_max, cq_min, number_of_users_above_90, number_of_users_under_90 = channelQuality(game_history)
    se = spectrumEfficiency(game_history, number_of_channels)
    number_of_used_channels, score_resued = clacReuse(game_history)

    
    ct_score = 1-ct/max_value_time
    
    ws = w_annc * ancc_score + w_ct * ct_score + w_cq * cq_mean + w_se * se ## <-- that is not ok 
    
    # reusese = reuseInGame(game_history)
    
    vec = [ancc, ct, 
           cq_mean,cq_median,  cq_max, cq_min, number_of_users_above_90, number_of_users_under_90, 
           se, ancc_score ,ct_score ,
           ws, number_of_used_channels, score_resued ]#, reuse]
    
    columns = ['ANCC', 'Convergence_Time', 'Channel_Quality_meanBased','Channel_Quality_median',
                   'Channel_Quality_max', 'Channel_Quality_min', 'CQ_above90', 'CQ_below90',
                   'Spectrum_Efficiency', 
                   'ancc_score', 'Convergence_Time_Score', 'Weighted_Score',
                   'number_of_used_channels', 'score_resued']
    df = pd.DataFrame([np.array(vec)],
                      columns = columns)
    
    folder_name = 'Inference_examination'
         
    if save_file:
        assert os.path.isdir(folder_name), "The folder Inference_examination is missing!!! "
        path = file_name + '.csv'
        
        df.to_csv(path_or_buf= path,  index = False)
    else:
        return vec
    
    
def clacANCC(game_history):
    """
    input = Data fram of the history game 
    out put = Average number of channel changes (ANCC)
    """
    all_agents_id =  game_history['agent_id'].unique()
    ncc_per_agent = []
    amount_of_decision_points = []
    for agent_i_d in all_agents_id:
        
        agent_i = game_history[game_history['agent_id'] == agent_i_d].reset_index()
        counter_channel_changes = 0
        n = len(agent_i)
        for i in range(n):
       
            if i == 0 :
                channel = agent_i['action'].iloc[i] # This is the initial channel 
                
            else:  
                c = agent_i['action'].iloc[i]
                
                if c != channel:
                    counter_channel_changes += 1
                    channel = c 
                    
        ncc_per_agent.append(counter_channel_changes)   
        amount_of_decision_points.append(counter_channel_changes/n)
        
    return np.mean(ncc_per_agent), 1 - np.mean(amount_of_decision_points)



def convergenceTime(game_history) :      
    
    #This convergence time is based on the last change of the last net 
    ## Note: The last change is depends on the last agent that play also!! 
    ## maybe it is a good idea to nodul tyhe time by 20 at each time --> to get the  round time
    # (each round == number of agents * 1)
    all_agents_id =  sorted(game_history['agent_id'].unique())
    convergence_time_per_agent = []
    max_value_time = 0
    for agent_i_d in all_agents_id:
        agent_i = game_history[game_history['agent_id'] == agent_i_d].reset_index()
        
        for i in range(len(agent_i)-1, -1, -1):
            
            if i == len(agent_i) - 1:
                last_channel = agent_i['action'].iloc[i]
                convergence_time = agent_i['time'].iloc[i]
            else:
                a = agent_i['action'].iloc[i]
                
                
                if a != last_channel:
                   break 
               
                else:
                    convergence_time = agent_i['time'].iloc[i]
        
        last_time_point= agent_i['time'].iloc[-1]
        convergence_time_per_agent.append(convergence_time)
        max_value_time = max(max_value_time, last_time_point)
        ## For examine the last round without changes 
        # convergence_time_per_agent.append(convergence_time // len(all_agents_id))
        
    return np.max(convergence_time_per_agent), max_value_time




def channelQuality(game_history):
    """
    input : game history as datafram
    output : mean-CQ, median-CQ
    """
    
    all_agents_id =  sorted(game_history['agent_id'].unique())
    channel_quality_per_agent = []
    
    counter_above_90 = 0
    counter_below_90 = 0
    for agent_i_d in all_agents_id:
        agent_i = game_history[game_history['agent_id'] == agent_i_d].reset_index() # check that 
        
        action = agent_i.iloc[-2]['action'] # was -2 
        
        channel_name = 'Ch_' + str(int(action))
        channel_quality  = agent_i.iloc[-1][channel_name]   
        
        if channel_quality >=0.9:
            counter_above_90 += 1
        else:
            counter_below_90 += 1
            
        channel_quality_per_agent.append(channel_quality)
        
    return np.mean(channel_quality_per_agent), np.median(channel_quality_per_agent), np.max(channel_quality_per_agent), \
        np.min(channel_quality_per_agent), counter_above_90/len(all_agents_id) , counter_below_90/len(all_agents_id)


def spectrumEfficiency(game_history, number_of_channels):

    all_agents_id =  sorted(game_history['agent_id'].unique())
    spectrun_efficiency_per_agent = []

    for agent_i_d in all_agents_id:
        agent_i = game_history[game_history['agent_id'] == agent_i_d].reset_index()
        
        spectrum = agent_i.iloc[-1]
        
        sum_squres = 0
        counter = 0 
        # print("number_of_channels:", number_of_channels)
        for i in range(number_of_channels):
            c = 'Ch_' + str(i)
            c_val = spectrum[c]
            
            if c_val != -1: # means the channel is optional 
                counter += 1
                sum_squres += c_val**2
         
            
        agent_se = np.sqrt(sum_squres) / np.sqrt(counter) 
        spectrun_efficiency_per_agent.append(agent_se)
                
    se = np.mean(spectrun_efficiency_per_agent)
    
    return se           
        
def clacReuse(game_history):
    all_agents_id =  sorted(game_history['agent_id'].unique())
    channel_used_at_the_end = []
    
    for agent_i_d in all_agents_id:
        agent_i = game_history[game_history['agent_id'] == agent_i_d].reset_index()
        
        channel_used_at_the_end.append(agent_i['action'].iloc[-1])
        
        
    unique_channels = np.unique(channel_used_at_the_end)
    
    number_of_used_channels = len(unique_channels)
    
    score_resued = 1 - number_of_used_channels/len(all_agents_id)
    
    return number_of_used_channels, score_resued
    
 
        
