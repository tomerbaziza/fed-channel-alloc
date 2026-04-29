import numpy as np 



def statistic_performance(n , history_length , env =None , worker = None, address_scen  = '', address_algo = ''):
    a = []
    b = []
    for i in range(n):
        print("Test: ", i)
        average_accumulated_reward_vec, average_changed_channels_vec = test_over_all_combination_5_24(nodes = 10,
                                                                                                      env = env,
                                                                                                      worker = worker,
                                                                                                      history_length  = history_length )
    
        a.append(average_accumulated_reward_vec)
        b.append(average_changed_channels_vec)
    
    return a, b 

def extract_convergence_time(agents):
    pass
    
    
def test_over_all_combination_5_24(nodes = 10, env = None, worker = None, history_length = None, address_scen  = '', address_algo = '',):

    python_env = env
    number = [i +2 for i in range(23)] 
    
    average_accumulated_reward_vec = []
    average_changed_channels_vec = []
    
    for i in number:
        number_of_nets = i 
        
        number_of_users_in_each_net = []
        net_center_location_and_std = []
        for k in range(number_of_nets):
            number_of_users_in_each_net.append(nodes)
            mean_x =  (2*np.random.random() - 1 ) * 100 # <-- location
            std_x = mean_x * 0.5
            mean_y =  (2*np.random.random() - 1 ) * 100
            std_y = mean_y * 0.1
            
            net_center_location_and_std.append((mean_x, mean_y , std_x, std_y))
            
        random_scenario = python_env(number_of_nets= number_of_nets ,
                                     number_of_users_in_each_net= number_of_users_in_each_net,
                                     net_center_location_and_std = net_center_location_and_std,
                                     possible_channels = 30,
                                     add_noise = False) #random.choice(scenarios)
        
        average_accumulated_reward_val, average_changed_channels,agents = worker(address_scen = address_scen,
                                                                           scenario = random_scenario,
                                                                           address_algo = address_algo,
                                                                           history_length = history_length ,
                                                                           training = False)
        
        average_accumulated_reward_vec.append(average_accumulated_reward_val)
        average_changed_channels_vec.append(average_changed_channels)
        
    return average_accumulated_reward_vec, average_changed_channels_vec