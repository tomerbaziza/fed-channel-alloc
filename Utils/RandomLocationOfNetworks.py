
import numpy as np 
from Utils.CalculateDis import calculate_dis_min


def set_random_location_of_networks(number_of_nets):
    net_center_location_and_std = []
    locations = []
    number_of_users_in_each_net = []
    
    for k in range(number_of_nets):
        number_of_users_in_each_net.append(np.random.randint(1, 16))  # Table II: M in {1,...,15}
        if k == 0:
            mean_x =  (2*np.random.random() - 1 ) * (number_of_nets) * 400
            mean_y =  (2*np.random.random() - 1 ) * (number_of_nets) * 400
            
        else:
            index = np.random.choice(len(locations))
            
            x_mean_chosen_net = locations[index][0]
            y_mean_chosen_net = locations[index][1]
            
            # randomize Radius and angle  
            r = np.random.randint(low = 50, high=500)
            theta =  np.random.randint(low = 0, high=2*np.pi)
            
            mean_x = r * np.cos(theta) + x_mean_chosen_net
            mean_y = r * np.sin(theta) + y_mean_chosen_net
            
        locations.append(np.array([mean_x, mean_y]))
        
        std_x = 50 
        std_y = 50 
        
        
        net_center_location_and_std.append((mean_x, mean_y , std_x, std_y))
        
    return number_of_users_in_each_net, net_center_location_and_std


    