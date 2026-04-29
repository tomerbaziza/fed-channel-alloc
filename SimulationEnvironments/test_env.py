from Pythonic_Environment import python_env
import pandas as pd 
import numpy as np 

            # No w we should be able to inser location from outside!! 


# Amud Anan data
file_path = 'LocationsUnits.csv'
df = pd.read_csv(file_path) ##
columns_to_include = ['X [UTM]', 'Y [UTM]']
filtered_df = df.loc[:, columns_to_include] ## just a X, Y location of each net  (The center)
file_path2 = 'Locations.csv'
df2 = pd.read_csv(file_path2) # This is actually the locations of each node, but with more colums 
columns_to_include2 = ['X [UTM]', 'Y [UTM]']
filtered_df2 = df2.loc[:, columns_to_include2] # This is the X, Y location

# Parameters
power1 = 32                   # user's power in dBm
power_in_watts = 10**(power1/10) /1000 # uwer's power in W
power = 10*np.log10(power_in_watts) # [32 - 30] dB
num_users = 6
num_channels = 30
sensitivity = -100           # interference should be less then -100dB, otherwise user is inteferred.
networks = []
locations = [np.array([row[0], row[1]]) for index, row in filtered_df.iterrows()]   #Amud anan
num_networks = len(locations)
users_locations_per_net = []
for i in range(num_networks):
    a = [[row[0], row[1]] for index, row in filtered_df2[i*6:(i+1)*6][:].iterrows()] 
    users_locations_per_net.append(np.array(a, dtype = np.float64))

#[(50,10,1,5), (90,50,5,5),
                               # (1000,100,50,80)]
            
############## Example where we define the ocation of each node in the game
env = python_env(number_of_nets= 20 , number_of_users_in_each_net= [6]*20,
                  net_center_location_and_std = None,
                  power = int(power),
                  users_locations_per_net = users_locations_per_net,
                  add_noise=False) # (mean_x, mean_y, std_x, std_y)


################3 Example where we define the mean and covariance of each network location
net_center_location_and_std = [(50,10,1,5),
                               (90,50,5,5),
                               (1000,100,50,80),
                               ]

env = python_env(number_of_nets= 3 , number_of_users_in_each_net= [6]*3,
                  net_center_location_and_std = net_center_location_and_std,
                  power = int(power),
                  users_locations_per_net = None,
                  add_noise=False) # (mean_x, mean_y, std_x, std_y)


s = env.reset()

env.plot_locations_per_nets()
for i in range(20):
    try:
        a = env.nets[i].noise_matrix 
        break
    except:
        print("i:", i, " is not")

