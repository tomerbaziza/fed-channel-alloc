# You need the following files: ('NodeStatus.csv'), ('Locations.csv'), ('Platforms.csv')
import pandas as pd 

node_status = pd.read_csv('NodeStatus.csv')

# node_status_active = node_status[node_status['// Time']<3000] #tHIS NODE ARE ACTIVE 

location = pd.read_csv('Locations.csv')

platform = pd.read_csv('Platforms.csv')


unique_net_id = node_status['Net Id'].unique()

location['net_id'] = [-999] * len(location)


for i in range(len(location)):
    
    if node_status['// Time'].iloc[i] < 3000:
        plat_id = location['Plat Index'].iloc[i]
        
        mac_adress = platform[platform['// plat Index'] == plat_id][' Mac Addr'].iloc[0]
        
        net_id = node_status[node_status['Mac Addr']==mac_adress]['Net Id'].iloc[0]
        
        location['net_id'].iloc[i] = net_id
    
location_active = location[location['net_id'] != -999]

location_active.to_csv('Location_brigade.csv')