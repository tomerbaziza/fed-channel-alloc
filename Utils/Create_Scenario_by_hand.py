

import numpy as np
import pandas as pd 


def create_Locatio_csv_file(mu_x, mu_y, name):

    n_nets = len(mu_x)
    
    std_x = 50
    std_y =  50
    number_of_users = 10
    
    for i in range(n_nets):
        mean_x = mu_x[i]
        mean_y = mu_y[i]
        
        population = np.random.multivariate_normal(mean = [mean_x, mean_y],
                                                           cov = [[std_x**2, 0],[0, std_y**2]],
                                                           size = number_of_users)
        
        
        population = np.concatenate((population, np.array([[i]*number_of_users]).T), axis = 1)
        
        df1 = pd.DataFrame(population,
                          columns = ['X [UTM]', 'Y [UTM]', 'net_id'])
        if i ==0 :
            df = df1 
            # pd.DataFrame(population,
            #                   columns = ['X [UTM]', 'Y [UTM]', 'net_id'])    
        
        
        else:
            data = [df,df1]
            df = pd.concat(data, ignore_index=True, sort=False)
            
    file = name +'.csv'
    
    df.to_csv(file)
    
    print('Done with the creation of ' + file)
    
    

########## Grid_4_5 distance 200 
########## number of users 10 at each net 



mu_x = [0] *5 + [150] * 5 + [300] * 5 + [450] * 5 
mu_y = [300, 150,0,-150,-300] * 4


name = 'Grid_4_5_distance_200'
create_Locatio_csv_file(mu_x, mu_y, name)


## The squre 200 

mu_x = [0,0,200,200]
mu_y =[0,200,0,200]
name ='Square_200'

create_Locatio_csv_file(mu_x, mu_y, name)


## Equilateral triangle (200)

mu_x = [0,100,200]
mu_y = [0, 173.2,0]

name ='Equilateral_triangle_200'

create_Locatio_csv_file(mu_x, mu_y, name)

## lace_10_200 (200)

mu_x = [0] * 15
mu_y = [i *150  for i in range(15)]

name ='lace_10_200'

create_Locatio_csv_file(mu_x, mu_y, name)



########## Grid_4_5 distance 200 
########## number of users 10 at each net 


mu_x = [0] *5 + [150] * 5 + [300] * 5 + [450] * 5 
mu_y = [300, 150,0,-150,-300] * 4

mu_x_middle= [75] * 4 + [225] *4 + [375] * 4 

mu_y_middle = [225, 75, -75,-225] * 4

mu_y += mu_y_middle
mu_x += mu_x_middle

name = 'DenseGridof_4_5_9'
create_Locatio_csv_file(mu_x, mu_y, name)


## Cyclic full mesh 
mu_x = [200, 141.42, 0 , -141.42, -200, -141.42,   0,   141.42]
mu_y = [0,  141.42, 200, 141.42,   0,   -141.42, -200, -141.42 ]

name = 'Cyclic_full_mesh'
create_Locatio_csv_file(mu_x, mu_y, name)


## Start of 5 aroud and one in the middle (radius = 350)
# only the middle fill all, and the rest does fill only the middle

mu_x = [0, 108.16, -283.16, -283.16, 108.16, 350]
mu_y = [0, 332.84, 205.73, -205.73, -332.87, 0]

name = 'Start_5_r_350'
create_Locatio_csv_file(mu_x, mu_y, name)


## One cube 
mu_x = [0,75,150,150,0]
mu_y = [0,75, 0, 150, 150]

name = 'Cube_with_middle_150'
create_Locatio_csv_file(mu_x, mu_y, name)