import os 
from Utils.utils import creat_player 
import pickle 
import tensorflow as tf 


def extract_a_number_in_a_str(s):
    local_number = ''
    n = len(local_number)
    max_value = -1
    dic = [str(i) for i in range(10)]
    
    for c in s:
        
        if c in dic:
            local_number += c
            n += 1
        elif n >0:
            max_value = max(max_value, int(local_number))
            local_number = ''
            n = 0
    if len(local_number) > 0:        
        max_value = max(max_value, int(local_number))
    if max_value > -1:
        return max_value
    
    else:
        print("There is no number in the given string!")
        return None
            
    
def train_model(trainig_iterations = 1, batch_size = 2, action_space = 30,
                learning_rate = 0.00025, history=2, optimizer = tf.keras.optimizers.Adam,
                max_experience = 100000,
                sensing_window = 5 ,
                number_of_layers = 3 ,
                number_of_nodes = 128,
                activation_fucntion = tf.nn.leaky_relu,
                mellowmax_constant = 0.02 ,
                gamma = 0.9 ,
                dropout = False,
                l2_regularization= False,
                i_d_folder = '',
                verbose = False):
    #create a global agent which train 
    # if not os.path.isdir('Global_Agnet'):
    # create folder 
    # os.makedirs('Global_Agnet')
    # print("Global Agent folder was created successfully!")
    ## create a global agent 
    agent = creat_player(number_of_actions = action_space,
                         history_length =  history,
                         learning_rate=learning_rate ,
                         max_experience =  max_experience,
                         batch_size = batch_size , 
                         sensing_window = sensing_window,
                         number_of_layers = number_of_layers ,
                         number_of_nodes = number_of_nodes,
                         activation_fucntion = activation_fucntion ,
                         mellowmax_constant = mellowmax_constant,
                         gamma = gamma,
                         dropout = dropout,
                         l2_regularization=l2_regularization,
                         i_d = 9999,
                         i_d_folder = i_d_folder,
                         verbose = verbose ) ## <--- wake up with the latest weights updated
    
        # with open('Global_Agnet/global_agent.pk', 'wb') as file:
            # pickle.dump(agent, file)

    
    # else:
        
        # load global agent 
        # with open('Global_Agent/global_agent.pk', 'rb') as file:
            # agent = pickle.load(file)
            
        
        
    # load the global experience replay buffer 
    
    folder_name = 'Global_RB_Storage_' + str(i_d_folder) 
    assert os.path.isdir(folder_name), 'There is no Global_RB_Storage folder to load from!!! folder name: ' + folder_name
    
    
    folder_path  = folder_name + '/'
    files = sorted(os.listdir(folder_path), key = lambda t: os.stat(folder_path + t).st_mtime) 
    
    assert not len(files) == 0 , ' The Global_RB_Storage folder is empty!'  
   
    latest_gmb_name = files[-1]
    ## Fist let's call the gmb
    path = folder_path + latest_gmb_name
    
    
    with open(path, 'rb') as file:    
        global_rb = pickle.load(file)
        
        
    # load the glbal memory buffer to agent 
    global_rb.define_new_batch_size(batch_size)
    
    agent.experience_replay_buffer = global_rb
     
    
    ### add trained weights;
   
    # if not os.path.isdir('Train_weights'):
    #     os.makedirs('Train_weights')
        
    # else:
    #     files = sorted(os.listdir("Train_weights/"), key = lambda t: os.stat("Train_weights/" + t).st_mtime)
    #     if len(files) > 0:
    #         latest_weights_name = files[-1]  
            
    #         # loads weights to agent
    #         path = "Train_weights/" + latest_weights_name
    #         with open(path, 'rb') as file:
    #             w = pickle.load(file)
                
    #         agent.load_given_weights(w) # <--here we load the trained weights
    
    # train the agent 
    if global_rb.count < batch_size:
        print("NO train was performed: global_rb.count < batch_size!!")
        return 
    
    for _ in range(trainig_iterations):
        cost = agent.learn()
        if verbose:
            print("cost:", cost)
            
    weights_folder = 'Train_weights_' + str(i_d_folder)
    
    if not os.path.isdir(weights_folder):#'Train_weights'):
        os.makedirs(weights_folder) #'Train_weights') 
        
    # save the weights
    path2weights_folder = weights_folder + '/'
    files = sorted(os.listdir(path2weights_folder), key = lambda t: os.stat(path2weights_folder + t).st_mtime)

    if len(files) == 0:
         file_name = 'trained_weights_based_on_1_simulations.pk'
     
    else:
        latest_weights_name = files[-1]
        
        number = extract_a_number_in_a_str(latest_weights_name)+1
        file_name = 'trained_weights_based_on_' + str(number) + '_simulations.pk'
        if verbose:
            print(latest_weights_name)
    path = path2weights_folder + file_name
    
    weights = agent.get_model_weights()
    
    with open(path, 'wb') as file:
        pickle.dump(weights, file)    
        
    #delete files 
    files = sorted(os.listdir(path2weights_folder), key = lambda t: os.stat(path2weights_folder + t).st_mtime)
    files = files[:-5]
    n = len(files)
    for i in range(n):
        file_name = files[i]
        number = extract_a_number_in_a_str(file_name)
        if number %50 != 0 :
            path = path2weights_folder + file_name
            os.remove(path)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        