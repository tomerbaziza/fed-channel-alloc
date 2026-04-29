import os 
from BuildingBlocks.Coordinator import coordinator
import tensorflow as tf 

def worker(address_scen, scenario, address_algo, mutex= None, training = True, 
           history_length = 1, replay_memory_size = 10000, number_of_layers= 3, 
           number_of_nodes = 128, learning_rate = 0.00025, 
           activation_fucntion = tf.nn.leaky_relu, mellowmax_constant = 0.02,
           gamma = 0.9,
           batch_size  = 64,
           dropout = False,
           l2_regularization = False,
           i_d_folder = '',
           epsilon = 0,
           verbose = False):
    # activate the bat file (opnet simulation)
    # os.chdir(address_scen)
    # os.startfile(scenario)
    # print("pass worker")
    ## activate the Coordinator
    average_accumulated_reward_val, average_changed_channels, agents, game_history = coordinator(environment = scenario,
                mutex = mutex,
                address_algo = address_algo,
                training = training,
                history_length= history_length,
                replay_memory_size = replay_memory_size,
                number_of_layers= number_of_layers, 
                number_of_nodes = number_of_nodes,
                learning_rate = learning_rate, 
                activation_fucntion = activation_fucntion,
                mellowmax_constant = mellowmax_constant,
                gamma = gamma,
                batch_size = batch_size,
                dropout = dropout,
                l2_regularization = l2_regularization,
                i_d_folder = i_d_folder,
                epsilon = epsilon,
                verbose = verbose)
    
    if verbose:
        print("Senario: %s finished" % (scenario))
    
    return average_accumulated_reward_val, average_changed_channels, agents, game_history