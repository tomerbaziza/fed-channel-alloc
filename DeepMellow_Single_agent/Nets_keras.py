import sys 
sys.path.append('/home/yaniv/DL/ReinforcementLearning/RLGit/Centrelized_approach_single_agent_V2/DeepMellow_Single_agent/Transformer/for_tf_2_9')
# from Transformer.for_tf_2_9.Transformer_models import Transformer_Encoder_Keras

import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np 
import pickle 

class Lstm_dual(object):
    def __init__(self, input_shape, K):
        # K - is output neurons number (last layer)
        x_inputs = Input(shape=input_shape)
        x = LSTM(128, activation=tf.nn.tanh)(x_inputs)
        x= Dense(128, activation=tf.nn.relu)(x)
        x1= Dense(128, activation=tf.nn.relu)(x)
        
        value = Dense(1, activation = tf.identity)(x1)
        advantage = Dense(K, activation=tf.identity)(x1)
        
        y = value + (advantage - tf.reduce_mean(advantage))
        self.model = Model(inputs = x_inputs, outputs = y)
        self.trainable_params = self.model.trainable_weights
        
    # @tf.function
    def forward(self, x):
        x = tf.squeeze(x, axis = 2)
        x = tf.transpose(x, (0,2,1))
        return self.model.call(x)#self.model.apply(x)
        
class LstmNet(object):
    def __init__(self, input_shape, K):
        """
        dictionary_details = {'LASTM':[[nuerons, activation],...],
                              'Dense': [[neurons, activation],....],
                              }
        K - is output neurons number (last layer)
        """
        x_inputs = Input(shape=input_shape)
        x = LSTM(128, activation=tf.nn.tanh)(x_inputs)
        x= Dense(128, activation=tf.nn.relu)(x)
        x= Dense(128, activation = tf.nn.relu)(x)
        y = Dense(K, activation=tf.identity)(x)

        self.model = Model(inputs = x_inputs, outputs = y)
        self.trainable_params = self.model.trainable_weights
   
    # @tf.function
    def forward(self, x):
        x = tf.squeeze(x, axis = 2)
        x = tf.transpose(x, (0,2,1))
        return self.model.call(x)#self.model.apply(x)
    
    
    def copy_params_from(self, params):
        for i in range(len(self.trainable_params)):
            self.trainable_params[i].assign(params[i].numpy())
            

class AnnNet(object):
    def __init__(self, input_shape, K):
        """
        dictionary_details = {'LASTM':[[nuerons, activation],...],
                              'Dense': [[neurons, activation],....],
                              }
        K - is output neurons number (last layer)
        """
        input_shape = (np.prod(input_shape),)
        x_inputs = Input(shape = input_shape)
        x = Dense(128, activation=tf.nn.relu)(x_inputs)
        x= Dense(128, activation=tf.nn.relu)(x)
        x= Dense(128, activation = tf.nn.relu)(x)
        y = Dense(K, activation=tf.identity)(x)

        self.model = Model(inputs = x_inputs, outputs = y)
        self.trainable_params = self.model.trainable_weights
   
    # @tf.function
    def forward(self, x):
        n,h,w,c = x.shape
        x = tf.reshape(x, shape = (n, h*w*c))
        return self.model.call(x)#self.model.apply(x)
    
    
    def copy_params_from(self, params):
        for i in range(len(self.trainable_params)):
            self.trainable_params[i].assign(params[i].numpy())


class AnnResNet(object):
    def __init__(self, input_shape, K, activation = tf.nn.leaky_relu):
        """
        dictionary_details = {'LASTM':[[nuerons, activation],...],
                              'Dense': [[neurons, activation],....],
                              }
        K - is output neurons number (last layer)
        """
        input_shape = (np.prod(input_shape),)
        x_inputs = Input(shape = input_shape)
        x1 = Dense(128, activation= activation )(x_inputs)
        x2= Dense(128, activation= activation )(x1)
        x3 = tf.add(x1, x2)
        x4= Dense(128, activation =  activation )(x3)
        x5 = tf.add(x3,x4)
        y = Dense(K, activation=tf.identity)(x5)

        self.model = Model(inputs = x_inputs, outputs = y)
        self.trainable_params = self.model.trainable_weights
   
    # @tf.function
    def forward(self, x):
        n,h,w,c = x.shape
        x = tf.reshape(x, shape=(n,h,c,w))
        x = tf.transpose(x,perm = (0,2,1,3))
        x = tf.reshape(x, shape = (n, h*w*c))
        return self.model.call(x)#self.model.apply(x)
    
    
    def copy_params_from(self, params):
        for i in range(len(self.trainable_params)):
            self.trainable_params[i].assign(params[i].numpy())            
            




class AnnResNet_tunable(object):
    def __init__(self, input_shape,
                 K,
                 number_of_layers,
                 number_of_nodes,
                 activation = tf.nn.leaky_relu,
                 dropout = False):
        """
        dictionary_details = {'LASTM':[[nuerons, activation],...],
                              'Dense': [[neurons, activation],....],
                              }
        K - is output neurons number (last layer)
        """
        
        input_shape = (np.prod(input_shape),)
        x_inputs = Input(shape = input_shape)
        x_old  = None
        for layer in range(number_of_layers):
            if layer == 0:
                x = Dense(number_of_nodes, activation= activation )(x_inputs)
                # x_old = x 
            
            else:
                x = Dense(number_of_nodes, activation= activation)(x)
                x = tf.add(x, x_old)
                # x_old = x
            
            if dropout:
                x = Dropout(0.2)(x)
            
            x_old = x 
            
        y = Dense(K, activation = tf.identity)(x)
        
        self.model = Model(inputs = x_inputs, outputs = y)
        self.trainable_params = self.model.trainable_weights
   
    # @tf.function
    def forward(self, x, training):
        n,h,w,c = x.shape
        x = tf.reshape(x, shape=(n,h,c,w))
        x = tf.transpose(x,perm = (0,2,1,3))
        x = tf.reshape(x, shape = (n, h*w*c))
        # if np.random.rand() < 0.01:
        #     print(training)
        return self.model.call(x,training=training)#self.model.apply(x)
    
    
    def copy_params_from(self, params):
        for i in range(len(self.trainable_params)):
            self.trainable_params[i].assign(params[i].numpy())            
            









# ## Checking the tunability of resnet network
# ann = AnnResNet_tunable((1,35), 30, 3, 128)
# ann_original = AnnResNet(input_shape = (1,35), K = 30)


# x = tf.random.normal(shape = (1,1,35,1))
# y1 = ann.forward(x)
# y2 = ann_original.forward(x)
# print("difference:",tf.reduce_sum(tf.abs(y1 - y2)))



# for w,w1 in zip(ann.trainable_params, ann_original.trainable_params):
#     w1.assign(w)
  


# y1 = ann.forward(x)
# y2 = ann_original.forward(x)
# print("difference:",tf.reduce_sum(tf.abs(y1 - y2)))











