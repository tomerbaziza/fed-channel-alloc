
import tensorflow as tf 
import numpy as np
import pickle  
import os 
import time 

class DeepMellow(object):
    def __init__(self,net, number_of_actions,gamma,
                 optimizer = tf.keras.optimizers.Adam(learning_rate= 0.00025), 
                 los_func = tf.keras.losses.Huber(), 
                 mellowmax_constant = None):
        """
        K = number of output nodes
        """
        self.w = mellowmax_constant
        self.gamma = gamma
        self.K = number_of_actions # action sapce
        self.net = net
        # the class of the loss 
        self.loss_func = los_func
        
        ## OPtimizer
        self.optimizer = optimizer
        
    # @tf.function       
    def forward(self, Z): 
        Z = self.net.forward(Z)
        return Z
         
    def predict(self, x):
        return self.forward(x)
    
    # @tf.function
    def cost(self, S, actions, G):
        prediction = self.forward(S) # S is the states, predictions is actions 
        
        # now take into account only the actions that you choose in the past 
        actual_chosen_actions = prediction * tf.one_hot(actions, self.K)
      
        selected_action_values = tf.reduce_sum(actual_chosen_actions, axis = [1])
        # print(selected_action_values, "g", G)
        costi = tf.reduce_mean(self.loss_func(y_true = G, y_pred = selected_action_values)) # in the paper it is L2 not huber
        
        return costi
    
    
    def sample_action(self, x, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            x = np.expand_dims(x, axis = 0)
            x = np.float32(x)
            a_pos = self.predict(x)
            return np.argmax(a_pos[0])
        
    # @tf.function
    def update_weights(self, states, actions, targets):
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            
            cost_i = self.cost(states, actions, targets)
            
        gradients = tape.gradient(cost_i, self.net.trainable_params) 
        # tf.print(cost_i, gradients)
        self.optimizer.apply_gradients(zip(gradients, self.net.trainable_params))
        return cost_i
    
    
    def copy_params_from(self, params):
        for i in range(len(self.trainable_params)):
            self.trainable_params[i].assign(params[i].numpy())
        # self.trainable_params = [param for param in model.trainable_params] # here is the problem 
        
        
    # def save(self, name ):
    #     params = [param.numpy() for param in self.trainable_params]
    #     file_name = 'weights/params' + name +'.pickle' 
    #     with open(file_name, 'wb') as f:
    #         pickle.dump(params, f)
            
    #     print('Saving wweights!')
        
        
    def save_weights(self):
        if not os.path.isdir('DeepMellow_weights'):
            os.makedirs('DeepMellow_weights')
        with open('DeepMellow_weights/trainable_params.pickle', 'wb') as file:
            trainable_params = []
            for w in self.net.trainable_params:
                trainable_params.append(w.numpy())
                
            pickle.dump(trainable_params, file)
        print("weights where saved properlly!!!")
        
        
    def load_weights(self):
        if os.path.isdir('DeepMellow_weights'):
            with open('DeepMellow_weights/trainable_params.pickle', 'rb') as file:
                trainable_params = pickle.load(file)
            for w, w2 in zip(trainable_params, self.net.trainable_params):
                w2.assign(w)
                
            print("weights were load successfully!!!")
                
        else:
            print("weights load where not successed!!!")
            print('pass this loading and start randomlly')
            time.sleep(10)
            
    
    def load_given_weights(self, given_weights):
        for w1, w2 in zip(self.net.trainable_params, given_weights):
            w1.assign(w2)
        
        

        
    def mellowMax(self, x, axis = 1):
        c = tf.reduce_max(x, axis = 1).numpy()
        x_shifted = x.numpy() - c[:,np.newaxis]
        exp_x = tf.math.exp(x_shifted*self.w) 
        mean_i = tf.reduce_mean(exp_x, axis = axis)
        log_i = tf.math.log(mean_i)
        return (log_i/self.w + c).numpy()

    
    def learn(self, experience_replay_buffer , batch_size = 64):# something with the batchsize id weird
        # SAmple experiences
        states, actions, rewards, next_states, dones = experience_replay_buffer.get_minibatch()
        
        # Calculate targets
        # change here require Mellowmax
        next_Qs = self.predict(next_states) # in R(n*Num_actions)
        next_Q = self.mellowMax(next_Qs, axis = 1)
        targets = rewards + np.invert(dones).astype(np.float32) * self.gamma * next_Q
        # Update model
        cost_i = self.update_weights(states, actions, targets)
        return cost_i


#######################
# class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

#   def __init__(self, initial_learning_rate):
#     self.initial_learning_rate = initial_learning_rate

#   def __call__(self, step):
#      return self.initial_learning_rate / (step + 1)

# optimizer = tf.keras.optimizers.SGD(learning_rate=MyLRSchedule(0.1))