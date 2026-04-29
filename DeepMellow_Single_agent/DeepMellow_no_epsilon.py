
import tensorflow as tf 
import numpy as np
import pickle  
import os 
import time 
import tensorflow_probability as tfp

def masking(obs):
    mask = []
    #print("State:", obs)
    return_same_channel = True
    for m in obs:
        if m == -1 or m == 0:
            mask.append(tf.float32.min)
        else:
            mask.append(0)
            return_same_channel = False
    return mask, return_same_channel
            
    
    
    
class DeepMellow(object):
    def __init__(self,net, number_of_actions,gamma,
                 lr = 0.00025,
                 optimizer = tf.keras.optimizers.Adam, 
                 los_func = tf.keras.losses.Huber(), 
                 mellowmax_constant = None,
                 l2_regularization = False):
        """
        K = number of output nodes
        """
        self.l2_regularization = l2_regularization
        self.w = mellowmax_constant
        self.gamma = gamma
        self.K = number_of_actions # action sapce
        self.net = net
        # the class of the loss 
        self.loss_func = los_func
        # self.distribution = tfp.distributions.Categorical
        ## OPtimizer
        self.optimizer = optimizer(learning_rate= lr)
        self.number_of_bits =  int(np.floor(np.log2(number_of_actions)) + 1)
        
    # @tf.function       
    def forward(self, Z, training): 
        Z = self.net.forward(x = Z, training = training)
        return Z
         
    def predict(self, x, training):
        return self.forward(x, training)
    
    # @tf.function
    def cost(self, S, actions, G):
        prediction = self.forward(S,training= False) # S is the states, predictions is actions 
        
        # now take into account only the actions that you choose in the past 
        # print("K:", self.K, )
        actual_chosen_actions = prediction * tf.one_hot(actions, self.K)
      
        selected_action_values = tf.reduce_sum(actual_chosen_actions, axis = [1])
        # print(selected_action_values, "g", G)
        costi = tf.reduce_mean(self.loss_func(y_true = G, y_pred = selected_action_values)) # in the paper it is L2 not huber
        if self.l2_regularization > 0.0:
            l2 = 0 
            counter =0 
            for w in self.net.trainable_params:
               val = tf.reduce_sum(
                   tf.square(w))
               
               l2 += val
               counter  += 1
               
            l2 = l2 / counter 
            costi += self.l2_regularization * l2
            # print("Boom I am in L2: ", self.l2_regularization, l2)
        return costi
        
        
    def sample_action(self, x, eps, training):

        x = np.expand_dims(x, axis = 0)
        x = np.float32(x)
        a_pos = self.predict(x, training)
        beta = 1
        
        logits = a_pos*beta
        
        
        #print(x.shape)
        
        """ Creating mask to avoid unpossibles actions !!!
        creating the mask depends on the last observation, 
        and have a symbole value of -1! 
        (1, 35, 1, 2)
        obs = x[0,num_of_bits:,0,-1] # should be (35,1) (#channels, 1, History), first 5 are location bit 
        mask = [tf.float32.min if m==-1 else 0 for m in obs.numpy()]
        
        logits = logits + mask 
        
        Now logits with impossible action has a zero probability to be chosen
        """
        
        obs = x[0, self.number_of_bits:,0,-1] # we start from 5 becasue we have binary incoding 
        #print(obs, obs.shape)
        mask,  return_same_channel = masking(obs) # [tf.float32.min if m==-1 or m == 0 else 0 for m in obs]
        
        #print("return_same_channel:", return_same_channel)
        #print(mask)
        if return_same_channel:
            channel_binary = x[0,: self.number_of_bits,0,-1]
            binary_str = ''
            
            for i in channel_binary:
                binary_str += str(int(i))
                
            a = int(binary_str, base = 2)
            # print("The original actions was:", a)
            
            return a 
       
        logits = logits + mask
        #print(logits)
        
        probs = tf.math.softmax(logits=logits) # beta = 9
        
        alpha = 0 # should be small number 
        probs = probs * (1-alpha) + alpha/(self.K) # check what is the number of action a \in [1,2,3,..., K]
        #print(probs)
        dis = tfp.distributions.Categorical(probs = probs)
        #time.sleep()
        #print("Probabilitis:", probs)
        if np.random.random() < eps:
            # a = np.argmax(dis.probs.numpy())
            action = dis.sample()
            a = action.numpy()[0]
            # always use this also in training 
        else:
            a = np.argmax(dis.probs.numpy())

        
        return a 
    
    
      
    # @tf.function
    def update_weights(self, states, actions, targets):
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            
            cost_i = self.cost(states, actions, targets)
            
        gradients = tape.gradient(cost_i, self.net.trainable_params) 
        self.optimizer.apply_gradients(zip(gradients, self.net.trainable_params))
        return cost_i
    
    
    def copy_params_from(self, params):
        for i in range(len(self.trainable_params)):
            self.trainable_params[i].assign(params[i].numpy())
        # self.trainable_params = [param for param in model.trainable_params] # here is the problem 
        
        
        
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
            
        
        
        
    def mellowMax(self, x, axis = 1):
        c = tf.reduce_max(x, axis = 1).numpy()
        x_shifted = x.numpy() - c[:,np.newaxis]
        exp_x = tf.math.exp(x_shifted*self.w) 
        mean_i = tf.reduce_mean(exp_x, axis = axis)
        log_i = tf.math.log(mean_i)
        return (log_i/self.w + c).numpy()

    def learn(self, experience_replay_buffer , batch_size = 64):# something with the batchsize id weird
        # SAmple experiences
        states, actions, rewards, next_states, dones, _ = experience_replay_buffer.get_minibatch()
        
        # Calculate targets
        # change here require Mellowmax
        next_Qs = self.predict(next_states, training = False) # in R(n*Num_actions)
        next_Q = self.mellowMax(next_Qs, axis = 1)
        targets = rewards + np.invert(dones).astype(np.float32) * self.gamma * next_Q
        # Update model
        cost_i = self.update_weights(states, actions, targets)
        
        return cost_i

    def load_given_weights(self, given_weights):
        for w1, w2 in zip(self.net.trainable_params, given_weights):
            w1.assign(w2)
        
#######################
# class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

#   def __init__(self, initial_learning_rate):
#     self.initial_learning_rate = initial_learning_rate

#   def __call__(self, step):
#      return self.initial_learning_rate / (step + 1)

# optimizer = tf.keras.optimizers.SGD(learning_rate=MyLRSchedule(0.1))