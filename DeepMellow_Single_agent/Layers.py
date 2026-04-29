import numpy as np 
import tensorflow as tf 

class DenseLayer(object):
    def __init__(self, M1, M2, f = tf.nn.relu, use_bias = True, layerNum = 0):
        
        self.use_bias = use_bias
        self.f = f
        # we need to use scaling using 0 and var = 1 ~N(0,1)
        # He normalization 
        W0= np.random.randn(M1, M2).astype(np.float32) * np.sqrt(2./float(M1))
        self.W = tf.Variable(initial_value = W0, name = 'W_dense_%i' % layerNum)
        
        self.params = [self.W]
        
        if self.use_bias:
            self.b = tf.Variable(initial_value = tf.zeros([M2,]), 
                                 name = 'b_dense_%i' % layerNum)
            
            self.params.append(self.b)
        
    def forward(self, X):
        Z = tf.matmul(X, self.W)
        
        if self.use_bias:
            Z += self.b
            
        return self.f(Z)
        
        
class ConvLayer(object):
    def __init__(self, mi, mo, filtersz = (4,4), stride = (2,2), f = tf.nn.relu, pad = 'VALID', add_bias = True, layerNum = 0):#SAME
        
        # mi = input feature map size
        # mo = ouput feature map size
        # stride -->(height, width)
        self.f = f
        self.add_bias = add_bias
        self.stride = stride
        self.pad = pad
        
        # also here I need to normelize properlly 
        # He normalization
        self.W = tf.Variable(initial_value = tf.random.normal(shape = [filtersz[0], filtersz[1], mi, mo],mean= 0.,
                                                              stddev = np.sqrt(2.0/(filtersz[0]*filtersz[1]*mi))),
                                                                 name = 'W_conv_%i' % layerNum)
        
        self.params = [self.W]
        
        if self.add_bias:
            self.b = tf.Variable(initial_value = tf.zeros(mo,), name = 'b_conv_%i' % layerNum)
            
            self.params.append(self.b)
            
    
    def forward(self, X):
        conv_out = tf.nn.conv2d(X, filters = self.W, strides = [1,self.stride[0], self.stride[1],1], padding=self.pad)
        
        if self.add_bias:
            conv_out = tf.nn.bias_add(conv_out, self.b)
            
        return self.f(conv_out)
    
    
    
    
class LSTM_tf(object):
    def __init__(self, M2,activation='tanh'):
        self.layer = tf.keras.layers.LSTM(M2,activation=activation)
        
        self.trainable_params = self.layer.get_weights()
        
    def forward(self, X):
        y = self.layer(X)
        return y
    
    
    
class LSTM(object):
    def __init__(self, M1, M2, activation = tf.nn.relu, i_d = 0, activation_func = tf.tanh, train_h0 = False, train_c0 = False):
        self.M1  = M1
        self.M2 = M2
        self.i_d = i_d
        self.f = activation_func
        # Forget gate variables 
        self.W_xf = tf.Variable(initial_value= tf.random.normal(shape = (M1, M2)) * tf.math.sqrt(2/M1), 
                             name  = 'lstm_W_xf' + str(i_d))
        
        self.W_hf = tf.Variable(initial_value = tf.random.normal(shape = (M2, M2)) * tf.math.sqrt(2/M2),
                               name = 'lstm_W_hf' + str(i_d))
        
        self.b_f = tf.Variable(initial_value = tf.zeros(shape = (M2,)), 
                              name = "lstm_b_f" + str(i_d))
        
        # input/update gate VAriables 
        self.W_xi = tf.Variable(initial_value = tf.random.normal(shape = (M1, M2)) * tf.math.sqrt(2/M1),
                                name = 'lstm_W_xi' +str(i_d))
        
        self.W_hi = tf.Variable(initial_value = tf.random.normal(shape = (M2, M2)) * tf.sqrt(2/M2), 
                               name  = 'lstm_W_hi')
        
        self.b_i = tf.Variable(initial_value = tf.zeros(shape = (M2, )), 
                               name = 'lstm_b_i' + str(i_d))
        
        # output gate variables
        
        self.W_xo = tf.Variable(initial_value = tf.random.normal(shape = (M1, M2)) * tf.math.sqrt(2/M1), 
                                name = 'lstm_W_xo' + str(i_d))
        
        self.W_ho = tf.Variable(initial_value = tf.random.normal(shape = (M2, M2)) * tf.sqrt(2/M2), 
                               name  = 'lstm_W_ho' + str(i_d))
        
        self.b_o = tf.Variable(initial_value = tf.zeros(shape = (M2, )), 
                               name = 'lstm_b_o' + str(i_d))
        
        # variables for the simpleRNN (cell store)
        
        self.W_xc = tf.Variable(initial_value = tf.random.normal(shape = (M1, M2)) * tf.math.sqrt(2/M1), 
                                name = 'lstm_W_xc' + str(i_d))
        
        self.W_hc = tf.Variable(initial_value = tf.random.normal(shape = (M2, M2)) * tf.sqrt(2/M2), 
                               name  = 'lstm_W_hc' + str(i_d))
        
        self.b_c = tf.Variable(initial_value = tf.zeros(shape = (M2, )), 
                               name = 'lstm_b_c' + str(i_d))
        
        # the initial fot t = 0 , this params in the original tensorflow are not trainable 
        # in this version of lstm layer they can be defined to be trainable or not 
        self.h0 = tf.Variable(initial_value = tf.zeros(shape  = (1,M2)), 
                              name = 'lstm_h0' + str(i_d), trainable = train_h0)
        
        self.c0 = tf.Variable(initial_value = tf.zeros(shape  = (1,M2)), 
                              name = 'lstm_c0' + str(i_d), trainable = train_c0)
        
        
        
        self.trainable_params = [self.W_xf, self.W_hf, self.b_f, 
                                 self.W_xi, self.W_hi, self.b_i,
                                 self.W_xo, self.W_ho, self.b_o,
                                 self.W_xc, self.W_hc, self.b_c,
                                 ]
        
        if train_h0:
            self.trainable_params.append(self.h0)
        
        if train_c0:
            self.trainable_params.append(self.c0)
            
            
    def reccurent(self, x_t, h_t_1, c_t_1):
        # forget gate 
        x_t = tf.expand_dims(x_t, axis = 0)
        # print(x_t.shape)
        f_t = tf.sigmoid(tf.matmul(x_t, self.W_xf) + tf.matmul(h_t_1, self.W_hf) + self.b_f)
        # print(f_t.shape)
        # input/output gate 
        i_t = tf.sigmoid(tf.matmul(x_t, self.W_xi) + tf.matmul(h_t_1, self.W_hi) + self.b_i)
        
        # output gate 
        o_t = tf.sigmoid(tf.matmul(x_t, self.W_xo) + tf.matmul(h_t_1, self.W_ho) + self.b_o)
        
        # cell store 
        c_t = tf.math.multiply(f_t, c_t_1) + tf.math.multiply( i_t, 
                                                              self.f(tf.matmul(x_t, self.W_xc) + tf.matmul(h_t_1, self.W_hc) + 
                                                                     self.b_c)
                                                              )
        h_t = tf.math.multiply(o_t, self.f(c_t))
        
        return h_t, c_t
    
    def forward(self, Z): # Z in R{N, T, D}
        
        N, T, D = tf.shape(Z).numpy()
        
        for i in range(N):
            z = Z[i,:,:]
            h_t = self.h0
            c_t = self.c0
         
            for j in range(T):
                h_t, c_t = self.reccurent(z[j,:], h_t, c_t)
                
                
            if i == 0:
                results = h_t
                
            else:
                results = tf.concat((results, h_t), axis = 0)
            
        return results                    