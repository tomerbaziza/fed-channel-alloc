
from Layers import ConvLayer, DenseLayer
import tensorflow as tf
import numpy  as np 

class CNN(object):
    def __init__(self, num_channels, dims_input_x_y , K, conv_layer_sizes, hidden_layer_sizes):
        
        # num_channels <-- the depth of the input 
        self.dims_input_x_y = dims_input_x_y # the size of the x,y dimention of the input  (numer of channels , width =1 )
        self.K = K # the number of output nodes of the net (number of actions)
        

        
            
        
        self.conv_layers = [ ]
        mi = num_channels
        count = 0 
        dims_x = dims_input_x_y[0]
        dims_y = dims_input_x_y[1]
        
        for mo, filtersz, stride in conv_layer_sizes:
            pad = 'VALID'
            layer = ConvLayer(mi, mo, filtersz, stride, layerNum= count, pad = pad)
            mi = mo
            self.conv_layers.append(layer)
            count += 1
            if pad == 'SAME':
                p_x = (stride *(dims_x/stride[0]) - dims_x + filtersz[0] - stride[0]) / 2
                p_y = (stride *(dims_y/stride[1]) - dims_y + filtersz[1] - stride[1]) / 2
            elif pad == 'VALID':
                p_x = 0#
                p_y = 0
                
            dims_x = np.ceil(1 + (dims_x - filtersz[0] + p_x + p_x)/stride[0]) ## just to know what is the size of the output (shoud be chaneg in case of other padding technique )
            dims_y = np.ceil(1 + (dims_y - filtersz[1] + p_y + p_y)/stride[1])
       
        # let's calculate the size of the input after fletten 
        M1 = int(dims_x * dims_y * mi )
        ## collect all the fully connected layers
        self.dense_layers = []
        # # architecture hiiden_layer_sizes = [[M2, f]]
        counter =0 
        for M2, f in hidden_layer_sizes:
            layer = DenseLayer(M1 , M2, f = f, layerNum = counter)
            self.dense_layers.append(layer)
            M1 = M2
            counter += 1
            
        # Now let's creat the last layer 
        layer = DenseLayer(M1, self.K, f = tf.identity) # linear
        self.dense_layers.append(layer)
        
        # collect the trainable params 
        # 1) collect from conv layers
        self.trainable_params = []
        for layer in self.conv_layers:
            self.trainable_params += layer.params
        
        # 2) collect from all the dense layers
        
        for layer in self.dense_layers:
            self.trainable_params += layer.params
            
    def forward(self, Z):
        # print(Z.shape)
        # convolution step 
        for layer in self.conv_layers:
            Z = layer.forward(Z)
            # print(Z.shape)
            
        n, h, w, c = Z.shape
        Z = tf.reshape(Z, shape = (int(n), int(h*w*c))
                       )
        # fully connected 
        # print(Z.shape)
        for layer in self.dense_layers:
            Z = layer.forward(Z)
            # print(Z.shape)
        return Z
    
        

    def copy_params_from(self, model):
        for i in range(len(self.trainable_params)):
            self.trainable_params[i].assign(model.trainable_params[i].numpy())
 
            

class ANN(object):
    def __init__(self, dims, K, hidden_layer_sizes):
        #  K, hidden_layer_sizes = [[M2,f], ....]
        # num_channels <-- the depth of the input 
        self.K = K # the number of output nodes of the net (number of actions)
        
        
            
       
        # let's calculate the size of the input after fletten 
        M1 = dims
        ## collect all the fully connected layers
        self.dense_layers = []
        # # architecture hiiden_layer_sizes = [[M2, f]]
        counter =0 
        for M2, f in hidden_layer_sizes:
            layer = DenseLayer(M1 , M2, f = f, layerNum = counter)
            self.dense_layers.append(layer)
            M1 = M2
            counter += 1
            
        # Now let's creat the last layer 
        layer = DenseLayer(M1, self.K, f = tf.identity) # linear
        self.dense_layers.append(layer)
        
        # collect the trainable params 
        self.trainable_params = []
        
        for layer in self.dense_layers:
            self.trainable_params += layer.params
            
    def forward(self, Z):
        n,h,w,c = np.shape(Z)
        # print(n,h,w,c)
        Z = Z.reshape(n,h*w*c)
        Z = np.float32(Z)
        for layer in self.dense_layers:
            Z = layer.forward(Z)
            
        return Z
    
        

    def copy_params_from(self, params):
        for i in range(len(self.trainable_params)):
            self.trainable_params[i].assign(params[i].numpy())
            

