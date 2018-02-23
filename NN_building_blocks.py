#BUILDING BLOCKS


import numpy as np
import math
import tensorflow as tf

def lrelu(x, alpha=0.1):

    """
    Implements the leakyRELU function:

    inputs X, returns X if X>0, returns alpha*X if X<0
    """


    return tf.maximum(alpha*x,x)

def evaluation(Y_pred, Y):

    """
    Returns the accuracy by comparing the convoluted output Y_hat
    with the labels of the samples Y

    """
    
    correct = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    return accuracy

def supervised_random_mini_batches(X, Y, mini_batch_size, seed):

    """
    Creates a list of random mini_batches from (X, Y)
    
    Arguments:
    X -- input data, of shape (number of examples, input size)
    Y -- true "label" one hot matrix of shape (number of examples, n_classes)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)
    m = X.shape[0]        #number of examples in set
    n_classes = Y.shape[1]
    mini_batches=[]

    permutation = list(np.random.permutation(m))
    
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]
    #partition of (shuffled_X, shuffled_Y) except the last mini_batch
    
    num_complete_mini_batches = math.floor(m/mini_batch_size)
    for k in range(num_complete_mini_batches):
        mini_batch_X = shuffled_X[k*mini_batch_size:(k+1)*mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k*mini_batch_size:(k+1)*mini_batch_size,:]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    # handling the case of last mini_batch < mini_batch_size    
    if m % mini_batch_size !=0:
        
        mini_batch_X = shuffled_X[mini_batch_size*num_complete_mini_batches:m,:]
        mini_batch_Y = shuffled_Y[mini_batch_size*num_complete_mini_batches:m,:]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def unsupervised_random_mini_batches(X, mini_batch_size, seed):

    """
    Creates a list of random mini_batches from (X, Y)
    
    Arguments:
    X -- input data, of shape (number of examples, input size)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of mini_batch_X
    """
    np.random.seed(seed)
    m = X.shape[0]        #number of examples in set
    mini_batches=[]

    permutation = list(np.random.permutation(m))
    
    shuffled_X = X[permutation,:]
    #partition of shuffled_X except the last mini_batch
    
    num_complete_mini_batches = math.floor(m/mini_batch_size)
    for k in range(num_complete_mini_batches):
        mini_batch_X = shuffled_X[k*mini_batch_size:(k+1)*mini_batch_size,:]
        mini_batches.append(mini_batch_X)
        
    # handling the case of last mini_batch < mini_batch_size    
    if m % mini_batch_size !=0:
        
        mini_batch_X = shuffled_X[mini_batch_size*num_complete_mini_batches:m,:]

        mini_batches.append(mini_batch_X)
    
    return mini_batches



# DENSELY CONNECTED NETWORKS

class DenseLayer(object):

    """
    Creates a dense layer

    Constructor inputs:
        - name of layer
        - mi: (int) dimension of input to layer
        - mo: (int) dimension of output of the activated output
        - apply_batch_norm: (bool) wether applying or not batch_normalization along the 0 axis of input
        - f: (function) activation function of the output
        - w_init: (tensorflow initializer) which initializer to use for the layer weights

    Created attributes:

        - W: (tf variable) (mi x mo) weight matrix
        - bi: (tf variable) (mo, ) bias vector
        - bo: (tf variable) (mi, ) bias vector
    
    Class methods:

        - forward:
            input: X (tf tensor) previous output layer
            outputs: output (tf tensor) activated layer output as A = act_f(X * W+ bi)

        - forwardT:
            input: X (tf tensor) previous output layer
            outputs: output (tf tensor) activated layer output as A = act_f(X * tf.transpose(W)+ bo)
    
        - set_session:
            Sets current session
            input:
                session: (tf session) current session
    """
    
    def __init__(self, name, mi, mo, apply_batch_norm, keep_prob, 
                f=tf.nn.relu, w_init=tf.random_normal_initializer(stddev=0.02)
                ):
        
                self.W = tf.get_variable(
                    "W_%s" %name,
                    shape=(mi,mo),
                    initializer=w_init,
                )
                
                self.bi = tf.get_variable(
                    "bi_%s" %name,
                    shape=(mo,),
                    initializer=tf.zeros_initializer(),
                )

                self.bo = tf.get_variable(
                    "bo_%s" %name,
                    shape=(mi,),
                    initializer=tf.zeros_initializer(),
                )
                
                self.f=f
                self.name=name
                self.apply_batch_norm=apply_batch_norm
                self.keep_prob = keep_prob
        
    def forward(self, X, reuse, is_training):
        
        if not is_training:
            self.keep_prob=1

        Z=tf.matmul(X,self.W)+self.bi

        if self.apply_batch_norm:
            Z=tf.contrib.layers.batch_norm(
                Z,
                decay=0.9,
                updates_collections=None,
                epsilon=1e-5,
                scale=True,
                is_training = is_training,
                reuse=reuse,
                scope = self.name,
            )
        
        output = tf.nn.dropout(Z, self.keep_prob)
        return self.f(output)

    def forwardT(self, X, reuse, is_training):
        
        if not is_training:
            self.keep_prob=1

        Z=tf.matmul(X,tf.transpose(self.W))+self.bo

        if self.apply_batch_norm:
            Z=tf.contrib.layers.batch_norm(
                Z,
                decay=0.9,
                updates_collections=None,
                epsilon=1e-5,
                scale=True,
                is_training = is_training,
                reuse=reuse,
                scope = self.name,
            )
        
        output = tf.nn.dropout(Z, self.keep_prob)
        return self.f(output)

    def set_session(self, session):
        
        self.session = session

#2D CONVOLUTIONAL NETWORKS

class AvgPool2D(object):

    """
    Performs average 2D pooling of the input, with no dropout by default
    Note that this layer trains no parameters

    Constructor input:
        - filter_sz: (int) width and height of the square pooling window 
        - stride: (int) horizontal and vertical value for the stride

        ?extend to square windows and stride?

    Class methods:

        forward:
            inputs:
                - X (tf tensor) previous activated layer output
                - output (tf tensor) average pooled layer output

    """

    def __init__(self, filter_sz, stride, keep_prob=1):

        self.filter_sz = filter_sz
        self.stride = stride
        self.keep_prob = keep_prob

    def forward(self, X, reuse, is_training):

        if not is_training:
            self.keep_prob=1

        output = X

        output = tf.nn.avg_pool(X, 
            ksize=[1, self.filter_sz,self.filter_sz,1],
            strides=[1,self.stride, self.stride, 1],
            padding = 'SAME'
            )
        output = tf.nn.dropout(output, self.keep_prob)

        return output

    def set_session(self, session):

        self.session = session

class MaxPool2D(object):

    """
    Performs max 2D pooling of the input, with no dropout by default
    Note that this layer trains no parameters

    Constructor input:
        - filter_sz: (int) width and height of the square pooling window 
        - stride: (int) horizontal and vertical value for the stride

        ?extend to square windows and stride?

    Class methods:

        forward:
            inputs:
                - X (tf tensor) previous activated layer output
                - output (tf tensor) max pooled layer output

    """

    def __init__(self, filter_sz, stride, keep_prob):

        self.filter_sz = filter_sz
        self.stride = stride
        self.keep_prob = keep_prob

    def forward(self, X, reuse, is_training):

        output = X

        if not is_training:
            self.keep_prob=1

        output = tf.nn.max_pool(output, 
            ksize=[1, self.filter_sz,self.filter_sz,1],
            strides=[1,self.stride, self.stride, 1],
            padding = 'SAME'
            )

        output = tf.nn.dropout(output, self.keep_prob)
        return output

    def set_session(self, session):

        self.session = session

class ConvLayer(object):
    
    """
    Performs 2D strided convlution on rectangular sized input tensor. 

    Constructor inputs:

        - mi: (int) input channels
        - mo: (int) output channels
        - filter_sz: (int) width and height of the convolution kernel
        - stride: (int) horizontal and vertical size of the stride
        - apply_batch_norm: (bool) whether applying batch normalization (along [0] axis of input tensor) or not
        - keep_prob: (int) dropout keep probability of propagated tensor
        - f: (function) activation layer function. tf.nn.relu by default
        - w_init: (tf initializer) initialization of the filter parameters, by default tf.truncated_normal_initializer(stddev=0.02)

    Class attributes

        - W: (tf tensor) variable tensor of dim (filter_sz, filter_sz, mi, mo) of trainable weights
        - b: (tf tensor) variable tensor of dim (mo, ) of trainable biases

    Class methods:

        - forward: 
            inputs:
                X: (tf tensor) input tensor of dim (batch_sz, n_W, n_H, mi) output of previous layer
                reuse: (bool) whether reusing the stored batch_normalization parameters or not
                is_training: (bool) flag that indicates whether we are in the process of training or not
            outputs:
                output: (tf tensor) activated output of dim (batch_sz, n_W, n_H, mo), input for next layer
        
        - set_session:
            Sets current session
            inputs:
                session: (tf session) current session


    """



    def __init__(
            self, name, mi, mo, filter_sz, stride, 
                 apply_batch_norm, keep_prob, f = tf.nn.relu,
                 w_init = tf.truncated_normal_initializer(stddev=0.02)
            ):
                
            
            self.W = tf.get_variable(
                "W_%s" %name,
                shape=(filter_sz,filter_sz, mi, mo),
                initializer=w_init,
            )
            
            self.b = tf.get_variable(
                "b_%s" %name,
                shape = (mo,),
                initializer=tf.zeros_initializer(),
            )
            
            self.name = name
            self.f = f
            self.stride = stride
            self.apply_batch_norm = apply_batch_norm
            self.keep_prob=keep_prob
        
    def forward(self, X, reuse, is_training):
        
        if not is_training:
            self.keep_prob=1

        conv_out = tf.nn.conv2d(
            X,
            self.W,
            strides=[1,self.stride,self.stride,1],
            padding='SAME'
        )
        conv_out = tf.nn.bias_add(conv_out,self.b)
        
        #applying batch_norm
        if self.apply_batch_norm:
            
            conv_out = tf.contrib.layers.batch_norm(
                conv_out,
                decay=0.9,
                updates_collections=None,
                epsilon=1e-5,
                scale=True,
                is_training = is_training,
                reuse=reuse,
                scope = self.name,
            )
        output = self.f(conv_out)  
        output = tf.nn.dropout(output, self.keep_prob)
        return output 
    
    def set_session(self, session):
        
        self.session = session

class DeconvLayer(object):
  
    """
    Performs 2D strided deconvlution on rectangular sized input tensor. 

    Constructor inputs:

        - mi: (int) input channels
        - mo: (int) output channels
        - output_shape: list = [int, int], list[0]([1]) is the witdth (height) of the output after deconvolution, the layer computes the resizing
                                automatically to match the output_shape. Default padding value is of 2 in both directions but can be modified.
        - filter_sz: (int) width and height of the convolution kernel
        - stride: (int) horizontal and vertical size of the stride
        - apply_batch_norm: (bool) whether applying batch normalization (along [0] axis of input tensor) or not
        - keep_prob: (int) dropout keep probability of propagated tensor
        - f: (function) activation layer function. tf.nn.relu by default
        - w_init: (tf initializer) initialization of the filter parameters, by default tf.truncated_normal_initializer(stddev=0.02)

    Class attributes

        - W: (tf tensor) variable tensor of dim (filter_sz, filter_sz, mo, mo) of trainable weights
        - W_id: (tf tensor) variable tensor of dim (1, 1, mi, mo) of trainable weights
        - b: (tf tensor) variable tensor of dim (mo, ) of trainable biases

    Class methods:

        - forward: 
            inputs:
                X: (tf tensor) input tensor of dim (batch_sz, n_W, n_H, mi) output of previous layer
                reuse: (bool) whether reusing the stored batch_normalization parameters or not
                is_training: (bool) flag that indicates whether we are in the process of training or not

                The deconvolution is computed in 3 steps according to: (cite article)
                1) conv2d with W_id is performed to match the output channels mo, with filter_sz=1 and stride=1
                2) the image is reshaped to reshape_size_H and W 
                3) conv2d with W is performed with input filter_sz and stride, output width and height will match output_shape

            outputs:
                output: (tf tensor) activated output of dim (batch_sz, n_W=output_shape[0], n_H = output_shape[1], mo), input for next layer
        
        - set_session:
            Sets current session
            inputs:
                session: (tf session) current session


    """  
    def __init__(self, name, 
                mi, mo, output_shape, 
                filter_sz, stride, 
                apply_batch_norm, keep_prob,
                f=tf.nn.relu, w_init = tf.random_normal_initializer(stddev=0.02)
                ):

                #using resize + conv2d and not conv2dt
                #mi: input channels
                #mo: output channels
                #output_shape: width and height of the output image
                
                #performs 2 convolutions, the first augments the number of channels
                #the second performs a convolution with != 1 kernel and stride
        
                self.W = tf.get_variable(
                    "W_%s" %name,
                    shape=(filter_sz, filter_sz, mo, mo),
                    initializer=w_init,
                )
                self.W_id = tf.get_variable(
                    'W_%s_id' %name,
                    shape=(1,1,mi, mo),
                    initializer=w_init,
                )

                self.b = tf.get_variable(
                    "b_%s" %name,
                    shape=(mo,),
                    initializer=tf.zeros_initializer(),
                )

                self.f = f
                self.stride = stride
                self.filter_sz = filter_sz
                self.name = name
                self.output_shape = output_shape
                self.apply_batch_norm = apply_batch_norm
                self.keep_prob = keep_prob
                
        
    def forward(self, X, reuse, is_training):

        if not is_training:
            self.keep_prob = 1
        
        padding_value = 2
        resized_shape_H = (self.output_shape[0] -1)*self.stride+ self.filter_sz-2*padding_value
        resized_shape_W = (self.output_shape[1] -1)*self.stride+ self.filter_sz-2*padding_value
        
        
        resized = tf.image.resize_images(X, 
                                          [resized_shape_H,
                                           resized_shape_W],
                                          method=tf.image.ResizeMethod.BILINEAR)
        
        #print('After first resize shape', resized.get_shape())
        
        output_id = tf.nn.conv2d(
            resized,
            self.W_id,
            strides=[1,1,1,1],
            padding='VALID'
        )

        #print('After id deconvolution shape', output_id.get_shape())
        
        paddings = tf.constant([[0,0],
                                [padding_value, padding_value], 
                                [padding_value, padding_value], 
                                [0,0]])
        
        output_id = tf.pad(output_id,
                          paddings, 'CONSTANT'
                          )
        
        #print('After padding', output_id.get_shape())

        conv_out = tf.nn.conv2d(
            output_id,
            self.W,
            strides=[1,self.stride,self.stride,1],
            padding='VALID'
        )

        #print('After deconvolution', conv_out.get_shape())
        conv_out = tf.nn.bias_add(conv_out,self.b)

        if self.apply_batch_norm:
            
            conv_out = tf.contrib.layers.batch_norm(
                conv_out,
                decay=0.9,
                updates_collections=None,
                epsilon=1e-5,
                scale=True,
                is_training = is_training,
                reuse=reuse,
                scope = self.name,
            )

        output = self.f(conv_out) 
        output = tf.nn.dropout(output, self.keep_prob)

        return output

    def set_session(self, session):
        
        self.session = session

class ConvBlock(object):

    """
    Performs series of 2D strided convolutions on rectangular sized input tensor. The convolution proceeds on two parallel
    paths: main path is composed by n subsequent convolutions while shortcut path has 1 convolution with different parameters and
    can be set as the identity convolution. 
    
    Constructor inputs:


        - block_id: progressive number of block in the network
        - init_mi: (int) input channels
        - sizes: (dict) python dictionary with keys:
                            
            sizes = { 'convblock_layer_n':[(n_c+1, kernel, stride, apply_batch_norm, keep_prob, act_f, weight initializer),
                               (n_c+2,,,,,,,),
                               (n_c+...,,,,,,,),
                               (n_c+ last,,,,,,,],

            'convblock_shortcut_layer_n':[(n_c+3, kernel, stride, apply_batch_norm, act_f, weight initializer)],
            'dense_layers':[(dim output, apply_batch_norm, keep_prob, act_f, weight initializer )]
            }

            - filter_sz: (int) width and height of the convolution kernel at that layer
            - stride: (int) horizontal and vertical size of the stride at that layer
            - apply_batch_norm: (bool) whether applying batch normalization (along [0] axis of input tensor) or not
            - keep_prob: (int) dropout keep probability of propagated tensor
            - f: (function) activation layer function. tf.nn.relu by default
            - w_init: (tf initializer) initialization of the filter parameters, by default tf.truncated_normal_initializer(stddev=0.02)
            
            sizes['convblock_layer_n'] is a list of tuples of layers specifications for the main path 
            sizes['convblock_shortcut_layer_n'] is a list composed by 1 tuple of layer specifications for the shortcut path
            sizes['dense_layers'] is a list of tuples of layers specifications for the densely connected part of the network
    
    Class attributes

        - conv_layers: (list) list of conv_layer objects
        - shortcut_layer: (list) conv_layer object

    Class methods:

        - forward: 
            inputs:
                X: (tf tensor) input tensor of dim (batch_sz, n_W, n_H, mi) output of previous layer
                reuse: (bool) whether reusing the stored batch_normalization parameters or not
                is_training: (bool) flag that indicates whether we are in the process of training or not
            outputs:
                output: (tf tensor) activated output of dim (batch_sz, n_W, n_H, mo), input for next layer
        
        - set_session:
            Sets current session
            inputs:
                session: (tf session) current session

    """

    def __init__(self,
                block_id,
                init_mi, sizes,
                ):
     
                #self.f=f
                self.conv_layers = []
                mi = init_mi
                self.block_id=block_id
                
                #build the block
                #main path
                count=0
                
                for mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init in sizes['convblock_layer_'+str(self.block_id)][:-1]:
                
                    name = 'convblock_{0}_layer_{1}'.format(block_id, count)
                    count += 1
                
                    layer = ConvLayer(name,
                                 mi, mo, filter_sz, stride, 
                                 apply_batch_norm, keep_prob,
                                 act_f, w_init)
                
                    self.conv_layers.append(layer)
                    mi = mo
                    
                name = 'convblock_{0}_layer_{1}'.format(block_id, count)


                mo, filter_sz, stride, apply_batch_norm, keep_prob, self.fin_act , w_init = sizes['convblock_layer_' +str(self.block_id)][-1]
            
                layer = ConvLayer(name,
                             mi, mo, filter_sz, stride,
                             apply_batch_norm, keep_prob,
                             lambda x: x, w_init)
            
                self.conv_layers.append(layer)
            
                #secondary path
                #set filter_sz = stride = 1 for an ID block
                mo, filter_sz, stride, apply_batch_norm, keep_prob, w_init = sizes['convblock_shortcut_layer_'+str(self.block_id)][0]
                name = 'convshortcut_layer_{0}'.format(block_id)

                self.shortcut_layer = ConvLayer(name,
                                           init_mi, mo, filter_sz,stride,
                                           apply_batch_norm, keep_prob,
                                           f=lambda x: x, w_init=w_init)
                
                self.sizes=sizes
        
    def output_dim(self, input_dim):
        
        dim = input_dim
        for _, _, stride, _, _, _, _, in self.sizes['convblock_layer_'+str(self.block_id)]:
            dim = int(np.ceil(float(dim)/stride))
  
        return dim

    
    def set_session(self, session):
        
        self.session = session
        
        self.shortcut_layer.set_session(session)
        
        for layer in self.conv_layers:
            layer.set_session(session)
    
    def forward(self, X, reuse, is_training):
        
        output = X
        shortcut_output = X
        #print('Convolutional block %i' %self.block_id)
        #print('Input shape ', X.get_shape() )
        i=0
        for layer in self.conv_layers:
            i+=1
            output = layer.forward(output, reuse, is_training) 
            #print('After layer %i' %i)
            #print('Output shape ', output.get_shape())
            
        shortcut_output = self.shortcut_layer.forward(shortcut_output,reuse, is_training)
        #print('Shortcut layer after convolution shape ', shortcut_output.get_shape())
        
        
        if (output.get_shape().as_list()[1], output.get_shape().as_list()[2]) != (shortcut_output.get_shape().as_list()[1], shortcut_output.get_shape().as_list()[2]):
            
            print('image size mismatch at conv block ' +str(self.block_id))
        
        if output.get_shape().as_list()[-1] != shortcut_output.get_shape().as_list()[-1]:
            
            print('image channels mismatch at conv block ' +str(self.block_id))
            
        else:
            
            output = output + shortcut_output
        
        return self.fin_act(output)

class DeconvBlock(object):

    """
    Performs series of 2D strided deconvolutions on rectangular sized input tensor. The convolution proceeds on two parallel
    paths: main path is composed by n subsequent deconvolutions while shortcut path has 1 deconvolution with different parameters and
    can be set as the identity deconvolution. 
    
    Constructor inputs:


        - block_id: progressive number of block in the network
        - init_mi: (int) input channels
        - sizes: (dict) python dictionary with keys:
                          
            sizes = { 

                        'deconvblock_layer_n':[(n_c+1, kernel, stride, apply_batch_norm, keep_prob, act_f, weight initializer),
                               (n_c+2,,,,,,,),
                               (n_c+...,,,,,,,),
                               (n_c+ last,,,,,,,],

                        'deconvblock_shortcut_layer_n':[(n_c+3, kernel, stride, apply_batch_norm, act_f, weight initializer)],
                        'dense_layers':[(dim output, apply_batch_norm, keep_prob, act_f, weight initializer )]
                     }

            - filter_sz: (int) width and height of the convolution kernel at that layer
            - stride: (int) horizontal and vertical size of the stride at that layer
            - apply_batch_norm: (bool) whether applying batch normalization (along [0] axis of input tensor) or not
            - keep_prob: (int) dropout keep probability of propagated tensor
            - f: (function) activation layer function. tf.nn.relu by default
            - w_init: (tf initializer) initialization of the filter parameters, by default tf.truncated_normal_initializer(stddev=0.02)
            
            sizes['deconvblock_layer_n'] is a list of tuples of layers specifications for the main path 
            sizes['deconvblock_shortcut_layer_n'] is a list composed by 1 tuple of layer specifications for the shortcut path
            sizes['dense_layers'] is a list of tuples of layers specifications for the densely connected part of the network
    
    Class attributes

        - conv_layers: (list) list of conv_layer objects
        - shortcut_layer: (list) conv_layer object

    Class methods:

        - forward: 
            inputs:
                X: (tf tensor) input tensor of dim (batch_sz, n_W, n_H, mi) output of previous layer
                reuse: (bool) whether reusing the stored batch_normalization parameters or not
                is_training: (bool) flag that indicates whether we are in the process of training or not
            outputs:
                output: (tf tensor) activated output of dim (batch_sz, n_W, n_H, mo), input for next layer
        
        - set_session:
            Sets current session
            inputs:
                session: (tf session) current session

    """
    
    def __init__(self,
                block_id,
                mi, output_sizes, 
                sizes):
     
                #self.f=f
                init_mi=mi
                self.deconv_layers = []
                self.block_id=block_id
                
                #output shapes has to be a dictionary of [n_H,n_W]

                #build the block
                #main path
                
                for i in range(len(sizes['deconvblock_layer_'+str(block_id)])-1):
                    
                    mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init = sizes['deconvblock_layer_'+str(block_id)][i]
                    name = 'deconvblock_{0}_layer_{1}'.format(block_id, i)

                    layer = DeconvLayer(name,mi, mo, output_sizes[block_id][i],
                                        filter_sz, stride, 
                                        apply_batch_norm, keep_prob,
                                        act_f, w_init)
                
                    self.deconv_layers.append(layer)
                    mi = mo
                    
                i+=1
                name = 'deconvblock_{0}_layer_{1}'.format(block_id, i)
                mo, filter_sz, stride, apply_batch_norm, keep_prob, self.fin_act, w_init = sizes['deconvblock_layer_' +str(block_id)][-1]
                

                layer = DeconvLayer(name, mi, mo, output_sizes[block_id][len(output_sizes[block_id])-1],
                                    filter_sz, stride,
                                    apply_batch_norm, keep_prob,
                                    lambda x: x, w_init)
            
                self.deconv_layers.append(layer)

                #secondary path
                mo, filter_sz, stride, apply_batch_norm, keep_prob, w_init = sizes['deconvblock_shortcut_layer_'+str(block_id)][0]
                name = 'deconvshortcut_layer_{0}'.format(block_id)
                
                self.shortcut_layer = DeconvLayer(name, init_mi, mo, output_sizes[block_id][len(output_sizes[block_id])-1],
                                           filter_sz,stride,
                                           apply_batch_norm, keep_prob,
                                           f=lambda x: x, w_init=w_init)

    def set_session(self, session):
        
        self.session = session
        
        self.shortcut_layer.set_session(session)
        
        for layer in self.deconv_layers:
            layer.set_session(session)    
    
    def forward(self, X, reuse, is_training):
        
        
        output = X
        shortcut_output = X
        
        #print('Deconvolutional block %i' %self.block_id)
        #print('Input shape ', X.get_shape() )
        i=0


        for layer in self.deconv_layers:
            i+=1
            output = layer.forward(output, reuse, is_training)  
            #print('After layer %i' %i)
            #print('Output shape ', output.get_shape())

        shortcut_output = self.shortcut_layer.forward(shortcut_output, reuse, is_training)
        #print('Shortcut layer after convolution shape ', shortcut_output.get_shape())
        
        if (output.get_shape().as_list()[1], output.get_shape().as_list()[2]) != (shortcut_output.get_shape().as_list()[1], shortcut_output.get_shape().as_list()[2]):
            
            print('Image size mismatch at deconv block ' +str(self.block_id))
        
        if output.get_shape().as_list()[-1] != shortcut_output.get_shape().as_list()[-1]:
            
            print('Image channels mismatch at deconv block ' +str(self.block_id))
            
            
        else:
            
            output = output + shortcut_output
        
        return self.fin_act(output)


    def output_dim(self, input_dim):
        
        dim = input_dim
        for _, _, stride, _, _, _, _, in self.sizes['deconvblock_layer_'+str(self.block_id)]:
            dim = int(np.ceil(float(dim)/stride))
  
        return dim
