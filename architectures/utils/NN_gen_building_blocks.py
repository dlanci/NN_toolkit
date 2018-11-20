#GENERATIVE MODELS BUILDING BLOCKS
rnd_seed=1

import numpy as np
import os 
import math

import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

st = tf.contrib.bayesflow.stochastic_tensor
Normal = tf.contrib.distributions.Normal
Bernoulli = tf.contrib.distributions.Bernoulli

from architectures.utils.NN_building_blocks import *
from architectures.utils.toolbox import *
#GANS

#(residual) convolution to (?,1) shape
class Discriminator(object):

    def __init__(self, X, d_sizes, d_name):

        self.residual=False

        for key in d_sizes:
            if not 'block' in key:
                self.residual=False
            else:
                self.residual=True

        _, dim_H, dim_W, mi = X.get_shape().as_list()

        if not self.residual:
            print('Convolutional Network architecture detected for discriminator '+ d_name)


            with tf.variable_scope('discriminator_'+d_name) as scope:
                #building discriminator convolutional layers

                self.d_conv_layers =[]
                count=0
                for mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init in d_sizes['conv_layers']:
                    
                    # make up a name - used for get_variable
                    name = "d_conv_layer_%s" % count
                    #print(name)
                    count += 1

                    layer = ConvLayer(name, mi, mo, 
                                      filter_sz, stride, 
                                      apply_batch_norm, keep_prob,
                                      act_f, w_init)

                    self.d_conv_layers.append(layer)
                    mi = mo

                    dim_H = int(np.ceil(float(dim_H) / stride))
                    dim_W = int(np.ceil(float(dim_W) / stride))
                        
                mi = mi * dim_H * dim_W

                    #building discriminator dense layers

                self.d_dense_layers = []
                for mo, apply_batch_norm, keep_prob, act_f, w_init in d_sizes['dense_layers']:
                    
                    name = 'd_dense_layer_%s' %count
                    #print(name)
                    count +=1
                    
                    layer = DenseLayer(name, mi, mo,
                                      apply_batch_norm, keep_prob,
                                      act_f, w_init)
                    mi = mo
                    self.d_dense_layers.append(layer)
                    
                    #final logistic layer

                name = 'd_dense_layer_%s' %count
                w_init_last = d_sizes['readout_layer_w_init']
                #print(name)
                self.d_final_layer = DenseLayer(name, mi, 1, 
                                                    False, keep_prob=1, 
                                                    act_f=lambda x: x, w_init=w_init_last)
                

                self.d_name=d_name
        else:

            print('Residual Convolutional Network architecture detected for discriminator'+ d_name)
            
            with tf.variable_scope('discriminator_'+d_name) as scope:
                #building discriminator convolutional layers

                self.d_blocks = []
                #count conv blocks
                d_steps = 0
                for key in d_sizes:
                    if 'conv' in key:
                        if not 'shortcut' in key:
                             d_steps+=1

                d_block_n=0
                d_layer_n=0
                

                for key in d_sizes:
                     
                    if 'block' and 'shortcut' in key:
                    
                        d_block = ConvBlock(d_block_n,
                                   mi, d_sizes,
                                   )
                        self.d_blocks.append(d_block)
                        
                        mo, _, _, _, _, _, _, _, = d_sizes['convblock_layer_'+str(d_block_n)][-1]
                        mi = mo
                        dim_H = d_block.output_dim(dim_H)
                        dim_W = d_block.output_dim(dim_W)
                        d_block_n+=1
                        
                    
                    if 'conv_layer' in key:

                        name = 'd_conv_layer_{0}'.format(d_layer_n)

                        mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init = d_sizes[key][0]


                        d_conv_layer = ConvLayer(name, mi, mo,
                                               filter_sz, stride,
                                               apply_batch_norm, keep_prob,
                                               act_f, w_init
                            )

                        self.d_blocks.append(d_conv_layer)

                        mi = mo
                        dim_W = int(np.ceil(float(dim_W) / stride))
                        dim_H = int(np.ceil(float(dim_H) / stride))
                        d_layer_n+=1
                
                assert d_block_n+d_layer_n==d_steps, '\nCheck keys in d_sizes, \n total convolution steps do not mach sum between convolutional blocks and convolutional layers'
                
                count=d_steps

                mi = mi * dim_H * dim_W

                #build dense layers
                
                self.d_dense_layers = []
                for mo, apply_batch_norm, keep_prob, act_f, w_init in d_sizes['dense_layers']:
                    
                    name = 'd_dense_layer_%s' %count
                    count +=1
                    
                    layer = DenseLayer(name,mi, mo,
                                      apply_batch_norm, keep_prob, 
                                      act_f, w_init)
                    mi = mo
                    self.d_dense_layers.append(layer)
                
                #final logistic layer
                name = 'd_dense_layer_%s' %count
                w_init_last = d_sizes['readout_layer_w_init']
                #print(name)
                self.d_final_layer = DenseLayer(name, mi, 1, 
                                                    False, keep_prob=1, 
                                                    act_f=lambda x: x, w_init=w_init_last)
                

                self.d_steps=d_steps
                self.d_name = d_name

    def d_forward(self, X, reuse = None, is_training=True):

        if not self.residual:
            print('Discriminator_'+self.d_name)
            print('Convolution')

            output = X
            print('Input for convolution shape ', X.get_shape())
            i=0
            for layer in self.d_conv_layers:
                i+=1
                # print('Convolution_layer_%i' %i)
                # print('Input shape', output.get_shape())
                output = layer.forward(output,
                                     reuse, 
                                     is_training)
                #print('After convolution shape', output.get_shape())
            
            output = tf.contrib.layers.flatten(output)
            #print('After flatten shape', output.get_shape())
            i=0
            for layer in self.d_dense_layers:
                #print('Dense weights %i' %i)
                #print(layer.W.get_shape())
                output = layer.forward(output,
                                       reuse,
                                       is_training)
                i+=1
                # print('After dense layer_%i' %i)
                # print('Shape', output.get_shape())

            logits = self.d_final_layer.forward(output, 
                                                reuse,
                                                is_training)
            print('Logits shape', logits.get_shape())
            return logits
        else:
            print('Redisual discriminator_'+self.d_name)
            print('Convolution')

            output = X

            i=0
            print('Input for convolution shape ', X.get_shape())
            for block in self.d_blocks:
                i+=1
                #print('Convolution_block_%i' %i)
                #print('Input shape', output.get_shape())
                output = block.forward(output,
                                         reuse,
                                         is_training)
                #print('After block shape', output.get_shape())
            
            
            output = tf.contrib.layers.flatten(output)
            #print('After flatten shape', output.get_shape())

            i=0
            for layer in self.d_dense_layers:
                #print('Dense weights %i' %i)
                #print(layer.W.get_shape())
                output = layer.forward(output,
                                       reuse,
                                       is_training)
                i+=1
                #print('After dense layer_%i' %i)
                #print('Shape', output.get_shape())

            logits = self.d_final_layer.forward(output, 
                                                reuse,
                                                is_training)
            print('Logits shape', logits.get_shape())
            return logits

#patchGAN architecture, convolution to (?, n_H_d, n_W_d, 1) shape
class pix2pixDiscriminator(object):

    def __init__(self, X, d_sizes, d_name):

        self.residual=False

        for key in d_sizes:
            if not 'block' in key:
                self.residual=False
            else:
                self.residual=True

        _, dim_H, dim_W, mi = X.get_shape().as_list()

        mi = 2*mi #takes as an input the concatenated true and fake images

        if not self.residual:
 
            print('Convolutional pix2pix Network architecture detected for discriminator '+ d_name)

            with tf.variable_scope('discriminator_'+d_name) as scope:
                
                #building discriminator convolutional layers
                self.d_conv_layers =[]

                count=0
                
                for mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init in d_sizes['conv_layers']:
                    
                    # make up a name - used for get_variable
                    name = "d_conv_layer_%s" % count
                    #print(name)
                    count += 1

                    layer = ConvLayer(name, mi, mo, 
                                      filter_sz, stride, 
                                      apply_batch_norm, keep_prob,
                                      act_f, w_init)

                    self.d_conv_layers.append(layer)
                    mi = mo
                    dim_H = int(np.ceil(float(dim_H) / stride))
                    dim_W = int(np.ceil(float(dim_W) / stride))
                        

                 #final unactivated conv layer
                filter_sz, stride, apply_batch_norm, keep_prob, w_init_last = d_sizes['readout_conv_layer'][0]
                count +=1
                name = 'last_conv_layer'
                self.last_conv_layer = ConvLayer(name, mi, 1,
                                      filter_sz, stride,
                                      apply_batch_norm, keep_prob,
                                      lambda x: x, w_init_last)
                self.d_name=d_name
        else:

            print('Residual Convolutional pix2pix Network architecture detected for discriminator'+ d_name)
            
            with tf.variable_scope('discriminator_'+d_name) as scope:
                #building discriminator convolutional layers

                self.d_blocks = []
                #count conv blocks
                d_steps = 0
                for key in d_sizes:
                    if 'conv' in key:
                        if not 'shortcut' in key:
                             d_steps+=1

                d_block_n=0
                d_layer_n=0
                

                for key in d_sizes:
                     
                    if 'block' and 'shortcut' in key:
                    
                        d_block = ConvBlock(d_block_n,
                                   mi, d_sizes,
                                   )
                        self.d_blocks.append(d_block)
                        
                        mo, _, _, _, _, _, _, _, = d_sizes['convblock_layer_'+str(d_block_n)][-1]
                        mi = mo
                        dim_H = d_block.output_dim(dim_H)
                        dim_W = d_block.output_dim(dim_W)
                        d_block_n+=1
                        
                    
                    if 'conv_layer' in key:

                        name = 'd_conv_layer_{0}'.format(d_layer_n)

                        mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init = d_sizes[key][0]


                        d_conv_layer = ConvLayer(name, mi, mo,
                                               filter_sz, stride,
                                               apply_batch_norm, keep_prob,
                                               act_f, w_init
                            )

                        self.d_blocks.append(d_conv_layer)

                        mi = mo
                        dim_W = int(np.ceil(float(dim_W) / stride))
                        dim_H = int(np.ceil(float(dim_H) / stride))
                        d_layer_n+=1
                
                assert d_block_n+d_layer_n==d_steps, '\nCheck keys in d_sizes, \n total convolution steps do not mach sum between convolutional blocks and convolutional layers'

                #final unactivated conv layer
                filter_sz, stride, apply_batch_norm, keep_prob, w_init_last = d_sizes['readout_conv_layer'][0]
                count +=1
                name = 'last_conv_layer'
                self.last_conv_layer = ConvLayer(name, mi, 1,
                                      filter_sz, stride,
                                      apply_batch_norm, keep_prob,
                                      lambda x: x, w_init_last)
                self.d_name=d_name

    def d_forward(self, X, samples, reuse = None, is_training=True):

        if not self.residual:
            print('Discriminator_'+self.d_name)

            output = tf.concat([X,samples],axis=3)
            print('Input for convolution shape ', X.get_shape())

            i=0
            for layer in self.d_conv_layers:
                i+=1
                # print('Convolution_layer_%i' %i)
                # print('Input shape', output.get_shape())
                output = layer.forward(output,
                                     reuse, 
                                     is_training)
                # print('After convolution shape', output.get_shape())

            logits = self.last_conv_layer.forward(output, 
                                                reuse,
                                                is_training)

            print('Logits shape', logits.get_shape())
            return logits
        else:
            print('Discriminator_'+self.d_name)

            output = tf.concat([X,samples],axis=3)
            print('Input for convolution shape ', X.get_shape())

            i=0
            for block in self.d_blocks:
                i+=1
                # print('Convolution_layer_%i' %i)
                # print('Input shape', output.get_shape())
                output = block.forward(output,
                                     reuse, 
                                     is_training)
                # print('After convolution shape', output.get_shape())

            logits = self.last_conv_layer.forward(output, 
                                                reuse,
                                                is_training)

            print('Logits shape', logits.get_shape())
            return logits

#convolution to (?, 1) shape with minibatch discrimination, outputs features for feature matching
class Discriminator_minibatch(object):

    def __init__(self, X, d_sizes, d_name):

        self.num_kernels=5
        self.kernel_dim=3

        _, dim_H, dim_W, mi = X.get_shape().as_list()


        print('Convolutional Network architecture detected for discriminator '+ d_name)


        with tf.variable_scope('discriminator_'+d_name) as scope:
            #building discriminator convolutional layers

            self.d_conv_layers =[]
            count=0
            for mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init in d_sizes['conv_layers']:
                
                # make up a name - used for get_variable
                name = "d_conv_layer_%s" % count
                #print(name)
                count += 1

                layer = ConvLayer(name, mi, mo, 
                                  filter_sz, stride, 
                                  apply_batch_norm, keep_prob,
                                  act_f, w_init)

                self.d_conv_layers.append(layer)
                mi = mo

                dim_H = int(np.ceil(float(dim_H) / stride))
                dim_W = int(np.ceil(float(dim_W) / stride))
                    
            mi = mi * dim_H * dim_W

                #building discriminator dense layers

            self.d_dense_layers = []
            for i, (mo, apply_batch_norm, keep_prob, act_f, w_init) in enumerate(d_sizes['dense_layers']):
                
                name = 'd_dense_layer_%s' %count
                #print(name)
                count +=1
                
                layer = DenseLayer(name, mi, mo,
                                  apply_batch_norm, keep_prob,
                                  act_f, w_init)
                
                self.d_dense_layers.append(layer)
                mi = mo
                if i == len(d_sizes['dense_layers'])-1:
                    name = "mb_disc_layer"
                    self.mb_layer = DenseLayer(name, mi, self.num_kernels*self.kernel_dim, False,
                                        keep_prob=1, act_f=lambda x:x, w_init=tf.truncated_normal_initializer(stddev=0.01))
                
    
            #final logistic layer

            name = 'd_dense_layer_%s' %count
            w_init_last = d_sizes['readout_layer_w_init']
            #print(name)
            #
            self.d_final_layer = DenseLayer(name, mi + self.num_kernels , 1, 
                                                False, keep_prob=1, 
                                                act_f=lambda x: x, w_init=w_init_last)
            

            self.d_name=d_name

            
    def d_forward(self, X, reuse = None, is_training=True):

        print('Discriminator_'+self.d_name)
        #print('Convolution')

        output = X
        print('Input for convolution shape ', X.get_shape())

        for i, layer in enumerate(self.d_conv_layers):
            # print('Convolution_layer_%i' %i)
            # print('Input shape', output.get_shape())
            output = layer.forward(output,
                                 reuse, 
                                 is_training)
            if i==np.ceil(len(self.d_conv_layers)/2):
                feature_output=output
            #print('After convolution shape', output.get_shape())
        
        output = tf.contrib.layers.flatten(output)
        #print('After flatten shape', output.get_shape())
        
        for i, layer in enumerate(self.d_dense_layers):
            #print('Dense weights %i' %i)
            #print(layer.W.get_shape())
            output = layer.forward(output,
                                   reuse,
                                   is_training)

            if i==len(self.d_dense_layers)-1:
                output_mb=self.mb_layer.forward(output, 
                                                reuse, 
                                                is_training)

            # print('After dense layer_%i' %i)
            # print('Shape', output.get_shape())

        activation = tf.reshape(output_mb, (-1, self.num_kernels, self.kernel_dim))
        diffs = tf.expand_dims(activation, 3) - tf.expand_dims(
                tf.transpose(activation, [1, 2, 0]), 0)

        eps = tf.expand_dims(tf.eye(tf.shape(X)[0], dtype=np.float32), 1)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2) + eps
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
        print('minibatch features shape', minibatch_features.get_shape())
        output=tf.concat([output, minibatch_features], 1)

        logits = self.d_final_layer.forward(output, 
                                            reuse,
                                            is_training)
        print('Logits shape', logits.get_shape())
        return logits, feature_output

class condDiscriminator(object):
    def __init__(self, X, dim_y, d_sizes, d_name):

        self.num_kernels=5
        self.kernel_dim=3

        self.residual=False

        for key in d_sizes:
            if not 'block' in key:
                self.residual=False
            else:
                self.residual=True

        _, dim_H, dim_W, mi = X.get_shape().as_list()

        mi = mi + dim_y
        self.dim_y=dim_y
        if not self.residual:
            print('Convolutional Network architecture detected for discriminator '+ d_name)


            with tf.variable_scope('discriminator_'+d_name) as scope:
                #building discriminator convolutional layers

                self.d_conv_layers =[]
                count=0
                for mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init in d_sizes['conv_layers']:
                    
                    # make up a name - used for get_variable
                    name = "d_conv_layer_%s" % count
                    #print(name)
                    count += 1

                    layer = ConvLayer(name, mi, mo, 
                                      filter_sz, stride, 
                                      apply_batch_norm, keep_prob,
                                      act_f, w_init)

                    self.d_conv_layers.append(layer)
                    mi = mo
                    mi = mi + dim_y

                    dim_H = int(np.ceil(float(dim_H) / stride))
                    dim_W = int(np.ceil(float(dim_W) / stride))
                        
                mi = mi * dim_H * dim_W

                    #building discriminator dense layers
                mi = mi + dim_y
                self.d_dense_layers = []
                for i, (mo, apply_batch_norm, keep_prob, act_f, w_init) in enumerate(d_sizes['dense_layers'], 0):
                    
                    name = 'd_dense_layer_%s' %count
                    #print(name)
                    count +=1
                    
                    layer = DenseLayer(name, mi, mo,
                                      apply_batch_norm, keep_prob,
                                      act_f, w_init)
                    mi = mo
                    mi = mi + dim_y
                    self.d_dense_layers.append(layer)

                    if i == len(d_sizes['dense_layers'])-1:
                        name = "mb_disc_layer"
                        self.mb_layer = DenseLayer(name, mi, self.num_kernels*self.kernel_dim, False,
                                        keep_prob=1, act_f=lambda x:x, w_init=tf.truncated_normal_initializer(stddev=0.01))
                    
                    #final logistic layer

                name = 'd_dense_layer_%s' %count
                w_init_last = d_sizes['readout_layer_w_init']
                #print(name)+self.num_kernels
                self.d_final_layer = DenseLayer(name, mi, 1, 
                                                    False, keep_prob=1, 
                                                    act_f=lambda x: x, w_init=w_init_last)
                

                self.d_name=d_name
                self.dim_y= dim_y
        else:

            print('Residual Convolutional Network architecture detected for discriminator'+ d_name)
            
            with tf.variable_scope('discriminator_'+d_name) as scope:
                #building discriminator convolutional layers

                self.d_blocks = []
                #count conv blocks
                d_steps = 0
                for key in d_sizes:
                    if 'conv' in key:
                        if not 'shortcut' in key:
                             d_steps+=1

                d_block_n=0
                d_layer_n=0
                

                for key in d_sizes:
                     
                    if 'block' and 'shortcut' in key:
                    
                        d_block = ConvBlock(d_block_n,
                                   mi, d_sizes,
                                   )
                        self.d_blocks.append(d_block)
                        
                        mo, _, _, _, _, _, _, _, = d_sizes['convblock_layer_'+str(d_block_n)][-1]
                        mi = mo
                        dim_H = d_block.output_dim(dim_H)
                        dim_W = d_block.output_dim(dim_W)
                        d_block_n+=1
                        
                    
                    if 'conv_layer' in key:

                        name = 'd_conv_layer_{0}'.format(d_layer_n)

                        mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init = d_sizes[key][0]


                        d_conv_layer = ConvLayer(name, mi, mo,
                                               filter_sz, stride,
                                               apply_batch_norm, keep_prob,
                                               act_f, w_init
                            )

                        self.d_blocks.append(d_conv_layer)

                        mi = mo
                        dim_W = int(np.ceil(float(dim_W) / stride))
                        dim_H = int(np.ceil(float(dim_H) / stride))
                        d_layer_n+=1
                
                assert d_block_n+d_layer_n==d_steps, '\nCheck keys in d_sizes, \n total convolution steps do not mach sum between convolutional blocks and convolutional layers'
                
                count=d_steps

                mi = mi * dim_H * dim_W

                #build dense layers
                
                self.d_dense_layers = []
                for mo, apply_batch_norm, keep_prob, act_f, w_init in d_sizes['dense_layers']:
                    
                    name = 'd_dense_layer_%s' %count
                    count +=1
                    
                    layer = DenseLayer(name,mi, mo,
                                      apply_batch_norm, keep_prob, 
                                      act_f, w_init)
                    mi = mo
                    self.d_dense_layers.append(layer)
                
                #final logistic layer
                name = 'd_dense_layer_%s' %count
                w_init_last = d_sizes['readout_layer_w_init']
                #print(name)
                self.d_final_layer = DenseLayer(name, mi, 1, 
                                                    False, keep_prob=1, 
                                                    act_f=lambda x: x, w_init=w_init_last)
                

                self.d_steps=d_steps
                self.d_name = d_name

    def d_forward(self, X, y, reuse = None, is_training=True):

        if not self.residual:
            print('Discriminator_'+self.d_name)
            print('Convolution')

            output = X 
            output = conv_concat(output, y, self.dim_y)  
            #output = conv_cond_concat(output, yb)
            print('Input for convolution shape ', output.get_shape())
            i=0
            for layer in self.d_conv_layers:
                i+=1
                #print('Convolution_layer_%i' %i)
                #print('Input shape', output.get_shape())
                output = layer.forward(output,
                                     reuse, 
                                     is_training)
                output = conv_concat(output, y, self.dim_y)
                if i==np.ceil(len(self.d_conv_layers)/2):
                    feature_output=output
                #print('After convolution shape', output.get_shape())
            
            output = tf.contrib.layers.flatten(output)
            output = lin_concat(output, y, self.dim_y)

            #print('After flatten shape', output.get_shape())
            i=0
            for layer in self.d_dense_layers:
                output = layer.forward(output,
                                       reuse,
                                       is_training)
                output = lin_concat(output, y, self.dim_y)
                if i==len(self.d_dense_layers)-1:
                    output_mb=self.mb_layer.forward(output, 
                                                reuse, 
                                                is_training)
                i+=1

                #print('After dense layer_%i' %i)
                #print('Shape', output.get_shape())
            activation = tf.reshape(output_mb, (-1, self.num_kernels, self.kernel_dim))
            diffs = tf.expand_dims(activation, 3) - tf.expand_dims(
                tf.transpose(activation, [1, 2, 0]), 0)

            eps = tf.expand_dims(tf.eye(tf.shape(X)[0], dtype=np.float32), 1)
            abs_diffs = tf.reduce_sum(tf.abs(diffs), 2) + eps
            minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
            print('minibatch features shape', minibatch_features.get_shape())
            #output=tf.concat([output, minibatch_features], 1)
            logits = self.d_final_layer.forward(output, 
                                                reuse,
                                                is_training)
            print('Logits shape', logits.get_shape())
            return logits, feature_output
        else:
            print('Redisual discriminator_'+self.d_name)
            print('Convolution')

            output = X

            i=0
            print('Input for convolution shape ', X.get_shape())
            for block in self.d_blocks:
                i+=1
                #print('Convolution_block_%i' %i)
                #print('Input shape', output.get_shape())
                output = block.forward(output,
                                         reuse,
                                         is_training)
                #print('After block shape', output.get_shape())
            
            
            output = tf.contrib.layers.flatten(output)
            #print('After flatten shape', output.get_shape())

            i=0
            for layer in self.d_dense_layers:
                #print('Dense weights %i' %i)
                #print(layer.W.get_shape())
                output = layer.forward(output,
                                       reuse,
                                       is_training)
                i+=1
                #print('After dense layer_%i' %i)
                #print('Shape', output.get_shape())

            logits = self.d_final_layer.forward(output, 
                                                reuse,
                                                is_training)
            print('Logits shape', logits.get_shape())
            return logits


#(residual) deconvolution to (?,n_H, n_W,n_C) image shape
class Generator(object):

    def __init__(self, Z, dim_H, dim_W, g_sizes, g_name):
        
        self.residual=False
        for key in g_sizes:
            if not 'block' in key:
                self.residual=False
            else :
                self.residual=True

        #dimensions of input
        latent_dims = g_sizes['z']

        #dimensions of output generated images
        dims_H =[dim_H]
        dims_W =[dim_W]
        mi = latent_dims

        if not self.residual:

            print('Convolutional architecture detected for generator ' + g_name)
            
            with tf.variable_scope('generator_'+g_name) as scope:
                
                #building generator dense layers
                self.g_dense_layers = []
                count = 0

                for mo, apply_batch_norm, keep_prob, act_f, w_init in g_sizes['dense_layers']:
                    name = 'g_dense_layer_%s' %count
                    #print(name)
                    count += 1
                    layer = DenseLayer(name, mi, mo, 
                                        apply_batch_norm, keep_prob,
                                        act_f=act_f , w_init=w_init
                                        )

                    self.g_dense_layers.append(layer)
                    mi = mo

                #deconvolutional layers
                #calculating the last dense layer mo 
            
                for _, _, stride, _, _, _, _, in reversed(g_sizes['conv_layers']):
                    
                    dim_H = int(np.ceil(float(dim_H)/stride))
                    dim_W = int(np.ceil(float(dim_W)/stride))
                    
                    dims_H.append(dim_H)
                    dims_W.append(dim_W)
        
                dims_H = list(reversed(dims_H))
                dims_W = list(reversed(dims_W))
                self.g_dims_H = dims_H
                self.g_dims_W = dims_W

                #last dense layer: projection
                projection, bn_after_project, keep_prob, act_f, w_init = g_sizes['projection'][0]
                
                mo = projection*dims_H[0]*dims_W[0]
                name = 'g_dense_layer_%s' %count
                count+=1
                #print(name)
                self.g_final_layer = DenseLayer(name, mi, mo, not bn_after_project, keep_prob, act_f, w_init)
                # self.g_dense_layers.append(layer)
                
                mi = projection
                self.g_conv_layers=[]
                
                for i, (mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init) in enumerate(g_sizes['conv_layers'] , 1):
                    name = 'g_conv_layer_%s' %count
                    count +=1

                    layer = DeconvLayer(
                      name, mi, mo, [dims_H[i], dims_W[i]],
                      filter_sz, stride, apply_batch_norm, keep_prob,
                      act_f, w_init
                    )

                    self.g_conv_layers.append(layer)
                    mi = mo

        if self.residual:

            print('Residual convolutional architecture detected for generator ' + g_name)
            with tf.variable_scope('generator_'+g_name) as scope:
                    
                    #dense layers
                    self.g_dense_layers = []
                    count = 0

                    mi = latent_dims

                    for mo, apply_batch_norm, keep_prob, act_f, w_init in g_sizes['dense_layers']:
                        name = 'g_dense_layer_%s' %count
                        count += 1
                        
                        layer = DenseLayer(
                            name, mi, mo,
                            apply_batch_norm, keep_prob,
                            f=act_f, w_init=w_init
                        )
                        self.g_dense_layers.append(layer)
                        mi = mo
                        
                    #checking generator architecture

                    g_steps = 0
                    for key in g_sizes:
                        if 'deconv' in key:
                            if not 'shortcut' in key:
                                 g_steps+=1

                    g_block_n=0
                    g_layer_n=0

                    for key in g_sizes:
                        if 'block' and 'shortcut' in key:
                            g_block_n+=1
                        if 'deconv_layer' in key:
                            g_layer_n +=1

                    assert g_block_n+g_layer_n==g_steps, '\nCheck keys in g_sizes, \n sum of generator steps do not coincide with sum of convolutional layers and convolutional blocks'

                    layers_output_sizes={}
                    blocks_output_sizes={}

                    #calculating the output size for each transposed convolutional step
                    for key, item in reversed(list(g_sizes.items())):

                        if 'deconv_layer' in key:
                            
                            _, _, stride, _, _, _, _, = g_sizes[key][0]
                            layers_output_sizes[g_layer_n-1]= [dim_H, dim_W]
                            
                            dim_H = int(np.ceil(float(dim_H)/stride))
                            dim_W = int(np.ceil(float(dim_W)/stride))
                            dims_H.append(dim_H)
                            dims_W.append(dim_W)
                            
                            g_layer_n -= 1

                          
                        if 'deconvblock_layer' in key:
                            
                            for _ ,_ , stride, _, _, _, _, in g_sizes[key]:
                            
                                dim_H = int(np.ceil(float(dim_H)/stride))
                                dim_W = int(np.ceil(float(dim_W)/stride))
                                dims_H.append(dim_H)
                                dims_W.append(dim_W)
                            
                            blocks_output_sizes[g_block_n-1] = [[dims_H[j],dims_W[j]] for j in range(1, len(g_sizes[key])+1)]
                            g_block_n -=1

                    dims_H = list(reversed(dims_H))
                    dims_W = list(reversed(dims_W))

                    #saving for later
                    self.g_dims_H = dims_H
                    self.g_dims_W = dims_W

                    #final dense layer
                    projection, bn_after_project, keep_prob, act_f, w_init = g_sizes['projection'][0]
                    
                    mo = projection*dims_H[0]*dims_W[0]
                    name = 'g_dense_layer_%s' %count
                    layer = DenseLayer(name, mi, mo, not bn_after_project, keep_prob, act_f, w_init)                    
                    self.g_dense_layers.append(layer)

                    #deconvolution input channel number
                    mi = projection
                    self.g_blocks=[]

                    block_n=0 #keep count of the block number
                    layer_n=0 #keep count of conv layer number
                    i=0
                    for key in g_sizes:
                        
                        if 'block' and 'shortcut' in key:
                        
                            g_block = DeconvBlock(block_n,
                                       mi, blocks_output_sizes, g_sizes,
                                       )
                            self.g_blocks.append(g_block)
                            
                            mo, _, _, _, _, _, _, = g_sizes['deconvblock_layer_'+str(block_n)][-1]
                            mi = mo
                            block_n+=1
                            count+=1 
                            i+=1
                            
                        if 'deconv_layer' in key:

                            name = 'g_conv_layer_{0}'.format(layer_n)

                            mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init = g_sizes[key][0]

                            g_conv_layer = DeconvLayer(
                                name, mi, mo, layers_output_sizes[layer_n],
                                filter_sz, stride, apply_batch_norm, keep_prob,
                                act_f, w_init
                            )
                            self.g_blocks.append(g_conv_layer)

                            mi=mo
                            layer_n+=1
                            count+=1 
                            i+=1

                    assert i==g_steps, 'Check convolutional layer and block building, steps in building do not coincide with g_steps'
                    assert g_steps==block_n+layer_n, 'Check keys in g_sizes'

        self.g_sizes=g_sizes
        self.g_name = g_name
        self.projection = projection
        self.bn_after_project = bn_after_project
   
    def g_forward(self, Z, reuse=None, is_training=True):

        if not self.residual:

            print('Generator_'+self.g_name)
            print('Deconvolution')
            #dense layers

            output = Z
            print('Input for deconvolution shape', Z.get_shape())
            i=0
            for layer in self.g_dense_layers:
                output = layer.forward(output, reuse, is_training)
                #print('After dense layer %i' %i)
                #print('shape: ', output.get_shape())
                i+=1

            output = self.g_final_layer.forward(output, reuse, is_training)

            output = tf.reshape(
                output,
                
                [-1, self.g_dims_H[0], self.g_dims_W[0], self.projection]
            
            )

            # print('Reshaped output after projection', output.get_shape())

            if self.bn_after_project:
                output = tf.contrib.layers.batch_norm(
                output,
                decay=0.9, 
                updates_collections=None,
                epsilon=1e-5,
                scale=True,
                is_training=is_training,
                reuse=reuse,
                scope='bn_after_project'
            )
            # passing to deconv blocks

            i=0
            for layer in self.g_conv_layers:
                i+=1
                output = layer.forward(output, reuse, is_training)
                #print('After deconvolutional layer %i' %i)
                #print('shape: ', output.get_shape())


            print('Deconvoluted output shape', output.get_shape())
            return output
        else:

            print('Generator_'+self.g_name)
            print('Deconvolution')
            #dense layers

            output = Z
            print('Input for deconvolution shape', Z.get_shape())
            i=0
            for layer in self.g_dense_layers:
                i+=1
                output = layer.forward(output, reuse, is_training)
                #print('After dense layer %i' %i)
                #print('shape: ', output.get_shape())

            
            output = tf.reshape(
                output,
                
                [-1, self.g_dims_H[0], self.g_dims_W[0], self.projection]
            
            )

            #print('Reshaped output after projection', output.get_shape())

            if self.bn_after_project:
                output = tf.contrib.layers.batch_norm(
                output,
                decay=0.9, 
                updates_collections=None,
                epsilon=1e-5,
                scale=True,
                is_training=is_training,
                reuse=reuse,
                scope='bn_after_project'
            )
            # passing to deconv blocks

            
            i=0
            for block in self.g_blocks:
                i+=1
                output = block.forward(output,
                                        reuse,
                                        is_training)
                #print('After deconvolutional block %i' %i)
                #print('shape: ', output.get_shape())
        

            print('Deconvoluted output shape', output.get_shape())
            return output

#conv / residual conv / deconv
class cycleGenerator(object):

    def __init__(self, X, dim_H, dim_W, g_sizes, g_name):
        
        #input shape
        _, input_dim_H, input_dim_W, input_n_C = X.get_shape().as_list()
        #output shape

        dims_H =[dim_H]
        dims_W =[dim_W]

        self.residual=False
        for key in g_sizes:
            if not 'block' in key:
                self.residual=False
            else :
                self.residual=True

        if not self.residual:

            print('Convolutional Network architecture detected for generator '+ g_name)

            with tf.variable_scope('generator_'+g_name) as scope:

                count = 0
                #checking generator architecture
                g_steps=0
                for key in g_sizes:
                    g_steps+=1

                g_convs = 0
                g_deconvs = 0
                for key in g_sizes:
                    if 'conv' in key:
                        if not 'deconv' in key:
                            g_convs+=1
                    if 'deconv' in key:
                         g_deconvs+=1
                    
                assert g_steps == g_convs + g_deconvs, '\nCheck keys in g_sizes, \n sum of generator steps do not coincide with sum of convolutional layers, convolutional blocks and deconv layers'

                #dimensions of output generated image
                deconv_layers_output_sizes={}
                    
                for key, item in reversed(list(g_sizes.items())):

                    if 'deconv_layer' in key:
                        
                        _, _, stride, _, _, _, _, = g_sizes[key][0]
                        deconv_layers_output_sizes[g_deconvs-1]= [dim_H, dim_W]
                        
                        dim_H = int(np.ceil(float(dim_H)/stride))
                        dim_W = int(np.ceil(float(dim_W)/stride))
                        dims_H.append(dim_H)
                        dims_W.append(dim_W)
                        
                        g_deconvs -= 1

                assert g_deconvs  == 0
                
                dims_H = list(reversed(dims_H))
                dims_W = list(reversed(dims_W))

                #saving for later
                self.g_dims_H = dims_H
                self.g_dims_W = dims_W


                #convolution input channel number
                mi = input_n_C

                self.g_conv_layers=[]
                self.g_deconv_layers=[]

                conv_layer_n=0 #keep count of conv layer number
                deconv_layer_n=0 #keep count of deconv layer number
                i=0 # keep count of the built blocks

                for key in g_sizes:

                    if 'conv_layer' in key:
                        if not 'deconv' in key:

                            name = 'g_conv_layer_{0}'.format(conv_layer_n)

                            mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init = g_sizes[key][0]

                            g_conv_layer = ConvLayer(
                                name, mi, mo, 
                                filter_sz, stride, apply_batch_norm, keep_prob,
                                act_f, w_init
                                )
                            self.g_conv_layers.append(g_conv_layer)
                            mi = mo
                            conv_layer_n +=1
                            count +=1
                            i+=1

                    if 'deconv_layer' in key:

                        name = 'g_deconv_layer_{0}'.format(deconv_layer_n)

                        mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init = g_sizes[key][0]
                        g_deconv_layer = DeconvLayer(
                            name, mi, mo, deconv_layers_output_sizes[deconv_layer_n],
                            filter_sz, stride, apply_batch_norm, keep_prob,
                            act_f, w_init
                        )
                        self.g_deconv_layers.append(g_deconv_layer)

                        mi=mo
                        deconv_layer_n+=1
                        count+=1 
                        i+=1

                assert i==conv_layer_n+deconv_layer_n, 'Check convolutional layer and block building, steps in building do not coincide with g_steps'

                #saving for later
                self.g_sizes=g_sizes
                self.g_name = g_name
       
        if self.residual:

            print('Residual Convolutional Network architecture detected for generator '+ g_name)

            with tf.variable_scope('generator_'+g_name) as scope:
                    
                count = 0

                #checking generator architecture
                g_steps=0
                for key in g_sizes:
                    g_steps+=1

                g_convs = 0
                g_deconvs = 0
                g_conv_blocks = 0
                #g_deconv_blocks = 0

                for key in g_sizes:

                    if 'conv' in key:
                        if not 'deconv' in key:
                                if not 'block' in key:
                                    g_convs+=1
                    if 'convblock' and 'shortcut' in key:
                            g_conv_blocks+=1
                        
                    if 'deconv' in key:
                        if not 'shortcut' in key:
                             g_deconvs+=1
                
                assert g_steps == g_convs +2*(g_conv_blocks)+ g_deconvs, '\nCheck keys in g_sizes, \n sum of generator steps do not coincide with sum of convolutional layers, convolutional blocks and deconv layers'

                #dimensions of output generated image

                deconv_layers_output_sizes={}
                

                for key, item in reversed(list(g_sizes.items())):

                    if 'deconv_layer' in key:
                        
                        _, _, stride, _, _, _, _, = g_sizes[key][0]
                        deconv_layers_output_sizes[g_deconvs-1]= [dim_H, dim_W]
                        
                        dim_H = int(np.ceil(float(dim_H)/stride))
                        dim_W = int(np.ceil(float(dim_W)/stride))
                        dims_H.append(dim_H)
                        dims_W.append(dim_W)
                        
                        g_deconvs -= 1

                assert g_deconvs  == 0
                
                dims_H = list(reversed(dims_H))
                dims_W = list(reversed(dims_W))


                #saving for later
                self.g_dims_H = dims_H
                self.g_dims_W = dims_W


                #convolution input channel number
                mi = input_n_C

                self.g_blocks=[]

                block_n=0 #keep count of the block number
                conv_layer_n=0 #keep count of conv layer number
                deconv_layer_n=0 #keep count of deconv layer number
                i=0 # keep count of the built blocks

                for key in g_sizes:

                    if 'conv_layer' in key:
                        if not 'deconv' in key:

                            name = 'g_conv_layer_{0}'.format(conv_layer_n)

                            mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init = g_sizes[key][0]

                            g_conv_layer = ConvLayer(
                                name, mi, mo, 
                                filter_sz, stride, apply_batch_norm, keep_prob,
                                act_f, w_init
                                )
                            self.g_blocks.append(g_conv_layer)
                            mi = mo
                            conv_layer_n +=1
                            count +=1
                            i+=1

                    
                    if 'block' and 'shortcut' in key:
                    
                        g_block = ConvBlock(block_n,
                                   mi, g_sizes,
                                   )
                        self.g_blocks.append(g_block)
                        
                        mo, _, _, _, _, _, _, = g_sizes['convblock_layer_'+str(block_n)][-1]
                        mi = mo
                        block_n+=1
                        count+=1 
                        i+=1
                        
                    if 'deconv_layer' in key:

                        name = 'g_deconv_layer_{0}'.format(deconv_layer_n)

                        mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init = g_sizes[key][0]
                        g_deconv_layer = DeconvLayer(
                            name, mi, mo, deconv_layers_output_sizes[deconv_layer_n],
                            filter_sz, stride, apply_batch_norm, keep_prob,
                            act_f, w_init
                        )
                        self.g_blocks.append(g_deconv_layer)

                        mi=mo
                        deconv_layer_n+=1
                        count+=1 
                        i+=1

                assert i==block_n+conv_layer_n+deconv_layer_n, 'Check convolutional layer and block building, steps in building do not coincide with g_steps'

                #saving for later
                self.g_sizes=g_sizes
                self.g_name = g_name
                    # return self.g_forward(Z)
       
    def g_forward(self, X, reuse=None, is_training=True):
        if not self.residual:
            print('Generator_'+self.g_name)
            #dense layers

            output = X
            print('Input for generator shape', X.get_shape())
            
            i=0

            for conv_layer in self.g_conv_layers:
                i+=1
                output = conv_layer.forward(output,
                                        reuse,
                                        is_training)
                #print('After block step%i' %i)
                #print('shape: ', output.get_shape())
            for deconv_layer in self.g_deconv_layers:

                i+=1
                output = deconv_layer.forward(output,
                                        reuse,
                                        is_training)

            print('Generator output shape', output.get_shape())
            return output

        if self.residual:
            print('Generator_'+self.g_name)
            #dense layers

            output = X
            print('Input for generator shape', X.get_shape())
            
            i=0


            for block in self.g_blocks:
                i+=1
                output = block.forward(output,
                                        reuse,
                                        is_training)
                #print('After block step%i' %i)
                #print('shape: ', output.get_shape())
        

            print('Generator output shape', output.get_shape())
            return output

#residual conv / residual deconv
class cycleGenerator_fullresidual(object):

    def __init__(self, X, dim_H, dim_W, g_sizes, g_name):
        
        _, input_dim_H, input_dim_W, input_n_C = X.get_shape().as_list()

        dims_H =[dim_H]
        dims_W =[dim_W]

        print('Residual Convolutional Network architecture (v2) detected for generator '+ g_name)

        with tf.variable_scope('generator_'+g_name) as scope:
                
            count = 0

            #checking generator architecture
            g_steps=0
            for key in g_sizes:
                g_steps+=1

            g_conv_blocks = 0
            g_deconv_blocks = 0

            for key in g_sizes:

                if 'conv' in key:
                    if not 'deconv' in key:
                            if 'block' in key:
                                g_conv_blocks+=1
                if 'deconv' in key:
                            if 'block' in key:
                                g_deconv_blocks+=1
            
            assert g_steps == g_conv_blocks+ g_deconv_blocks, '\nCheck keys in g_sizes, \n sum of generator steps do not coincide with sum of convolutional blocks and deconvolutional blocks'
            
            #dimensions of output generated image
            g_deconv_blocks=g_deconv_blocks//2
            deconv_blocks_output_sizes={}
            for key, item in reversed(list(g_sizes.items())):

                if 'deconvblock_layer' in key:
                    
                    for _ ,_ , stride, _, _, _, _, in g_sizes[key]:
                    
                        dim_H = int(np.ceil(float(dim_H)/stride))
                        dim_W = int(np.ceil(float(dim_W)/stride))
                        dims_H.append(dim_H)
                        dims_W.append(dim_W)
                    
                    deconv_blocks_output_sizes[g_deconv_blocks-1] = [[dims_H[j],dims_W[j]] for j in range(1, len(g_sizes[key])+1)]
                    g_deconv_blocks -=1

            dims_H = list(reversed(dims_H))
            dims_W = list(reversed(dims_W))
            
            assert g_deconv_blocks==0

            #convolution input channel number
            mi = input_n_C

            self.g_blocks=[]

            convblock_n=0 #keep count of the conv block number
            deconvblock_n=0 #keep count of deconv block number

            i=0 # keep count of the built blocks

            for key in g_sizes:

                if 'convblock_layer' in key:
                    if not 'deconv' in key:

                        g_block = ConvBlock(convblock_n,
                                   mi, g_sizes,
                                   )
                        self.g_blocks.append(g_block)
                        
                        mo, _, _, _, _, _, _, = g_sizes['convblock_layer_'+str(convblock_n)][-1]
                        mi = mo
                        convblock_n+=1
                        i+=1

                
                if 'deconvblock_layer' in key:

                        g_block = DeconvBlock(deconvblock_n,
                                   mi, deconv_blocks_output_sizes, g_sizes,
                                   )
                        self.g_blocks.append(g_block)
                        
                        mo, _, _, _, _, _, _, = g_sizes['deconvblock_layer_'+str(deconvblock_n)][-1]
                        mi = mo
                        deconvblock_n+=1
                        i+=1

            assert i==convblock_n+deconvblock_n, 'Check convolutional layer and block building, steps in building do not coincide with g_steps'

            #saving for later
            self.g_dims_H = dims_H
            self.g_dims_W = dims_W
            self.g_sizes=g_sizes
            self.g_name = g_name
       
    def g_forward(self, X, reuse=None, is_training=True):

        print('Generator_'+self.g_name)
        #dense layers

        output = X
        print('Input for generator shape', X.get_shape())
        
        i=0

        for block in self.g_blocks:
            i+=1
            output = block.forward(output,
                                    reuse,
                                    is_training)
            #print('After block step%i' %i)
            #print('shape: ', output.get_shape())

        print('Generator output shape', output.get_shape())
        return output

#pix2pix architecture, u_net
#works with same dim of input and output, implement different dimensions
class pix2pixGenerator(object):

    def __init__(self, X, output_dim_H, output_dim_W, g_enc_sizes, g_dec_sizes, g_name):
        
        _, input_dim_H, input_dim_W, input_n_C = X.get_shape().as_list()

        enc_dims_H=[input_dim_H]
        enc_dims_W=[input_dim_W]
        enc_dims_nC=[input_n_C]

        output_n_C=input_n_C
        mi = input_n_C

        with tf.variable_scope('generator_'+g_name) as scope:
            
            #building generator encoder convolutional layers
            self.g_enc_conv_layers =[]
            enc_dims=[]
            for conv_count, (mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init) in enumerate(g_enc_sizes['conv_layers'], 1):
                
                name = "g_conv_layer_%s" % conv_count
                layer = ConvLayer(name, mi, mo, 
                                  filter_sz, stride, 
                                  apply_batch_norm, keep_prob,
                                  act_f, w_init)

                input_dim_H = int(np.ceil(float(input_dim_H)/stride))
                input_dim_W = int(np.ceil(float(input_dim_W)/stride))

                enc_dims_H.append(input_dim_H)
                enc_dims_W.append(input_dim_W)
                enc_dims_nC.append(mo)
                self.g_enc_conv_layers.append(layer)

                mi = mo
            
            dec_dims_H = [output_dim_H]
            dec_dims_W = [output_dim_W]

            #building generator decoder deconvolutional layers
            #calculate outputsize for each deconvolution step
            for _, _, stride, _, _, _, _ in reversed(g_dec_sizes['deconv_layers']):
                
                output_dim_H = int(np.ceil(float(output_dim_H)/stride))
                output_dim_W = int(np.ceil(float(output_dim_W)/stride))
                
                dec_dims_H.append(output_dim_H)
                dec_dims_W.append(output_dim_W)
    
            dec_dims_H = list(reversed(dec_dims_H))
            dec_dims_W = list(reversed(dec_dims_W))

            self.g_dec_conv_layers=[]
            
            #number of channels of last convolution and of first transposed convolution
            # the layer will be reshaped to have dimensions [?, 1, 1, mi*enc_dims_W[-1]*enc_dims_H[-1]]
            mi=mi*enc_dims_W[-1]*enc_dims_H[-1]
            self.n_C_last=mi

            for deconv_count, (mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init) in enumerate(g_dec_sizes['deconv_layers'], 1):
                
                if deconv_count == 1:
                    name = 'g_deconv_layer_%s' %deconv_count
                    #print(name)
                    
                    layer = DeconvLayer(
                      name, mi, mo, [dec_dims_H[deconv_count], dec_dims_W[deconv_count]],
                      filter_sz, stride, apply_batch_norm, keep_prob,
                      act_f, w_init
                    )

                    self.g_dec_conv_layers.append(layer)
                    mi = mo

                if deconv_count > 1:
                    name = 'g_deconv_layer_%s' %deconv_count
                    #print(name)
                    
                    layer = DeconvLayer(
                      name, 2*mi, mo, [dec_dims_H[deconv_count], dec_dims_W[deconv_count]],
                      filter_sz, stride, apply_batch_norm, keep_prob,
                      act_f, w_init
                    )

                    self.g_dec_conv_layers.append(layer)
                    mi = mo


            assert conv_count==deconv_count, '\n Number of convolutional and deconvolutional layers do not coincide in \n encoder and decoder part of generator '+g_name
            
            # self.g_dims_H = dec_dims_H
            # self.g_dims_W = dims_W
            self.conv_count=conv_count
            self.deconv_count=deconv_count
            self.g_name=g_name
       
    def g_forward(self, X, reuse=None, is_training=True):

        print('Generator_'+self.g_name)
        
        output = X
        print('Input for generator encoder shape', X.get_shape())

        skip_conv_outputs=[]
        #convolutional encoder layers

        for i, layer in enumerate(self.g_enc_conv_layers, 1):
            output = layer.forward(output,
                                    reuse,
                                    is_training)

            skip_conv_outputs.append(output)
            # print('After conv layer%i' %i)
            # print('shape: ', output.get_shape())

        assert i == self.conv_count

        if (output.get_shape().as_list()[1], output.get_shape().as_list()[2]) != (1, 1):
            output = tf.reshape(
                output,
                [-1, 1, 1, self.n_C_last]
            )
        
        print('Output of generator encoder, \n and input for generator decoder shape', output.get_shape())

        for i, layer in enumerate(self.g_dec_conv_layers, 1):

            skip_layer=self.conv_count - i
            if i > 1:
                #print('After deconv layer %i' %i)
                #print('main path', output.get_shape())
                #print('secondary path', skip_conv_outputs[skip_layer].get_shape())
                output = tf.concat([output, skip_conv_outputs[skip_layer]], axis =3)
                #print('After concat shape', output.get_shape())
            output = layer.forward(output,
                                   reuse,
                                   is_training)

            # print('After deconv layer %i' %i)
            # print('Shape', output.get_shape()) 

        assert i == self.deconv_count

        print('Generator output shape', output.get_shape())
        return output

#same as pix2pix with input noise
class bicycleGenerator(object):

    def __init__(self, X, output_dim_H, output_dim_W, g_sizes_enc, g_sizes_dec, g_name):
        
        _, input_dim_H, input_dim_W, input_n_C = X.get_shape().as_list()

        enc_dims_H=[input_dim_H]
        enc_dims_W=[input_dim_W]
        enc_dims_nC=[input_n_C]

        output_n_C=input_n_C
        self.latent_dims = g_sizes_enc['latent_dims']
        mi = input_n_C + self.latent_dims

        with tf.variable_scope('generator_'+g_name) as scope:
            
            #building generator encoder convolutional layers
            self.g_enc_conv_layers =[]
            enc_dims=[]
            for conv_count, (mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init) in enumerate(g_sizes_enc['conv_layers'], 1):
                
                name = "g_conv_layer_%s" % conv_count
                layer = ConvLayer(name, mi, mo, 
                                  filter_sz, stride, 
                                  apply_batch_norm, keep_prob,
                                  act_f, w_init)

                input_dim_H = int(np.ceil(float(input_dim_H)/stride))
                input_dim_W = int(np.ceil(float(input_dim_W)/stride))

                enc_dims_H.append(input_dim_H)
                enc_dims_W.append(input_dim_W)
                enc_dims_nC.append(mo)
                self.g_enc_conv_layers.append(layer)

                mi = mo
            
            dec_dims_H = [output_dim_H]
            dec_dims_W = [output_dim_W]

            #building generator decoder deconvolutional layers
            #calculate outputsize for each deconvolution step
            for _, _, stride, _, _, _, _ in reversed(g_sizes_dec['deconv_layers']):
                
                output_dim_H = int(np.ceil(float(output_dim_H)/stride))
                output_dim_W = int(np.ceil(float(output_dim_W)/stride))
                
                dec_dims_H.append(output_dim_H)
                dec_dims_W.append(output_dim_W)
    
            dec_dims_H = list(reversed(dec_dims_H))
            dec_dims_W = list(reversed(dec_dims_W))

            self.g_dec_conv_layers=[]
            
            #number of channels of last convolution and of first transposed convolution
            # the layer will be reshaped to have dimensions [?, 1, 1, mi*enc_dims_W[-1]*enc_dims_H[-1]]
            mi=mi*enc_dims_W[-1]*enc_dims_H[-1]
            self.n_C_last=mi

            for deconv_count, (mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init) in enumerate(g_sizes_dec['deconv_layers'], 1):
                
                if deconv_count == 1:
                    name = 'g_deconv_layer_%s' %deconv_count
                    #print(name)
                    
                    layer = DeconvLayer(
                      name, mi, mo, [dec_dims_H[deconv_count], dec_dims_W[deconv_count]],
                      filter_sz, stride, apply_batch_norm, keep_prob,
                      act_f, w_init
                    )

                    self.g_dec_conv_layers.append(layer)
                    mi = mo

                if deconv_count > 1:
                    name = 'g_deconv_layer_%s' %deconv_count
                    #print(name)
                    
                    layer = DeconvLayer(
                      name, 2*mi, mo, [dec_dims_H[deconv_count], dec_dims_W[deconv_count]],
                      filter_sz, stride, apply_batch_norm, keep_prob,
                      act_f, w_init
                    )

                    self.g_dec_conv_layers.append(layer)
                    mi = mo


            assert conv_count==deconv_count, '\n Number of convolutional and deconvolutional layers do not coincide in \n encoder and decoder part of generator '+g_name
            
            # self.g_dims_H = dec_dims_H
            # self.g_dims_W = dims_W
            self.conv_count=conv_count
            self.deconv_count=deconv_count
            self.g_name=g_name
       
    def g_forward(self, X, z, reuse=None, is_pretraining=None, is_training=True):

        if is_pretraining:
            z=X
        elif not is_pretraining:
            z = tf.reshape(z, [tf.shape(X)[0], 1, 1, self.latent_dims])
            z = tf.tile(z, [1, tf.shape(X)[1], tf.shape(X)[2], 1]) 

        print('Generator_'+self.g_name)
        
        output = X
        output=tf.concat([X,z], axis=3)

        print('Input for generator encoded shape', X.get_shape())

        skip_conv_outputs=[]
        #convolutional encoder layers

        for i, layer in enumerate(self.g_enc_conv_layers, 1):
            output = layer.forward(output,
                                    reuse,
                                    is_training)

            skip_conv_outputs.append(output)
            #print('After conv layer%i' %i)
            #print('shape: ', output.get_shape())

        assert i == self.conv_count

        if (output.get_shape().as_list()[1], output.get_shape().as_list()[2]) != (1, 1):
            output = tf.reshape(
                output,
                [-1, 1, 1, self.n_C_last]
            )
        
        print('Output of generator encoder, \n and input for generator decoder shape', output.get_shape())

        for i, layer in enumerate(self.g_dec_conv_layers, 1):

            skip_layer=self.conv_count - i
            if i > 1:
                #print('After deconv layer %i' %i)
                #print('main path', output.get_shape())
                #print('secondary path', skip_conv_outputs[skip_layer].get_shape())
                output = tf.concat([output, skip_conv_outputs[skip_layer]], axis =3)
                #print('After concat shape', output.get_shape())
            output = layer.forward(output,
                                   reuse,
                                   is_training)

            # print('After deconv layer %i' %i)
            # print('Shape', output.get_shape()) 

        assert i == self.deconv_count

        print('Generator output shape', output.get_shape())
        return output

class condGenerator(object):
    def __init__(self, dim_y, dim_H, dim_W, g_sizes, g_name):

        self.residual=False
        for key in g_sizes:
            if not 'block' in key:
                self.residual=False
            else :
                self.residual=True

        #dimensions of input
        latent_dims = g_sizes['z']

        #dimensions of output generated images
        dims_H =[dim_H]
        dims_W =[dim_W]
        mi = latent_dims + dim_y

        if not self.residual:

            print('Convolutional architecture detected for generator ' + g_name)
            
            with tf.variable_scope('generator_'+g_name) as scope:
                
                #building generator dense layers
                self.g_dense_layers = []
                count = 0

                for mo, apply_batch_norm, keep_prob, act_f, w_init in g_sizes['dense_layers']:
                    name = 'g_dense_layer_%s' %count
                    #print(name)
                    count += 1
                    layer = DenseLayer(name, mi, mo, 
                                        apply_batch_norm, keep_prob,
                                        act_f=act_f , w_init=w_init
                                        )

                    self.g_dense_layers.append(layer)
                    mi = mo
                    mi = mi + dim_y

                #deconvolutional layers
                #calculating the last dense layer mo 
            
                for _, _, stride, _, _, _, _, in reversed(g_sizes['conv_layers']):
                    
                    dim_H = int(np.ceil(float(dim_H)/stride))
                    dim_W = int(np.ceil(float(dim_W)/stride))
                    
                    dims_H.append(dim_H)
                    dims_W.append(dim_W)
        
                dims_H = list(reversed(dims_H))
                dims_W = list(reversed(dims_W))
                self.g_dims_H = dims_H
                self.g_dims_W = dims_W

                #last dense layer: projection
                projection, bn_after_project, keep_prob, act_f, w_init = g_sizes['projection'][0]
                
                mo = (projection)*dims_H[0]*dims_W[0]
                name = 'g_dense_layer_%s' %count
                count+=1
                #print(name)
                self.g_final_layer = DenseLayer(name, mi, mo, not bn_after_project, keep_prob, act_f, w_init)
                # self.g_dense_layers.append(layer)
                
                mi = projection+dim_y
                self.g_conv_layers=[]
                
                for i, (mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init) in enumerate(g_sizes['conv_layers'] , 1):
                    name = 'g_conv_layer_%s' %count
                    count +=1

                    layer = DeconvLayer(
                      name, mi, mo, [dims_H[i], dims_W[i]],
                      filter_sz, stride, apply_batch_norm, keep_prob,
                      act_f, w_init
                    )

                    self.g_conv_layers.append(layer)
                    mi = mo
                    mi = mi + dim_y

        if self.residual:

            print('Residual convolutional architecture detected for generator ' + g_name)
            with tf.variable_scope('generator_'+g_name) as scope:
                    
                    #dense layers
                    self.g_dense_layers = []
                    count = 0

                    mi = latent_dims

                    for mo, apply_batch_norm, keep_prob, act_f, w_init in g_sizes['dense_layers']:
                        name = 'g_dense_layer_%s' %count
                        count += 1
                        
                        layer = DenseLayer(
                            name, mi, mo,
                            apply_batch_norm, keep_prob,
                            f=act_f, w_init=w_init
                        )
                        self.g_dense_layers.append(layer)
                        mi = mo
                        
                    #checking generator architecture

                    g_steps = 0
                    for key in g_sizes:
                        if 'deconv' in key:
                            if not 'shortcut' in key:
                                 g_steps+=1

                    g_block_n=0
                    g_layer_n=0

                    for key in g_sizes:
                        if 'block' and 'shortcut' in key:
                            g_block_n+=1
                        if 'deconv_layer' in key:
                            g_layer_n +=1

                    assert g_block_n+g_layer_n==g_steps, '\nCheck keys in g_sizes, \n sum of generator steps do not coincide with sum of convolutional layers and convolutional blocks'

                    layers_output_sizes={}
                    blocks_output_sizes={}

                    #calculating the output size for each transposed convolutional step
                    for key, item in reversed(list(g_sizes.items())):

                        if 'deconv_layer' in key:
                            
                            _, _, stride, _, _, _, _, = g_sizes[key][0]
                            layers_output_sizes[g_layer_n-1]= [dim_H, dim_W]
                            
                            dim_H = int(np.ceil(float(dim_H)/stride))
                            dim_W = int(np.ceil(float(dim_W)/stride))
                            dims_H.append(dim_H)
                            dims_W.append(dim_W)
                            
                            g_layer_n -= 1

                          
                        if 'deconvblock_layer' in key:
                            
                            for _ ,_ , stride, _, _, _, _, in g_sizes[key]:
                            
                                dim_H = int(np.ceil(float(dim_H)/stride))
                                dim_W = int(np.ceil(float(dim_W)/stride))
                                dims_H.append(dim_H)
                                dims_W.append(dim_W)
                            
                            blocks_output_sizes[g_block_n-1] = [[dims_H[j],dims_W[j]] for j in range(1, len(g_sizes[key])+1)]
                            g_block_n -=1

                    dims_H = list(reversed(dims_H))
                    dims_W = list(reversed(dims_W))

                    #saving for later
                    self.g_dims_H = dims_H
                    self.g_dims_W = dims_W

                    #final dense layer
                    projection, bn_after_project, keep_prob, act_f, w_init = g_sizes['projection'][0]
                    
                    mo = projection*dims_H[0]*dims_W[0]
                    name = 'g_dense_layer_%s' %count
                    layer = DenseLayer(name, mi, mo, not bn_after_project, keep_prob, act_f, w_init)                    
                    self.g_dense_layers.append(layer)

                    #deconvolution input channel number
                    mi = projection
                    self.g_blocks=[]

                    block_n=0 #keep count of the block number
                    layer_n=0 #keep count of conv layer number
                    i=0
                    for key in g_sizes:
                        
                        if 'block' and 'shortcut' in key:
                        
                            g_block = DeconvBlock(block_n,
                                       mi, blocks_output_sizes, g_sizes,
                                       )
                            self.g_blocks.append(g_block)
                            
                            mo, _, _, _, _, _, _, = g_sizes['deconvblock_layer_'+str(block_n)][-1]
                            mi = mo
                            block_n+=1
                            count+=1 
                            i+=1
                            
                        if 'deconv_layer' in key:

                            name = 'g_conv_layer_{0}'.format(layer_n)

                            mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init = g_sizes[key][0]

                            g_conv_layer = DeconvLayer(
                                name, mi, mo, layers_output_sizes[layer_n],
                                filter_sz, stride, apply_batch_norm, keep_prob,
                                act_f, w_init
                            )
                            self.g_blocks.append(g_conv_layer)

                            mi=mo
                            layer_n+=1
                            count+=1 
                            i+=1

                    assert i==g_steps, 'Check convolutional layer and block building, steps in building do not coincide with g_steps'
                    assert g_steps==block_n+layer_n, 'Check keys in g_sizes'

        self.g_sizes=g_sizes
        self.g_name = g_name
        self.projection = projection
        self.bn_after_project = bn_after_project
        self.dim_y = dim_y
   
    def g_forward(self, Z, y, reuse=None, is_training=True):

        if not self.residual:

            print('Generator_'+self.g_name)
            print('Deconvolution')
            #dense layers

            output = Z
            output = lin_concat(output, y, self.dim_y)
            print('Input for deconvolution shape', output.get_shape())
            i=0
            for layer in self.g_dense_layers:

                
                output = layer.forward(output, reuse, is_training)
                output= lin_concat(output, y, self.dim_y)
                #print('After dense layer and concat %i' %i)
                #print('shape: ', output.get_shape())
                i+=1

            output = self.g_final_layer.forward(output, reuse, is_training)

            output = tf.reshape(
                output,
                
                [-1, self.g_dims_H[0], self.g_dims_W[0], self.projection]
            
            )
            #print('Reshaped output after projection', output.get_shape())

            if self.bn_after_project:
                output = tf.contrib.layers.batch_norm(
                output,
                decay=0.9, 
                updates_collections=None,
                epsilon=1e-5,
                scale=True,
                is_training=is_training,
                reuse=reuse,
                scope='bn_after_project'
            )
            # passing to deconv blocks
            output = conv_concat(output, y, self.dim_y)
            #print('After reshape and concat', output.get_shape())
            i=0
            for layer in self.g_conv_layers[:-1]:
                i+=1
                output = layer.forward(output, reuse, is_training)
                output = conv_concat(output, y, self.dim_y)
                #print('After deconvolutional layer and concat %i' %i)
                #print('shape: ', output.get_shape())

            output=self.g_conv_layers[-1].forward(output, reuse, is_training)
            print('Deconvoluted output shape', output.get_shape())
            return output
        else:

            print('Generator_'+self.g_name)
            print('Deconvolution')
            #dense layers

            output = Z
            print('Input for deconvolution shape', Z.get_shape())
            i=0
            for layer in self.g_dense_layers:
                i+=1
                output = layer.forward(output, reuse, is_training)
                #print('After dense layer %i' %i)
                #print('shape: ', output.get_shape())

            
            output = tf.reshape(
                output,
                
                [-1, self.g_dims_H[0], self.g_dims_W[0], self.projection]
            
            )

            #print('Reshaped output after projection', output.get_shape())

            if self.bn_after_project:
                output = tf.contrib.layers.batch_norm(
                output,
                decay=0.9, 
                updates_collections=None,
                epsilon=1e-5,
                scale=True,
                is_training=is_training,
                reuse=reuse,
                scope='bn_after_project'
            )
            # passing to deconv blocks

            
            i=0
            for block in self.g_blocks:
                i+=1
                output = block.forward(output,
                                        reuse,
                                        is_training)
                #print('After deconvolutional block %i' %i)
                #print('shape: ', output.get_shape())
        

            print('Deconvoluted output shape', output.get_shape())
            return output


#VARIATIONAL_AUTOENCODERS

class denseEncoder:

    def __init__(self, X, e_sizes, name):

        latent_dims = e_sizes['z']

        _, mi = X.get_shape().as_list()

        with tf.variable_scope('encoder'+name) as scope:

            self.e_layers=[]

            count=0
            for mo, apply_batch_norm, keep_prob, act_f, w_init in e_sizes['dense_layers']:

                name = 'layer_{0}'.format(count)
                count +=1

                layer = DenseLayer(name, mi, mo,
                        apply_batch_norm, keep_prob,
                        act_f, w_init
                        )

                self.e_layers.append(layer)
                mi = mo

            name = 'layer_{0}'.format(count)

            last_enc_layer = DenseLayer(name, mi, 2*latent_dims,
                False, 1, act_f=lambda x:x, w_init=e_sizes['last_layer_weight_init'])

            self.latent_dims = latent_dims
            self.e_layers.append(last_enc_layer)

    def encode(self, X, reuse = None, is_training=False):

        output=X
        for layer in self.e_layers:
            output = layer.forward(output, reuse, is_training)

        self.means = output[:,:self.latent_dims]
        self.stddev = tf.nn.softplus(output[:,self.latent_dims:])+1e-6

        with st.value_type(st.SampleValue()):
            Z = st.StochasticTensor(Normal(loc=self.means, scale=self.stddev))
        
        return Z

class denseDecoder:

    def __init__(self, Z, latent_dims, dim, d_sizes, name):

        mi = latent_dims

        with tf.variable_scope('decoder'+name) as scope:

            self.d_layers = []
            count = 0
            for mo, apply_batch_norm, keep_prob, act_f, w_init in d_sizes['dense_layers']:

                name = 'layer_{0}'.format(count)
                count += 1

                layer = DenseLayer(name, mi, mo,
                        apply_batch_norm, keep_prob,
                        act_f, w_init
                        )

                self.d_layers.append(layer)
                mi = mo

            name = 'layer_{0}'.format(count)

            last_dec_layer = DenseLayer(name, mi, dim, False, 1,
                act_f=lambda x:x, w_init=d_sizes['last_layer_weight_init']
                )

            self.d_layers.append(last_dec_layer)

    def decode(self, Z, reuse=None, is_training=False):

        output=Z

        for layer in self.d_layers:
             output = layer.forward(output, reuse, is_training)
        
        return output

class bicycleEncoder(object):

    def __init__(self, X, e_sizes, e_name):

        _, dim_H, dim_W, mi = X.get_shape().as_list()
        latent_dims=e_sizes['latent_dims']

        self.residual=False
        for key in e_sizes:
            if 'block' in key:
                self.residual=True

        if not self.residual:
            print('Convolutional Network architecture detected for encoder '+ e_name)


            with tf.variable_scope('encoder_'+e_name) as scope:
                #building discriminator convolutional layers

                self.e_conv_layers =[]
                count=0
                for mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init in e_sizes['conv_layers']:
                    
                    # make up a name - used for get_variable
                    name = "e_conv_layer_%s" % count
                    #print(name)
                    count += 1

                    layer = ConvLayer(name, mi, mo, 
                                      filter_sz, stride, 
                                      apply_batch_norm, keep_prob,
                                      act_f, w_init)

                    self.e_conv_layers.append(layer)
                    mi = mo

                    dim_H = int(np.ceil(float(dim_H) / stride))
                    dim_W = int(np.ceil(float(dim_W) / stride))
                        
                mi = mi * dim_H * dim_W

                #building encoder dense layers

                self.e_dense_layers = []
                for mo, apply_batch_norm, keep_prob, act_f, w_init in e_sizes['dense_layers']:
                    
                    name = 'e_dense_layer_%s' %count
                    #print(name)
                    count +=1
                    
                    layer = DenseLayer(name, mi, mo,
                                      apply_batch_norm, keep_prob,
                                      act_f, w_init)
                    mi = mo
                    self.e_dense_layers.append(layer)
                    
                    #final logistic layer

                name = 'e_last_dense_layer_mu'
                w_init_last = e_sizes['readout_layer_w_init']
                #print(name)
                self.e_final_layer_mu = DenseLayer(name, mi, latent_dims, 
                                                    False, keep_prob=1, 
                                                    act_f=lambda x: x, w_init=w_init_last)
                name = 'e_last_dense_layer_sigma'
                self.e_final_layer_sigma = DenseLayer(name, mi, latent_dims, 
                                                    False, keep_prob=1, 
                                                    act_f=lambda x: x, w_init=w_init_last)

                self.e_name=e_name
                self.latent_dims=latent_dims
        else:
            print('Residual Convolutional Network architecture detected for Encoder'+ e_name)
            
            with tf.variable_scope('encoder_'+e_name) as scope:
                #building discriminator convolutional layers

                self.e_blocks = []
                #count conv blocks
                e_steps = 0
                for key in e_sizes:
                    if 'conv' in key:
                        if not 'shortcut' in key:
                             e_steps+=1

                e_block_n=0
                e_layer_n=0
                

                for key in e_sizes:
                     
                    if 'block' and 'shortcut' in key:
                    
                        e_block = ConvBlock(e_block_n,
                                   mi, e_sizes,
                                   )
                        self.e_blocks.append(e_block)
                        
                        mo, _, _, _, _, _, _, = e_sizes['convblock_layer_'+str(e_block_n)][-1]
                        mi = mo
                        dim_H = e_block.output_dim(dim_H)
                        dim_W = e_block.output_dim(dim_W)
                        e_block_n+=1
                        
                    
                    if 'conv_layer' in key:

                        name = 'e_conv_layer_{0}'.format(e_layer_n)

                        mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init = e_sizes[key][0]


                        e_conv_layer = ConvLayer(name, mi, mo,
                                               filter_sz, stride,
                                               apply_batch_norm, keep_prob,
                                               act_f, w_init
                            )

                        self.e_blocks.append(e_conv_layer)

                        mi = mo
                        dim_W = int(np.ceil(float(dim_W) / stride))
                        dim_H = int(np.ceil(float(dim_H) / stride))
                        e_layer_n+=1
                
                assert e_block_n+e_layer_n==e_steps, '\nCheck keys in d_sizes, \n total convolution steps do not mach sum between convolutional blocks and convolutional layers'
                
                count=e_steps

                mi = mi * dim_H * dim_W

                #building encoder dense layers

                self.e_dense_layers = []
                for mo, apply_batch_norm, keep_prob, act_f, w_init in e_sizes['dense_layers']:
                    
                    name = 'e_dense_layer_%s' %count
                    #print(name)
                    count +=1
                    
                    layer = DenseLayer(name, mi, mo,
                                      apply_batch_norm, keep_prob,
                                      act_f, w_init)
                    mi = mo
                    self.e_dense_layers.append(layer)
                    
                    #final logistic layer

                
                w_init_last = e_sizes['readout_layer_w_init']
                #print(name)
                name = 'e_last_dense_layer_mu'
                self.e_final_layer_mu = DenseLayer(name, mi, latent_dims, 
                                                    'bn', keep_prob=1, 
                                                    act_f=lambda x: x, w_init=w_init_last)
                name = 'e_last_dense_layer_sigma'
                self.e_final_layer_sigma = DenseLayer(name, mi, latent_dims, 
                                                    'bn', keep_prob=1, 
                                                    act_f=lambda x: x, w_init=w_init_last)

                self.e_name=e_name
                self.latent_dims=latent_dims

    def e_forward(self, X, reuse = None, is_training=True):

        if not self.residual:
            print('Encoder_'+self.e_name)
            print('Convolution')

            output = X
            print('Input for convolution shape ', X.get_shape())
            i=0
            for layer in self.e_conv_layers:
                i+=1
                # print('Convolution_layer_%i' %i)
                # print('Input shape', output.get_shape())
                output = layer.forward(output,
                                     reuse, 
                                     is_training)
                #print('After convolution shape', output.get_shape())
            
            output = tf.contrib.layers.flatten(output)
            #print('After flatten shape', output.get_shape())
            i=0
            for layer in self.e_dense_layers:
                #print('Dense weights %i' %i)
                #print(layer.W.get_shape())
                output = layer.forward(output,
                                       reuse,
                                       is_training)
                i+=1
                # print('After dense layer_%i' %i)
                # print('Shape', output.get_shape())

            mu = self.e_final_layer_mu.forward(output, 
                                                reuse, 
                                                is_training)

            log_sigma = self.e_final_layer_sigma.forward(output, 
                                                reuse, 
                                                is_training)

            z = mu + tf.random_normal(shape=tf.shape(self.latent_dims))*tf.exp(log_sigma)

            print('Encoder output shape', z.get_shape())
            return z, mu, log_sigma
        else:
            print('Residual encoder_'+self.e_name)
            print('Convolution')

            output = X

            i=0
            print('Input for convolution shape ', X.get_shape())
            for block in self.e_blocks:
                i+=1
                #print('Convolution_block_%i' %i)
                #print('Input shape', output.get_shape())
                output = block.forward(output,
                                         reuse,
                                         is_training)
                #print('After block shape', output.get_shape())
            
            
            output = tf.contrib.layers.flatten(output)
            #print('After flatten shape', output.get_shape())

            i=0
            for layer in self.e_dense_layers:
                #print('Dense weights %i' %i)
                #print(layer.W.get_shape())
                output = layer.forward(output,
                                       reuse,
                                       is_training)
                i+=1
                # print('After dense layer_%i' %i)
                # print('Shape', output.get_shape())

            mu = self.e_final_layer_mu.forward(output, 
                                                reuse, 
                                                is_training)

            log_sigma = self.e_final_layer_sigma.forward(output, 
                                                reuse, 
                                                is_training)

            z = mu + tf.random_normal(shape=tf.shape(self.latent_dims))*tf.exp(log_sigma)

            print('Encoder output shape', z.get_shape())
            return z, mu, log_sigma

# class convEncoder:

# class convDecoder:

# class resEncoder:

# class resDecoder:   

