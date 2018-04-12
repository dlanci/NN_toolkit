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

#GANS

class Discriminator(object):

    def __init__(self, X, d_sizes, d_name):
       
        _, dim_H, dim_W, mi = X.get_shape().as_list()


        for key in d_sizes:
            if not 'block' in key:
                print('Convolutional Network architecture detected')
            else:
                print('Check network architecture')
            break

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
                                                False, 1, 
                                                lambda x: x, w_init_last)
            

            self.d_name=d_name


    def d_forward(self, X, reuse = None, is_training=True):
            print('Discriminator_'+self.d_name)
            print('Convolution')

            output = X
            print('Input for convolution shape ', X.get_shape())
            i=0
            for layer in self.d_conv_layers:
                i+=1
                print('Convolution_layer_%i' %i)
                print('Input shape', output.get_shape())
                output = layer.forward(output,
                                     reuse, 
                                     is_training)
                print('After convolution shape', output.get_shape())
            
            output = tf.contrib.layers.flatten(output)
            print('After flatten shape', output.get_shape())
            i=0
            for layer in self.d_dense_layers:
                print('Dense weights %i' %i)
                print(layer.W.get_shape())
                output = layer.forward(output,
                                       reuse,
                                       is_training)
                i+=1
                print('After dense layer_%i' %i)
                print('Shape', output.get_shape())
  
            logits = self.d_final_layer.forward(output, 
                                                reuse,
                                                is_training)
            print('Logits shape', logits.get_shape())
            return logits

class resDiscriminator(object):

    def __init__(self, X, d_sizes, d_name):
        
        _, dim_H, dim_W, mi = X.get_shape().as_list()

        for key in d_sizes:
            if 'block' in key:
                print('Residual Network architecture detected')
                break
    

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
                    
                    mo, _, _, _, _, _, _, = d_sizes['convblock_layer_'+str(d_block_n)][-1]
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
                                                False, 1, 
                                                lambda x: x, w_init_last)
            

            self.d_steps=d_steps
            self.d_name = d_name
            
    def d_forward(self, X, reuse = None, is_training=True):
            print('Discriminator_'+self.d_name)
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

class Generator(object):

    def __init__(self, Z, dim_H, dim_W, g_sizes, g_name):
        
        #output images size better way to pass it?

        #dimensions of input
        latent_dims = g_sizes['z']
        dims_H =[dim_H]
        dims_W =[dim_W]

        mi = latent_dims

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
                                    f=act_f , w_init=w_init
                                    )

                self.g_dense_layers.append(layer)
                mi = mo
                
                
            #deconvolutional layers
        
            for _, _, stride, _, _, _, _ in reversed(g_sizes['conv_layers']):
                
                dim_H = int(np.ceil(float(dim_H)/stride))
                dim_W = int(np.ceil(float(dim_W)/stride))
                
                dims_H.append(dim_H)
                dims_W.append(dim_W)
    
            dims_H = list(reversed(dims_H))
            dims_W = list(reversed(dims_W))
            self.g_dims_H = dims_H
            self.g_dims_W = dims_W

            projection, bn_after_project, keep_prob, act_f, w_init = g_sizes['projection'][0]
            
            mo = projection*dims_H[0]*dims_W[0]
        
            name = 'g_dense_layer_%s' %count
            count+=1
            #print(name)

            self.g_final_layer = DenseLayer(name, mi, mo, not bn_after_project, keep_prob, act_f, w_init)
            # self.g_dense_layers.append(layer)
            
            
            mi = projection
            self.g_conv_layers=[]
            
            
            for i in range(len(g_sizes['conv_layers'])):
                name = 'g_conv_layer_%s' %count
                count +=1
                #print(name)
                mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init = g_sizes['conv_layers'][i]
                
                layer = DeconvLayer(
                  name, mi, mo, [dims_H[i+1], dims_W[i+1]],
                  filter_sz, stride, apply_batch_norm, keep_prob,
                  act_f, w_init
                )

                self.g_conv_layers.append(layer)
                mi = mo
                
            self.g_sizes=g_sizes
            self.g_name = g_name
            self.projection = projection
            self.bn_after_project = bn_after_project
   
    def g_forward(self, Z, reuse=None, is_training=True):
        print('Generator_'+self.g_name)
        print('Deconvolution')
        #dense layers

        output = Z
        print('Input for deconvolution shape', Z.get_shape())
        i=0
        for layer in self.g_dense_layers:
            print(i)
            output = layer.forward(output, reuse, is_training)
            print('After dense layer %i' %i)
            print('shape: ', output.get_shape())
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
            print('After deconvolutional layer %i' %i)
            print('shape: ', output.get_shape())


        print('Deconvoluted output shape', output.get_shape())
        return output

class resGenerator(object):

    def __init__(self, Z, dim_H, dim_W, g_sizes, g_name):
        
        latent_dims = g_sizes['z']

        mi = latent_dims

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
                
                #assert g_steps == self.d_steps, '\nUnmatching discriminator/generator architecture'
                

                g_block_n=0
                g_layer_n=0

                for key in g_sizes:
                    if 'block' and 'shortcut' in key:
                        g_block_n+=1
                    if 'deconv_layer' in key:
                        g_layer_n +=1

                assert g_block_n+g_layer_n==g_steps, '\nCheck keys in g_sizes, \n sum of generator steps do not coincide with sum of convolutional layers and convolutional blocks'

                #dimensions of output generated image
                dims_H =[dim_H]
                dims_W =[dim_W]

                layers_output_sizes={}
                blocks_output_sizes={}

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
                
                #print(name)

                layer = DenseLayer(name, mi, mo, not bn_after_project, keep_prob, act_f, w_init)
                # self.g_dense_layers.append(layer)

                
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
                #saving for later
                self.g_sizes=g_sizes
                self.g_name = g_name
                self.projection = projection
                self.bn_after_project = bn_after_project
                # return self.g_forward(Z)
   
    def g_forward(self, Z, reuse=None, is_training=True):
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

class cycleGenerator(object):

    def __init__(self, X, dim_H, dim_W, g_sizes, g_name):
        
        _, input_dim_H, input_dim_W, input_n_C = X.get_shape().as_list()

        dims_H =[dim_H]
        dims_W =[dim_W]

        for key in g_sizes:

            if 'block' in key:
                print('Residual Convolutional Network architecture detected')
            else:
                print('Check network architecture')
            break

        with tf.variable_scope('generator_'+g_name) as scope:
                
                #dense layers
                
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
                        # if not 'deconv' in key:
                            g_conv_blocks+=1

                    # if 'deconvblock' and 'shortcut' in key:
                    #     g_deconv_blocks+=1
                        
                    if 'deconv' in key:
                        if not 'shortcut' in key:
                             g_deconvs+=1
                
                assert g_steps == g_convs +2*(g_conv_blocks)+ g_deconvs, '\nCheck keys in g_sizes, \n sum of generator steps do not coincide with sum of convolutional layers, convolutional blocks and deconv layers'

                #dimensions of output generated image
                
                #conv_layers_output_sizes={}
                blocks_output_sizes={}
                deconv_layers_output_sizes={}
                

                for key, item in reversed(list(g_sizes.items())):

                    # if 'conv_layer' in key:
                    #     if not 'deconv' in key:
                    #         _, _, stride, _, _, _, _, = g_sizes[key][0]
                    #         conv_layers_output_sizes[g_convs-1] = [dim_H, dim_W]

                    #         dim_H = int(np.ceil(float(dim_H)/stride))
                    #         dim_W = int(np.ceil(float(dim_W)/stride))
                    #         dims_H.append(dim_H)
                    #         dims_W.append(dim_W)

                    #         g_convs -= 1


                      
                    # if 'convblock_layer' in key:
                        
                    #     for _ ,_ , stride, _, _, _, _, in g_sizes[key]:
                        
                    #         dim_H = int(np.ceil(float(dim_H)/stride))
                    #         dim_W = int(np.ceil(float(dim_W)/stride))
                    #         dims_H.append(dim_H)
                    #         dims_W.append(dim_W)
                        
                    #     blocks_output_sizes[g_conv_blocks-1] = [[dims_H[j],dims_W[j]] for j in range(1, len(g_sizes[key])+1)]
                    #     g_conv_blocks -=1

                    if 'deconv_layer' in key:
                        
                        
                        _, _, stride, _, _, _, _, = g_sizes[key][0]
                        deconv_layers_output_sizes[g_deconvs-1]= [dim_H, dim_W]
                        
                        dim_H = int(np.ceil(float(dim_H)/stride))
                        dim_W = int(np.ceil(float(dim_W)/stride))
                        dims_H.append(dim_H)
                        dims_W.append(dim_W)
                        
                        g_deconvs -= 1

                #assert g_convs  == 0
                #assert g_conv_blocks == 0
                assert g_deconvs  == 0
                
                dims_H = list(reversed(dims_H))
                dims_W = list(reversed(dims_W))

                # print(conv_layers_output_sizes)
                # print(blocks_output_sizes)

                #saving for later
                self.g_dims_H = dims_H
                self.g_dims_W = dims_W


                #deconvolution input channel number
                mi = input_n_C

                self.g_blocks=[]

                block_n=0 #keep count of the block number
                conv_layer_n=0 #keep count of conv layer number
                deconv_layer_n=0 #keep count of deconv layer number
                i=0 # keep count of the built blocks


                for key in g_sizes:

                    if 'conv_layer' in key:
                        if not 'deconv' in key

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
                        print(deconv_layer_n)
                        print(deconv_layers_output_sizes[deconv_layer_n])
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
            print('Generator_'+self.g_name)
            #dense layers

            output = X
            print('Input for generator shape', X.get_shape())
            
            i=0

            print(self.g_blocks)


            for block in self.g_blocks:
                i+=1
                output = block.forward(output,
                                        reuse,
                                        is_training)
                print('After block step%i' %i)
                if i == 11 or i ==13 or i==15:
                    print(block.name)
                print('shape: ', output.get_shape())
        

            print('Generator output shape', output.get_shape())
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
                False, 1, f=lambda x:x, w_init=e_sizes['last_layer_weight_init'])

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
                f=lambda x:x, w_init=d_sizes['last_layer_weight_init']
                )

            self.d_layers.append(last_dec_layer)

    def decode(self, Z, reuse=None, is_training=False):

        output=Z

        for layer in self.d_layers:
             output = layer.forward(output, reuse, is_training)
        
        return output

# class convEncoder:

# class convDecoder:

# class resEncoder:

# class resDecoder:   