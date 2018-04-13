#NETWORK ARCHITECTURES

import numpy as np
import os 
import math

import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

from architectures.utils.NN_building_blocks import *
from architectures.utils.NN_gen_building_blocks import *

def lrelu(x, alpha=0.1):
    return tf.maximum(alpha*x,x)

# some dummy constants
LEARNING_RATE = None
LEARNING_RATE_D = None
LEARNING_RATE_G = None
BETA1 = None
BATCH_SIZE = None
EPOCHS = None
SAVE_SAMPLE_PERIOD = None
PATH = None
SEED = None
rnd_seed=1


#can't seem to work on mnist
class resDCVAE(object):

    def __init__(self, n_H, n_W, n_C, e_sizes, d_sizes,
        lr=LEARNING_RATE, beta1=BETA1,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        save_sample=SAVE_SAMPLE_PERIOD, path=PATH):
        
        #size of every layer in the encoder
        #up to the latent layer, decoder
        #will have reverse shape
        self.n_H = n_H
        self.n_W = n_W
        self.n_C = n_C
        
        self.e_sizes = e_sizes
        self.d_sizes = d_sizes
        self.latent_dims = e_sizes['z']

        
        self.X = tf.placeholder(
            tf.float32,
            shape=(None, n_H, n_W, n_C),
            name='X'
        )
        
        self.batch_sz = tf.placeholder(
            tf.int32,
            shape=(),
            name='batch_sz'
        )
        
        #builds the encoder and outputs a Z distribution
        self.Z = self.build_encoder(self.X, self.e_sizes)
        
        #builds decoder from Z distribution
        logits = self.build_decoder(self.Z, self.d_sizes)
        
        #builds X_hat distribution from decoder output
        self.X_hat_distribution = Bernoulli(logits=logits)
        
        
        #posterior predictive
        
        with tf.variable_scope('encoder') as scope:
            scope.reuse_variables
            self.Z_dist = self.encode(
                self.X, reuse=True, is_training=False,
            )#self.X or something on purpose?

                                                   
        with tf.variable_scope('decoder') as scope:
            scope.reuse_variables()
            sample_logits = self.decode(
                self.Z_dist, reuse=True, is_training=False,
            )
            
        self.posterior_predictive_dist = Bernoulli(logits=sample_logits)
        self.posterior_predictive = self.posterior_predictive_dist.sample()
        self.posterior_predictive_probs = tf.nn.sigmoid(sample_logits)
        
        #prior predictive from prob

        standard_normal = Normal(
          loc=np.zeros(self.latent_dims, dtype=np.float32),
          scale=np.ones(self.latent_dims, dtype=np.float32)
        )

        Z_std = standard_normal.sample(1)

        with tf.variable_scope('decoder') as scope:
            scope.reuse_variables()
            logits_from_prob = self.decode(
                Z_std, reuse=True, is_training=False,
            )
        
        prior_predictive_dist = Bernoulli(logits=logits_from_prob)
        self.prior_predictive = prior_predictive_dist.sample()
        self.prior_predictive_probs = tf.nn.sigmoid(logits_from_prob)


        # prior predictive from input

        self.Z_input = tf.placeholder(tf.float32, shape=(None, self.latent_dims))
        
        with tf.variable_scope('decoder') as scope:
            scope.reuse_variables()    
            logits_from_input = self.decode(
                self.Z_input, reuse=True, is_training=False,
            )
        
        input_predictive_dist = Bernoulli(logits=logits_from_input)
        self.prior_predictive_from_input= input_predictive_dist.sample()
        self.prior_predictive_from_input_probs = tf.nn.sigmoid(logits_from_input)

        
        #cost
        kl = tf.reduce_sum(
            tf.contrib.distributions.kl_divergence(
                self.Z.distribution,
                standard_normal),
            1
        )
        
        
        expected_log_likelihood = tf.reduce_sum(
              self.X_hat_distribution.log_prob(self.X),
              1
        )
        
        self.loss = tf.reduce_sum(expected_log_likelihood - kl)
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
        ).minimize(-self.loss)

        #saving for later
        self.lr = lr
        self.batch_size=batch_size
        self.epochs = epochs
        self.path = path
        self.save_sample = save_sample
      
    def build_encoder(self, X, e_sizes):
        
        with tf.variable_scope('encoder') as scope:
            
            mi = self.n_C
            dim_H = self.n_H
            dim_W = self.n_W
            

            for key in e_sizes:
                if 'block' in key:
                    print('Residual Network architecture detected')
                    break

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

            #build dense layers
            
            self.e_dense_layers = []
            for mo, apply_batch_norm, keep_prob, act_f, w_init in e_sizes['dense_layers']:
                
                name = 'e_dense_layer_%s' %count
                count +=1
                
                layer = DenseLayer(name,mi, mo,
                                  apply_batch_norm, keep_prob, 
                                  act_f, w_init)
                mi = mo
                self.e_dense_layers.append(layer)
            
            #final logistic layer
            name = 'e_dense_layer_%s' %count

            last_enc_layer = DenseLayer(name, mi, 2*self.latent_dims, False, 1,
             f=lambda x: x, w_init=e_sizes['last_layer_weight_init'])
            
            self.e_dense_layers.append(last_enc_layer)            

            self.e_steps=e_steps

            return self.encode(X)
        
    def encode(self, X, reuse=None, is_training=True):

        #propagate X until end of encoder
        output=X

        i=0
        for block in self.e_blocks:
            i+=1
            # print('Convolution_block_%i' %i)
            # print('Input shape', output.get_shape())
            output = block.forward(output,
                                     reuse,
                                     is_training)
            # print('After block shape', output.get_shape())
        
        
        output = tf.contrib.layers.flatten(output)
        # print('After flatten shape', output.get_shape())

        i=0
        for layer in self.e_dense_layers:
            # print('Dense weights %i' %i)
            # print(layer.W.get_shape())
            output = layer.forward(output,
                                   reuse,
                                   is_training)
            i+=1
            # print('After dense layer_%i' %i)
            # print('Shape', output.get_shape())
        
        
        #get means and stddev from last encoder layer
        self.means = output[:, :self.latent_dims]
        self.stddev = tf.nn.softplus(output[:,self.latent_dims:])+1e-6
        
        # get a sample of Z, we need to use a stochastic tensor
        # in order for the errors to be backpropagated past this point
        
        with st.value_type(st.SampleValue()):
            Z = st.StochasticTensor(Normal(loc=self.means, scale=self.stddev))
        
        return Z

    def build_decoder(self, Z, d_sizes):

        with tf.variable_scope('decoder') as scope:

            #dimensions of input
            #dense layers
            self.d_dense_layers = []
            count = 0

            mi = self.latent_dims

            for mo, apply_batch_norm, keep_prob, act_f, w_init in d_sizes['dense_layers']:
                name = 'd_dense_layer_%s' %count
                count += 1
                
                layer = DenseLayer(
                    name, mi, mo,
                    apply_batch_norm, keep_prob,
                    f=act_f, w_init=w_init
                )
                self.d_dense_layers.append(layer)
                mi = mo
                
            #checking generator architecture

            d_steps = 0
            for key in d_sizes:
                if 'deconv' in key:
                    if not 'shortcut' in key:
                         d_steps+=1
            
            assert d_steps == self.e_steps, '\nUnmatching discriminator/generator architecture'
            

            d_block_n=0
            d_layer_n=0

            for key in d_sizes:
                if 'block' and 'shortcut' in key:
                    d_block_n+=1
                if 'deconv_layer' in key:
                    d_layer_n +=1

            assert d_block_n+d_layer_n==d_steps, '\nCheck keys in g_sizes, \n sum of generator steps do not coincide with sum of convolutional layers and convolutional blocks'

            #dimensions of output generated image
            dims_W = [self.n_W]
            dims_H = [self.n_H]

            dim_H = self.n_H
            dim_W = self.n_W


            layers_output_sizes={}
            blocks_output_sizes={}

            for key, item in reversed(list(d_sizes.items())):

                if 'deconv_layer' in key:
                    
                    _, _, stride, _, _, _, _, = d_sizes[key][0]
                    layers_output_sizes[d_layer_n-1]= [dim_H, dim_W]
                    
                    dim_H = int(np.ceil(float(dim_H)/stride))
                    dim_W = int(np.ceil(float(dim_W)/stride))
                    dims_H.append(dim_H)
                    dims_W.append(dim_W)
                    
                    d_layer_n -= 1

                  
                if 'deconvblock_layer' in key:
                    
                    for _ ,_ , stride, _, _, _, _, in d_sizes[key]:
                    
                        dim_H = int(np.ceil(float(dim_H)/stride))
                        dim_W = int(np.ceil(float(dim_W)/stride))
                        dims_H.append(dim_H)
                        dims_W.append(dim_W)
                    
                    blocks_output_sizes[d_block_n-1] = [[dims_H[j],dims_W[j]] for j in range(1, len(d_sizes[key])+1)]
                    d_block_n -=1

            dims_H = list(reversed(dims_H))
            dims_W = list(reversed(dims_W))

            #saving for later
            self.d_dims_H = dims_H
            self.d_dims_W = dims_W

            #final dense layer

            projection, bn_after_project, keep_prob, act_f, w_init = d_sizes['projection'][0]
                
            mo = projection*dims_W[0]*dims_H[0]

            #final dense layer
            name = 'dec_layer_%s' %count

            layer = DenseLayer(name, mi, mo, not bn_after_project, keep_prob, act_f, w_init)
            self.d_dense_layers.append(layer)


            #deconvolution input channel number
            mi = projection

            self.d_blocks=[]

            block_n=0 #keep count of the block number
            layer_n=0 #keep count of conv layer number
            i=0
            for key in d_sizes:
                
                if 'block' and 'shortcut' in key:
                
                    d_block = DeconvBlock(block_n,
                               mi, blocks_output_sizes, d_sizes,
                               )
                    self.d_blocks.append(d_block)
                    
                    mo, _, _, _, _, _, _, = d_sizes['deconvblock_layer_'+str(block_n)][-1]
                    mi = mo
                    block_n+=1
                    count+=1 
                    i+=1
                    
                if 'deconv_layer' in key:

                    name = 'd_conv_layer_{0}'.format(layer_n)

                    mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init = d_sizes[key][0]

                    d_conv_layer = DeconvLayer(
                        name, mi, mo, layers_output_sizes[layer_n],
                        filter_sz, stride, apply_batch_norm, keep_prob,
                        act_f, w_init
                    )
                    self.d_blocks.append(d_conv_layer)

                    mi=mo
                    layer_n+=1
                    count+=1 
                    i+=1

            assert i==d_steps, 'Check convolutional layer and block building, steps in building do not coincide with g_steps'
            assert d_steps==block_n+layer_n, 'Check keys in g_sizes'
            #saving for later
            self.d_sizes=d_sizes

            self.bn_after_project = bn_after_project
            self.projection = projection
            
            return self.decode(Z)
        
    def decode(self, Z, reuse=None, is_training=True):

        output = Z

        i=0
        for layer in self.d_dense_layers:
            i+=1
            output = layer.forward(output, reuse, is_training)


        
        output = tf.reshape(
            output,
            
            [-1, self.d_dims_H[0], self.d_dims_W[0], self.projection]
        
        )


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
        for block in self.d_blocks:
            i+=1
            output = block.forward(output,
                                    reuse,
                                    is_training)

    

        return output

    def set_session(self, session):
        
        self.session = session
        
        for layer in self.e_blocks:
            layer.set_session(session)
        for layer in self.e_dense_layers:
            layer.set_session(session)
            
        for layer in self.d_blocks:
            layer.set_session(session) 
        for layer in self.d_dense_layers:
            layer.set_session(session)  
        
    def fit(self, X):

        SEED = 1

        costs = []
        N = len(X)
        n_batches = N // self.batch_size

        

        print('\n ****** \n')
        print('Training residual convolutional VAE with a total of ' +str(N)+' samples distributed in batches of size '+str(self.batch_size)+'\n')
        print('The learning rate set is '+str(self.lr)+', and every ' +str(self.save_sample)+ ' iterations a generated sample will be saved to '+ self.path)
        print('\n ****** \n')
        total_iters=0

        for epoch in range(self.epochs):
            
            t0 = datetime.now()
            print('Epoch: {0}'.format(epoch))

            SEED = SEED + 1

            batches = unsupervised_random_mini_batches(X, self.batch_size, SEED)

            for X_batch in batches:

                feed_dict = {
                            self.X: X_batch, self.batch_sz: self.batch_size
                            }

                _, c = self.session.run(
                            (self.train_op, self.loss),
                            feed_dict=feed_dict
                    )

                c /= self.batch_size
                costs.append(c)

                total_iters += 1

                if total_iters % self.save_sample ==0:
                    print("At iteration: %d  -  dt: %s - cost: %.2f" % (total_iters, datetime.now() - t0, c))
                    print('Saving a sample...')
                        
                    probs = [self.prior_predictive_sample_with_probs()  for i in range(64)]  
                    
                    for i in range(64):
                        plt.subplot(8,8,i+1)
                        plt.imshow(probs[i].reshape(28,28), cmap='gray')
                        plt.subplots_adjust(wspace=0.2,hspace=0.2)
                        plt.axis('off')
                        
                    fig = plt.gcf()
                    fig.set_size_inches(4,4)
                    plt.savefig(self.path+'/samples_at_iter_%d.png' % total_iters,dpi=150)

        plt.clf()
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iteration')
        plt.title('learning rate=' + str(self.lr))
        plt.show()
        
        print('Parameters trained')

    def prior_predictive_with_input(self, Z):
        return self.session.run(
          self.prior_predictive_from_input_probs,
          feed_dict={self.Z_input: Z}
        )

    def posterior_predictive_sample(self, X):
        # returns a sample from p(x_new | X)
        return self.session.run(self.posterior_predictive_probs, feed_dict={self.X: X, self.batch_sz:BATCH_SIZE})

    def prior_predictive_sample_with_probs(self):
        # returns a sample from p(x_new | z), z ~ N(0, 1)
        return self.session.run(self.prior_predictive_probs)