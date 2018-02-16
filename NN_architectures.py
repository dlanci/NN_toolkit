
import scipy as sp
import numpy as np
import os 
import pickle
import math

import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime


st = tf.contrib.bayesflow.stochastic_tensor
Normal = tf.contrib.distributions.Normal
Bernoulli = tf.contrib.distributions.Bernoulli


from NN_building_blocks import *

def lrelu(x, alpha=0.1):
    return tf.maximum(alpha*x,x)

# some dummy constants
LEARNING_RATE = 1
BETA1 = 1
BATCH_SIZE = 1
EPOCHS = 1
SAVE_SAMPLE_PERIOD = 1
PATH = 'string'
SEED = 1

#NETWORK ARCHITECTURES

#class DNN:

#check if it works with metric function
class CNN:
    
    def __init__(

        self, n_W, n_H, n_C, sizes,
        lr=LEARNING_RATE, beta1=BETA1,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        save_sample=SAVE_SAMPLE_PERIOD, path=PATH
        ):

        """
        Positional args:

        

        Keyword args:



        """
        self.n_classes = sizes['n_classes']
        self.n_H = n_H
        self.n_W = n_W
        self.n_C = n_C

        self.conv_sizes = sizes

        self.keep_prob = tf.placeholder(
            tf.float32
            )
        
        self.X = tf.placeholder(
            tf.float32,
            shape=(None, n_W, n_H, n_C),
            name = 'X'
            )

        self.X_input = tf.placeholder(
            tf.float32,
            shape=(None, n_W, n_H, n_C),
            name = 'X'
            )

        self.batch_sz=tf.placeholder(
            tf.int32,
            shape=(),
            name='batch_sz',
            )

        self.Y = tf.placeholder(
            tf.float32,
            shape=(None, self.n_classes),
            name='Y'
        )

        self.Y_hat = self.build_CNN(self.X, self.conv_sizes)

        cost = tf.nn.softmax_cross_entropy_with_logits(
                logits= self.Y_hat,
                labels= self.Y
            )

        self.loss = tf.reduce_mean(cost)

        self.train_op = tf.train.AdamOptimizer(
            lr,
            beta1=beta1
            ).minimize(self.loss
            )

        self.X_input = tf.placeholder(
            tf.float32,
            shape = (None, n_W, n_H, n_C),
            name='X_input'
        )

        #convolve from input
        with tf.variable_scope('convolutional') as scope:
            scope.reuse_variables()
            self.Y_hat_from_test = self.convolve(
                self.X_input, reuse=True, is_training=False, keep_prob=1
            )

        #saving for later
        self.lr = lr
        self.batch_size=batch_size
        self.epochs = epochs
        self.path = path

    def build_CNN(self, X, conv_sizes):

        with tf.variable_scope('convolutional') as scope:
            #keep track of dims for dense layers
            mi = self.n_C
            dim_W = self.n_W
            dim_H = self.n_H

            self.conv_layers = []

            count = 0
            
            # n=0
            # for key in conv_sizes:
            #     if 'conv' in key:
            #         n+=1

            n = len(conv_sizes)-1 # count the number of layers leaving out n_classes key
            
            for key in conv_sizes:
                if not 'block' in key:
                    print('Convolutional network architecture detected')
                else:
                    print('Check network architecture')
                break
            
            for key in conv_sizes:
                if 'conv' in key:

                    mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init = conv_sizes[key][0]

                    name = 'conv_layer_{0}'.format(count)
                    count += 1

                    layer = ConvLayer(name,
                        mi, mo, filter_sz, stride,
                        apply_batch_norm, keep_prob,
                        f=act_f, w_init=w_init
                    )
                    self.conv_layers.append(layer)

                    mi=mo

                    dim_W = int(np.ceil(float(dim_W) / stride))
                    dim_H = int(np.ceil(float(dim_H) / stride))

                if 'pool' in key:
                    count+=1
                    if 'max' in key:
                        
                        filter_sz, stride, keep_prob = conv_sizes[key][0]

                        layer = MaxPool2D(filter_sz, stride, keep_prob)

                        dim_W = int(np.ceil(float(dim_W) / stride))
                        dim_H = int(np.ceil(float(dim_H) / stride))

                    if 'avg' in key:

                        filter_sz, stride, keep_prob =conv_sizes[key][0]

                        layer = AvgPool2D(filter_sz, stride, keep_prob)

                        dim_W = int(np.ceil(float(dim_W) / stride))
                        dim_H = int(np.ceil(float(dim_H) / stride))
                    
                    self.conv_layers.append(layer)

            mi = mi * dim_W * dim_H
            self.dense_layers = []
            for mo, apply_batch_norm, keep_prob, act_f, w_init in conv_sizes['dense_layers']:

                name = 'dense_layer_{0}'.format(count)
                count += 1

                layer = DenseLayer(name,mi, mo,
                                  apply_batch_norm, keep_prob,
                                  act_f, w_init)
                mi = mo
                self.dense_layers.append(layer)


            readout_layer =  DenseLayer('readout_layer', 
                                        mi, self.n_classes,
                                        False, 1, tf.nn.softmax, 
                                        tf.random_uniform_initializer())

            self.dense_layers.append(readout_layer)

            return self.convolve(X)

    def convolve(self, X, reuse=None, is_training=True, keep_prob=0.5):
        
        print('Convolution')
        print('Input for convolution', X.get_shape())
        
        output=X
        i=0
        for layer in self.conv_layers:
            i+=1
            output = layer.forward(output, reuse, is_training)
            print('After convolution', i)
            print(output.get_shape())
            

        print('After convolution shape', output.get_shape())

        output = tf.contrib.layers.flatten(output)

        print('After flatten shape', output.get_shape())
        for layer in self.dense_layers:

            i+=1
            output = layer.forward(output, reuse, is_training)
            print('After dense layer {0}, shape {1}'.format(i, output.get_shape()))

        print('Logits shape', output.get_shape())
        return output

    def set_session(self, session):

        self.session=session

        for layer in self.conv_layers:
            layer.set_session(session)

    def fit(self, X, Y):

        SEED = 1

        N = X.shape[0]
        n_batches = N // self.batch_size

        print('\n ****** \n')
        print('Training CNN with a total of ' +str(N)+' samples distributed in batches of size '+str(self.batch_size)+'\n')
        print('The learning rate set is '+str(self.lr))
        print('\n ****** \n')

        costs = []
        for epoch in range(self.epochs):

            SEED = SEED + 1

            batches = supervised_random_mini_batches(X, Y, self.batch_size, SEED)

            for batch in batches:

                (X_batch, Y_batch) = batch

                feed_dict = {

                            self.X: X_batch,
                            self.Y: Y_batch,
                            self.batch_sz: self.batch_size,
                            
                            }

                _, c = self.session.run(
                            (self.train_op, self.loss),
                            feed_dict=feed_dict
                    )

                c /= self.batch_size
                costs.append(c)

            if epoch % 10 ==0:

                #define a metric/accuracy

                print('At epoch %d, cost: %f' %(epoch, c))

        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iteration')
        plt.title('learning rate=' + str(self.lr))
        plt.show()
        
        print('Parameters trained')

        #get samples at test time
    
    def predicted_Y_hat(self, X):
        return self.session.run(
            self.rec_sample,
            feed_dict={self.X_input:X}
        )

#check if it works with metric function
class resCNN:
    
    def __init__(

        self, n_W, n_H, n_C, sizes,
        lr=LEARNING_RATE, beta1=BETA1,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        save_sample=SAVE_SAMPLE_PERIOD, path=PATH
        ):

        """
        Positional args:


        Keyword args:


        """
        self.n_classes = sizes['n_classes']
        self.n_H = n_H
        self.n_W = n_W
        self.n_C = n_C

        self.conv_sizes = sizes

        self.keep_prob = tf.placeholder(
            tf.float32
            )

        self.X = tf.placeholder(
            tf.float32,
            shape=(None, n_W, n_H, n_C),
            name = 'X'
            )

        self.X_input = tf.placeholder(
            tf.float32,
            shape=(None, n_W, n_H, n_C),
            name = 'X'
            )

        self.batch_sz=tf.placeholder(
            tf.int32,
            shape=(),
            name='batch_sz',
            )

        self.Y = tf.placeholder(
            tf.float32,
            shape=(None, self.n_classes),
            name='Y'
        )

        self.Y_hat = self.build_resCNN(self.X, self.conv_sizes)

        cost = tf.nn.softmax_cross_entropy_with_logits(
                logits= self.Y_hat ,
                labels= self.Y
            )

        self.loss = tf.reduce_mean(cost)

        self.train_op = tf.train.AdamOptimizer(
            lr,
            beta1=beta1
            ).minimize(self.loss
            )

        self.X_input = tf.placeholder(
            tf.float32,
            shape = (None, n_W, n_H, n_C),
            name='X_input'
        )

        #convolve from input
        with tf.variable_scope('convolutional') as scope:
            scope.reuse_variables()
            self.Y_hat_from_test = self.convolve(
                self.X_input, reuse=True, is_training=False,
            )

        #saving for later
        self.lr = lr
        self.batch_size=batch_size
        self.epochs = epochs
        self.path = path

    def build_resCNN(self, X, conv_sizes):
        
        with tf.variable_scope('convolutional') as scope:
            
            #dimensions of input
            mi = self.n_C

            dim_W = self.n_W
            dim_H = self.n_H

            
            for key in conv_sizes:
                if 'block' in key:
                    print('Residual Network architecture detected')
                else:
                    print('Check network architecture')
                break
            
            self.conv_blocks = []
            #count conv blocks
            # n_conv_blocks = 0
            # for key in sizes:
            #     if 'block' in key:
            #         n_conv_blocks+=1

            #build convblocks
            i=0

            count=0
            for key in conv_sizes:
                count+=1 
                if 'block' and 'shortcut' in key:

                    conv_block = ConvBlock(i,
                               mi, conv_sizes,
                               )
                    self.conv_blocks.append(conv_block)
                    
                    mo, _, _, _, _, _, _, = conv_sizes['convblock_layer_'+str(i)][-1]
                    mi = mo
                    dim_H = conv_block.output_dim(dim_H)
                    dim_W = conv_block.output_dim(dim_W)
                    i+=1

                if 'pool' in key:

                    if 'max' in key:

                        filter_sz, stride, keep_prob =conv_sizes[key][0]

                        maxpool_layer = MaxPool2D(filter_sz, stride, keep_prob)

                        self.conv_blocks.append(maxpool_layer)

                        dim_W = int(np.ceil(float(dim_W) / stride))
                        dim_H = int(np.ceil(float(dim_H) / stride))
                        
                    if 'avg' in key:

                        filter_sz, stride, keep_prob =conv_sizes[key][0]

                        avgpool_layer = AvgPool2D(filter_sz, stride, keep_prob)

                        self.conv_blocks.append(avg_layer)
                        
                        dim_W = int(np.ceil(float(dim_W) / stride))
                        dim_H = int(np.ceil(float(dim_H) / stride))
                   

            mi = mi * dim_W * dim_H
            self.dense_layers = []

            for mo, apply_batch_norm, keep_prob, act_f, w_init in conv_sizes['dense_layers']:
                print(mi,mo)
                name = 'dense_layer_{0}'.format(count)
                count += 1

                layer = DenseLayer(name,mi, mo,
                                  apply_batch_norm, keep_prob,
                                  act_f, w_init)
                mi = mo
                self.dense_layers.append(layer)

            readout_layer =  DenseLayer('readout_layer', 
                                        mi, self.n_classes,
                                        False, 1, tf.nn.softmax, 
                                        tf.random_uniform_initializer())

            self.dense_layers.append(readout_layer)


            return self.convolve(X)
            
    def convolve(self, X, reuse = None, is_training=True):

        print('Convolution')
        print('Input for convolution shape ', X.get_shape())

        output = X

        i=0
        for block in self.conv_blocks:
            i+=1
            print('Convolution_block_%i' %i)
            print('Input shape', output.get_shape())
            output = block.forward(output,
                                     reuse,
                                     is_training)
        
        
        output = tf.contrib.layers.flatten(output)
        print('After flatten shape', output.get_shape())

        i=0
        for layer in self.dense_layers:
            i+=1
            print('Dense weights %i' %i)
            print(layer.W.get_shape())
            output = layer.forward(output,
                                   reuse,
                                   is_training)

            print('After dense layer_%i' %i)
            print('Shape', output.get_shape())

        print('Logits shape', output.get_shape())
        return output

    def set_session(self, session):

        self.session=session

        for layer in self.conv_blocks:
            layer.set_session(session)

    def fit(self, X, Y):

        SEED = 1

        N = X.shape[0]
        n_batches = N // self.batch_size

        print('\n ****** \n')
        print('Training CNN with a total of ' +str(N)+' samples distributed in batches of size '+str(self.batch_size)+'\n')
        print('The learning rate set is '+str(self.lr))
        print('\n ****** \n')

        costs = []
        for epoch in range(self.epochs):

            SEED = SEED + 1

            batches = supervised_random_mini_batches(X, Y, self.batch_size, SEED)

            for batch in batches:

                (X_batch, Y_batch) = batch

                feed_dict = {

                            self.X: X_batch,
                            self.Y: Y_batch,
                            self.batch_sz: self.batch_size,
                            
                            }

                _, c = self.session.run(
                            (self.train_op, self.loss),
                            feed_dict=feed_dict
                    )

                c /= self.batch_size
                costs.append(c)

            if epoch % 10 ==0:

                #define a metric/accuracy

                print('At epoch %d, cost: %f' %(epoch, c))

        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iteration')
        plt.title('learning rate=' + str(self.lr))
        plt.show()
        
        print('Parameters trained')

        #get samples at test time
    
    def predicted_Y_hat(self, X):
        return self.session.run(
            self.rec_sample,
            feed_dict={self.X_input:X}
        )

#check if works by creating some samples
class DAE:

    def __init__(
        self, dim, e_sizes, d_sizes, an_id=0, 
        lr=LEARNING_RATE, beta1=BETA1,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        save_sample=SAVE_SAMPLE_PERIOD, path=PATH
        ):
        """
        Positional args

        Keyword args


        """

        self.dim = dim
        self.an_id = an_id
        self.latent_dims=e_sizes['z']
        self.e_sizes=e_sizes
        self.d_sizes=d_sizes

        self.e_last_act_f = e_sizes['last_act_f']
        self.d_last_act_f = d_sizes['last_act_f']

        self.X = tf.placeholder(
                tf.float32,
                shape=(None, self.dim),
                name='X'
            )
            
        self.batch_sz = tf.placeholder(
                tf.float32,
                shape=(),
                name='batch_sz'
            )

        self.Z=self.build_encoder(self.X, self.e_sizes)

        logits = self.build_decoder(self.Z, self.d_sizes)

        self.X_hat = self.d_last_act_f(logits)

        cost = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.X,
                    logits=logits
                )

        self.loss= tf.reduce_sum(cost) #mean?           

        self.train_op = tf.train.AdamOptimizer(
            lr,
            beta1=beta1
            ).minimize(self.loss)

        #test time

        self.X_input = tf.placeholder(
                tf.float32,
                shape = (None, self.dim),
                name='X_input'
            )

        #encode at test time

        with tf.variable_scope('encoder') as scope:
            scope.reuse_variables()
            self.Z_input = self.encode(
                self.X_input, reuse=True, is_training=False
            )

        #decode from encoded at test time

        with tf.variable_scope('decoder') as scope:
            scope.reuse_variables()
            X_decoded = self.decode(
                self.Z_input, reuse=True, is_training=False
                )
            self.X_decoded = tf.nn.sigmoid(X_decoded)


        #saving for later
        self.lr = lr
        self.batch_size=batch_size
        self.epochs = epochs
        self.path = path
        self.save_sample = save_sample

    def build_encoder(self, X, e_sizes):

        with tf.variable_scope('encoder') as scope:

            #dimensions of input
            mi = self.dim

            self.e_layers = []

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
            last_enc_layer = DenseLayer(name, mi, self.latent_dims, False, 1,
                self.e_last_act_f, w_init=tf.random_normal_initializer(stddev=0.02)
                )
            self.e_layers.append(last_enc_layer)

            return self.encode(X)

    def encode(self, X, reuse=None, is_training=True):

        Z=X
        for layer in self.e_layers:
            Z=layer.forward(Z, reuse, is_training)
        return Z

    def build_decoder(self, Z, sizes):

        with tf.variable_scope('decoder') as scope:

            mi = self.latent_dims

            self.d_layers = []
            count = 0
            for mo, apply_batch_norm, keep_prob, act_f, w_init in self.d_sizes['dense_layers']:

                name = 'layer_{0}'.format(count)
                count += 1

                layer = DenseLayer(name, mi, mo,
                        apply_batch_norm, keep_prob,
                        act_f, w_init
                        )

                self.d_layers.append(layer)
                mi = mo

            name = 'layer_{0}'.format(count)

            last_dec_layer = DenseLayer(name, mi, self.dim, False, 1,
                f=lambda x:x, w_init=tf.random_normal_initializer(stddev=0.02)
                )

            self.d_layers.append(last_dec_layer)


            return self.decode(Z)

    def decode(self, Z, reuse=None, is_training=True):

        X=Z

        for layer in self.d_layers:
             X = layer.forward(X, reuse, is_training)
        return X

    def set_session(self, session):

        self.session = session

        for layer in self.d_layers:
            layer.set_session(self.session)

        for layer in self.e_layers:
            layer.set_session(self.session)

    def fit(self, X):

        SEED = 1

        costs = []
        N = len(X)
        n_batches = N // self.batch_size

        total_iters=0

        print('\n ****** \n')
        print('Training deep AE with a total of ' +str(N)+' samples distributed in batches of size '+str(self.batch_size)+'\n')
        print('The learning rate set is '+str(self.lr)+', and every ' +str(self.save_sample)+ ' epoch a generated sample will be saved to '+ self.path)
        print('\n ****** \n')

        for epoch in range(self.epochs):

            # print('Epoch: {0}'.format(epoch))

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

            if epoch % self.save_sample ==0:


                print('At epoch %d, cost: %f' %(epoch, c))

        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iteration')
        plt.title('learning rate=' + str(self.lr))
        plt.show()
        
        print('Parameters trained')

    def get_sample(self, X):
        return self.session.run(
            self.X_decoded, feed_dict={self.X_input:X, self.batch_sz:1}
            )

#check if works by creating some samples
class DVAE:

    def __init__(
        self, dim, e_sizes, d_sizes, an_id=0, 
        lr=LEARNING_RATE, beta1=BETA1,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        save_sample=SAVE_SAMPLE_PERIOD, path=PATH
        ):
        """
        Positional args


        Keyword args


        """
        self.dim = dim
        self.e_sizes=e_sizes
        self.d_sizes=d_sizes

        self.latent_dims=e_sizes['z']
        self.d_last_act_f = d_sizes['last_act_f']

        self.X = tf.placeholder(
                tf.float32,
                shape=(None, self.dim),
                name='X'
            )
            
        self.batch_sz = tf.placeholder(
                tf.float32,
                shape=(),
                name='batch_sz'
            )

        self.Z=self.build_encoder(self.X, self.e_sizes)


        logits = self.build_decoder(self.Z, self.d_sizes)


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
        
        self.elbo = tf.reduce_sum(expected_log_likelihood - kl)
        
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
        ).minimize(-self.elbo)          


        #test time

        self.X_input = tf.placeholder(
                tf.float32,
                shape = (None, self.dim),
                name='X_input'
            )

        #encode at test time

        with tf.variable_scope('encoder') as scope:
            scope.reuse_variables()
            self.Z_input = self.encode(
                self.X_input, reuse=True, is_training=False
            )
        #decode from encoded at test time

        with tf.variable_scope('decoder') as scope:
            scope.reuse_variables()
            X_decoded = self.decode(
                self.Z_input, reuse=True, is_training=False
                )
            self.X_decoded = tf.nn.sigmoid(X_decoded)

        #saving for later
        self.lr = lr
        self.batch_size=batch_size
        self.epochs = epochs
        self.path = path
        self.save_sample = save_sample

    def build_encoder(self, X, e_sizes):

        with tf.variable_scope('encoder') as scope:

            #dimensions of input
            mi = self.dim
            self.e_layers = []

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

            last_enc_layer = DenseLayer(name, mi, 2*self.latent_dims,
                False, 1, f=lambda x:x, w_init=tf.random_normal_initializer())

            self.e_layers.append(last_enc_layer)

            return self.encode(X)

    def encode(self, X, reuse=None, is_training=False):

        output=X
        for layer in self.e_layers:
            output = layer.forward(output, reuse, is_training)

        self.means = output[:,:self.latent_dims]
        self.stddev = tf.nn.softplus(output[:,self.latent_dims:])+1e-6

        with st.value_type(st.SampleValue()):
            Z = st.StochasticTensor(Normal(loc=self.means, scale=self.stddev))
        
        return Z

    def build_decoder(self, Z, d_sizes):

        with tf.variable_scope('decoder') as scope:

            mi = self.latent_dims

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

            last_dec_layer = DenseLayer(name, mi, self.dim, False, 1,
                f=lambda x:x, w_init=tf.random_normal_initializer(stddev=0.02)
                )

            self.d_layers.append(last_dec_layer)

            return self.decode(Z)

    def decode(self, Z, reuse=None, is_training=False):

        output=Z

        for layer in self.d_layers:
             output = layer.forward(output, reuse, is_training)
        
        return output

    def set_session(self, session):

        self.session = session

        for layer in self.d_layers:
            layer.set_session(self.session)

        for layer in self.e_layers:
            layer.set_session(self.session)

    def fit(self, X):

        SEED = 1

        costs = []
        N = len(X)
        n_batches = N // self.batch_size

        total_iters=0

        print('\n ****** \n')
        print('Training deep VAE with a total of ' +str(N)+' samples distributed in batches of size '+str(self.batch_size)+'\n')
        print('The learning rate set is '+str(self.lr)+', and every ' +str(self.save_sample)+ ' epoch a generated sample will be saved to '+ self.path)
        print('\n ****** \n')


        for epoch in range(self.epochs):

            SEED = SEED + 1

            batches = unsupervised_random_mini_batches(X, self.batch_size, SEED)

            for X_batch in batches:

                feed_dict = {
                            self.X: X_batch, self.batch_sz: self.batch_size
                            }

                _, c = self.session.run(
                            (self.train_op, self.elbo),
                            feed_dict=feed_dict
                    )

                c /= self.batch_size

                costs.append(c)

            total_iters +=1

            if total_iters % self.save_sample ==0:

                print('At epoch %d, cost: %f' %(epoch, c))

                #         print("on iter {0}, cost: {0}".format(total_iters, c))
                #         print('Saving a sample...')
                        
                #         n = 8
                        
                #         samples = sample(Z, n*n)
                        
                #         for i in range(64):
                #             plt.subplot(8,8,i+1)
                #             plt.imshow(samples[i].reshape(d,d), cmap='gray')
                #             plt.subplots_adjust(wspace=0.2,hspace=0.2)
                #             plt.axis('off')
                        
                #         fig = plt.gcf()
                #         fig.set_size_inches(5, 5)
                #         plt.savefig(self.path+'/samples_at_iter_%d.png' % total_iters,dpi=300)

        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iteration')
        plt.title('Learning rate=' + str(self.lr))
        plt.show()
        
        print('Parameters trained')

    def sample(self, Z, n):
        samples = self.session.run(
          self.prior_predictive_from_input_probs,
          feed_dict={self.Z_input: Z, self.batch_sz: n}
        )
        return samples

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
        return self.session.run((self.prior_predictive, self.prior_predictive_probs))

#check if works by creating some samples
#check support for rectangular images
class DCVAE:

    def __init__(self, n_W, n_H, n_C, e_sizes, d_sizes,
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
            shape=(None, n_W, n_H, n_C),
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
        
        self.elbo = tf.reduce_sum(expected_log_likelihood - kl)
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=LEARNING_RATE,
            beta1=BETA1,
        ).minimize(-self.elbo)

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
            
            self.e_conv_layers=[]
            count = 0
            
            for mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init in e_sizes['conv_layers']:
                
                name = 'e_conv_layer_%s' %count
                count += 1
                
                layer = ConvLayer(name, mi, mo,
                                  filter_sz, stride, 
                                  apply_batch_norm, keep_prob,
                                  act_f, w_init)

                self.e_conv_layers.append(layer)
                mi = mo
                
                #print('Dim:', dim)
                dim_H = int(np.ceil(float(dim_H)/stride))
                dim_W = int(np.ceil(float(dim_W)/stride))
            
            mi = mi*dim_H*dim_W
            
            self.e_dense_layers=[]
            
            for mo, apply_batch_norm, keep_prob, act_f, w_init in e_sizes['dense_layers']:
                
                name = 'e_dense_layer_%s' %count
                count +=1
                
                layer = DenseLayer(name, mi, mo, 
                                   apply_batch_norm, keep_prob,
                                   act_f, w_init)

                self.e_dense_layers.append(layer)
                
                mi = mo
        
            #no activation of last layer and need 2
            #times as many units (M means and M stddevs)
            name = 'e_conv_layer_%s' %count
            last_enc_layer = DenseLayer(name, mi, 2*self.latent_dims, False, 1, f=lambda x: x)
            self.e_dense_layers.append(last_enc_layer)
            
            return self.encode(X)
        
    def encode(self, X, reuse=None, is_training=True):
        #propagate X until end of encoder
        output=X

        for layer in self.e_conv_layers:
            output = layer.forward(output, reuse, is_training)
        
        output = tf.contrib.layers.flatten(output)
        
        for layer in self.e_dense_layers:
            output = layer.forward(output, reuse, is_training)
        
        
        #get means and stddev from last encoder layer
        self.means = output[:, :self.latent_dims]
        self.stddev = tf.nn.softplus(output[:,self.latent_dims:])+1e-6
        
        # get a sample of Z, we need to use a stochastic tensor
        # in order for the errors to be backpropagated past this point
        
        with st.value_type(st.SampleValue()):
            Z = st.StochasticTensor(Normal(loc=self.means, scale=self.stddev))
        
        return Z
    
        #build decoder
    
    def build_decoder(self, Z, d_sizes):
        
        with tf.variable_scope('decoder') as scope:
            
            dims_H=[self.n_H]
            dims_W=[self.n_W]
            
            dim_H = self.n_H
            dim_W = self.n_W
            
            for _, _, stride, _, _, _, _, in reversed(d_sizes['conv_layers']):
                dim_H = int(np.ceil(float(dim_H)/stride))
                dim_W = int(np.ceil(float(dim_W)/stride))
                
                dims_H.append(dim_H)
                dims_W.append(dim_W)
            
            dims_H = list(reversed(dims_H))
            dims_W = list(reversed(dims_W))
            #print('Decoder dims:', dims)
            self.d_dims_H = dims_H
            self.d_dims_W = dims_W
            
            mi = self.latent_dims
            self.d_dense_layers =[]
            
            count=0
            for mo, apply_batch_norm, keep_prob, act_f, w_init in d_sizes['dense_layers']:
                
                name = 'd_dense_layer_%s' %count
                count +=1
                
                layer = DenseLayer(name, mi, mo, 
                                   apply_batch_norm, keep_prob,
                                   act_f, w_init)
                self.d_dense_layers.append(layer)
                mi = mo
                
            mo = d_sizes['projection']*dims_W[0]*dims_H[0]


            #final dense layer
            name = 'dec_layer_%s' %count
            last_dec_layer = DenseLayer(name, mi, mo, not d_sizes['bn_after_project'], 1)
            self.d_dense_layers.append(last_dec_layer)
            
            
            #fractionally strided layers
            
            mi = d_sizes['projection']
            self.d_conv_layers=[]
            

            activation_functions = []

            for _, _, _, _, _, act_f, _ in d_sizes['conv_layers'][:-1]:
                activation_functions.append(act_f)

            activation_functions.append(d_sizes['output_activation'])
            
            for i in range(len(d_sizes['conv_layers'])):
                name = 'fs_convlayer_%s' %i
                
                mo, filter_sz, stride, apply_batch_norm, keep_prob, _, w_init = d_sizes['conv_layers'][i]
                f = activation_functions[i]
                
                layer = DeconvLayer(
                  name, mi, mo, [dims_H[i+1], dims_W[i+1]],
                  filter_sz, stride,
                  apply_batch_norm, keep_prob, 
                  f, w_init
                )

                self.d_conv_layers.append(layer)
                mi = mo
            
            return self.decode(Z)
    
    def decode(self, Z, reuse=None, is_training=True):
        
        #dense layers
        output = Z
        
        for layer in self.d_dense_layers:
            output = layer.forward(output, reuse, is_training)

        output = tf.reshape(
            output,
            [-1, self.d_dims_W[0],self.d_dims_H[0],self.d_sizes['projection']]
        )

        if self.d_sizes['bn_after_project']:
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
        #passing to fs-convolutional layers   
        
        for layer in self.d_conv_layers:

            output = layer.forward(output, reuse, is_training)
            
        return output
    
    def set_session(self, session):
        
        self.session = session
        
        for layer in self.e_conv_layers:
            layer.set_session(session)
        for layer in self.e_dense_layers:
            layer.set_session(session)
            
        for layer in self.d_dense_layers:
            layer.set_session(session) 
        for layer in self.d_conv_layers:
            layer.set_session(session)  
        
    def fit(self, X):

        SEED = 1

        costs = []
        N = len(X)
        n_batches = N // self.batch_size

        total_iters=0

        print('\n ****** \n')
        print('Training deep convolutional VAE with a total of ' +str(N)+' samples distributed in '+str(n_batches)+' batches, each of size '+str(self.batch_size)+'\n')
        print('The learning rate set is '+str(self.lr)+', and every ' +str(self.save_sample)+ ' epoch a generated sample will be saved to '+ self.path)
        print('\n ****** \n')

        for epoch in range(self.epochs):

            SEED = SEED + 1

            batches = unsupervised_random_mini_batches(X, self.batch_size, SEED)

            for X_batch in batches:

                feed_dict = {
                            self.X: X_batch, self.batch_sz: self.batch_size
                            }

                _, c = self.session.run(
                            (self.train_op, self.elbo),
                            feed_dict=feed_dict
                    )

                c /= self.batch_size
                
            total_iters +=1

            if total_iters % self.save_sample == 0:
                print("On iteration %d, cost: %.3f" %(total_iters, c))
                print('Saving a sample...')
                    
                    
                n = 8 #number of images per side
                x_values = np.linspace(-3,3,n)
                y_values = np.linspace(-3,3,n)

                flat_image = np.empty((28*n,28*n))

                Z=[]
                for i, x in enumerate(x_values):
                    for j, y in enumerate(y_values):
                        z=[x, y]
                        Z.append(z)
        
                samples = self.sample(Z, n*n)
                
                k = 0
                for i, x in enumerate(x_values):
                    for j, y in enumerate(y_values):  
                        x_recon = samples[k]
                        k+=1
        
                        x_recon=x_recon.reshape(28,28)
                        flat_image[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = x_recon
                
                plt.imshow(flat_image,cmap='gray')
                plt.show()
                plt.savefig(
                    self.path+'samples_at_iter_%d.png' % total_iters,
                )
                                
            plt.clf()
            plt.plot(costs, label='cost vs iteration')
            plt.show()
            plt.savefig(self.path+'cost vs iteration.png')

    def sample(self, Z, n):
        samples = self.session.run(
          self.prior_predictive_from_input_probs,
          feed_dict={self.Z_input: Z, self.batch_sz: n}
        )
        return samples

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
        return self.session.run((self.prior_predictive, self.prior_predictive_probs))

#debug
#check if works by creating some samples
#check support for rectangular images
class resDCVAE:

    def __init__(self, n_W, n_H, n_C, e_sizes, d_sizes,
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
            shape=(None, n_W, n_H, n_C),
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
        
        self.elbo = tf.reduce_sum(expected_log_likelihood - kl)
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=LEARNING_RATE,
            beta1=BETA1,
        ).minimize(-self.elbo)

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
                else:
                    print('Check network architecture')
                break

            self.e_blocks = []
            #count conv blocks

            n_conv_blocks = 0
            for key in d_sizes:
                if 'block' in key:
                    n_conv_blocks+=1

            #build conv blocks
            for i in range(n_conv_blocks//2):
            
                e_block = ConvBlock(i,
                               mi, e_sizes,
                               )
                self.e_blocks.append(e_block)
                mo, _, _, _, _, _, _, = e_sizes['convblock_layer_'+str(i)][-1]
                mi = mo
                dims_H=e_block.output_dim(dim_H)
                dims_W=e_block.output_dim(dim_W)
        
            mi = mi * dims_H * dims_W
            count = i
            
            self.e_dense_layers = []
            for mo, apply_batch_norm, keep_prob, act_f, w_init in e_sizes['dense_layers']:
                
                name = 'e_dense_layer_%s' %count
                count +=1
                
                layer = DenseLayer(name, mi, mo, 
                                  apply_batch_norm, keep_prob,
                                  act_f, w_init)
                mi = mo
                self.e_dense_layers.append(layer)
        
            #no activation of last layer and need 2
            #times as many units (M means and M stddevs)
            name = 'e_conv_layer_%s' %count
            last_enc_layer = DenseLayer(name, mi, 2*self.latent_dims, False, 1, f=lambda x: x)
            self.e_dense_layers.append(last_enc_layer)


            return self.encode(X)
        
    def encode(self, X, reuse=None, is_training=True):

        print('Convolution')
        #propagate X until end of encoder
        output=X

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
        
        
        #get means and stddev from last encoder layer
        self.means = output[:, :self.latent_dims]
        self.stddev = tf.nn.softplus(output[:,self.latent_dims:])+1e-6
        
        # get a sample of Z, we need to use a stochastic tensor
        # in order for the errors to be backpropagated past this point
        
        with st.value_type(st.SampleValue()):
            Z = st.StochasticTensor(Normal(loc=self.means, scale=self.stddev))
        
        print('Latent space dimensions (/2)', output.get_shape())
        return Z

    def build_decoder(self, Z, d_sizes):

        with tf.variable_scope('decoder') as scope:
            
            #dimensions of input
            dims_W = [self.n_W]
            dims_H = [self.n_H]

            dim_H = self.n_W
            dim_W = self.n_W

            mi = self.latent_dims
            
            for _, _, stride, _ in reversed(d_sizes['conv_layers']):
                dim_H = int(np.ceil(float(dim_H)/stride))
                dim_W = int(np.ceil(float(dim_W)/stride))
                
                dims_H.append(dim_H)
                dims_W.append(dim_W)
            
            dims_H = list(reversed(dims_H))
            dims_W = list(reversed(dims_W))
            #print('Decoder dims:', dims)
            self.d_dims_H = dims_H
            self.d_dims_W = dims_W

            #dense layers
            self.d_dense_layers = []
            count = 0

            for mo, apply_batch_norm, keep_prob, act_f, w_init in d_sizes['dense_layers']:
                name = 'd_dense_layer_%s' %count
                count += 1
                
                layer = DenseLayer(name,mi, mo,
                                   apply_batch_norm, keep_prob,
                                   f=act_f, w_init=w_init
                )
                self.d_dense_layers.append(layer)
                mi = mo
                    
            #calculate height and width of image at every layer            
            #final dense layer
            mo = d_sizes['projection']*dims_H[0]*dims_W[0]
            
            name = 'd_dense_layer_%s' %count
            layer = DenseLayer(name, mi, mo, not d_sizes['bn_after_project'], 1)
            self.d_dense_layers.append(layer)
            

            #input channel number
            mi = g_sizes['projection']
            #count deconv blocks
            n_conv_blocks = 0
            for key in g_sizes:
                if 'shortcut' in key:
                    n_conv_blocks+=1
            
            activation_functions = []

            for _, _, _, _, _, act_f, _ in d_sizes['conv_layers'][:-1]:
                activation_functions.append(act_f)

            activation_functions.append(d_sizes['output_activation'])
            #deconvolutional blocks

            self.d_blocks=[]

            output_sizes = {}

            for i in range(n_conv_blocks):
                output_sizes[i] = [
                    dims_W[i+1:i+1+len(d_sizes['deconvblock_layer_'+str(i)])],
                    dims_H[i+1:i+1+len(d_sizes['deconvblock_layer_'+str(i)])]
                ]
            
            #activation_functions = [tf.nn.relu]*(n_conv_blocks-1)+[g_sizes['output_activation']]

            for i in range(n_conv_blocks):
            
                name = 'd_block_layer_%s' %i
            
                #f = activation_functions[i]
                d_block = DeconvBlock(i, 
                                      mi,
                                      output_sizes[i], 
                                      d_sizes, 
                )
            
                self.d_blocks.append(d_block)
                mo, _, _, _, _, _, = d_sizes['deconvblock_layer_'+str(i)][-1]
                mi = mo
                    
            self.d_sizes=d_sizes
            
            return self.decode(Z)
        
    def decode(self, Z, reuse=None, is_training=True):

        print('Deconvolution')
        #dense layers

        output = Z
        print('Input for deconvolution shape', Z.get_shape())
        i=0
        for layer in self.d_dense_layers:
            i+=1
            output = layer.forward(output, reuse, is_training)
            #print('After dense layer %i' %i)
            #print('shape: ', output.get_shape())

        
        output = tf.reshape(
            output,
            
            [-1, self.d_dims_W[0], self.d_dims_H[0], self.d_sizes['projection']]
        
        )

        #print('Reshaped output after projection', output.get_shape())

        if self.d_sizes['bn_after_project']:
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
            #print('After deconvolutional block %i' %i)
            #print('shape: ', output.get_shape())
    

        print('Deconvoluted output shape', output.get_shape())
        return output

    def set_session(self, session):
        
        self.session = session
        
        for layer in self.e_conv_layers:
            layer.set_session(session)
        for layer in self.e_dense_layers:
            layer.set_session(session)
            
        for layer in self.d_dense_layers:
            layer.set_session(session) 
        for layer in self.d_conv_layers:
            layer.set_session(session)  
        
    def fit(self, X):

        SEED = 1

        costs = []
        N = len(X)
        n_batches = N // self.batch_size

        total_iters=0

        print('\n ****** \n')
        print('Training deep convolutional VAE with a total of ' +str(N)+' samples distributed in '+str(n_batches)+' batches, each of size '+str(self.batch_size)+'\n')
        print('The learning rate set is '+str(self.lr)+', and every ' +str(self.save_sample)+ ' epoch a generated sample will be saved to '+ self.path)
        print('\n ****** \n')

        for epoch in range(self.epochs):

            SEED = SEED + 1

            batches = unsupervised_random_mini_batches(X, self.batch_size, SEED)

            for X_batch in batches:

                feed_dict = {
                            self.X: X_batch, self.batch_sz: self.batch_size
                            }

                _, c = self.session.run(
                            (self.train_op, self.elbo),
                            feed_dict=feed_dict
                    )

                c /= self.batch_size
                
            total_iters +=1

            if total_iters % self.save_sample == 0:
                print("On iteration %d, cost: %.3f" %(total_iters, c))
                print('Saving a sample...')
                    
                    
                n = 8 #number of images per side
                x_values = np.linspace(-3,3,n)
                y_values = np.linspace(-3,3,n)

                flat_image = np.empty((28*n,28*n))

                Z=[]
                for i, x in enumerate(x_values):
                    for j, y in enumerate(y_values):
                        z=[x, y]
                        Z.append(z)
        
                samples = self.sample(Z, n*n)
                
                k = 0
                for i, x in enumerate(x_values):
                    for j, y in enumerate(y_values):  
                        x_recon = samples[k]
                        k+=1
        
                        x_recon=x_recon.reshape(28,28)
                        flat_image[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = x_recon
                
                plt.imshow(flat_image,cmap='gray')
                plt.show()
                plt.savefig(
                    self.path+'samples_at_iter_%d.png' % total_iters,
                )
                                
            plt.clf()
            plt.plot(costs, label='cost vs iteration')
            plt.show()
            plt.savefig(self.path+'cost vs iteration.png')

    def sample(self, Z, n):
        samples = self.session.run(
          self.prior_predictive_from_input_probs,
          feed_dict={self.Z_input: Z, self.batch_sz: n}
        )
        return samples

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
        return self.session.run((self.prior_predictive, self.prior_predictive_probs))

#class aeDNN: 

#class aeCNN: 

#check if works by creating some samples
#check rectangular images support
class DCGAN:
    
    def __init__(
        self,
        n_W, n_H, n_C,
        d_sizes,g_sizes,
        lr=LEARNING_RATE, beta1=BETA1,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        save_sample=SAVE_SAMPLE_PERIOD, path=PATH
        ):

        """

        Positional arguments:

            - width of (square) image
            - number of channels of input image
            - discriminator sizes

                a python dict of the kind
                    d_sizes = { 'conv_layers':[(n_c+1, kernel, stride, apply_batch_norm, weight initializer, act_f),
                                                   (,,,,),
                                                   ],
                                'dense_layers':[(n_o, apply_bn, weight_init, act_f)]
                                }
            - generator sizes

                a python dictionary of the kind

                    g_sizes = { 
                                'z':latent_space_dim,
                                'projection': int,
                                'bn_after_project':bool

                                'conv_layers':[(n_c+1, kernel, stride, apply_batch_norm, weight initializer, act_f),
                                                   (,,,,),
                                                   ],
                                'dense_layers':[(n_o, apply_bn, weight_init, act_f)]
                                'activation':function
                                }

        Keyword arguments:

            - lr = LEARNING_RATE (float32)
            - beta1 = ema parameter for adam opt (float32)
            - batch_size (int)
            - save_sample = after how many batches iterations to save a sample (int)
            - path = relative path for saving samples

        """

        self.n_H = n_H
        self.n_W = n_W
        self.n_C = n_C
        
        self.latent_dims = g_sizes['z']
        
        #input data
        
        self.X = tf.placeholder(
            tf.float32,
            shape=(None, 
                   n_W, n_H, n_C),
            name='X',
        )
        
        self.Z = tf.placeholder(
            tf.float32,
            shape=(None, 
                   self.latent_dims),
            name='Z'    
        )

        self.batch_sz = tf.placeholder(
            tf.int32, 
            shape=(), 
            name='batch_sz'
        )
        
        logits = self.build_discriminator(self.X, d_sizes)
        
        self.sample_images = self.build_generator(self.Z, g_sizes)
        
        # get sample logits
        with tf.variable_scope('discriminator') as scope:
            scope.reuse_variables()
            sample_logits = self.d_forward(self.sample_images, reuse=True)
                
        # get sample images for test time
        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()
            self.sample_images_test = self.g_forward(
                self.Z, reuse=True, is_training=False
            )
            
        #cost building
        
        #Discriminator cost
        self.d_cost_real = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits,
            labels=tf.ones_like(logits)
        )
        
        self.d_cost_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=sample_logits,
            labels=tf.zeros_like(sample_logits)
        )
        
        self.d_cost = tf.reduce_mean(self.d_cost_real) + tf.reduce_mean(self.d_cost_fake)
        
        #Generator cost
        self.g_cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=sample_logits,
                labels=tf.ones_like(sample_logits)
            )
        )
        
        #Measure accuracy of the discriminator

        real_predictions = tf.cast(logits>0,tf.float32)
        fake_predictions = tf.cast(sample_logits<0,tf.float32)
        
        num_predictions=2.0*BATCH_SIZE
        num_correct = tf.reduce_sum(real_predictions)+tf.reduce_sum(fake_predictions)
        
        self.d_accuracy = num_correct/num_predictions
        
        #optimizers
        self.d_params =[t for t in tf.trainable_variables() if t.name.startswith('d')]
        self.g_params =[t for t in tf.trainable_variables() if t.name.startswith('g')]
        
        self.d_train_op = tf.train.AdamOptimizer(
        
            lr,
            beta1=beta1,
        
        ).minimize(
            self.d_cost,
            var_list=self.d_params
        )
        
        self.g_train_op = tf.train.AdamOptimizer(
        
            LEARNING_RATE,
            beta1=BETA1,
        
        ).minimize(
            self.g_cost,
            var_list=self.g_params
        )

        self.batch_size=batch_size
        self.epochs=epochs
        self.save_sample=save_sample
        self.path=path
        self.lr = lr
        
    def build_discriminator(self, X, d_sizes):
        
        with tf.variable_scope('discriminator') as scope:
            
            #dimensions of input
            mi = self.n_C
            dim_H = self.n_H
            dim_W = self.n_W
            # dims=[dim]
            
            for key in d_sizes:
                if not 'block' in key:
                    print('Convolutional Network architecture detected')
                else:
                    print('Check network architecture')
                break
            
            self.d_convlayers =[]
            count=0

            #building discriminator convolutional layers
            
            for mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init in d_sizes['conv_layers']:
                
                # make up a name - used for get_variable
                name = "convlayer_%s" % count
                count += 1

                layer = ConvLayer(name, mi, mo, 
                                  filter_sz, stride, 
                                  apply_batch_norm, keep_prob,
                                  act_f, w_init)

                self.d_convlayers.append(layer)
                mi = mo
                dim_H = int(np.ceil(float(dim_H) / stride))
                dim_W = int(np.ceil(float(dim_W) / stride))
                
            mi = mi * dim_H * dim_W

            #building discriminator dense layers
            
            self.d_dense_layers = []
            for mo, apply_batch_norm, keep_prob, act_f, w_init in d_sizes['dense_layers']:
                
                name = 'd_dense_layer_%s' %count
                count +=1
                
                layer = DenseLayer(name, mi, mo,
                                  apply_batch_norm, keep_prob,
                                  act_f, w_init)
                mi = mo
                self.d_dense_layers.append(layer)
            
            #final logistic layer
            name = 'd_dense_layer_%s' %count

            self.d_final_layer = DenseLayer(name, mi, 1, 
                                            False, 1, 
                                            lambda x: x)
            
            return self.d_forward(X)
            
    def d_forward(self, X, reuse = None, is_training=True):

            print('Convolution')

            output = X
            print('Input for convolution shape ', X.get_shape())
            i=0
            for layer in self.d_convlayers:
                i+=1
                #print('Convolution_layer_%i' %i)
                #print('Input shape', output.get_shape())
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
                #print('After dense layer_%i' %i)
                #print('Shape', output.get_shape())
  
            logits = self.d_final_layer.forward(output, 
                                                reuse,
                                                is_training)
            print('Logits shape', logits.get_shape())
            return logits
        
    def build_generator(self, Z, g_sizes):
        
        with tf.variable_scope('generator') as scope:
            
            #dimensions of input
            dims_H =[self.n_H]
            dims_W =[self.n_W]

            dim_H = self.n_H
            dim_W = self.n_W

            mi = self.latent_dims
            
            #building generator dense layers
            self.g_dense_layers = []
            count = 0

            for mo, apply_batch_norm, keep_prob, act_f, w_init in g_sizes['dense_layers']:
                name = 'g_dense_layer_%s' %count
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

            
            mo = g_sizes['projection']*dims_H[0]*dims_W[0]
        
            name = 'g_dense_layer_%s' %count
            layer = DenseLayer(name, mi, mo, not g_sizes['bn_after_project'], 1)
            self.g_dense_layers.append(layer)
            
            
            mi = g_sizes['projection']
            self.g_convlayers=[]
            
            #num_relus = len(g_sizes['conv_layers']) - 1
            activation_functions = []

            for _, _, _, _, _, act_f, _, in g_sizes['conv_layers'][:-1]:
                activation_functions.append(act_f)

            activation_functions.append(g_sizes['output_activation'])
            
            for i in range(len(g_sizes['conv_layers'])):
                name = 'fs_convlayer_%s' %i
                
                mo, filter_sz, stride, apply_batch_norm, keep_prob, _, w_init = g_sizes['conv_layers'][i]
                f = activation_functions[i]
                
                layer = DeconvLayer(
                  name, mi, mo, [dims_H[i+1], dims_W[i+1]],
                  filter_sz, stride, apply_batch_norm, keep_prob,
                  f, w_init
                )

                self.g_convlayers.append(layer)
                mi = mo
                
            self.g_sizes=g_sizes
            
            return self.g_forward(Z)
        
    def g_forward(self, Z, reuse=None, is_training=True):

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
            
            [-1, self.g_dims_H[0], self.g_dims_W[0], self.g_sizes['projection']]
        
        )

        #print('Reshaped output after projection', output.get_shape())

        if self.g_sizes['bn_after_project']:
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
        for layer in self.g_convlayers:
            i+=1
            output = layer.forward(output, reuse, is_training)
            #print('After deconvolutional layer %i' %i)
            #print('shape: ', output.get_shape())


        print('Deconvoluted output shape', output.get_shape())
        return output
    
    def set_session(self, session):
        
        self.session = session
        
        for layer in self.d_convlayers:
            layer.set_session(session)
                
        for layer in self.d_dense_layers:
            layer.set_session(session)
        
        for layer in self.g_convlayers:
            layer.set_session(session)
                
        for layer in self.g_dense_layers:
            layer.set_session(session)
    
    def fit(self, X):
        
        d_costs = []
        g_costs = []

        N = len(X)
        n_batches = N // self.batch_size
    
        total_iters=0

        print('\n ****** \n')
        print('Training DCGAN with a total of ' +str(N)+' samples distributed in batches of size '+str(self.batch_size)+'\n')
        print('The learning rate set is '+str(self.lr)+', and every ' +str(self.save_sample)+ ' epoch a generated sample will be saved to '+ self.path)
        print('\n ****** \n')

        for i in range(self.epochs):
            
            print('Epoch:', i)
            np.random.shuffle(X)
            
            for j in range(n_batches):
                
                t0 = datetime.now()
                X_batch = X[j*self.batch_size:(j+1)*self.batch_size]
                
                Z = np.random.uniform(-1,1, size= (self.batch_size, self.latent_dims))
                
                _, d_cost, d_acc = self.session.run(
                    (self.d_train_op, self.d_cost, self.d_accuracy),
                    feed_dict={self.X: X_batch, self.Z:Z, self.batch_sz: self.batch_size},
                )
                
                d_costs.append(d_cost)
                
                #train the generator averaging two costs if the
                #discriminator learns too fast
                
                _, g_cost1 =  self.session.run(
                    (self.g_train_op, self.g_cost),
                    feed_dict={self.Z:Z, self.batch_sz:self.batch_size},
                )
                
                _, g_cost2 =  self.session.run(
                    (self.g_train_op, self.g_cost),
                    feed_dict={self.Z:Z, self.batch_sz:self.batch_size},
                )
        
                g_costs.append((g_cost1 + g_cost2)/2) # just use the avg            
            
                total_iters += 1
                if total_iters % self.save_sample ==0:
                    print("  batch: %d/%d  -  dt: %s - d_acc: %.2f" % (j+1, n_batches, datetime.now() - t0, d_acc))
                    print('Saving a sample...')
                    
                    Z = np.random.uniform(-1,1, size=(self.batch_size,self.latent_dims))
                    
                    samples = self.sample(Z)#shape is (64,D,D,color)
                    
                    w = self.n_W
                    h = self.n_H
                    samples = samples.reshape(self.batch_size, h, w)
                    
                    
                    for i in range(64):
                        plt.subplot(8,8,i+1)
                        plt.imshow(samples[i].reshape(h,w), cmap='gray')
                        plt.subplots_adjust(wspace=0.2,hspace=0.2)
                        plt.axis('off')
                    
                    fig = plt.gcf()
                    fig.set_size_inches(5,7)
                    plt.savefig(self.path+'/samples_at_iter_%d.png' % total_iters,dpi=300)


                    
            plt.clf()
            plt.plot(d_costs, label='discriminator cost')
            plt.plot(g_costs, label='generator cost')
            plt.legend()
            plt.savefig(self.path+'/cost vs iteration.png')
    
    def sample(self, Z):
        
        samples = self.session.run(
            self.sample_images_test, 
            feed_dict={self.Z:Z, self.batch_sz: self.batch_size})

        return samples 

    def get_sample(self, Z):
        
        one_sample = self.session.run(
            self.sample_images_test, 
            feed_dict={self.Z:Z, self.batch_sz: 1})

        return one_sample 

#add rectangular images support
#add residual support
#debug
class resDCGAN:
    
    def __init__(
        self,
        n_W, n_C,
        d_sizes,g_sizes,
        lr=LEARNING_RATE, beta1=BETA1,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        save_sample=SAVE_SAMPLE_PERIOD, path=PATH
        ):

        """

        Positional arguments:

            - width of (square) image
            - number of channels of input image
            - discriminator sizes

                a python dict of the kind
                    d_sizes = { 'convblocklayer_n':[(n_c+1, kernel, stride, apply_batch_norm, weight initializer),
                                                   (,,,,),
                                                   (,,,,),
                                                   ],
                                'convblock_shortcut_layer_n':[(,,,)],
                                'dense_layers':[(n_o, apply_bn, weight_init)]
                                }
            - generator sizes

                a python dictionary of the kind

                    g_sizes = { 
                                'z':latent_space_dim,
                                'projection': int,
                                'bn_after_project':bool

                                'deconvblocklayer_n':[(n_c+1, kernel, stride, apply_batch_norm, weight initializer),
                                                   (,,,,),
                                                   (,,,,),
                                                   ],
                                'deconvblock_shortcut_layer_n':[(,,,)],
                                'dense_layers':[(n_o, apply_bn, weight_init)]
                                'activation':function
                                }

        Keyword arguments:

            - lr = LEARNING_RATE (float32)
            - beta1 = ema parameter for adam opt (float32)
            - batch_size (int)
            - save_sample = after how many batches iterations to save a sample (int)
            - path = relative path for saving samples

        """

        self.n_W = n_W
        self.n_C = n_C
        
        self.latent_dims = g_sizes['z']
        
        #input data
        
        self.X = tf.placeholder(
            tf.float32,
            shape=(None, 
                   n_W, n_W, n_C),
            name='X',
        )
        
        self.Z = tf.placeholder(
            tf.float32,
            shape=(None, 
                   self.latent_dims),
            name='Z'    
        )

        self.batch_sz = tf.placeholder(
            tf.int32, 
            shape=(), 
            name='batch_sz'
        )
        
        logits = self.build_discriminator(self.X, d_sizes)
        
        self.sample_images = self.build_generator(self.Z, g_sizes)
        
        # get sample logits
        with tf.variable_scope('discriminator') as scope:
            scope.reuse_variables()
            sample_logits = self.d_forward(self.sample_images, reuse=True)
                
        # get sample images for test time
        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()
            self.sample_images_test = self.g_forward(
                self.Z, reuse=True, is_training=False
            )
            
        #cost building
        
        #Discriminator cost
        self.d_cost_real = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits,
            labels=tf.ones_like(logits)
        )
        
        self.d_cost_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=sample_logits,
            labels=tf.zeros_like(sample_logits)
        )
        
        self.d_cost = tf.reduce_mean(self.d_cost_real) + tf.reduce_mean(self.d_cost_fake)
        
        #Generator cost
        self.g_cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=sample_logits,
                labels=tf.ones_like(sample_logits)
            )
        )
        
        #Measure accuracy of the discriminator

        real_predictions = tf.cast(logits>0,tf.float32)
        fake_predictions = tf.cast(sample_logits<0,tf.float32)
        
        num_predictions=2.0*batch_size
        num_correct = tf.reduce_sum(real_predictions)+tf.reduce_sum(fake_predictions)
        
        self.d_accuracy = num_correct/num_predictions
        
        #optimizers
        self.d_params =[t for t in tf.trainable_variables() if t.name.startswith('d')]
        self.g_params =[t for t in tf.trainable_variables() if t.name.startswith('g')]
        
        self.d_train_op = tf.train.AdamOptimizer(
        
            lr,
            beta1=beta1,
        
        ).minimize(
            self.d_cost,
            var_list=self.d_params
        )
        
        self.g_train_op = tf.train.AdamOptimizer(
        
            lr,
            beta1=beta1,
        
        ).minimize(
            self.g_cost,
            var_list=self.g_params
        )
        self.batch_size=batch_size
        self.epochs=epochs
        self.save_sample=save_sample
        self.path=path
        self.lr = lr
        
    def build_discriminator(self, X, d_sizes):
        
        with tf.variable_scope('discriminator') as scope:
            
            #dimensions of input
            mi = self.n_C
            dim = self.n_W
            dims=[dim]
            
            for key in d_sizes:
                if 'block' in key:
                    print('Residual Network architecture detected')
                else:
                    print('Check network architecture')
                break
            
            self.d_blocks = []
            #count conv blocks
            n_conv_blocks = 0
            for key in d_sizes:
                if 'block' in key:
                    n_conv_blocks+=1
         
            #build conv blocks
            for i in range(n_conv_blocks//2):
            
                d_block = ConvBlock(i,
                               mi, d_sizes,
                               )
                self.d_blocks.append(d_block)
                mo, _, _, _, _, _, = d_sizes['convblock_layer_'+str(i)][-1]
                mi = mo
                dims.append(d_block.output_dim(dim))
        
            mi = mi * dims[-1] * dims[-1]
            count = i

            #build dense layers
            
            self.d_dense_layers = []
            for mo, apply_batch_norm, act_f, w_init in d_sizes['dense_layers']:
                
                name = 'd_dense_layer_%s' %count
                count +=1
                
                layer = DenseLayer(name,
                                  mi, mo,
                                  apply_batch_norm, 
                                  act_f, w_init)
                mi = mo
                self.d_dense_layers.append(layer)
            
            #final logistic layer
            name = 'd_dense_layer_%s' %count

            self.d_final_layer = DenseLayer(name, mi, 1, False, lambda x: x)
            
            return self.d_forward(X)
            
    def d_forward(self, X, reuse = None, is_training=True):

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
        
    def build_generator(self, Z, g_sizes):
        
        with tf.variable_scope('generator') as scope:
            
            #dimensions of input
            dims =[self.n_W]
            dim = self.n_W
            mi = self.latent_dims
            
            #dense layers
            self.g_dense_layers = []
            count = 0

            for mo, apply_batch_norm, act_f, w_init in g_sizes['dense_layers']:
                name = 'g_dense_layer_%s' %count
                count += 1
                
                layer = DenseLayer(
                    name,
                    mi, mo,
                    apply_batch_norm, 
                    f=act_f, w_init=w_init
                )
                self.g_dense_layers.append(layer)
                mi = mo

            #deconvolutional layers
            #count deconv blocks
            n_conv_blocks = 0
            for key in g_sizes:
                if 'block' in key:
                    n_conv_blocks+=1
                    
            #calculate height and width of image at every layer
            for i in range(n_conv_blocks//2):
                
                for _, _, stride, _, _, _, in reversed(g_sizes['deconvblock_layer_'+str(i)]):
                    dim = int(np.ceil(float(dim)/stride))
                    dims.append(dim)
    
            dims = list(reversed(dims))
            print('blocks dims: ', dims)
            self.g_dims = dims
            
            #final dense layer
            mo = g_sizes['projection']*dims[0]*dims[0]
            
            name = 'g_dense_layer_%s' %count
            layer = DenseLayer(name, mi, mo, not g_sizes['bn_after_project'])
            self.g_dense_layers.append(layer)
            
            #input channel number
            mi = g_sizes['projection']

            self.g_blocks=[]
            
            #activation_functions = [tf.nn.relu]*(n_conv_blocks-1)+[g_sizes['output_activation']]

            for i in range(n_conv_blocks//2):
            
                name = 'g_block_layer_%s' %i
            
                #f = activation_functions[i]
                g_block = DeconvBlock(i, 
                                      mi,
                                      dims[i+1:i+1+len(g_sizes['deconvblock_layer_'+str(i)])], 
                                      g_sizes, 
                )
            
                self.g_blocks.append(g_block)
                mo, _, _, _, _, _, = g_sizes['deconvblock_layer_'+str(i)][-1]
                mi = mo
                    
            self.g_sizes=g_sizes
            
            return self.g_forward(Z)
        
    def g_forward(self, Z, reuse=None, is_training=True):

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
            
            [-1, self.g_dims[0], self.g_dims[0], self.g_sizes['projection']]
        
        )

        #print('Reshaped output after projection', output.get_shape())

        if self.g_sizes['bn_after_project']:
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
    
    def set_session(self, session):
        
        self.session = session
        
        for block in self.d_blocks:
            block.set_session(session)
                
        for layer in self.d_dense_layers:
            layer.set_session(session)
        
        for block in self.g_blocks:
            block.set_session(session)
                
        for layer in self.g_dense_layers:
            layer.set_session(session)
    
    def fit(self, X):
        
        d_costs = []
        g_costs = []

        N = len(X)
        n_batches = N // self.batch_size
        
        print('\n ****** \n')
        print('Training resDCGAN with a total of ' +str(N)+' samples distributed in batches of size '+str(self.batch_size)+'\n')
        print('The learning rate set is '+str(self.lr)+', and every ' +str(self.save_sample)+ ' epoch a generated sample will be saved to '+ self.path)
        print('\n ****** \n')

        total_iters=0
        
        for i in range(self.epochs):
            
            print('Epoch:', i)
            np.random.shuffle(X)
            
            for j in range(n_batches):
                
                t0 = datetime.now()
                X_batch = X[j*self.batch_size:(j+1)*self.batch_size]
                
                Z = np.random.uniform(-1,1, size= (self.batch_size, self.latent_dims))
                
                _, d_cost, d_acc = self.session.run(
                    (self.d_train_op, self.d_cost, self.d_accuracy),
                    feed_dict={self.X: X_batch, self.Z:Z, self.batch_sz: self.batch_size},
                )
                
                d_costs.append(d_cost)
                
                #train the generator averaging two costs if the
                #discriminator learns too fast
                
                _, g_cost1 =  self.session.run(
                    (self.g_train_op, self.g_cost),
                    feed_dict={self.Z:Z, self.batch_sz:self.batch_size},
                )
                
                _, g_cost2 =  self.session.run(
                    (self.g_train_op, self.g_cost),
                    feed_dict={self.Z:Z, self.batch_sz:self.batch_size},
                )
        
                g_costs.append((g_cost1 + g_cost2)/2) # just use the avg            
            
                total_iters += 1
                if total_iters % self.save_sample ==0:
                    print("  batch: %d/%d  -  dt: %s - d_acc: %.2f" % (j+1, n_batches, datetime.now() - t0, d_acc))
                    print('Saving a sample...')
                    
                    Z = np.random.uniform(-1,1, size=(self.batch_size,self.latent_dims))
                    
                    samples = self.sample(Z)#shape is (64,D,D,color)
                    
                    d = self.n_W
                    samples = samples.reshape(self.batch_size, d, d)
                    
                    
                    for i in range(64):
                        plt.subplot(8,8,i+1)
                        plt.imshow(samples[i].reshape(d,d), cmap='gray')
                        plt.subplots_adjust(wspace=0.2,hspace=0.2)
                        plt.axis('off')
                    
                    fig = plt.gcf()
                    fig.set_size_inches(5, 5)
                    plt.savefig(self.path+'/samples_at_iter_%d.png' % total_iters,dpi=300)


                    
            plt.clf()
            plt.plot(d_costs, label='discriminator cost')
            plt.plot(g_costs, label='generator cost')
            plt.legend()
            plt.savefig(self.path+'/cost vs iteration.png')
    
    def sample(self, Z):
        
        samples = self.session.run(
            self.sample_images_test, 
            feed_dict={self.Z:Z, self.batch_sz: self.batch_size})

        return samples 

    def get_sample(self, Z):
        
        one_sample = self.session.run(
            self.sample_images_test, 
            feed_dict={self.Z:Z, self.batch_sz: 1})

        return one_sample 

# class cycleGAN:


        
