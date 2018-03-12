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


#tested on mnist
class DAE(object):

    """
    Builds densely connected deep autoencoder. Regularization implemented 
    with dropout, no regularization parameter implemented yet. Minimization through
    AdamOptimizer (adaptive learning rate).

    The minimized loss function is the reduced sum of the sigmoid cross entropy with logis
    over all the barch samples.

    Constructor inputs:

        -Positional arguments:
            - dim: (int) input features 
            - e_sizes: (dict)
            - d_sizes: (dict)
        
        -Keyword arguments

            - an_id: (int) number useful for stacked ae
            - lr: (float32) learning rate arg for the AdamOptimizer
            - beta1: (float32) beta1 arg for the AdamOptimizer
            - batch_size: (int) size of each batch
            - epochs: (int) number of times the training has to be repeated over all the batches
            - save_sample: (int) after how many iterations of the training algorithm performs the evaluations in fit function
            - path: (str) path for saving the session checkpoint

    Class attributes:
        
        - X: (tf placeholder) input tensor of shape (batch_size, input features)
        - Y: (tf placeholder) label tensor of shape (batch_size, n_classes) (one_hot encoding)
        - Y_hat: (tf tensor) shape=(batch_size, n_classes) predicted class (one_hot)
        - loss: (tf scalar) reduced mean of cost computed with softmax cross entropy with logits
        - train_op: gradient descent algorithm with AdamOptimizer

    

    Class methods:
        - get_sample:

    """


    def __init__(
        self, dim, e_sizes, d_sizes, an_id=0, 
        lr=LEARNING_RATE, beta1=BETA1,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        save_sample=SAVE_SAMPLE_PERIOD, path=PATH, seed=SEED, img_height=None, img_width=None
        ):


        self.dim = dim
        self.an_id = an_id
        self.latent_dims=e_sizes['z']
        self.e_sizes=e_sizes
        self.d_sizes=d_sizes

        self.seed = seed

        self.img_height=img_height
        self.img_width=img_width

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
            learning_rate=lr,
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
                self.e_last_act_f, w_init=tf.random_normal_initializer(stddev=0.02, seed=rnd_seed)
                )
            self.e_layers.append(last_enc_layer)

            return self.encode(X)

    def encode(self, X, reuse=None, is_training=True):

        Z=X
        for layer in self.e_layers:
            Z=layer.forward(Z, reuse, is_training)
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
                f=lambda x:x, w_init=tf.random_normal_initializer(stddev=0.02, seed=rnd_seed)
                )

            self.d_layers.append(last_dec_layer)


            return self.decode(Z)

    def decode(self, Z, reuse=None, is_training=True):

        X=Z

        for layer in self.d_layers:
             X = layer.forward(X, reuse, is_training)
        return X

    # def get_logits(self, X):

    def set_session(self, session):

        self.session = session

        for layer in self.d_layers:
            layer.set_session(self.session)

        for layer in self.e_layers:
            layer.set_session(self.session)

    def fit(self, X):

        """
        Function is called if the flag is_training is set on TRAIN. If a model already is present
        continues training the one already present, otherwise initialises all params from scratch.
        
        Performs the training over all the epochs, at when the number of epochs of training
        is a multiple of save_sample, prints the cost at that epoch. When training has gone through all
        the epochs, plots a plot of the cost versus epoch. 

        Positional arguments:

            - X_train: (ndarray) size=(train set size, input features) training sample set
            
        """


        seed = self.seed

        costs = []
        N = len(X)
        n_batches = N // self.batch_size

        total_iters=0

        print('\n ****** \n')
        print('Training deep AE with a total of ' +str(N)+' samples distributed in batches of size '+str(self.batch_size)+'\n')
        print('The learning rate set is '+str(self.lr)+', and every ' +str(self.save_sample)+ ' epochs the training cost will be printed')
        print('\n ****** \n')

        for epoch in range(self.epochs):

            # print('Epoch: {0}'.format(epoch))

            seed += 1

            batches = unsupervised_random_mini_batches(X, self.batch_size, seed)

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

            if epoch % self.save_sample == 0:


                print('At epoch %d, cost: %f' %(epoch, c))

        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iteration')
        plt.title('learning rate=' + str(self.lr))
        plt.show()
        
        print('Parameters trained')

    def get_sample(self, X):
        """
        Input X

        takes an X, encodes and then decodes it reproducing a X_hat

        Outputs X_hat
        """
        return self.session.run(
            self.X_decoded, feed_dict={self.X_input:X, self.batch_sz:1}
            )