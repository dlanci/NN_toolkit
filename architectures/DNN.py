#NETWORK ARCHITECTURES

import numpy as np
import os 
import math

import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

from architectures.utils.NN_building_blocks import *

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

#untested yet
class DNN(object):

    """
    Builds densely connected deep neural network. Regularization implemented 
    with dropout, no regularization parameter implemented yet. Minimization through
    AdamOptimizer (adaptive learning rate)
    
    Constructor inputs:

        -Positional arguments:
            - dim: (int) input features
            - sizes: (dict) python dictionary containing the size of the
                    dense layers and the number of classes for classification
            
                sizes = {'dense_layer_n' :[mo, apply_batch_norm, keep_probability, act_f, w_init]
                        'n_classes': n_classes
                }

                mo: (int) size of layer n
                apply_batch_norm: (bool) apply batch norm at layer n
                keep_probability: (float32) probability of activation of output
                act_f: (function) activation function for layer n
                w_init: (tf initializer) random initializer for weights at layer n
                n_classes: number of classes

        -Keyword arguments

            -lr: (float32) learning rate arg for the AdamOptimizer
            -beta1: (float32) beta1 arg for the AdamOptimizer
            -batch_size: (int) size of each batch
            -epochs: (int) number of times the training has to be repeated over all the batches
            -save_sample: (int) after how many iterations of the training algorithm performs the evaluations in fit function
            -path: (str) path for saving the session checkpoint

    Class attributes:

        - X: (tf placeholder) input tensor of shape (batch_size, input features)
        - Y: (tf placeholder) label tensor of shape (batch_size, n_classes) (one_hot encoding)
        - Y_hat: (tf tensor) shape=(batch_size, n_classes) predicted class (one_hot)
        - loss: (tf scalar) reduced mean of cost computed with softmax cross entropy with logits
        - train_op: gradient descent algorithm with AdamOptimizer

    """
    def __init__(self,
        dim, sizes,
        lr=LEARNING_RATE, beta1=BETA1,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        save_sample=SAVE_SAMPLE_PERIOD, path=PATH, seed=SEED):

        self.seed=seed
        self.n_classes = sizes['n_classes']
        self.dim = dim

        self.sizes=sizes


        self.X = tf.placeholder(
            tf.float32,
            shape=(None, dim),
            name = 'X_data'
            )

        self.X_input = tf.placeholder(
            tf.float32,
            shape=(None, dim),
            name = 'X_input'
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

        self.Y_hat = self.build_NN(self.X, self.conv_sizes)


        cost = tf.nn.softmax_cross_entropy_with_logits(
                logits= self.Y_hat,
                labels= self.Y
            )

        self.loss = tf.reduce_mean(cost)

        self.train_op = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1
            ).minimize(self.loss
            )



        #convolve from input
        with tf.variable_scope('classification') as scope:
            scope.reuse_variables()
            self.Y_hat_from_test = self.convolve(
                self.X_input, reuse=True, is_training=False, keep_prob=1
            )

        self.accuracy = evaluation(self.Y_hat_from_test, self.Y)

        #saving for later
        self.lr = lr
        self.batch_size=batch_size
        self.epochs = epochs
        self.path = path
        self.save_sample = save_sample
        

    def build_NN(self, X, sizes):

        with tf.variable_scope('classification') as scope:

            mi = self.dim
            self.dense_layers = []
            count = 0

            for mo, apply_batch_norm, keep_prob, act_f, w_init in sizes['dense_layers']:

                name = 'dense_layer_{0}'.format(count)
                count += 1

                layer = DenseLayer(name,mi, mo,
                                  apply_batch_norm, keep_prob,
                                  act_f, w_init)
                mi = mo
                self.dense_layers.append(layer)

            #readout layer
            readout_layer =  DenseLayer('readout_layer', 
                                        mi, self.n_classes,
                                        False, 1, tf.nn.softmax, 
                                        tf.random_uniform_initializer(seed=self.seed))

            self.dense_layers.append(readout_layer)

            return self.propagate(X)

    def propagate(self, X, reuse=None, is_training=True):

        print('Propagation')
        print('Input for propagation', X.get_shape())

        output = X

        for layer in self.dense_layers:
            output.layer.forward(output, reuse, is_training)

        print('Logits shape', output.get_shape())
        return output

    def set_session(self, session):

        for layer in self.dense_layers:

            layer.set_session(session)

    def fit(self, X_train, Y_train, X_test, Y_test):

        """
        Function is called if the flag is_training is set on TRAIN. If a model already is present
        continues training the one already present, otherwise initialises all params from scratch.
        
        Performs the training over all the epochs, at when the number of epochs of training
        is a multiple of save_sample prints out training cost, train and test accuracies
        
        Plots a plot of the cost versus epoch. 

        Positional arguments:

            - X_train: (ndarray) size=(train set size, input features) training sample set
            - X_test: (ndarray) size=(test set size, input features) test sample set

            - Y_train: (ndarray) size=(train set size, input features) training labels set
            - Y_test: (ndarray) size=(test set size, input features) test labels set



        """
        seed = self.seed
        

        N = X_train.shape[0]
        test_size = X_test.shape[0]

        n_batches = N // self.batch_size

        print('\n ****** \n')
        print('Training CNN for '+str(self.epochs)+' epochs with a total of ' +str(N)+ ' samples\ndistributed in ' +str(n_batches)+ ' batches of size '+str(self.batch_size)+'\n')
        print('The learning rate set is '+str(self.lr))
        print('\n ****** \n')

        costs = []
        for epoch in range(self.epochs):

            train_acc = 0
            test_acc =0
            train_accuracies=[]
            test_accuracies=[]

            seed += 1

            train_batches = supervised_random_mini_batches(X_train, Y_train, self.batch_size, seed)
            test_batches = supervised_random_mini_batches(X_test, Y_test, self.batch_size, seed)
            
            for train_batch in train_batches:

                (X_train, Y_train) = train_batch

                feed_dict = {

                            self.X: X_train,
                            self.Y: Y_train,
                            self.batch_sz: self.batch_size,
                            
                            }

                _, c = self.session.run(
                            (self.train_op, self.loss),
                            feed_dict=feed_dict
                    )

                train_acc = self.session.run(
                    self.accuracy, feed_dict={self.X_input:X_train, self.Y:Y_train}

                    )


                c /= self.batch_size
                costs.append(c)
                train_accuracies.append(train_acc)

            train_acc = np.array(train_accuracies).mean()

            #model evaluation
            if epoch % self.save_sample ==0:

                for test_batch in test_batches:

                    (X_test_batch, Y_test_batch) = test_batch


                    feed_dict={        
                                self.X_input: X_test_batch,
                                self.Y: Y_test_batch,

                                }

                    test_acc = self.session.run(
                            self.accuracy,
                            feed_dict=feed_dict
                                   
                        )

                    test_accuracies.append(test_acc)

                test_acc = np.array(test_accuracies).mean()

                print('Evaluating performance on train/test sets')
                print('At iteration {0}, train cost: {1:.4g}, train accuracy {2:.4g}'.format(epoch, c, train_acc))
                print('test accuracy {0:.4g}'.format(test_acc))
          

        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iteration')
        plt.title('learning rate=' + str(self.lr))
        plt.show()
        
        print('Parameters trained')

    
    def predicted_Y_hat(self, X):


        pred = tf.nn.softmax(self.Y_hat_from_test)
        output = self.session.run(
            pred, 
            feed_dict={self.X_input:X}
            )
        return output