#NETWORK ARCHITECTURES

import numpy as np
import os 
import math

import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

from NN_building_blocks import *
from NN_gen_building_blocks import *

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

#CLASSIFICATION MODELS


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

#tested on mnist
class CNN(object):
    
    """
    Builds convolutional neural network. Regularization implemented 
    with dropout, no regularization parameter implemented yet. Minimization through
    AdamOptimizer (adaptive learning rate). Supports convolution, max_pooling and avg_pooling
    
    Constructor inputs:

        -Positional arguments:
            - dims of input image: (n_H (rows))*(n_W (columns))*(n_C (input channels))

            - sizes: (dict) python dictionary containing the size of the
                    convolutional layers and the number of classes for classification
            
                sizes = {'conv_layer_n' :[(mo, filter_sz, stride, apply_batch_norm, keep_probability, act_f, w_init)]
                        'maxpool_layer_n':[(filter_sz, stride, keep_prob)]
                        'avgpool_layer_n':[(filter_sz, stride, keep_prob)]
                        
                        'n_classes': n_classes

                }
                
                convolution and pooling layers can be in any order, the last key has to be 'n_classes'

                mo: (int) number of output channels after convolution
                filter_sz: (int) size of the kernel
                stride: (int) stride displacement
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
    def __init__(

        self, n_H, n_W, n_C, sizes,
        lr=LEARNING_RATE, beta1=BETA1,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        save_sample=SAVE_SAMPLE_PERIOD, path=PATH, seed = SEED
        ):

        self.seed = seed

        self.n_classes = sizes['n_classes']
        self.n_H = n_H
        self.n_W = n_W
        self.n_C = n_C

        self.conv_sizes = sizes
        
        self.X = tf.placeholder(
            tf.float32,
            shape=(None, n_H, n_W, n_C),
            name = 'X_data'
            )

        self.X_input = tf.placeholder(
            tf.float32,
            shape=(None, n_H, n_W, n_C),
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

        self.Y_hat = self.build_CNN(self.X, self.conv_sizes)

        #add regularization
        #reg = 0


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
        with tf.variable_scope('convolutional') as scope:
            scope.reuse_variables()
            self.Y_hat_from_test = self.convolve(
                self.X_input, reuse=True, is_training=False
            )

        self.accuracy = evaluation(self.Y_hat_from_test, self.Y)

        #saving for later
        self.lr = lr
        self.batch_size=batch_size
        self.epochs = epochs
        self.path = path
        self.save_sample = save_sample
        

    def build_CNN(self, X, conv_sizes):

        with tf.variable_scope('convolutional') as scope:
            #keep track of dims for dense layers
            mi = self.n_C
            dim_W = self.n_W
            dim_H = self.n_H

            self.conv_layers = []

            count = 0
            
            n = len(conv_sizes)-1 # count the number of layers leaving out n_classes key
            
            for key in conv_sizes:
                if not 'block' in key:
                    print('Convolutional network architecture detected')
                else:
                    print('Check network architecture')
                break

            #convolutional layers
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

            #dense layers
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

            #readout layer
            readout_layer =  DenseLayer('readout_layer', 
                                        mi, self.n_classes,
                                        False, 1, tf.nn.softmax, 
                                        tf.random_uniform_initializer(seed=self.seed))

            self.dense_layers.append(readout_layer)

            return self.convolve(X)

    def convolve(self, X, reuse=None, is_training=True):
        
        print('Convolution')
        print('Input for convolution', X.get_shape())
        
        output=X
        i=0
        for layer in self.conv_layers:
            i+=1
            output = layer.forward(output, reuse, is_training)
            #print('After convolution', i)
            #print(output.get_shape())
            

        #print('After convolution shape', output.get_shape())

        output = tf.contrib.layers.flatten(output)

        #print('After flatten shape', output.get_shape())
        for layer in self.dense_layers:

            i+=1
            output = layer.forward(output, reuse, is_training)
            #print('After dense layer {0}, shape {1}'.format(i, output.get_shape()))

        print('Logits shape', output.get_shape())

        return output

    def set_session(self, session):

        self.session=session

        for layer in self.conv_layers:
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

            seed += 1

            train_batches = supervised_random_mini_batches(X_train, Y_train, self.batch_size, seed)
            test_batches = supervised_random_mini_batches(X_test, Y_test, self.batch_size, seed)
            train_acc = 0
            test_acc =0
            train_accuracies=[]
            test_accuracies=[]

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

                train_accuracies.append(train_acc)
                costs.append(c)

            train_acc = np.array(train_accuracies).mean()
            #model evaluation
            if epoch % self.save_sample ==0:
                                
                
                for test_batch in test_batches:

                    (X_test_batch, Y_test_batch) = test_batch
                    #print(X_test_batch.sum(),Y_test_batch.sum())
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
                print('At epoch {0}, train cost: {1:.4g}, train accuracy {2:.4g}'.format(epoch, c, train_acc))
                print('test accuracy {0:.4g}'.format(test_acc))
          

        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iteration')
        plt.title('learning rate=' + str(self.lr))
        plt.show()
        
        print('Parameters trained')

        #get samples at test time
    
    def predicted_Y_hat(self, X):
        pred = tf.nn.softmax(self.Y_hat_from_test)
        output = self.session.run(
            pred, 
            feed_dict={self.X_input:X}
            )
        return output

#tested on mnist
class resCNN(object):
    
    """
    Builds residual convolutional neural network. Regularization implemented 
    with dropout, no regularization parameter implemented yet. Minimization through
    AdamOptimizer (adaptive learning rate). Supports convolution, max_pooling and avg_pooling
    
    Constructor inputs:

        -Positional arguments:
            - dims of input image: (n_W (rows))*(n_H (colums))*(n_C (input channels))

            - sizes: (dict) python dictionary containing the size of the
                    convolutional blocks and the number of classes for classification
            
                sizes = {'convblock_layer_n' :[(mo, filter_sz, stride, apply_batch_norm, keep_probability, act_f, w_init),
                                                (),
                                                (),]
                        'maxpool_layer_n':[(filter_sz, stride, keep_prob)]
                        'avgpool_layer_n':[(filter_sz, stride, keep_prob)]
                        
                        'n_classes': n_classes

                }
                
                convolution blocks and pooling layers can be in any order, the last key has to be 'n_classes'

                mo: (int) number of output channels after convolution
                filter_sz: (int) size of the kernel
                stride: (int) stride displacement
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
    def __init__(

        self, n_H, n_W, n_C, sizes,
        lr=LEARNING_RATE, beta1=BETA1,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        save_sample=SAVE_SAMPLE_PERIOD, path=PATH, seed=SEED
        ):

        self.n_classes = sizes['n_classes']
        self.n_H = n_H
        self.n_W = n_W
        self.n_C = n_C

        self.conv_sizes = sizes
        self.seed = seed

        self.keep_prob = tf.placeholder(
            tf.float32
            )

        self.X = tf.placeholder(
            tf.float32,
            shape=(None, n_H, n_W, n_C),
            name = 'X'
            )

        self.X_input = tf.placeholder(
            tf.float32,
            shape=(None, n_H, n_W, n_C),
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
            learning_rate=lr,
            beta1=beta1
            ).minimize(self.loss
            )

        #convolve from input
        with tf.variable_scope('convolutional') as scope:
            scope.reuse_variables()
            self.Y_hat_from_test = self.convolve(
                self.X_input, reuse=True, is_training=False,
            )

        self.accuracy = evaluation(self.Y_hat_from_test, self.Y)

        #saving for later
        self.lr = lr
        self.batch_size=batch_size
        self.epochs = epochs
        self.path = path
        self.save_sample = save_sample

    def build_resCNN(self, X, conv_sizes):
        
        with tf.variable_scope('convolutional') as scope:
            
            #dimensions of input
            mi = self.n_C

            dim_W = self.n_W
            dim_H = self.n_H

            
            for key in conv_sizes:
                if 'block' in key:
                    print('Residual Network architecture detected')
                break
            
            self.conv_blocks = []
            #count conv blocks

            steps=0
            for key in conv_sizes:
                if 'conv' in key:
                    if not 'shortcut' in key:
                        steps+=1
                if 'pool' in key:
                    steps+=1


            #build convblocks
            block_n=0
            layer_n=0
            pool_n=0
            
            for key in conv_sizes:
                
                if 'block' and 'shortcut' in key:

                    conv_block = ConvBlock(block_n,
                               mi, conv_sizes,
                               )
                    self.conv_blocks.append(conv_block)
                    
                    mo, _, _, _, _, _, _, = conv_sizes['convblock_layer_'+str(block_n)][-1]
                    mi = mo
                    dim_H = conv_block.output_dim(dim_H)
                    dim_W = conv_block.output_dim(dim_W)
                    block_n+=1

                if 'conv_layer' in key:

                    name = 'conv_layer_{0}'.format(layer_n)

                    mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init = conv_sizes[key][0]


                    conv_layer = ConvLayer(name, mi, mo,
                                           filter_sz, stride,
                                           apply_batch_norm, keep_prob,
                                           act_f, w_init
                        )

                    self.conv_blocks.append(conv_layer)

                    mi = mo
                    dim_W = int(np.ceil(float(dim_W) / stride))
                    dim_H = int(np.ceil(float(dim_H) / stride))
                    layer_n+=1  

                if 'pool' in key:
                    pool_n+=1

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
                    
            assert steps == pool_n + block_n + layer_n, 'Check conv_sizes keys'
            count = steps
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
                                        tf.random_uniform_initializer(seed=rnd_seed))

            self.dense_layers.append(readout_layer)


            return self.convolve(X)
            
    def convolve(self, X, reuse = None, is_training=True):

        print('Convolution')
        print('Input for convolution shape ', X.get_shape())

        output = X

        i=0
        for block in self.conv_blocks:
            i+=1
            # print('Convolution_block_%i' %i)
            # print('Input shape', output.get_shape())
            output = block.forward(output,
                                     reuse,
                                     is_training)
        
        
        output = tf.contrib.layers.flatten(output)
        # print('After flatten shape', output.get_shape())

        i=0
        for layer in self.dense_layers:
            i+=1
            # print('Dense weights %i' %i)
            output = layer.forward(output,
                                   reuse,
                                   is_training)

            # print('After dense layer_%i' %i)
            # print('Shape', output.get_shape())

        print('Logits shape', output.get_shape())
        return output

    def set_session(self, session):

        self.session=session

        for layer in self.conv_blocks:
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
        print('Training residual CNN for '+str(self.epochs)+' epochs with a total of ' +str(N)+ ' samples\ndistributed in ' +str(n_batches)+ ' batches of size '+str(self.batch_size)+'\n')
        print('The learning rate set is '+str(self.lr))
        print('\n ****** \n')

        costs = []
        for epoch in range(self.epochs):

            seed = seed + 1

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
                    
                print('Evaluating performance on train/test sets')
                print('At iteration {0}, train cost: {1:.4g}, train accuracy {2:.4g}'.format(epoch, c, train_acc))
                print('test accuracy {0:.4g}'.format(test_acc))
          

        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iteration')
        plt.title('learning rate=' + str(self.lr))
        plt.show()
        
        print('Parameters trained')

        #get samples at test time
    
    def predicted_Y_hat(self, X):
        pred = tf.nn.softmax(self.Y_hat_from_test)
        output = self.session.run(
            pred, 
            feed_dict={self.X_input:X}
            )
        return output

#GENERATIVE MODELS

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

#tested on mnist
class DVAE(object):

	"""
	Builds densely connected deep variational autoencoder. Regularization implemented 
	with dropout, no regularization parameter implemented yet. Minimization through
	AdamOptimizer (adaptive learning rate).

	The minimized loss function is the elbo function (explain further)

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
	    - posterior_predictive_sample:
	    - prior_predictive_sample_with_probs:

	"""

	def __init__(
		self, dim, e_sizes, d_sizes, an_id=0, 
		lr=LEARNING_RATE, beta1=BETA1,
		batch_size=BATCH_SIZE, epochs=EPOCHS,
		save_sample=SAVE_SAMPLE_PERIOD, path=PATH, seed=SEED, img_height=None, img_width=None
		):
		"""
		Positional args


		Keyword args

		"""
		self.dim = dim
		self.e_sizes=e_sizes
		self.d_sizes=d_sizes
		self.img_height=img_height
		self.img_width=img_width
		self.seed = seed

		self.latent_dims=e_sizes['z']
		#self.d_last_act_f = d_sizes['last_act_f']

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

		self.E = denseEncoder(self.X, e_sizes, 'A')

		with tf.variable_scope('encoder_A') as scope:

		    self.Z = self.E.encode(self.X)

		self.D = denseDecoder(self.Z, self.latent_dims, dim, d_sizes, 'A')

		with tf.variable_scope('decoder_A') as scope:

		    logits = self.D.decode(self.Z)


		self.X_hat_distribution = Bernoulli(logits=logits)

		# posterior predictive
		# take samples from X_hat

		with tf.variable_scope('encoder_A') as scope:
		    scope.reuse_variables
		    self.Z_dist = self.E.encode(
		        self.X, reuse=True, is_training=False,
		    )#self.X or something on purpose?                                            
		with tf.variable_scope('decoder_A') as scope:
		    scope.reuse_variables()
		    sample_logits = self.D.decode(
		        self.Z_dist, reuse=True, is_training=False,
		    )


		self.posterior_predictive_dist = Bernoulli(logits=sample_logits)
		self.posterior_predictive = self.posterior_predictive_dist.sample(seed=self.seed)
		self.posterior_predictive_probs = tf.nn.sigmoid(sample_logits)

		# prior predictive 
		# take sample from a Z ~ N(0, 1)
		# and put it through the decoder

		standard_normal = Normal(
		  loc=np.zeros(self.latent_dims, dtype=np.float32),
		  scale=np.ones(self.latent_dims, dtype=np.float32)
		)

		Z_std = standard_normal.sample(1, seed=self.seed)

		with tf.variable_scope('decoder_A') as scope:
		    scope.reuse_variables()
		    logits_from_prob = self.D.decode(
		        Z_std, reuse=True, is_training=False,
		    )

		prior_predictive_dist = Bernoulli(logits=logits_from_prob)
		self.prior_predictive = prior_predictive_dist.sample()
		self.prior_predictive_probs = tf.nn.sigmoid(logits_from_prob)


		#cost
		kl = tf.reduce_sum(
		    tf.contrib.distributions.kl_divergence(
		        self.Z.distribution,
		        standard_normal),
		    1
		)

		# equivalent
		# expected_log_likelihood = -tf.nn.sigmoid_cross_entropy_with_logits(
		#   labels=self.X,
		#   logits=posterior_predictive_logits
		# )
		# expected_log_likelihood = tf.reduce_sum(expected_log_likelihood, 1)

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

	def set_session(self, session):

		self.session = session

		for layer in self.D.d_layers:
			layer.set_session(self.session)

		for layer in self.E.e_layers:
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


		print('\n ****** \n')
		print('Training deep VAE with a total of ' +str(N)+' samples distributed in batches of size '+str(self.batch_size)+'\n')
		print('The learning rate set is '+str(self.lr)+', and every ' +str(self.save_sample)+ ' iterations a generated sample will be saved to '+ self.path)
		print('\n ****** \n')
		total_iters=0

		for epoch in range(self.epochs):

			t0 = datetime.now()
			print('Epoch: {0}'.format(epoch))

			seed+=1

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

				total_iters += 1
                
				if total_iters % self.save_sample ==0:
					print("At iteration: %d  -  dt: %s - cost: %.2f" % (total_iters, datetime.now() - t0, c))
					print('Saving a sample...')

					probs = []
					for i in range(64):
						probs.append(self.prior_predictive_sample())
						self.seed+=1

					for i in range(64):
						plt.subplot(8,8,i+1)
						plt.imshow(probs[i].reshape(self.img_height,self.img_width), cmap='gray')
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

	def posterior_predictive_sample(self, X):
		# returns a sample from p(x_new | X)
		return self.session.run(self.posterior_predictive_probs, feed_dict={self.X: X, self.batch_sz:self.batch_size})

	def prior_predictive_sample(self):
		# returns a sample from p(x_new | z), z ~ N(0, 1)
		return self.session.run(self.prior_predictive_probs)

#tested on mnist
class DCVAE(object):

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
        self.Z=self.build_encoder(self.X, self.e_sizes)


        logits = self.build_decoder(self.Z, self.d_sizes)


        self.X_hat_distribution = Bernoulli(logits=logits)

        # posterior predictive
        # take samples from X_hat
        
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
        self.posterior_predictive = self.posterior_predictive_dist.sample(seed=self.seed)
        self.posterior_predictive_probs = tf.nn.sigmoid(sample_logits)

        # prior predictive 
        # take sample from a Z ~ N(0, 1)
        # and put it through the decoder

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


        #cost
        kl = tf.reduce_sum(
            tf.contrib.distributions.kl_divergence(
                self.Z.distribution,
                standard_normal),
            1
        )

        # equivalent
        # expected_log_likelihood = -tf.nn.sigmoid_cross_entropy_with_logits(
        #   labels=self.X,
        #   logits=posterior_predictive_logits
        # )
        # expected_log_likelihood = tf.reduce_sum(expected_log_likelihood, 1)

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
            last_enc_layer = DenseLayer(name, mi, 2*self.latent_dims, False, 1,
             f=lambda x: x, w_init=e_sizes['last_layer_weight_init'])
            
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
            
            for i in range(len(d_sizes['conv_layers'])):

                name = 'fs_convlayer_%s' %i
                
                mo, filter_sz, stride, apply_batch_norm, keep_prob, act_f, w_init = d_sizes['conv_layers'][i]
                
                layer = DeconvLayer(
                  name, mi, mo, [dims_H[i+1], dims_W[i+1]],
                  filter_sz, stride,
                  apply_batch_norm, keep_prob, 
                  act_f, w_init
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
            [-1, self.d_dims_H[0],self.d_dims_W[0],self.d_sizes['projection']]
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

        print('\n ****** \n')
        print('Training deep convolutional VAE with a total of ' +str(N)+' samples distributed in batches of size '+str(self.batch_size)+'\n')
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
                        
                    probs = [self.prior_predictive_sample()  for i in range(64)]  
                    
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

    def sample(self, Z, n):
        samples = self.session.run(
          self.prior_predictive_from_input_probs,
          feed_dict={self.Z_input: Z, self.batch_sz: n}
        )
        return samples

    def posterior_predictive_sample(self, X):
        # returns a sample from p(x_new | X)
        return self.session.run(self.posterior_predictive_probs, feed_dict={self.X: X, self.batch_sz:self.batch_size})

    def prior_predictive_sample(self):
        # returns a sample from p(x_new | z), z ~ N(0, 1)
        return self.session.run(self.prior_predictive_probs)

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
             f=lambda x: x, w_init=tf.random_normal_initializer())
            
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
            mo = d_sizes['projection']*dims_H[0]*dims_W[0]
            name = 'd_dense_layer_%s' %count

            layer = DenseLayer(name, mi, mo, not d_sizes['bn_after_project'], 1)
            self.d_dense_layers.append(layer)

            #deconvolution input channel number
            mi = d_sizes['projection']

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

            
            return self.decode(Z)
        
    def decode(self, Z, reuse=None, is_training=True):

        output = Z

        i=0
        for layer in self.d_dense_layers:
            i+=1
            output = layer.forward(output, reuse, is_training)


        
        output = tf.reshape(
            output,
            
            [-1, self.d_dims_H[0], self.d_dims_W[0], self.d_sizes['projection']]
        
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

#tested on mnist
class DCGAN(object):
    
    def __init__(
        self,
        n_H, n_W, n_C,
        d_sizes,g_sizes,
        lr_g=LEARNING_RATE_G, lr_d=LEARNING_RATE_D, beta1=BETA1,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        save_sample=SAVE_SAMPLE_PERIOD, path=PATH, seed=SEED,
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
        self.seed=seed
        self.latent_dims = g_sizes['z']
        
        #input data
        
        self.X = tf.placeholder(
            tf.float32,
            shape=(None, 
                   n_H, n_W, n_C),
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

        D = Discriminator(self.X, d_sizes, 'A')
        
        with tf.variable_scope('discriminator_A') as scope:
            
            logits = D.d_forward(self.X)

        G = Generator(self.Z, self.n_H, self.n_W, g_sizes, 'A')

        with tf.variable_scope('generator_A') as scope:

            self.sample_images = G.g_forward(self.Z)
        
        # get sample logits
        with tf.variable_scope('discriminator_A') as scope:
            scope.reuse_variables()
            sample_logits = D.d_forward(self.sample_images, reuse=True)
                
        # get sample images for test time
        with tf.variable_scope('generator_A') as scope:
            scope.reuse_variables()
            self.sample_images_test = G.g_forward(
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
            learning_rate=lr_d,
            beta1=beta1,
        ).minimize(
            self.d_cost,
            var_list=self.d_params
        )
        
        self.g_train_op = tf.train.AdamOptimizer(
            learning_rate=lr_g,
            beta1=beta1,
        ).minimize(
            self.g_cost,
            var_list=self.g_params
        )

        self.batch_size=batch_size
        self.epochs=epochs
        self.save_sample=save_sample
        self.path=path
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.D=D
        self.G=G

                
    def set_session(self, session):
        
        self.session = session
        
        for layer in self.D.d_conv_layers:
            layer.set_session(session)
                
        for layer in self.D.d_dense_layers:
            layer.set_session(session)
        
        for layer in self.G.g_conv_layers:
            layer.set_session(session)
                
        for layer in self.G.g_dense_layers:
            layer.set_session(session)
    
    def fit(self, X):

        seed = self.seed
        d_costs = []
        g_costs = []

        N = len(X)
        n_batches = N // self.batch_size
    
        total_iters=0

        print('\n ****** \n')
        print('Training DCGAN with a total of ' +str(N)+' samples distributed in batches of size '+str(self.batch_size)+'\n')
        print('The learning rate set for the generator is '+str(self.lr_g)+' while for the discriminator is '+str(self.lr_d)+', and every ' +str(self.save_sample)+ ' epoch a generated sample will be saved to '+ self.path)
        print('\n ****** \n')

        for epoch in range(self.epochs):

            seed +=1

            print('Epoch:', epoch)
            
            batches = unsupervised_random_mini_batches(X, self.batch_size, seed)

            for X_batch in batches:
                
                t0 = datetime.now()
                
                np.random.seed(seed)
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
                    print("At iter: %d  -  dt: %s - d_acc: %.4f" % (total_iters, datetime.now() - t0, d_acc))
                    print('Saving a sample...')
                    
                    np.random.seed(seed)
                    Z = np.random.uniform(-1,1, size=(64,self.latent_dims))
                    
                    samples = self.sample(Z)#shape is (64,D,D,color)
                    
                    w = self.n_W
                    h = self.n_H
                    samples = samples.reshape(64, h, w)
                    
                    
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

#tested on mnist (use greater lr on generator than on discriminator)
class resDCGAN(object):
    
    def __init__(
        self, 
        n_H, n_W, n_C,
        d_sizes,g_sizes,
        lr_g=LEARNING_RATE_G, lr_d=LEARNING_RATE_D, beta1=BETA1,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        save_sample=SAVE_SAMPLE_PERIOD, path=PATH, seed=SEED
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

        self.seed = seed
        self.n_W = n_W
        self.n_H = n_H
        self.n_C = n_C
        
        self.latent_dims = g_sizes['z']
        
        #input data
        
        self.X = tf.placeholder(
            tf.float32,
            shape=(None, 
                   n_H, n_W, n_C),
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
        
        D = resDiscriminator(self.X, d_sizes, 'A')
        
        with tf.variable_scope('discriminator_A') as scope:
            
            logits = D.d_forward(self.X)

        G = resGenerator(self.Z, self.n_H, self.n_W, g_sizes, 'A')

        with tf.variable_scope('generator_A') as scope:

            self.sample_images = G.g_forward(self.Z)
        
        # get sample logits
        with tf.variable_scope('discriminator_A') as scope:
            scope.reuse_variables()
            sample_logits = D.d_forward(self.sample_images, reuse=True)
                
        # get sample images for test time
        with tf.variable_scope('generator_A') as scope:
            scope.reuse_variables()
            self.sample_images_test = G.g_forward(
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
            learning_rate=lr_d,
            beta1=beta1,
        ).minimize(
            self.d_cost,
            var_list=self.d_params
        )
        
        self.g_train_op = tf.train.AdamOptimizer(
            learning_rate=lr_g,
            beta1=beta1,
        ).minimize(
            self.g_cost,
            var_list=self.g_params
        )
        self.batch_size=batch_size
        self.epochs=epochs
        self.save_sample=save_sample
        self.path=path
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.D=D
        self.G=G
    
    def set_session(self, session):
        
        self.session = session
        
        for block in self.D.d_blocks:
            block.set_session(session)
                
        for layer in self.D.d_dense_layers:
            layer.set_session(session)
        
        for block in self.G.g_blocks:
            block.set_session(session)
                
        for layer in self.G.g_dense_layers:
            layer.set_session(session)
    
    def fit(self, X):

        seed = self.seed
        d_costs = []
        g_costs = []

        N = len(X)
        n_batches = N // self.batch_size
    
        total_iters=0

        print('\n ****** \n')
        print('Training residual DCGAN with a total of ' +str(N)+' samples distributed in batches of size '+str(self.batch_size)+'\n')
        print('The learning rate set for the generator is '+str(self.lr_g)+' while for the discriminator is '+str(self.lr_d)+', and every ' +str(self.save_sample)+ ' epoch a generated sample will be saved to '+ self.path)
        print('\n ****** \n')

        for epoch in range(self.epochs):

            seed +=1

            print('Epoch:', epoch)
            
            batches = unsupervised_random_mini_batches(X, self.batch_size, seed)

            for X_batch in batches:
                
                t0 = datetime.now()
                np.random.seed(seed)
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

        
                g_costs.append((g_cost1+g_cost2)/2) # just use the avg            
            
                total_iters += 1
                if total_iters % self.save_sample ==0:
                    print("At iter: %d  -  dt: %s - d_acc: %.2f" % (total_iters, datetime.now() - t0, d_acc))
                    print('Saving a sample...')

                    np.random.seed(seed)
                    Z = np.random.uniform(-1,1, size=(64,self.latent_dims))
                    
                    samples = self.sample(Z)#shape is (64,D,D,color)
                    
                    w = self.n_W
                    h = self.n_H
                    samples = samples.reshape(64, h, w)
                    
                    
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

class cycleGAN(object):
    
    def __init__(
        self, 
        n_H, n_W, n_C,
        d_sizes_A, d_sizes_B, g_sizes_A, g_sizes_B,
        lr_g=LEARNING_RATE_G, lr_d=LEARNING_RATE_D, beta1=BETA1,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        save_sample=SAVE_SAMPLE_PERIOD, path=PATH, seed=SEED
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
        self.seed=seed
        self.n_W = n_W
        self.n_H = n_H
        self.n_C = n_C
        
        #self.latent_dims = g_sizes['z']
        
        #input data
        
        self.input_A = tf.placeholder(
            tf.float32,
            shape=(None, 
                   n_H, n_W, n_C),
            name='X_A',
        )

        self.input_B = tf.placeholder(
            tf.float32,
            shape=(None, 
                   n_H, n_W, n_C),
            name='X_B',
        )
        
        # self.Z = tf.placeholder(
        #     tf.float32,
        #     shape=(None, 
        #            self.latent_dims),
        #     name='Z'    
        # )

        self.batch_sz = tf.placeholder(
            tf.int32, 
            shape=(), 
            name='batch_sz'
        )

        D_A = resDiscriminator(self.input_A, d_sizes_A, 'A')
        D_B = resDiscriminator(self.input_B, d_sizes_B, 'B')

        G_A_to_B = cycleGenerator(self.input_A, self.n_H, self.n_W, g_sizes_A, 'A_to_B')
        G_B_to_A = cycleGenerator(self.input_B, self.n_H, self.n_W, g_sizes_B, 'B_to_A')
        

        #first cycle (A to B)
        with tf.variable_scope('discriminator_A') as scope:
            
            self.logits_A = D_A.d_forward(self.input_A)

        with tf.variable_scope('generator_A_to_B') as scope:

            self.sample_images_B = G_A_to_B.g_forward(self.input_A)

        with tf.variable_scope('discriminator_B') as scope:
            scope.reuse_variables()
            self.sample_logits_B = D_B.d_forward(self.sample_images_B, reuse=True)
        
        with tf.variable_scope('generator_B_to_A') as scope:
            scope.reuse_variables()
            self.cycl_A = G_B_to_A.g_forward(self.sample_images_B, reuse=True)

        #second cycle (B to A)
        with tf.variable_scope('discriminator_B') as scope:
            
            self.logits_B = D_B.d_forward(self.input_B)

        with tf.variable_scope('generator_B_to_A') as scope:

            self.sample_images_A = G_B_to_A.g_forward(self.input_B)

        with tf.variable_scope('discriminator_A') as scope:
            scope.reuse_variables()
            self.sample_logits_A = D_A.d_forward(self.sample_images_A, reuse=True)
        
        with tf.variable_scope('generator_A_to_B') as scope:
            scope.reuse_variables()
            self.cycl_B = G_A_to_B.g_forward(self.sample_images_A, reuse=True)

        # get sample images for test time
        with tf.variable_scope('generator_A_to_B') as scope:
            scope.reuse_variables()
            self.sample_images_test = G_A_to_B.g_forward(
                self.input_A, reuse=True, is_training=False
            )
            
        #cost building
        
        #Discriminators cost
        #cost is low if real images are predicted as real (1)
        d_cost_real_A = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits_A,
            labels=tf.ones_like(logits_A)
        )
        #cost is low if fake generated images are predicted as fake (0)
        d_cost_fake_A = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.sample_logits_A,
            labels=tf.zeros_like(sample_logits_A)
        )
        
        #discriminator_A cost
        self.d_cost_A = tf.reduce_mean(d_cost_real_A) + tf.reduce_mean(d_cost_fake_A)
        

        d_cost_real_B = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits_B,
            labels=tf.ones_like(logits_B)
        )
        
        d_cost_fake_B = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.sample_logits_B,
            labels=tf.zeros_like(sample_logits_B)
        )
        #discriminator_B cost
        self.d_cost_B = tf.reduce_mean(d_cost_real_B) + tf.reduce_mean(d_cost_fake_B)

        #averaging the two discriminators cost
        #self.d_cost = (d_cost_A+d_cost_B)/2

        #Generator cost 
        #cost is low if logits from discriminator A are predicted as true (1)
        g_cost_A = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.sample_logits_A,
                labels=tf.ones_like(self.sample_logits_A)
            )
        )
        #cost is low if logits from discriminator B are predicted as true (1)
        g_cost_B = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.sample_logits_B,
                labels=tf.ones_like(self.sample_logits_B)
            )
        )
        #cycle cost is low if cyclic images are similar to input images (in both sets)
        g_cycle_cost = tf.reduce_mean(tf.abs(self.input_A-self.cycl_A)) + tf.reduce_mean(tf.abs(self.input_B-self.cycl_B))


        self.g_cost_A = g_cost_A + 10*g_cycle_cost
        self.g_cost_B = g_cost_B + 10*g_cycle_cost


        
        #Measure accuracy of the discriminators
        #discriminator A

        real_predictions_A = tf.cast(logits_A>0,tf.float32)
        fake_predictions_A = tf.cast(sample_logits_A<0,tf.float32)
        
        num_predictions=2.0*batch_size
        num_correct_A = tf.reduce_sum(real_predictions_A)+tf.reduce_sum(fake_predictions_A)

        self.d_accuracy_A = num_correct_A/num_predictions

        real_predictions_B = tf.cast(logits_B>0,tf.float32)
        fake_predictions_B = tf.cast(sample_logits_B<0,tf.float32)
        
        num_predictions=2.0*batch_size
        num_correct_B = tf.reduce_sum(real_predictions_B)+tf.reduce_sum(fake_predictions_B)
        
        self.d_accuracy_B = num_correct_B/num_predictions

        # self.model_vars = tf.trainable_variables()
        
        #optimizers
        self.d_params_A =[t for t in tf.trainable_variables() if 'discriminator_A' in t.name]
        self.d_params_B =[t for t in tf.trainable_variables() if 'discriminator_B' in t.name]

        self.g_params_A =[t for t in tf.trainable_variables() if 'B_to_A' in t.name]
        self.g_params_B =[t for t in tf.trainable_variables() if 'A_to_B' in t.name]
        

        self.d_train_op_A = tf.train.AdamOptimizer(
            learning_rate=lr_d,
            beta1=beta1,
        ).minimize(
            self.d_cost_A,
            var_list=self.d_params_A
        )

        self.d_train_op_B = tf.train.AdamOptimizer(
            learning_rate=lr_d,
            beta1=beta1,
        ).minimize(
            self.d_cost_B,
            var_list=self.d_params_B
        )
        
        self.g_train_op_A = tf.train.AdamOptimizer(
            learning_rate=lr_g,
            beta1=beta1,
        ).minimize(
            self.g_cost_A,
            var_list=self.g_params_A
        )

        self.g_train_op_B = tf.train.AdamOptimizer(
            learning_rate=lr_g,
            beta1=beta1,
        ).minimize(
            self.g_cost_B,
            var_list=self.g_params_B
        )

        self.batch_size=batch_size
        self.epochs=epochs
        self.save_sample=save_sample
        self.path=path
        self.lr_g = lr_g
        self.lr_d = lr_d

        self.D_A=D_A
        self.D_B=D_B
        self.G_A_to_B=G_A_to_B
        self.G_B_to_A=G_B_to_A
    
    def set_session(self, session):
        
        self.session = session
        
        for block in self.D_A.d_blocks:
            block.set_session(session)
                
        for layer in self.D_A.d_dense_layers:
            layer.set_session(session)
        
        for block in self.G_A_to_B.g_blocks:
            block.set_session(session)
                
        for layer in self.G_A_to_B.g_dense_layers:
            layer.set_session(session)

        for block in self.D_B.d_blocks:
            block.set_session(session)
                
        for layer in self.D_B.d_dense_layers:
            layer.set_session(session)
        
        for block in self.G_B_to_A.g_blocks:
            block.set_session(session)
                
        for layer in self.G_B_to_A.g_dense_layers:
            layer.set_session(session)
    
    def fit(self, X):

        seed = self.seed
        d_costs_A = []
        g_costs_A = []

        d_costs_B = []
        g_costs_B = []

        N = len(X)
        n_batches = N // self.batch_size
    
        total_iters=0

        print('\n ****** \n')
        print('Training cycle GAN with a total of ' +str(N)+' samples distributed in batches of size '+str(self.batch_size)+'\n')
        print('The learning rate set for the generator is '+str(self.lr_g)+' while for the discriminator is '+str(self.lr_d)+', and every ' +str(self.save_sample)+ ' epoch a generated sample will be saved to '+ self.path)
        print('\n ****** \n')

        for epoch in range(self.epochs):

            seed +=1

            print('Epoch:', epoch)
            
            batches_A = unsupervised_random_mini_batches(X_A, self.batch_size, seed)
            batches_B = unsupervised_random_mini_batches(X_B, self.batch_size, seed)

            for X_batch_A, X_batch_B in zip(batches_A,batches_B):
                
                t0 = datetime.now()
                

                #optimize generator_A 

                #train the generator averaging two costs if the
                #discriminator learns too fast
                
                _, g_cost_A1 =  self.session.run(
                    (self.g_train_op_A, self.g_cost_A),
                    feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:self.batch_size},
                )
                
                _, g_cost_A2 =  self.session.run(
                    (self.g_train_op_A, self.g_cost_A),
                    feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:self.batch_size},
                )

                g_costs_A.append((g_cost_A1+g_cost_A2)/2) # just use the avg    

                #optimize discriminator_B

                _, d_cost_B, d_acc_B = self.session.run(
                    (self.d_train_op_B, self.d_cost_B, self.d_accuracy_B),
                    feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:self.batch_size},
                )

                d_costs_B.append(d_cost_B)

                #optimize generator_B

                _, g_cost_B1 =  self.session.run(
                    (self.g_train_op_B, self.g_cost_B),
                    feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:self.batch_size},
                )
                
                _, g_cost_B2 =  self.session.run(
                    (self.g_train_op_B, self.g_cost_B),
                    feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:self.batch_size},
                )

                g_costs_B.append((g_cost_B1+g_cost_B2)/2) # just use the avg    

                #optimize discriminator_A

                
                _, d_cost_A, d_acc_A = self.session.run(
                    (self.d_train_op_A, self.d_cost_A, self.d_accuracy_A),
                    feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:self.batch_size},
                )

                d_costs_A.append(d_cost_A)
                
                        
            
                total_iters += 1
                # if total_iters % self.save_sample ==0:
                #     print("At iter: %d  -  dt: %s - d_acc: %.2f" % (total_iters, datetime.now() - t0, d_acc))
                #     print('Saving a sample...')
                    
                #     Z = np.random.uniform(-1,1, size=(self.batch_size,self.latent_dims))
                    
                #     samples = self.sample(Z)#shape is (64,D,D,color)
                    
                #     w = self.n_W
                #     h = self.n_H
                #     samples = samples.reshape(self.batch_size, h, w)
                    
                    
                #     for i in range(64):
                #         plt.subplot(8,8,i+1)
                #         plt.imshow(samples[i].reshape(h,w), cmap='gray')
                #         plt.subplots_adjust(wspace=0.2,hspace=0.2)
                #         plt.axis('off')
                    
                #     fig = plt.gcf()
                #     fig.set_size_inches(5,7)
                #     plt.savefig(self.path+'/samples_at_iter_%d.png' % total_iters,dpi=300)


                    
            plt.clf()
            plt.plot(d_costs_A, label='discriminator cost')
            plt.plot(g_costs_B, label='generator cost')
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
