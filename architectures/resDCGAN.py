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