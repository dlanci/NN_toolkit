#NETWORK ARCHITECTURES

import numpy as np
import os 
import math

import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

from architectures.utils.NN_building_blocks import *
from architectures.utils.NN_gen_building_blocks import *
from architectures.utils.toolbox import *

def lrelu(x, alpha=0.2):
    return tf.maximum(alpha*x,x)

# some dummy constants
LEARNING_RATE = None
BETA1 = None
COST_TYPE=None
BATCH_SIZE = None
EPOCHS = None
SAVE_SAMPLE_PERIOD = None
PATH = None
SEED = None
rnd_seed=1
PREPROCESS=None
LAMBDA=.01
EPS=1e-6

DISCR_STEPS=None
GEN_STEPS=None

min_true=None
max_true=None

min_reco=None
max_reco=None

n_H=None
n_W=None
n_C=None

d_sizes=None
g_sizes=None


#tested on mnist
class DCGAN(object):
    
    def __init__(
        self,
        n_H=n_H, n_W=n_H, n_C=n_C,
        min_reco=min_reco, max_reco=max_reco,
        d_sizes= d_sizes,g_sizes=g_sizes,
        lr=LEARNING_RATE, beta1=BETA1, preprocess=PREPROCESS,
        cost_type=COST_TYPE,
        discr_steps=DISCR_STEPS, gen_steps=GEN_STEPS,
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
        self.gen_steps=gen_steps
        self.discr_steps=discr_steps

        self.min_reco=min_reco 
        self.max_reco=max_reco

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

        self.lr = tf.placeholder(
            tf.float32, 
            shape=(), 
            name='lr'
        )


        D = Discriminator_minibatch(self.X, d_sizes, 'A')

        G = Generator(self.Z, self.n_H, self.n_W, g_sizes, 'A')
        
        with tf.variable_scope('discriminator_A') as scope:
            
            logits_real, feature_output_real = D.d_forward(self.X)

        with tf.variable_scope('generator_A') as scope:

            sample_images = G.g_forward(self.Z)
        
        # get sample logits
        with tf.variable_scope('discriminator_A') as scope:
            scope.reuse_variables()
            logits_fake, feature_output_fake = D.d_forward(sample_images, reuse=True)
                
        # get sample images for test time
        with tf.variable_scope('generator_A') as scope:
            scope.reuse_variables()
            self.sample_images_test = G.g_forward(
                self.Z, reuse=True, is_training=False
            )

        predicted_real= tf.nn.sigmoid(logits_real)
        predicted_real=tf.maximum(tf.minimum(predicted_real, 0.99), 0.00)
            
        predicted_fake=tf.nn.sigmoid(logits_fake)
        predicted_fake=tf.maximum(tf.minimum(predicted_fake, 0.99), 0.00)
        #parameters list

        self.d_params =[t for t in tf.trainable_variables() if t.name.startswith('d')]
        self.g_params =[t for t in tf.trainable_variables() if t.name.startswith('g')]
        
        #cost building
        
        if cost_type == 'GAN':

            # self.d_cost_real = tf.nn.sigmoid_cross_entropy_with_logits(
            #     logits=logits_real,
            #     labels=tf.ones_like(logits_real)
            # )
            
            # self.d_cost_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            #     logits=logits_fake,
            #     labels=tf.zeros_like(logits_fake)
            # )
            
            # self.d_cost = tf.reduce_mean(self.d_cost_real) + tf.reduce_mean(self.d_cost_fake)
            
            # #Generator cost
            # self.g_cost = tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(
            #         logits=logits_fake,
            #         labels=tf.ones_like(logits_fake)
            #     )
            # )
            #Discriminator cost
            self.d_cost_real = -tf.reduce_mean(tf.log(predicted_real))
            self.d_cost_fake = -tf.reduce_mean(tf.log(1 - predicted_fake))
            self.d_cost = self.d_cost_real+self.d_cost_fake
            
            #Generator cost
            self.g_cost =tf.reduce_mean(-tf.log(predicted_fake))

        if cost_type == 'WGAN-gp':

            self.d_cost_real= -tf.reduce_mean(logits_real)
            self.d_cost_fake_lr_GAN =  tf.reduce_mean(logits_fake)
            self.d_cost_GAN= self.d_cost_real+self.d_cost_fake_lr_GAN
 
            epsilon= tf.random_uniform(
                    [self.batch_sz, 1, 1, 1], 
                    minval=0.,
                    maxval=1.,
                    )

            interpolated = epsilon*self.X+(1-epsilon)*sample_images

            with tf.variable_scope('discriminator_A') as scope:
                scope.reuse_variables()
                disc_interpolates = D.d_forward(interpolates,reuse = True)

            gradients = tf.gradients(disc_interpolates,[interpolated])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1)**2)
            self.d_cost+=10*gradient_penalty

        if cost_type == 'FEATURE':

            self.d_cost_real = -tf.reduce_mean(tf.log(predicted_real))
            self.d_cost_fake = -tf.reduce_mean(tf.log(1 - predicted_fake))
            self.d_cost = self.d_cost_real+self.d_cost_fake
            #GENERATOR COSTS
            #self.g_cost =tf.reduce_mean(-tf.log(predicted_fake))
            self.g_cost=0.01*tf.sqrt(tf.reduce_sum(tf.pow(feature_output_real-feature_output_fake,2)))

        self.d_train_op = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
            ).minimize(
            self.d_cost,
            var_list=self.d_params
        )
    
        self.g_train_op = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
            ).minimize(
            self.g_cost,
            var_list=self.g_params
            )
        #Measure accuracy of the discriminator

        real_predictions = tf.cast(predicted_real>0,tf.float32)
        fake_predictions = tf.cast(predicted_fake<0,tf.float32)
        
        num_predictions=2.0*batch_size
        num_correct = tf.reduce_sum(real_predictions)+tf.reduce_sum(fake_predictions)
        
        self.d_accuracy = num_correct/num_predictions

        self.cost_type=cost_type
        self.batch_size=batch_size
        self.epochs=epochs
        self.save_sample=save_sample
        self.path=path
        self.D=D
        self.G=G

        self.lr=lr
                
    def set_session(self, session):
        
        self.session = session
        
        for layer in self.D.d_conv_layers:
            layer.set_session(session)
                
        # for layer in self.D.d_dense_layers:
        #     layer.set_session(session)
        
        for layer in self.G.g_conv_layers:
            layer.set_session(session)
                
        for layer in self.G.g_dense_layers:
            layer.set_session(session)
    
    def fit(self, X, validating_size):

        _all = X

        gen_steps=self.gen_steps
        discr_steps=self.discr_steps

        m = _all.shape[0]
        train = _all[0:m-validating_size]

        validating = _all[m-validating_size:m]

        seed=self.seed
        d_costs = []
        g_costs = []

        N = len(train)
        n_batches = N // self.batch_size
    
        total_iters=0

        print('\n ****** \n')
        print('Training improved DCGAN with a total of ' +str(N)+' samples distributed in batches of size '+str(self.batch_size)+'\n')
        print('The learning rate is '+str(self.lr)+', and every ' +str(self.save_sample)+ ' epoch a generated sample will be saved to '+ self.path)
        print('\n ****** \n')

        for epoch in range(self.epochs):

            seed +=1

            print('Epoch:', epoch)
            
            batches = unsupervised_random_mini_batches(train, self.batch_size, seed)

            for X_batch in batches:
                bs = X_batch.shape[0]
                t0 = datetime.now()
                
                np.random.seed(seed)

                d_cost=0

                for i in range(discr_steps):

                    Z = np.random.uniform(-1,1, size= (bs, self.latent_dims))
                    
                    _, d_cost, d_acc = self.session.run(
                        (self.d_train_op, self.d_cost, self.d_accuracy),
                        feed_dict={self.X: X_batch, self.Z:Z, self.batch_sz: bs},
                    )
                    d_cost+=d_cost

                
                d_costs.append(d_cost/discr_steps)
                
                #train the generator averaging two costs if the
                #discriminator learns too fast
                g_cost=0
                for i in range(gen_steps):
                    _, g_cost =  self.session.run(
                        (self.g_train_op, self.g_cost),
                        feed_dict={self.X: X_batch, self.Z:Z, self.batch_sz:bs},
                    )
                    g_cost+=g_cost
                
                g_costs.append(g_cost/gen_steps) 
            
                total_iters += 1
                if total_iters % self.save_sample ==0:
                    print("At iter: %d  -  dt: %s - d_acc: %.4f" % (total_iters, datetime.now() - t0, d_acc))
                    print("Discriminator cost {0:.4g}, Generator cost {1:.4g}".format(d_cost/discr_steps, g_cost/gen_steps))
                    print('Saving a sample...')
                    plt.clf() 
                    np.random.seed(seed)
                    Z = np.random.uniform(-1,1, size=(16,self.latent_dims))
                    
                    samples = self.sample(Z)#shape is (64,D,D,color)
                    samples = denormalise(samples, self.min_reco, self.max_reco)
                    
                    w = self.n_W
                    h = self.n_H
                    samples = np.sum(samples,axis=3)
                    samples = samples.reshape(16, h, w)
                    
                    for i in range(16):
                        plt.subplot(4,4,i+1)
                        plt.imshow(samples[i].reshape(h,w))
                        plt.subplots_adjust(wspace=0.2,hspace=0.2)
                        plt.axis('off')
                    
                    fig = plt.gcf()
                    fig.set_size_inches(7,7)
                    plt.savefig(self.path+'/samples_at_iter_%d.png' % total_iters,dpi=300)

                    plt.clf() 
                    plt.plot(d_costs, label='Discriminator cost')
                    plt.plot(g_costs, label='Generator cost')
                    plt.legend()
                    
                    fig = plt.gcf()
                    fig.set_size_inches(10,5)
                    plt.savefig(self.path+'/cost vs iteration.png',dpi=300)
    
    def sample(self, Z):
        
        many_samples = self.session.run(
            self.sample_images_test, 
            feed_dict={self.Z:Z, self.batch_sz: self.batch_size})

        return many_samples 

    def get_sample(self, Z):
        
        one_sample = self.session.run(
            self.sample_images_test, 
            feed_dict={self.Z:Z, self.batch_sz: 1})

        return one_sample 