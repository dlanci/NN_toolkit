import numpy as np
import os 
import math

import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

from architectures.utils.NN_building_blocks import *
from architectures.utils.NN_gen_building_blocks import *
from architectures.utils.toolbox import *


#some hyperparameters of the network
LEARNING_RATE = None
BETA1 = None
COST_TYPE=None
BATCH_SIZE = None
EPOCHS = None
SAVE_SAMPLE_PERIOD = None
PATH = None
SEED = None
rnd_seed=1
preprocess=None
LAMBDA=.01
EPS=1e-10
CYCL_WEIGHT=None
GAN_WEIGHT=None
DISCR_STEPS=None
GEN_STEPS=None

min_true=None
max_true=None

min_reco=None
max_reco=None

n_H_A=None
n_W_A=None
n_W_B=None
n_H_B=None
n_C=None

d_sizes=None
g_sizes_enc=None
g_sizes_dec=None



class pix2pix(object):

    def __init__(

        self,
        n_H_A=n_H_A, n_W_A=n_W_A,
        n_H_B=n_H_B, n_W_B=n_W_B, n_C=n_C,
        min_true=min_true, max_true=max_true, 
        min_reco=min_reco, max_reco=max_reco,
        d_sizes=d_sizes, g_sizes_enc=g_sizes_enc, g_sizes_dec=g_sizes_dec,
        lr=LEARNING_RATE, beta1=BETA1, preprocess=preprocess,
        cost_type=COST_TYPE, gan_weight=GAN_WEIGHT, cycl_weight=CYCL_WEIGHT,
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


        self.max_reco=max_reco
        self.min_reco = min_reco

        self.max_true=max_true
        self.min_true=min_true

        self.seed=seed

        self.n_W_A = n_W_A
        self.n_H_A = n_H_A

        self.n_W_B = n_W_B
        self.n_H_B = n_H_B

        self.n_C = n_C

        self.batch_sz = tf.placeholder(
            tf.int32, 
            shape=(), 
            name='batch_sz'
        )

        self.input_A = tf.placeholder(
        tf.float32,
        shape=(None, 
           n_H_A, n_W_A, n_C),
        name='X_A',
        )

        self.input_B = tf.placeholder(
        tf.float32,
        shape=(None, 
           n_H_B, n_W_B, n_C),
        name='X_B',
        )

        self.input_test_A = tf.placeholder(
            tf.float32,
            shape=(None,
                   n_H_A, n_W_A, n_C),
            name='X_test_A'
        )

        D = pix2pixDiscriminator(self.input_A, d_sizes, 'B')
        G = pix2pixGenerator(self.input_A, n_H_B, n_W_B, g_sizes_enc, g_sizes_dec, 'A_to_B')

        with tf.variable_scope('generator_A_to_B') as scope:

            sample_images = G.g_forward(self.input_A)

        with tf.variable_scope('discriminator_B') as scope:

            predicted_real = D.d_forward(self.input_A, self.input_B)
            
        with tf.variable_scope('discriminator_B') as scope:
            scope.reuse_variables()
            predicted_fake = D.d_forward(self.input_A, sample_images, reuse=True)

        #get sample images at test time
        with tf.variable_scope('generator_A_to_B') as scope:
            scope.reuse_variables()
            self.sample_images_test_A_to_B = G.g_forward(
                self.input_test_A, reuse=True, is_training=False
            )

        self.d_params = [t for t in tf.trainable_variables() if 'discriminator' in t.name]
        self.g_params = [t for t in tf.trainable_variables() if 'generator' in t.name]


        if cost_type == 'GAN':

            #Discriminator cost
            
            predicted_real= tf.nn.sigmoid(predicted_real)
            predicted_fake=tf.nn.sigmoid(predicted_fake)

            d_cost_real = tf.log(predicted_real + EPS)

            #d_cost_fake is low if fake images are predicted as real
            d_cost_fake = tf.log(1 - predicted_fake +EPS)

            self.d_cost =  tf.reduce_mean(-(d_cost_real + d_cost_fake))


            # #Discriminator cost
            # #d_cost_real is low if real images are predicted as real
            # d_cost_real = tf.nn.sigmoid_cross_entropy_with_logits(
            #     logits = predicted_real,
            #     labels = tf.ones_like(predicted_real)-0.01
            # )
            # #d_cost_fake is low if fake images are predicted as real
            # d_cost_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            #     logits = predicted_fake,
            #     labels = tf.zeros_like(predicted_fake)+0.01
            #     )

            # self.d_cost = tf.reduce_mean(d_cost_real)+ tf.reduce_mean(d_cost_fake)

            #Generator cost 
            #g_cost is low if logits from discriminator on samples generated by generator are predicted as true (1)
            self.g_cost_GAN = tf.reduce_mean(-tf.log(predicted_fake + EPS))

            # self.g_cost_GAN = tf.reduce_mean(
            #     tf.nn.sigmoid_cross_entropy_with_logits(
            #         logits=predicted_fake,
            #         labels=tf.ones_like(predicted_fake)-0.01
            #     )
            # )
            self.g_cost_l1 = tf.reduce_mean(tf.square(self.input_B - sample_images))
            #self.g_cost_sum = tf.abs(tf.reduce_sum(self.input_B)-tf.reduce_sum(sample_images))
            self.g_cost=gan_weight*self.g_cost_GAN + cycl_weight*self.g_cost_l1

        if cost_type == 'WGAN-gp':


            self.g_cost_GAN = -tf.reduce_mean(predicted_fake)

            self.g_cost_l1 = tf.reduce_mean(tf.abs(self.input_B - sample_images))
            self.g_cost=gan_weight*self.g_cost_GAN+cycl_weight*self.g_cost_l1


            self.d_cost = tf.reduce_mean(predicted_fake) - tf.reduce_mean(predicted_real)
            
            alpha = tf.random_uniform(
                shape=[self.batch_sz,self.n_H_A,self.n_W_A,self.n_C],
                minval=0.,
                maxval=1.
            )

            # interpolates_1 = alpha*self.input_A+(1-alpha)*sample_images
            interpolates = alpha*self.input_B+(1-alpha)*sample_images

            with tf.variable_scope('discriminator_B') as scope:
                scope.reuse_variables()
                disc_interpolates = D.d_forward(self.input_A, interpolates,reuse = True)

            gradients = tf.gradients(disc_interpolates,[interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            self.gradient_penalty = tf.reduce_mean((slopes-1)**2)
            self.d_cost+=LAMBDA*self.gradient_penalty

        self.d_train_op = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
            beta2=0.9
            ).minimize(
            self.d_cost,
            var_list=self.d_params
        )

        self.g_train_op = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
            beta2=0.9
            ).minimize(
            self.g_cost,
            var_list=self.g_params
        )

        #saving for later
        self.batch_size=batch_size
        self.epochs=epochs
        self.save_sample=save_sample
        self.path=path
        self.lr = lr

        self.D=D
        self.G=G

        self.sample_images=sample_images
        self.preprocess=preprocess
        self.cost_type=cost_type
        self.cycl_weight=cycl_weight

        self.gen_steps=gen_steps
        self.discr_steps=discr_steps

    def set_session(self,session):

        self.session = session

        for layer in self.D.d_conv_layers:
            layer.set_session(session)

        for layer in self.G.g_enc_conv_layers:
            layer.set_session(session)

        for layer in self.G.g_dec_conv_layers:
            layer.set_session(session)

    def fit(self, X_A, X_B, validating_size):

        all_A = X_A
        all_B = X_B
        gen_steps = self.gen_steps
        discr_steps = self.discr_steps

        m = X_A.shape[0]
        train_A = all_A[0:m-validating_size]
        train_B = all_B[0:m-validating_size]

        validating_A = all_A[m-validating_size:m]
        validating_B = all_B[m-validating_size:m]

        seed=self.seed

        d_costs=[]
        d_gps=[]
        g_costs=[]
        g_GANs=[]
        g_l1s=[]
        N=len(train_A)
        n_batches = N // self.batch_size

        total_iters=0

        print('\n ****** \n')
        print('Training pix2pix (from 1611.07004) GAN with a total of ' +str(N)+' samples distributed in '+ str((N)//self.batch_size) +' batches of size '+str(self.batch_size)+'\n')
        print('The validation set consists of {0} images'.format(validating_A.shape[0]))
        print('The learning rate is '+str(self.lr)+', and every ' +str(self.save_sample)+ ' batches a generated sample will be saved to '+ self.path)
        print('\n ****** \n')

        for epoch in range(self.epochs):

            seed+=1
            print('Epoch:', epoch)

            batches_A = unsupervised_random_mini_batches(train_A, self.batch_size, seed)
            batches_B = unsupervised_random_mini_batches(train_B, self.batch_size, seed)

            for X_batch_A, X_batch_B in zip(batches_A, batches_B):

                bs=X_batch_A.shape[0]

                t0 = datetime.now()

                g_cost=0
                g_GAN=0
                g_l1=0
                for i in range(gen_steps):

                    _, g_cost, g_GAN, g_l1 = self.session.run(
                    (self.g_train_op, self.g_cost, self.g_cost_GAN, self.g_cost_l1),
                    feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:bs},
                    )
                    g_cost+=g_cost
                    g_GAN+=g_GAN
                    g_l1+=g_l1

                g_costs.append(g_cost/gen_steps)
                g_GANs.append(g_GAN/gen_steps)
                g_l1s.append(self.cycl_weight*g_l1/gen_steps)

                d_cost=0
                d_gp=0
                for i in range(discr_steps):

                    if self.cost_type=='WGAN-gp':
                        _, d_cost, d_gp = self.session.run(
                        (self.d_train_op, self.d_cost, self.gradient_penalty),
                        feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:bs},
                        )

                        d_gp+=d_gp


                    else:
                        _, d_cost = self.session.run(
                        (self.d_train_op, self.d_cost),
                        feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:bs},
                        )
                    
                    d_cost+=d_cost
                    

                    
                d_costs.append(d_cost/discr_steps)
                if self.cost_type=='WGAN-gp':
                    d_gps.append(LAMBDA*d_gp/discr_steps)

                total_iters+=1
                if total_iters % self.save_sample ==0:

                    plt.clf()
                    print("At iter: %d  -  dt: %s" % (total_iters, datetime.now() - t0))
                    print("Discriminator cost {0:.4g}, Generator cost {1:.4g}".format(d_costs[-1], g_costs[-1]))
                    print('Saving a sample...')

                    if self.preprocess!=False:
                        draw_nn_sample(validating_A, validating_B, 1, self.preprocess,
                                        self.min_true, self.max_true, 
                                        self.min_reco, self.max_reco,
                                        f=self.get_sample_A_to_B, is_training=True,
                                        total_iters=total_iters, PATH=self.path)
                    else:
                        draw_nn_sample(validating_A, validating_B, 1, self.preprocess,
                                        f=self.get_sample_A_to_B, is_training=True,
                                        total_iters=total_iters, PATH=self.path)
                    plt.clf()
                    plt.subplot(1,2,1)
                    plt.plot(d_costs, label='Discriminator GAN cost')
                    plt.plot(g_GANs, label='Generator GAN cost')
                    plt.xlabel('Epoch')
                    plt.ylabel('Cost')
                    plt.legend()
                    
                    plt.subplot(1,2,2)
                    plt.plot(g_costs, label='Generator total cost')
                    plt.plot(g_GANs, label='Generator GAN cost')
                    plt.plot(g_l1s, label='Generator l1 cycle cost')
                    plt.xlabel('Epoch')
                    plt.ylabel('Cost')
                    plt.legend()

                    fig = plt.gcf()
                    fig.set_size_inches(15,5)
                    plt.savefig(self.path+'/cost_iteration_gen_disc_B_to_A.png',dpi=150)


    def get_sample_A_to_B(self, Z):
        
        one_sample = self.session.run(
            self.sample_images_test_A_to_B, 
            feed_dict={self.input_test_A:Z, self.batch_sz: 1})

        return one_sample 

    def get_samples_A_to_B(self, Z):
        
        many_samples = self.session.run(
            self.sample_images_test_A_to_B, 
            feed_dict={self.input_test_A:Z, self.batch_sz: Z.shape[0]})

        return many_samples 