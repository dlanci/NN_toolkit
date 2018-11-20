import numpy as np
import os 
import math

import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

from architectures.utils.NN_building_blocks import *
from architectures.utils.NN_gen_building_blocks import *

def lrelu(x, alpha=0.2):
    return tf.maximum(alpha*x,x)


pretrain=None
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
CYCL_WEIGHT=None
LATENT_WEIGHT=None
KL_WEIGHT=None
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
e_sizes=None


class bicycle_GAN(object):
    #fix args
    def __init__(

        self, 
        n_H_A=n_H_A, n_W_A=n_W_A,
        n_H_B=n_H_B, n_W_B=n_W_B, n_C=n_C,
        min_true=min_true, max_true=max_true, 
        min_reco=min_reco, max_reco=max_reco,
        d_sizes=d_sizes, g_sizes_enc=g_sizes_enc, g_sizes_dec=g_sizes_dec, e_sizes=e_sizes, 
        pretrain=pretrain, lr=LEARNING_RATE, beta1=BETA1, preprocess=PREPROCESS,
        cost_type=COST_TYPE, cycl_weight=CYCL_WEIGHT, latent_weight=LATENT_WEIGHT, kl_weight=KL_WEIGHT,
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

        latent_dims=e_sizes['latent_dims']

        self.min_true=min_true
        self.max_true=max_true

        self.min_reco=min_reco 
        self.max_reco=max_reco
        self.seed=seed

        self.n_W_A = n_W_A
        self.n_H_A = n_H_A

        self.n_W_B = n_W_B
        self.n_H_B = n_H_B
        self.n_C = n_C 

        #input data
        
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

        self.z = tf.placeholder(
            tf.float32,
            shape=(None,
                    latent_dims)
        )

        self.input_test_A = tf.placeholder(
                    tf.float32,
                    shape=(None, 
                           n_H_A, n_W_A, n_C),
                    name='X_test_A',
                )


        G = bicycleGenerator(self.input_A, self.n_H_B, self.n_W_B, g_sizes_enc, g_sizes_dec, 'A_to_B')

        D = Discriminator_minibatch(self.input_B, d_sizes, 'B')
        #D = Discriminator(self.input_B, d_sizes, 'B')

        E = bicycleEncoder(self.input_B, e_sizes, 'B')

        if pretrain: 

            with tf.variable_scope('generator_A_to_B') as scope:
                sample_A_to_B = G.g_forward(self.input_A, self.z, is_pretraining=True, is_training=True)

            with tf.variable_scope('discriminator_B') as scope:

                logits_real, feature_output_real = D.d_forward(self.input_B)
                
            with tf.variable_scope('discriminator_B') as scope:
                scope.reuse_variables()
                logits_fake , feature_output_fake = D.d_forward(sample_A_to_B, reuse=True)

            with tf.variable_scope('generator_A_to_B') as scope:
                scope.reuse_variables()
                self.test_images_A_to_B = G.g_forward(
                    self.input_test_A, self.z, reuse=True, is_pretraining=True, is_training=False
                    )

            #parameters lists
            self.d_params =[t for t in tf.trainable_variables() if 'discriminator' in t.name]
            self.g_params =[t for t in tf.trainable_variables() if 'generator' in t.name] 

            predicted_real= tf.nn.sigmoid(logits_real)
            predicted_real=tf.maximum(tf.minimum(predicted_real, 0.99), 0.00)
            
            predicted_fake=tf.nn.sigmoid(logits_fake)
            predicted_fake=tf.maximum(tf.minimum(predicted_fake, 0.99), 0.00)

            
            if cost_type=='GAN':
                ##GAN LOSS

                #DISCRIMINATOR COSTS
                self.d_cost_real = -tf.reduce_mean(tf.log(predicted_real))
                self.d_cost_fake_lr_GAN = -tf.reduce_mean(tf.log(1 - predicted_fake))
                self.d_cost=self.d_cost_real+self.d_cost_fake_lr_GAN
                #GENERATOR COSTS
                self.g_cost_lr_GAN = tf.reduce_mean(-tf.log(predicted_fake))
                

            if cost_type=='WGAN':
                #WGAN LOSS
                #DISCRIMINATOR COST

                self.d_cost_real= -tf.reduce_mean(logits_real)
                self.d_cost_fake_lr_GAN =  tf.reduce_mean(logits_fake)
                self.d_cost_GAN= self.d_cost_real+self.d_cost_fake_lr_GAN

                #GP
                epsilon= tf.random_uniform(
                    [self.batch_sz, 1, 1, 1], 
                    minval=0.,
                    maxval=1.,
                    )
                interpolated = epsilon*self.input_A + (1-epsilon)*sample_A_to_B
                with tf.variable_scope('discriminator_B') as scope:
                    scope.reuse_variables()
                    logits_interpolated, feature_output_interpolated= D.d_forward(interpolated, reuse=True)

                gradients = tf.gradients(logits_interpolated, [interpolated], name='D_logits_intp')[0]
                grad_l2= tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
                self.grad_penalty=tf.reduce_mean(tf.square(grad_l2-1.0))
                self.d_cost=self.d_cost_GAN+10*self.grad_penalty

                #GENERATOR COST
                self.g_cost_lr_GAN= - tf.reduce_mean(logits_fake)

            if cost_type=='FEATURE':
                #DISCRIMINATOR COSTS
                self.d_cost_real = -tf.reduce_mean(tf.log(predicted_real))
                self.d_cost_fake_lr_GAN = -tf.reduce_mean(tf.log(1 - predicted_fake))
                self.d_cost=self.d_cost_real+self.d_cost_fake_lr_GAN
                #GENERATOR COSTS
                self.g_cost_lr_GAN = tf.sqrt(tf.reduce_sum(tf.pow(feature_output_real-feature_output_fake,2)))

            #CYCLIC WEIGHT
            
            self.g_cost_cycl=tf.reduce_mean(tf.abs(self.input_B- sample_A_to_B ))+tf.sqrt(tf.reduce_sum(tf.pow(feature_output_real-feature_output_fake,2)))
            self.g_cost=self.g_cost_lr_GAN + cycl_weight*self.g_cost_cycl 

            self.d_train_op = tf.train.AdamOptimizer(
                    learning_rate=lr, 
                    beta1=beta1,
                    beta2=0.9,
                ).minimize(
                    self.d_cost, 
                    var_list=self.d_params
                )

            self.g_train_op = tf.train.AdamOptimizer(
                    learning_rate=lr, 
                    beta1=beta1,
                    beta2=0.9,
                ).minimize(
                    self.g_cost, 
                    var_list=self.g_params
                )

            real_predictions = tf.cast(logits_real>0,tf.float32)
            fake_predictions = tf.cast(logits_fake<0,tf.float32)
            
            num_predictions=2.0*batch_size
            num_correct = tf.reduce_sum(real_predictions)+tf.reduce_sum(fake_predictions)

            self.d_accuracy= num_correct/num_predictions

            self.D=D
            self.G=G
            self.cycl_weight=cycl_weight
            self.latent_dims=latent_dims

        if not pretrain:

            with tf.variable_scope('encoder_B') as scope:
                z_encoded, z_encoded_mu, z_encoded_log_sigma = E.e_forward(self.input_B)
            
            with tf.variable_scope('generator_A_to_B') as scope:
                sample_A_to_B_encoded = G.g_forward(self.input_A, z_encoded)

            with tf.variable_scope('generator_A_to_B') as scope:
                scope.reuse_variables()
                sample_A_to_B = self.sample_A_to_B = G.g_forward(self.input_A, self.z, reuse=True)
            
            with tf.variable_scope('encoder_B') as scope:
                scope.reuse_variables()
                z_recon, z_recon_mu, z_recon_log_sigma = E.e_forward(sample_A_to_B, reuse=True)

            with tf.variable_scope('discriminator_B') as scope:

                logits_real, feature_output_real = D.d_forward(self.input_B)

                
            with tf.variable_scope('discriminator_B') as scope:
                scope.reuse_variables()
                logits_fake, feature_output_fake = D.d_forward(sample_A_to_B, reuse=True)
                logits_fake_encoded, feature_output_fake_encoded = D.d_forward(sample_A_to_B_encoded, reuse=True)

            with tf.variable_scope('generator_A_to_B') as scope:
                scope.reuse_variables()
                self.test_images_A_to_B = G.g_forward(
                    self.input_test_A, self.z, reuse=True, is_training=False
                    )

            #parameters lists
            self.d_params =[t for t in tf.trainable_variables() if 'discriminator' in t.name]
            self.e_params =[t for t in tf.trainable_variables() if 'encoder' in t.name]
            self.g_params =[t for t in tf.trainable_variables() if 'generator' in t.name]
            
            predicted_real= tf.nn.sigmoid(logits_real)
            predicted_real=tf.maximum(tf.minimum(predicted_real, 0.99), 0.00)

            predicted_fake=tf.nn.sigmoid(logits_fake)
            predicted_fake=tf.maximum(tf.minimum(predicted_fake, 0.99), 0.00)

            predicted_fake_encoded = tf.nn.sigmoid(logits_fake_encoded)
            predicted_fake_encoded =tf.maximum(tf.minimum(predicted_fake_encoded, 0.99), 0.00)

            #GAN LOSS
            if cost_type=='GAN': 
                #DISCRIMINATOR LOSSES
                self.d_cost_real = -tf.reduce_mean(tf.log(predicted_real))
                self.d_cost_fake_lr_GAN = -tf.reduce_mean(tf.log(1 - predicted_fake ))
                self.d_cost_fake_vae_GAN = -tf.reduce_mean(tf.log(1 - predicted_fake_encoded ))

                #GENERATOR LOSSES

                self.g_cost_vae_GAN = -tf.reduce_mean(tf.log(predicted_fake_encoded))
                self.g_cost_lr_GAN = -tf.reduce_mean(tf.log(predicted_fake))
                # #X ENTROPY LOSS

                #DISCRIMINATOR LOSSES
                # self.d_cost_real = tf.nn.sigmoid_cross_entropy_with_logits(
                #     logits=predicted_real_B,
                #     labels=tf.ones_like(predicted_real_B)
                # )
                
                # self.d_cost_fake_lr_GAN = tf.nn.sigmoid_cross_entropy_with_logits(
                #     logits=predicted_fake,
                #     labels=tf.zeros_like(predicted_fake)
                # )

                # self.d_cost_fake_vae_GAN = tf.nn.sigmoid_cross_entropy_with_logits(
                #     logits=predicted_fake,
                #     labels=tf.zeros_like(predicted_fake_encoded)
                # )

                #GENERATOR LOSSES
                # self.g_cost_lr_GAN = tf.reduce_mean(
                #     tf.nn.sigmoid_cross_entropy_with_logits(
                #         logits=predicted_fake,
                #         labels=tf.ones_like(predicted_fake)
                #     )
                # )

                # self.g_cost_vae_GAN = tf.reduce_mean(
                #     tf.nn.sigmoid_cross_entropy_with_logits(
                #         logits=predicted_fake_encoded,
                #         labels=tf.ones_like(predicted_fake_encoded)
                #     )
                # )
            #WGAN LOSS
            if cost_type=='WGAN':
                #DISCRIMINATOR
                self.d_cost_real= -tf.reduce_mean(logits_real)
                self.d_cost_fake_vae_GAN = tf.reduce_mean(logits_fake_encoded)
                self.d_cost_fake_lr_GAN =  tf.reduce_mean(logits_fake)

                #GP
                epsilon= tf.random_uniform(
                    [self.batch_sz, 1, 1, 1], 
                    minval=0.,
                    maxval=1.,
                    )
                interpolated = epsilon*self.input_A + (1-epsilon/2)*sample_A_to_B + (1-epsilon/2)*sample_A_to_B_encoded 
                with tf.variable_scope('discriminator_B') as scope:
                    scope.reuse_variables()
                    logits_interpolated= D.d_forward(self.input_A, interpolated, reuse=True)

                gradients = tf.gradients(logits_interpolated, [interpolated], name='D_logits_intp')[0]
                grad_l2= tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
                self.grad_penalty=tf.reduce_mean(tf.square(grad_l2-1.0))

                #GENERATOR
                self.g_cost_vae_GAN= - tf.reduce_mean(logits_fake_encoded)
                self.g_cost_lr_GAN= - tf.reduce_mean(logits_fake)
                
            if cost_type=='FEATURE': 
                #DISCRIMINATOR LOSSES
                self.d_cost_real = -tf.reduce_mean(tf.log(predicted_real))
                self.d_cost_fake_lr_GAN = -tf.reduce_mean(tf.log(1 - predicted_fake ))
                self.d_cost_fake_vae_GAN = -tf.reduce_mean(tf.log(1 - predicted_fake_encoded ))

                #GENERATOR LOSSES

                self.g_cost_lr_GAN = tf.sqrt(tf.reduce_sum(tf.pow(feature_output_real-feature_output_fake,2)))
                self.g_cost_vae_GAN = tf.sqrt(tf.reduce_sum(tf.pow(feature_output_real-feature_output_fake_encoded,2)))
                # #X ENTROPY LOSS

            #CYCLIC WEIGHT

            self.g_cost_cycl = tf.reduce_mean(tf.abs(self.input_B - sample_A_to_B_encoded))
            self.g_feature_match= 0.5*tf.sqrt(tf.reduce_sum(tf.pow(feature_output_real-feature_output_fake,2)))+0.5*tf.sqrt(tf.reduce_sum(tf.pow(feature_output_real-feature_output_fake_encoded,2)))
            
            self.g_4_cells_cycl=tf.reduce_mean(
                    tf.abs(
                    tf.cast(
                    tf.convert_to_tensor(
                    [
                    tf.nn.top_k(
                        tf.reshape(
                                self.input_B[i],
                        [-1]), 
                    k=4)[0] -
                    tf.nn.top_k(
                        tf.reshape(
                                sample_A_to_B[i],
                        [-1]), 
                    k=4)[0] 
                    for i in range(16)]
            )
            ,dtype=tf.float32)
            )
            )
            self.g_4_cells_pos=tf.reduce_mean(tf.abs(tf.cast(tf.convert_to_tensor(
                    [
                    tf.nn.top_k(
                        tf.reshape(
                                self.input_B[i],
                        [-1]), 
                    k=4)[1] -
                    tf.nn.top_k(
                        tf.reshape(
                                sample_A_to_B[i],
                        [-1]), 
                    k=4)[1] 
                    for i in range(16)]
            ), dtype=tf.float32)
            )
            )
            
            
            

            self.g_4_cells_cycl_encoded=tf.reduce_mean(
                    tf.abs(
                    tf.cast(
                    tf.convert_to_tensor(
                    [
                    tf.nn.top_k(
                        tf.reshape(
                                self.input_B[i],
                        [-1]), 
                    k=4)[0] -
                    tf.nn.top_k(
                        tf.reshape(
                                sample_A_to_B_encoded[i],
                        [-1]), 
                    k=4)[0] 
                    for i in range(16)]
            )
            ,dtype=tf.float32)
            )
            )
            self.g_4_cells_pos_encoded=tf.reduce_mean(tf.abs(tf.cast(tf.convert_to_tensor(
                    [
                    tf.nn.top_k(
                        tf.reshape(
                                self.input_B[i],
                        [-1]), 
                    k=4)[1] -
                    tf.nn.top_k(
                        tf.reshape(
                                sample_A_to_B_encoded[i],
                        [-1]), 
                    k=4)[1] 
                    for i in range(16)]
            ), dtype=tf.float32)
            )
            )
            


            
            #ENCODER COSTS

            self.e_cost_latent_cycle = tf.reduce_mean(tf.abs(self.z - z_recon))
            self.e_cost_kl = -0.5 * tf.reduce_mean(+1 - 2*np.log(0.02) + 2*z_encoded_log_sigma - (z_encoded_mu ** 2 + tf.exp(z_encoded_log_sigma)**2)/(0.02)**2)
            #self.e_cost_kl = 0.5 * tf.reduce_mean(1 + 2*z_encoded_log_sigma - (z_encoded_mu ** 2 + tf.exp(z_encoded_log_sigma)**2))
            #TOTAL COSTS
            #self.d_cost = self.d_cost_fake_vae_GAN + self.d_cost_fake_lr_GAN + self.d_cost_real
            self.d_cost = self.d_cost_fake_vae_GAN + self.d_cost_fake_lr_GAN + self.d_cost_real # + 10*self.grad_penalty
            self.g_cost = self.g_cost_vae_GAN + self.g_cost_lr_GAN + cycl_weight*(self.g_cost_cycl + self.g_feature_match) + (cycl_weight/0.5)*(self.g_4_cells_cycl + self.g_4_cells_cycl_encoded) + latent_weight*self.e_cost_latent_cycle 
            self.e_cost = self.g_cost_vae_GAN + cycl_weight*(self.g_cost_cycl) + (cycl_weight/0.5)*(self.g_4_cells_cycl + self.g_4_cells_cycl_encoded) + latent_weight*self.e_cost_latent_cycle + kl_weight*self.e_cost_kl

            self.d_train_op = tf.train.AdamOptimizer(
                    learning_rate=lr, 
                    beta1=beta1,
                    beta2=0.9,
                ).minimize(
                    self.d_cost, 
                    var_list=self.d_params
                )

            self.g_train_op = tf.train.AdamOptimizer(
                    learning_rate=lr, 
                    beta1=beta1,
                    beta2=0.9,
                ).minimize(
                    self.g_cost, 
                    var_list=self.g_params
                )

            self.e_train_op = tf.train.AdamOptimizer(
                    learning_rate=lr, 
                    beta1=beta1,
                    beta2=0.9,
                ).minimize(
                    self.e_cost, 
                    var_list=self.e_params
                )

            real_predictions = tf.cast(logits_real>0,tf.float32)
            fake_predictions = tf.cast(logits_fake<0,tf.float32)
            
            num_predictions=2.0*batch_size
            num_correct = tf.reduce_sum(real_predictions)+tf.reduce_sum(fake_predictions)

            self.d_accuracy= num_correct/num_predictions

            self.D=D
            self.G=G
            self.E=E

            self.latent_weight=latent_weight
            self.cycl_weight=cycl_weight
            self.kl_weight=kl_weight
            self.latent_dims=latent_dims

        self.batch_size=batch_size
        self.epochs=epochs
        self.save_sample=save_sample
        self.path=path
        self.lr = lr

        self.preprocess=preprocess
        self.cost_type=cost_type
        self.gen_steps=gen_steps
        self.discr_steps=discr_steps
        self.pretrain=pretrain

    def set_session(self, session):

        self.session = session

        for layer in self.D.d_conv_layers:
            layer.set_session(session)

        for layer in self.G.g_enc_conv_layers:
            layer.set_session(session)

        for layer in self.G.g_dec_conv_layers:
            layer.set_session(session)

        if not self.pretrain:
            for layer in self.E.e_blocks:
                layer.set_session(session)

            for layer in self.E.e_dense_layers:
                layer.set_session(session)

    def fit(self, X_A, X_B, validating_size):

        all_A = X_A
        all_B = X_B

        gen_steps=self.gen_steps
        discr_steps=self.discr_steps

        m = X_A.shape[0]
        train_A = all_A[0:m-validating_size]
        train_B = all_B[0:m-validating_size]

        validating_A = all_A[m-validating_size:m]
        validating_B = all_B[m-validating_size:m]

        seed=self.seed

        d_costs=[]
        d_costs_vae_GAN=[]
        d_costs_lr_GAN=[]
        d_costs_GAN=[ ]

        g_costs=[]
        g_costs_lr_GAN=[]
        g_costs_vae_GAN=[]
        g_costs_cycl=[]

        e_costs=[]
        e_costs_kl=[]
        e_costs_latent_cycle=[]

        N=len(train_A)
        n_batches = N // self.batch_size

        total_iters=0

        if self.pretrain==True:

            print('\n ****** \n')
            print('Pretraining bicycleGAN (pix2pix part) with a total of ' +str(N)+' samples distributed in '+ str((N)//self.batch_size) +' batches of size '+str(self.batch_size)+'\n')
            print('The validation set consists of {0} images'.format(validating_A.shape[0]))
            print('The learning rate is '+str(self.lr)+', and every ' +str(self.save_sample)+ ' batches a generated sample will be saved to '+ self.path+'/pretrained/')
            print('\n ****** \n')

            for epoch in range(self.epochs):

                seed+=1
                print('Epoch:', epoch)

                batches_A = unsupervised_random_mini_batches(train_A, self.batch_size, seed)
                batches_B = unsupervised_random_mini_batches(train_B, self.batch_size, seed)

                for X_batch_A, X_batch_B in zip(batches_A, batches_B)[:-1]:

                    bs=X_batch_A.shape[0]

                    t0 = datetime.now()


                    g_cost=0
                    g_cost_cycl=0
                    g_cost_lr_GAN=0

                    d_cost=0
                    d_cost_GAN=0


                    for i in range(discr_steps):

                        sample_z = np.zeros(shape=(bs, self.latent_dims))
                        #sample_z = np.random.normal(size=(bs, self.latent_dims))

                        _, d_acc, d_cost = self.session.run(

                            (self.d_train_op, self.d_accuracy, self.d_cost),

                            feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, 
                                    self.z:sample_z, self.batch_sz:bs
                                    },
                        )
                        d_cost+=d_cost
                        
                        if self.cost_type=='WGAN':
                            d_cost_GAN= self.session.run(

                                (self.d_cost_GAN),
                                feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, 
                                    self.z:sample_z, self.batch_sz:bs
                                    },

                                )
                            d_cost_GAN+=d_cost_GAN


                    d_costs.append(d_cost/discr_steps)

                    if self.cost_type=='WGAN':
                        d_costs_GAN.append(d_cost_GAN/discr_steps)

                    for i in range(gen_steps):

                        sample_z = np.zeros(shape=(bs, self.latent_dims))
                        #sample_z = np.random.normal(size=(bs, self.latent_dims))

                        _, g_cost, g_cost_cycl, g_cost_lr_GAN= self.session.run(
                            
                            (self.g_train_op, self.g_cost, self.g_cost_cycl, 
                            self.g_cost_lr_GAN),
                            
                            feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, 
                                        self.z:sample_z, self.batch_sz:bs
                                        },
                        )

                        g_cost+=g_cost
                        g_cost_cycl+=g_cost_cycl
                        g_cost_lr_GAN+=g_cost_lr_GAN

                    g_costs.append(g_cost/gen_steps)
                    g_costs_lr_GAN.append(g_cost_lr_GAN/gen_steps)
                    g_costs_cycl.append(self.cycl_weight*g_cost_cycl/gen_steps)

                                    
                    total_iters+=1
                    if total_iters % self.save_sample==0:
                        plt.clf()
                        print("At iter: %d  -  dt: %s - d_acc: %.2f" % (total_iters, datetime.now() - t0, d_acc))
                        print("Discriminator cost {0:.4g}, Generator cost {1:.4g}".format(d_cost, g_cost))
                        print('Saving a sample...')


                        if self.preprocess!=False:
                            draw_nn_sample(validating_A, validating_B, 1, self.preprocess,
                                            self.min_true, self.max_true, 
                                            self.min_reco, self.max_reco,
                                            f=self.get_sample_A_to_B, is_training=True,
                                            total_iters=total_iters, PATH=self.path+'/pretrained/')
                        else:
                            draw_nn_sample(validating_A, validating_B, 1, self.preprocess,
                                            f=self.get_sample_A_to_B, is_training=True,
                                            total_iters=total_iters, PATH=self.path+'/pretrained/')

                        plt.clf()
                        plt.subplot(1,3,1)
                        plt.plot(d_costs, label='Discriminator total cost')
                        plt.xlabel('Batch')
                        plt.ylabel('Cost')
                        plt.legend()

                        plt.subplot(1,3,2)
                        if self.cost_type=='WGAN':
                            plt.plot(d_costs_GAN, label='Discriminator GAN cost')
                        plt.plot(g_costs_lr_GAN, label='Generator GAN cost')
                        plt.xlabel('Batch')
                        plt.ylabel('Cost')
                        plt.legend()

                        plt.subplot(1,3,3)
                        plt.plot(g_costs, label='Generator total cost')
                        plt.plot(g_costs_cycl, label='Generator cyclic cost')
                        plt.xlabel('Batch')
                        plt.ylabel('Cost')
                        plt.legend()

                        fig = plt.gcf()
                        fig.set_size_inches(15,5)
                        plt.savefig(self.path+'/pretrained/cost_iteration.png',dpi=80)

        if not self.pretrain:

            print('\n ****** \n')
            print('Training bicycleGAN with a total of ' +str(N)+' samples distributed in '+ str((N)//self.batch_size) +' batches of size '+str(self.batch_size)+'\n')
            print('The validation set consists of {0} images'.format(validating_A.shape[0]))
            print('The learning rate is '+str(self.lr)+', and every ' +str(self.save_sample)+ ' batches a generated sample will be saved to '+ self.path)
            print('\n ****** \n')
            
            for epoch in range(self.epochs):

                seed+=1
                print('Epoch:', epoch)

                batches_A = unsupervised_random_mini_batches(train_A, self.batch_size, seed)
                batches_B = unsupervised_random_mini_batches(train_B, self.batch_size, seed)

                for X_batch_A, X_batch_B in zip(batches_A[:-1], batches_B[:-1]):

                    bs=X_batch_A.shape[0]

                    t0 = datetime.now()


                    e_cost=0
                    e_cost_latent_cycle=0
                    e_cost_kl=0

                    g_cost=0
                    g_cost_cycl=0
                    g_cost_lr_GAN=0
                    g_cost_vae_GAN=0
                    #cluster_diff=0

                    d_cost=0
                    d_cost_vae_GAN=0
                    d_cost_lr_GAN=0


                    for i in range(discr_steps):

                        sample_z = np.random.normal(size=(bs, self.latent_dims))*0.02

                        _, d_acc, d_cost, d_cost_vae_GAN, d_cost_lr_GAN = self.session.run(

                            (self.d_train_op, self.d_accuracy, self.d_cost, self.d_cost_fake_vae_GAN, self.d_cost_fake_lr_GAN),

                            feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, 
                                    self.z:sample_z, self.batch_sz:bs
                                    },
                        )
                        d_cost+=d_cost
                        d_cost_vae_GAN+=d_cost_vae_GAN
                        d_cost_lr_GAN+=d_cost_lr_GAN

                    d_costs.append(d_cost/discr_steps)
                    d_costs_vae_GAN.append(d_cost_vae_GAN/discr_steps)
                    d_costs_lr_GAN.append(d_cost_lr_GAN/discr_steps)

                    for i in range(gen_steps):

                        sample_z = np.random.normal(size=(bs, self.latent_dims))*0.02

                        _, _, e_cost, g_cost, g_cost_cycl, g_cost_lr_GAN, g_cost_vae_GAN, e_cost_latent_cycle, e_cost_kl  = self.session.run(
                            
                            (self.g_train_op, self.e_train_op, self.e_cost, self.g_cost, 
                            self.g_cost_cycl, self.g_cost_lr_GAN, self.g_cost_vae_GAN, self.e_cost_latent_cycle, self.e_cost_kl
                                ),
                            
                            feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, 
                                        self.z:sample_z, self.batch_sz:bs
                                        },
                        )

                        e_cost+=e_cost
                        e_cost_latent_cycle+=e_cost_latent_cycle
                        e_cost_kl+=e_cost_kl


                        g_cost+=g_cost
                        g_cost_cycl+=g_cost_cycl
                        g_cost_lr_GAN+=g_cost_lr_GAN
                        g_cost_vae_GAN+=g_cost_vae_GAN
                        #cluster_diff+=cluster_diff

                    e_costs.append(e_cost)
                    e_costs_latent_cycle.append(self.latent_weight*e_cost_latent_cycle/gen_steps)
                    e_costs_kl.append(self.kl_weight*e_cost_kl/gen_steps)

                    g_costs.append(g_cost)
                    g_costs_vae_GAN.append(g_cost_vae_GAN/gen_steps)
                    g_costs_lr_GAN.append(g_cost_lr_GAN/gen_steps)
                    g_costs_cycl.append(self.cycl_weight*g_cost_cycl/gen_steps)
                    #cluster_diffs.append(self.cycl_weight*cluster_diff/gen_steps)

                         
                    total_iters+=1
                    if total_iters % self.save_sample==0:
                        plt.clf()
                        print("At iter: %d  -  dt: %s - d_acc: %.2f" % (total_iters, datetime.now() - t0, d_acc))
                        print("Discriminator cost {0:.4g}, Generator cost {1:.4g}, VAE Cost {2:.4g}, KL divergence cost {3:.4g}".format(d_cost, g_cost, e_cost, e_cost_kl))
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
                        plt.subplot(2,4,1)
                        plt.plot(d_costs, label='Discriminator total cost')
                        plt.plot(d_costs_lr_GAN, label='Discriminator of image with encoded noise cost')
                        plt.plot(d_costs_vae_GAN, label='Discriminator of image with input noise cost')
                        plt.xlabel('Batch')
                        plt.ylabel('Cost')
                        plt.legend()

                        plt.subplot(2,4,2)
                        plt.plot(g_costs, label='Generator total cost')
                        plt.plot(g_costs_cycl, label='Generator cyclic cost')
                        #plt.plot(g_costs_GAN, label='GAN cost (encoded noise image)')
                        #plt.plot(g_costs_vae_GAN, label='GAN cost (input noise image)')
                        plt.xlabel('Batch')
                        plt.ylabel('Cost')
                        plt.legend()

                        plt.subplot(2,4,3)
                        plt.plot(e_costs, label='VAE cost')
                        #plt.plot(e_costs_kl, label='KL cost')
                        #plt.plot(e_costs_latent_cycle, label='Latent space cyclic cost')
                        plt.xlabel('Batch')
                        plt.ylabel('Cost')
                        plt.legend()


                        plt.subplot(2,4,6)
                        #plt.plot(g_costs, label='Generator cost')
                        #plt.plot(g_costs_cycl, label='Generator cyclic cost')
                        plt.plot(g_costs_lr_GAN, label='GAN cost (encoded noise image)')
                        plt.plot(g_costs_vae_GAN, label='GAN cost (input noise image)')
                        plt.xlabel('Batch')
                        plt.ylabel('Cost')
                        plt.legend()

                        plt.subplot(2,4,7)
                        plt.plot(e_costs_latent_cycle, label='Latent space cyclic cost')
                        plt.xlabel('Batch')
                        plt.ylabel('Cost')
                        plt.legend()

                        plt.subplot(2,4,8) 
                        plt.plot(e_costs_kl, label='KL cost')
                        #plt.plot(e_costs_latent_cycle, label='Latent space cyclic cost')
                        plt.xlabel('Batch')
                        plt.ylabel('Cost')
                        plt.legend()


                        fig = plt.gcf()
                        fig.set_size_inches(20,10)
                        plt.savefig(self.path+'/cost_iteration.png',dpi=80)

    def get_sample_A_to_B(self, X):


        z = np.random.normal(size=(1, self.latent_dims))*0.02

        one_sample = self.session.run(
            self.test_images_A_to_B, 
            feed_dict={self.input_test_A:X, self.z:z, self.batch_sz: 1})

        return one_sample 

    def get_samples_A_to_B(self, X):

        
        z = np.random.normal(size=(X.shape[0], self.latent_dims))*0.02
        
        many_samples = self.session.run(
            self.test_images_A_to_B, 
            feed_dict={self.input_test_A:X, self.z:z, self.batch_sz: X.shape[0]})

        return many_samples





