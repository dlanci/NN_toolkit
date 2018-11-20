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
EPS=1e-10
CYCL_WEIGHT=None
LATENT_WEIGHT=None
KL_WEIGHT=None
DISCR_STEPS=None
GEN_STEPS=None
max_true=None
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
        max_true=max_true, max_reco=max_reco,
        d_sizes=d_sizes, g_sizes_enc=g_sizes_enc, g_sizes_dec=g_sizes_dec, e_sizes=e_sizes, 
        lr=LEARNING_RATE, beta1=BETA1, preprocess=PREPROCESS,
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
        self.max_true=max_true
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


        G = bicycleGenerator(self.input_A, self.n_H_B, self.n_W_B, g_sizes_enc, g_sizes_dec, 'A_to_B')

        D = pix2pixDiscriminator(self.input_B, d_sizes, 'B')

        E = bicycleEncoder(self.input_B, e_sizes, 'B')


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
            logits_real = D.d_forward(self.input_A, self.input_B)

        with tf.variable_scope('discriminator_B') as scope:
            scope.reuse_variables()
            logits_fake = D.d_forward(self.input_A, sample_A_to_B, reuse=True)
            logits_fake_encoded = D.d_forward(self.input_A, sample_A_to_B_encoded, reuse=True)

        self.input_test_A = tf.placeholder(
                    tf.float32,
                    shape=(None, 
                           n_H_A, n_W_A, n_C),
                    name='X_test_A',
                )

        with tf.variable_scope('generator_A_to_B') as scope:
            scope.reuse_variables()
            self.test_images_A_to_B = G.g_forward(
                self.input_test_A, self.z, reuse=True, is_training=False
                )

        #parameters lists
        self.d_params =[t for t in tf.trainable_variables() if 'discriminator' in t.name]
        self.e_params =[t for t in tf.trainable_variables() if 'encoder' in t.name]
        self.g_params =[t for t in tf.trainable_variables() if 'generator' in t.name]
        
        D_real = tf.nn.sigmoid(logits_real)
        D_fake = tf.nn.sigmoid(logits_fake)
        D_fake_encoded = tf.nn.sigmoid(logits_fake_encoded)

        d_loss_vae_gan =tf.reduce_mean(tf.squared_difference(D_real, 0.9)) + tf.reduce_mean(tf.square(D_fake_encoded))
        d_loss_lr_gan = tf.reduce_mean(tf.squared_difference(D_real, 0.9)) + tf.reduce_mean(tf.square(D_fake))

        g_loss_vae_gan=tf.reduce_mean(tf.squared_difference(D_fake_encoded, 0.9))
        g_loss_gan = tf.reduce_mean(tf.squared_difference(D_fake, 0.9))
        g_loss_cycl=cycl_weight * tf.reduce_mean(tf.abs(self.input_B-sample_A_to_B_encoded))
         

        e_loss_latent_cycle = tf.reduce_mean(tf.abs(self.z - z_recon))
        self.e_loss_kl = 0.5 * tf.reduce_mean(-1 - 2*z_encoded_log_sigma + z_encoded_mu ** 2 + tf.exp(z_encoded_log_sigma)**2)

        self.d_loss = d_loss_vae_gan + d_loss_lr_gan - tf.reduce_mean(tf.squared_difference(D_real, 0.9))
        self.g_loss = g_loss_vae_gan + g_loss_gan + cycl_weight*g_loss_cycl + latent_weight*e_loss_latent_cycle
        self.e_loss = g_loss_vae_gan + cycl_weight *g_loss_cycl + latent_weight*e_loss_latent_cycle + kl_weight *self.e_loss_kl

        self.d_train_op = tf.train.AdamOptimizer(
                learning_rate=lr, 
                beta1=beta1,
                beta2=0.9,
            ).minimize(
                self.d_loss, 
                var_list=self.d_params
            )

        self.g_train_op = tf.train.AdamOptimizer(
                learning_rate=lr, 
                beta1=beta1,
                beta2=0.9,
            ).minimize(
                self.g_loss, 
                var_list=self.g_params
            )

        self.e_train_op = tf.train.AdamOptimizer(
                learning_rate=lr, 
                beta1=beta1,
                beta2=0.9,
            ).minimize(
                self.e_loss, 
                var_list=self.e_params
            )

        self.batch_size=batch_size
        self.epochs=epochs
        self.save_sample=save_sample
        self.path=path
        self.lr = lr

        self.D=D
        self.G=G
        self.E=E

        self.preprocess=preprocess
        self.cost_type=cost_type

        self.latent_weight=latent_weight
        self.cycl_weight=cycl_weight
        self.kl_weight=kl_weight
        self.latent_dims=latent_dims

    def set_session(self, session):

        self.session = session

        for layer in self.D.d_conv_layers:
            layer.set_session(session)

        for layer in self.G.g_enc_conv_layers:
            layer.set_session(session)

        for layer in self.G.g_dec_conv_layers:
            layer.set_session(session)

        for layer in self.E.e_blocks:
            layer.set_session(session)

        for layer in self.E.e_dense_layers:
            layer.set_session(session)

    def fit(self, X_A, X_B, validating_size):

        all_A = X_A
        all_B = X_B

        m = X_A.shape[0]
        train_A = all_A[0:m-validating_size]
        train_B = all_B[0:m-validating_size]

        validating_A = all_A[m-validating_size:m]
        validating_B = all_B[m-validating_size:m]

        seed=self.seed

        d_losses=[]
        g_losses=[]
        e_losses=[]
        e_loss_kls=[]

        N=len(train_A)
        n_batches = N // self.batch_size

        total_iters=0

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

            for X_batch_A, X_batch_B in zip(batches_A, batches_B):

                bs=X_batch_A.shape[0]

                t0 = datetime.now()

                sample_z = np.random.normal(size=(bs, self.latent_dims))

                _, _, _, e_loss, g_loss, d_loss, e_loss_kl = self.session.run(
                    
                    (self.d_train_op, self.g_train_op, self.e_train_op, self.e_loss, self.g_loss, self.d_loss, self.e_loss_kl),
                    
                    feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, 
                                self.z:sample_z, self.batch_sz:bs
                                },
                    )

                e_losses.append(e_loss)
                g_losses.append(g_loss)
                d_losses.append(d_loss)
                e_loss_kls.append(e_loss_kl)

                total_iters+=1
                if total_iters % self.save_sample==0:
                    print("At iter: %d  -  dt: %s " % (total_iters, datetime.now() - t0))
                    print("Discriminator cost {0:.4g}, Generator cost {1:.4g}, VAE Cost {2:.4g}, KL divergence loss {3:.4g}".format(d_loss, g_loss, e_loss, e_loss_kl))
                    print('Saving a sample...')


                    if self.preprocess=='normalise':

                        draw_nn_sample(validating_A, validating_B, 1, self.preprocess,
                            self.max_true, self.max_reco, f=self.get_sample_A_to_B, is_training=True,
                            total_iters=total_iters, PATH=self.path)
                    else:
                        draw_nn_sample(validating_A, validating_B, 1, self.preprocess,
                                        f=self.get_sample_A_to_B, is_training=True,
                                        total_iters=total_iters, PATH=self.path)

            plt.clf()
            plt.subplot(1,2,1)
            plt.plot(d_losses, label='Discriminator cost')
            plt.plot(g_losses, label='Generator cost')
            plt.plot(e_losses, label='VAE cost')
            plt.plot(e_loss_kls, label='Total cost')
            plt.xlabel('Epoch')
            plt.ylabel('Cost')
            plt.legend()


            fig = plt.gcf()
            fig.set_size_inches(15,5)
            plt.savefig(self.path+'/cost_iteration.png',dpi=150)



    def get_sample_A_to_B(self, X):


        z = np.random.normal(size=(1, self.latent_dims))

        one_sample = self.session.run(
            self.test_images_A_to_B, 
            feed_dict={self.input_test_A:X, self.z:z, self.batch_sz: 1})

        return one_sample 

    def get_samples_A_to_B(self, X):

        
        z = np.random.normal(size=(X.shape[0], self.latent_dims))
        
        many_samples = self.session.run(
            self.test_images_A_to_B, 
            feed_dict={self.input_test_A:X, self.z:z, self.batch_sz: X.shape[0]})

        return many_samples





