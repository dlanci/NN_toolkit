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
preprocess=None
LAMBDA=.01
EPS=1e-10
CYCL_WEIGHT=None
GAN_WEIGHT=None
DISCR_STEPS=None
GEN_STEPS=None
max_true=None
max_reco=None

n_H_A=None
n_W_A=None
n_W_B=None
n_H_B=None
n_C=None

d_sizes_A=None
d_sizes_B=None
g_sizes_A=None
g_sizes_B=None

class cycleGAN_fullresidual(object):
    
    def __init__(
        self, 
        n_H_A=n_H_A, n_W_A=n_W_A,
        n_H_B=n_H_B, n_W_B=n_W_B, n_C=n_C,
        max_true=max_true, max_reco=max_reco,
        d_sizes_A=d_sizes_A, d_sizes_B=d_sizes_B, g_sizes_A=g_sizes_A, g_sizes_B=g_sizes_B,
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

        D_A = Discriminator(self.input_A, d_sizes_A, 'A')
        D_B = Discriminator(self.input_B, d_sizes_B, 'B')

        G_A_to_B = cycleGenerator_fullresidual(self.input_A, self.n_H_B, self.n_W_B, g_sizes_A, 'A_to_B')
        G_B_to_A = cycleGenerator_fullresidual(self.input_B, self.n_H_A, self.n_W_A, g_sizes_B, 'B_to_A')
        

        #first cycle (A to B)
        with tf.variable_scope('discriminator_A') as scope:
            
            logits_A = D_A.d_forward(self.input_A)

        with tf.variable_scope('generator_A_to_B') as scope:

            sample_images_B = G_A_to_B.g_forward(self.input_A)

        #second cycle (B to A)
        with tf.variable_scope('discriminator_B') as scope:
            
            logits_B = D_B.d_forward(self.input_B)

        with tf.variable_scope('generator_B_to_A') as scope:

            sample_images_A = G_B_to_A.g_forward(self.input_B)


        with tf.variable_scope('discriminator_A') as scope:
            scope.reuse_variables()
            sample_logits_A = D_A.d_forward(sample_images_A, reuse=True)

        with tf.variable_scope('discriminator_B') as scope:
            scope.reuse_variables()
            sample_logits_B = D_B.d_forward(sample_images_B, reuse=True)


        with tf.variable_scope('generator_A_to_B') as scope:
            scope.reuse_variables()
            cycl_B = G_A_to_B.g_forward(sample_images_A, reuse=True)


        with tf.variable_scope('generator_B_to_A') as scope:
            scope.reuse_variables()
            cycl_A = G_B_to_A.g_forward(sample_images_B, reuse=True)


        self.input_test_A = tf.placeholder(
            tf.float32,
            shape=(None, 
                   n_H_A, n_W_A, n_C),
            name='X_test_A',
        )
        # get sample images for test time
        with tf.variable_scope('generator_A_to_B') as scope:
            scope.reuse_variables()
            self.sample_images_test_A_to_B = G_A_to_B.g_forward(
                self.input_test_A, reuse=True, is_training=False
            )

        self.input_test_B = tf.placeholder(
            tf.float32,
            shape=(None, 
                   n_H_B, n_W_B, n_C),
            name='X_test_B',
        )

        with tf.variable_scope('generator_B_to_A') as scope:
            scope.reuse_variables()
            self.sample_images_test_B_to_A = G_B_to_A.g_forward(
                self.input_test_B, reuse=True, is_training=False
            )
        #parameters lists
        self.d_params_A =[t for t in tf.trainable_variables() if 'discriminator_A' in t.name]
        self.d_params_B =[t for t in tf.trainable_variables() if 'discriminator_B' in t.name]

        self.g_params_A =[t for t in tf.trainable_variables() if 'B_to_A' in t.name]
        self.g_params_B =[t for t in tf.trainable_variables() if 'A_to_B' in t.name]
        
        #cost building

        if cost_type == 'GAN':
        
            #Discriminators cost

            #Discriminator_A cost
            #cost is low if real images are predicted as real (1) 
            d_cost_real_A = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_A,
                labels=tf.ones_like(logits_A)
            )
            # #cost is low if fake generated images are predicted as fake (0)
            d_cost_fake_A = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=sample_logits_A,
                labels=tf.zeros_like(sample_logits_A)
            )

            
            self.d_cost_A = tf.reduce_mean(d_cost_real_A) + tf.reduce_mean(d_cost_fake_A)
            
            #Discriminator_B cost
            d_cost_real_B = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_B,
                labels=tf.ones_like(logits_B)
            )
            
            d_cost_fake_B = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=sample_logits_B,
                labels=tf.zeros_like(sample_logits_B)
            )

            self.d_cost_B = tf.reduce_mean(d_cost_real_B) + tf.reduce_mean(d_cost_fake_B)
            
            #Generator cost 
            #cost is low if logits from discriminator A on samples generated by G_B_to_A 
            #are predicted as true (1)
            self.g_GAN_cost_A = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=sample_logits_A,
                    labels=tf.ones_like(sample_logits_A)
                )
            )
            #cost is low if logits from discriminator B on samples generated by G_A_to_B 
            #are predicted as true (1)
            self.g_GAN_cost_B = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=sample_logits_B,
                    labels=tf.ones_like(sample_logits_B)
                )
            )

            #cycle cost is low if cyclic images are similar to input images (in both sets)
            self.g_cycle_cost_A = tf.reduce_mean(tf.abs(self.input_A-cycl_A)) 
            self.g_cycle_cost_B = tf.reduce_mean(tf.abs(self.input_B-cycl_B))

            g_cycle_cost= self.g_cycle_cost_A+self.g_cycle_cost_B
            self.g_cost_A = gan_weight*self.g_GAN_cost_A + cycl_weight*g_cycle_cost
            self.g_cost_B = gan_weight*self.g_GAN_cost_B + cycl_weight*g_cycle_cost


        if cost_type == 'WGAN-gp':

            #Discriminators cost
            #Discriminator A

            self.d_cost_A = tf.reduce_mean(sample_logits_A) - tf.reduce_mean(logits_A)
            self.d_cost_B = tf.reduce_mean(sample_logits_B) - tf.reduce_mean(logits_B)

            self.g_cost_A = -tf.reduce_mean(sample_logits_A)
            self.g_cost_B = -tf.reduce_mean(sample_logits_B)

            # g_cycle_cost_A = tf.reduce_mean(tf.abs(self.input_A-cycl_A)) 
            # g_cycle_cost_B = tf.reduce_mean(tf.abs(self.input_B-cycl_B))

            # g_cycle_cost= g_cycle_cost_A+g_cycle_cost_B
            # self.g_cost_A = g_cost_A + 0*g_cycle_cost
            # self.g_cost_B = g_cost_B + 0*g_cycle_cost
            
            
            alpha_A = tf.random_uniform(
                shape=[self.batch_sz,self.n_H_A,self.n_W_A,self.n_C],
                minval=0.,
                maxval=1.
                )

            interpolates_A = alpha_A*self.input_A+(1-alpha_A)*sample_images_A

            with tf.variable_scope('discriminator_A') as scope:
                scope.reuse_variables()
                disc_A_interpolates = D_A.d_forward(interpolates_A,reuse = True)

            gradients_A = tf.gradients(disc_A_interpolates,[interpolates_A])[0]
            slopes_A = tf.sqrt(tf.reduce_sum(tf.square(gradients_A), reduction_indices=[1]))
            gradient_penalty_A = tf.reduce_mean((slopes_A-1)**2)
            self.d_cost_A+=LAMBDA*gradient_penalty_A

            alpha_B = tf.random_uniform(
                shape=[self.batch_sz,self.n_H_B,self.n_W_B,self.n_C],
                minval=0.,
                maxval=1.
                )

            interpolates_B = alpha_B*self.input_B+(1-alpha_B)*sample_images_B

            with tf.variable_scope('discriminator_B') as scope:
                scope.reuse_variables()
                disc_B_interpolates = D_B.d_forward(interpolates_B,reuse = True)

            gradients_B = tf.gradients(disc_B_interpolates,[interpolates_B])[0]
            slopes_B = tf.sqrt(tf.reduce_sum(tf.square(gradients_B), reduction_indices=[1]))
            gradient_penalty_B = tf.reduce_mean((slopes_B-1)**2)
            self.d_cost_B+=LAMBDA*gradient_penalty_B

        
        self.d_train_op_A = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
            beta2=0.9,
        ).minimize(
            self.d_cost_A,
            var_list=self.d_params_A
        )

        self.d_train_op_B = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
            beta2=0.9,
        ).minimize(
            self.d_cost_B,
            var_list=self.d_params_B
        )
        
        self.g_train_op_A = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
            beta2=0.9,
        ).minimize(
            self.g_cost_A,
            var_list=self.g_params_A
        )

        self.g_train_op_B = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
            beta2=0.9,
        ).minimize(
            self.g_cost_B,
            var_list=self.g_params_B
        )
        #Measure accuracy of the discriminators
        
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

        self.batch_size=batch_size
        self.epochs=epochs
        self.save_sample=save_sample
        self.path=path
        self.lr = lr

        self.D_A=D_A
        self.D_B=D_B
        self.G_A_to_B=G_A_to_B
        self.G_B_to_A=G_B_to_A

        self.sample_images_B=sample_images_B
        self.sample_images_A=sample_images_A

        self.preprocess=preprocess
        self.cost_type=cost_type

        self.gen_steps=gen_steps
        self.discr_steps=discr_steps
        self.cycl_weight=cycl_weight
    
    def set_session(self, session):
        
        self.session = session
        
        for block in self.D_A.d_conv_layers:
            block.set_session(session)
                
        for layer in self.D_A.d_dense_layers:
            layer.set_session(session)

        for layer in self.G_A_to_B.g_blocks:
            layer.set_session(session)

        for block in self.D_B.d_conv_layers:
            block.set_session(session)
                
        for layer in self.D_B.d_dense_layers:
            layer.set_session(session)
        
        for block in self.G_B_to_A.g_blocks:
            block.set_session(session)
                    
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

        seed = self.seed

        d_costs_A=[]
        d_gps_A=[]
        g_GANs_A=[]
        g_cycles_A=[]
        g_costs_A=[]

        d_costs_B=[]
        d_gps_B=[]
        g_GANs_B=[]
        g_cycles_B=[]
        g_costs_B=[]

        N = len(train_A)
        n_batches = N // self.batch_size
    
        total_iters=0

        print('\n ****** \n')
        print('Training cycle GAN with residual arichitecture and a total of ' +str(N)+' samples distributed in '+ str((N)//self.batch_size) +' batches of size '+str(self.batch_size)+'\n')
        print('The validation set consists of {0} images'.format(validating_A.shape[0]))
        print('The learning rate set is '+str(self.lr)+', and every ' +str(self.save_sample)+ ' batches a generated sample will be saved to '+ self.path)
        print('\n ****** \n')

        for epoch in range(self.epochs):

            seed +=1

            print('Epoch:', epoch)
            
            batches_A = unsupervised_random_mini_batches(X_A, self.batch_size, seed)
            batches_B = unsupervised_random_mini_batches(X_B, self.batch_size, seed)

            for X_batch_A, X_batch_B in zip(batches_A,batches_B):
                
                bs = X_batch_A.shape[0]

                t0 = datetime.now()
                
                #optimize generator_A
                g_cost_A=0
                g_GAN_A=0
                g_cycle_A=0

                for i in range(gen_steps):

                    _, g_cost_A, g_GAN_A, g_cycle_A =  self.session.run(
                        (self.g_train_op_A, self.g_cost_A, self.g_GAN_cost_A, self.g_cycle_cost_A),
                        feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:bs},
                    )
                    g_cost_A+=g_cost_A
                    g_GAN_A+=g_GAN_A
                    g_cycle_A+=g_cycle_A
                
                g_costs_A.append(g_cost_A/gen_steps)
                g_GANs_A.append(g_GAN_A/gen_steps)
                g_cycles_A.append(self.cycl_weight*g_cycle_A/gen_steps)     


                #optimize discriminator_B
                d_cost_B=0
                d_gp_B=0
                for i in range(discr_steps):

                    if self.cost_type=='WGAN-gp':
                        _, d_cost_B, d_acc_B, d_gp_B = self.session.run(
                        (self.d_train_op_B, self.d_cost_B, self.d_accuracy_B, self.gradient_penalty_B),
                        feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:bs},
                        )

                        d_gp_B+=d_gp_B

                    else:

                        _, d_cost_B, d_acc_B, = self.session.run(
                            (self.d_train_op_B, self.d_cost_B, self.d_accuracy_B),
                            feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:bs},
                        )

                    d_cost_B+=d_cost_B

                d_costs_B.append(d_cost_B/discr_steps)
                d_gps_B.append(LAMBDA*d_gp_B/discr_steps)

                #optimize generator_B
                g_cost_B=0
                g_GAN_B=0
                g_cycle_B=0
                for i in range(gen_steps):
                    _, g_cost_B, g_GAN_B, g_cycle_B =  self.session.run(
                        (self.g_train_op_B, self.g_cost_B, self.g_GAN_cost_B, self.g_cycle_cost_B),
                        feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:bs},
                    )
                    g_cost_B+=g_cost_B
                    g_GAN_B+=g_GAN_B
                    g_cycle_B+=g_cycle_B

                g_costs_B.append(g_cost_B/gen_steps)
                g_GANs_B.append(g_GAN_B/gen_steps)
                g_cycles_B.append(self.cycl_weight*g_cycle_B/gen_steps) 

                d_cost_A=0
                d_gp_A=0
                for i in range(discr_steps):
                    #optimize Discriminator_A 
                    if self.cost_type=='WGAN-gp':
                        _, d_cost_A, d_acc_A, d_gp_A = self.session.run(
                        (self.d_train_op_A, self.d_cost_A, self.d_accuracy_A, self.gradient_penalty_A),
                        feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:bs},
                        )

                        d_gp_A+=d_gp_A


                    else:

                        _, d_cost_A, d_acc_A, = self.session.run(
                            (self.d_train_op_A, self.d_cost_A, self.d_accuracy_A),
                            feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:bs},
                        )

                    d_cost_A+=d_cost_A

                d_costs_A.append(d_cost_A/discr_steps)
                d_gps_A.append(LAMBDA*d_gp_A/discr_steps)

                total_iters += 1
                if total_iters % self.save_sample ==0:
                    print("At iter: %d  -  dt: %s - d_acc_A: %.2f" % (total_iters, datetime.now() - t0, d_acc_A))
                    print("At iter: %d  -  dt: %s - d_acc_B: %.2f" % (total_iters, datetime.now() - t0, d_acc_B))
                    print("Discrimator_A cost {0:.4g}, Generator_A_to_B cost {1:.4g}".format(d_cost_A, g_cost_A))
                    print("Discrimator_B cost {0:.4g}, Generator_B_to_A cost {1:.4g}".format(d_cost_B, g_cost_B))
                    print('Saving a sample...')
                    
                    #A is true
                    #B is reco
                    
                    if self.preprocess=='normalise':
                        draw_nn_sample(validating_A, validating_B, 1, self.preprocess,
                                        self.max_true, self.max_reco,
                                        f=self.get_sample_A_to_B, is_training=True,
                                        total_iters=total_iters, PATH=self.path)
                    else:
                        draw_nn_sample(validating_A, validating_B, 1, self.preprocess,
                                        f=self.get_sample_A_to_B, is_training=True,
                                        total_iters=total_iters, PATH=self.path)

            plt.clf()
            plt.subplot(1,2,1)
            plt.plot(d_costs_A, label='Discriminator A cost')
            plt.plot(d_gps_A, label='Gradient penalty A')
            plt.plot(g_GANs_A, label='Generator B to A GAN cost')
            plt.xlabel('Epoch')
            plt.ylabel('Cost')
            plt.legend()
            
            plt.subplot(1,2,2)
            plt.plot(g_costs_A, label='Generator B to A total cost')
            plt.plot(g_GANs_A, label='Generator B to A GAN cost')
            plt.plot(g_cycles_B, label='Generators B to A to B cycle cost')
            plt.xlabel('Epoch')
            plt.ylabel('Cost')
            plt.legend()

            fig = plt.gcf()
            fig.set_size_inches(15,5)
            plt.savefig(self.path+'/cost_iteration_gen_disc_B_to_A.png',dpi=150)

            plt.clf()
            plt.subplot(1,2,1)
            plt.plot(d_costs_B, label='Discriminator B cost')
            plt.plot(d_gps_B, label='Gradient penalty B')
            plt.plot(g_GANs_B, label='Generator A to B GAN cost')
            plt.xlabel('Epoch')
            plt.ylabel('Cost')
            plt.legend()

            plt.subplot(1,2,2)
            plt.plot(g_costs_B, label='Generator A to B total cost')
            plt.plot(g_GANs_B, label='Generator A to B GAN cost')
            plt.plot(g_cycles_B, label='Generators B to A to B cycle cost')
            plt.xlabel('Epoch')
            plt.ylabel('Cost')
            plt.legend()

            fig = plt.gcf()
            fig.set_size_inches(15,5)
            plt.savefig(self.path+'/cost_iteration_gen_disc_A_to_B.png',dpi=150)
    


    def get_sample_A_to_B(self, Z):
        
        one_sample = self.session.run(
            self.sample_images_test_A_to_B, 
            feed_dict={self.input_test_A:Z, self.batch_sz: 1})

        return one_sample 

    def get_sample_B_to_A(self, Z):
        
        one_sample = self.session.run(
            self.sample_images_test_B_to_A, 
            feed_dict={self.input_test_B:Z, self.batch_sz: 1})

        return one_sample 

    def get_samples_A_to_B(self, Z):
        
        many_samples = self.session.run(
            self.sample_images_test_A_to_B, 
            feed_dict={self.input_test_A:Z, self.batch_sz: Z.shape[0]})

        return many_samples 

    def get_samples_B_to_A(self, Z):
        
        many_samples = self.session.run(
            self.sample_images_test_B_to_A, 
            feed_dict={self.input_test_B:Z, self.batch_sz: Z.shape[0]})

        return many_samples 

