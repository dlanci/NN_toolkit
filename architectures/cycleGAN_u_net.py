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

mean_A=None
std_A=None

mean_B=None
std_B=None

n_H_A=None
n_W_A=None
n_W_B=None
n_H_B=None
n_C=None

d_sizes_A=None
d_sizes_B=None
g_sizes_enc_A=None
g_sizes_dec_A=None
g_sizes_enc_B=None
g_sizes_dec_B=None

class cycleGAN_u_net(object):
    
    def __init__(
        self, 
        n_H_A=n_H_A, n_W_A=n_W_A,
        n_H_B=n_H_B, n_W_B=n_W_B, n_C=n_C,
        mean_A=mean_A, std_A=std_A, 
        mean_B=mean_B, std_B=std_B,
        d_sizes_A=d_sizes_A, d_sizes_B=d_sizes_B, 
        g_sizes_enc_A=g_sizes_enc_A, g_sizes_dec_A=g_sizes_dec_A,
        g_sizes_enc_B=g_sizes_enc_B, g_sizes_dec_B=g_sizes_dec_B,
        lr=LEARNING_RATE, beta1=BETA1, preprocess=preprocess,
        cost_type=COST_TYPE, gan_weight=GAN_WEIGHT, cycl_weight=CYCL_WEIGHT,
        discr_steps=DISCR_STEPS, gen_steps=GEN_STEPS,
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

        self.mean_A=mean_A
        self.std_A=std_A
        self.mean_B=mean_B
        self.std_B=std_B
        
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

        # get sample images for test time
        self.input_test_A = tf.placeholder(
            tf.float32,
            shape=(None, 
                   n_H_A, n_W_A, n_C),
            name='X_test_A',
        )
        self.input_test_B = tf.placeholder(
            tf.float32,
            shape=(None, 
                   n_H_B, n_W_B, n_C),
            name='X_test_B',
        )


        D_A = Discriminator(self.input_A, d_sizes_A, 'A')
        D_B = Discriminator(self.input_B, d_sizes_B, 'B')

        G_A_to_B = pix2pixGenerator(self.input_A, self.n_H_B, self.n_W_B, g_sizes_enc_A, g_sizes_dec_A, 'A_to_B')
        G_B_to_A = pix2pixGenerator(self.input_B, self.n_H_A, self.n_W_A, g_sizes_enc_B, g_sizes_dec_B, 'B_to_A')
        

        # A -> B'
        with tf.variable_scope('generator_A_to_B') as scope:

            sample_images_B = G_A_to_B.g_forward(self.input_A)

        # B -> A'
        with tf.variable_scope('generator_B_to_A') as scope:

            sample_images_A = G_B_to_A.g_forward(self.input_B)

        # B' -> A'
        with tf.variable_scope('generator_B_to_A') as scope:
            scope.reuse_variables()
            cycl_A = G_B_to_A.g_forward(sample_images_B, reuse=True)

        with tf.variable_scope('generator_B_to_A') as scope:
            scope.reuse_variables()
            self.sample_images_test_A = G_B_to_A.g_forward(
                self.input_test_B, reuse=True, is_training=False
            )

        # A' -> B'
        with tf.variable_scope('generator_A_to_B') as scope:
            scope.reuse_variables()
            cycl_B = G_A_to_B.g_forward(sample_images_A, reuse=True)

        with tf.variable_scope('generator_A_to_B') as scope:
            scope.reuse_variables()
            self.sample_images_test_B = G_A_to_B.g_forward(
                self.input_test_A, reuse=True, is_training=False
            )

        #Discriminator of images os set B

        with tf.variable_scope('discriminator_B') as scope:
            logits_real_B = D_B.d_forward(self.input_B)     
            #logits_real_B = D_B.d_forward(self.input_A, self.input_B)
        
        with tf.variable_scope('discriminator_B') as scope:
            scope.reuse_variables()
            logits_fake_B = D_B.d_forward(cycl_B, reuse=True)
            #logits_fake_B = D_B.d_forward(self.input_A, cycl_B, reuse=True)

        #Discriminator of images os set A

        with tf.variable_scope('discriminator_A') as scope:
            logits_real_A = D_A.d_forward(self.input_A)
            #logits_real_A = D_A.d_forward(self.input_B, self.input_A)

        with tf.variable_scope('discriminator_A') as scope:
            scope.reuse_variables()
            logits_fake_A = D_A.d_forward(cycl_A, reuse=True)
            #logits_fake_A = D_A.d_forward(self.input_B, sample_images_A, reuse=True)


        #parameters lists
        self.d_params_A =[t for t in tf.trainable_variables() if 'discriminator_A' in t.name]
        self.d_params_B =[t for t in tf.trainable_variables() if 'discriminator_B' in t.name]

        self.d_params = [t for t in tf.trainable_variables() if 'discriminator' in t.name]

        self.g_params_A =[t for t in tf.trainable_variables() if 'B_to_A' in t.name]
        self.g_params_B =[t for t in tf.trainable_variables() if 'A_to_B' in t.name]
        
        self.g_params = [t for t in tf.trainable_variables() if 'generator' in t.name]

        predicted_fake_A=tf.nn.sigmoid(logits_fake_A)
        predicted_real_A=tf.nn.sigmoid(logits_real_A)

        predicted_fake_B=tf.nn.sigmoid(logits_fake_B)
        predicted_real_B=tf.nn.sigmoid(logits_real_B)


        #cost building
        epsilon = 1e-4
        if cost_type == 'GAN':

            self.d_cost_real_A = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_real_A,
                labels=(1-epsilon)*tf.ones_like(logits_real_A)
            )
            
            self.d_cost_fake_A = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_fake_A,
                labels=epsilon+tf.zeros_like(logits_fake_A)
            )
            
            self.d_cost_A = tf.reduce_mean(self.d_cost_real_A) + tf.reduce_mean(self.d_cost_fake_A)
            
            #Generator cost
            self.g_cost_GAN_A = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_fake_A,
                    labels=(1-epsilon)*tf.ones_like(logits_fake_A)
                )
            )

            # #Discriminator cost
            # self.d_cost_real_A = tf.reduce_mean(-tf.log(predicted_real_A + epsilon))
            # self.d_cost_fake_A = tf.reduce_mean(-tf.log(1 + epsilon - predicted_fake_A))
            # self.d_cost_A = self.d_cost_real_A+self.d_cost_fake_A
            
            # #Generator cost
            # self.g_cost_GAN_A = tf.reduce_mean(-tf.log(predicted_fake_A + epsilon))


            self.d_cost_real_B = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_real_B,
                labels=(1-epsilon)*tf.ones_like(logits_real_B)
            )
            
            self.d_cost_fake_B = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_fake_B,
                labels=epsilon+tf.zeros_like(logits_fake_B)
            )
            
            self.d_cost_B = tf.reduce_mean(self.d_cost_real_B) + tf.reduce_mean(self.d_cost_fake_B)
            
            #Generator cost
            self.g_cost_GAN_B = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_fake_B,
                    labels=(1-epsilon)*tf.ones_like(logits_fake_B)
                )
            )

            self.d_cost=(self.d_cost_B+self.d_cost_A)/2.

            # #Discriminator cost
            # self.d_cost_real_B = tf.reduce_mean(-tf.log(predicted_real_B + epsilon))
            # self.d_cost_fake_B = tf.reduce_mean(-tf.log(1 + epsilon - predicted_fake_B))
            # self.d_cost_B = self.d_cost_real_B+self.d_cost_fake_B
            
            # #Generator cost
            # self.g_cost_GAN_B = tf.reduce_mean(-tf.log(predicted_fake_B + epsilon))
            
            #cycle cost is low if cyclic images are similar to input images (in both sets)
            self.g_cost_cycle_A = tf.reduce_mean(tf.abs(self.input_A-cycl_A)) 
            self.g_cost_cycle_B = tf.reduce_mean(tf.abs(self.input_B-cycl_B))


            g_cost_cycle= self.g_cost_cycle_A+self.g_cost_cycle_B
            self.g_cost_A = gan_weight*self.g_cost_GAN_A + cycl_weight*g_cost_cycle
            self.g_cost_B = gan_weight*self.g_cost_GAN_B + cycl_weight*g_cost_cycle

            self.g_cost=gan_weight*(self.g_cost_GAN_A+self.g_cost_GAN_B)+cycl_weight*g_cost_cycle

        self.d_train_op_A = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
        ).minimize(
            self.d_cost_A,
            var_list=self.d_params_A
        )

        self.d_train_op_B = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
        ).minimize(
            self.d_cost_B,
            var_list=self.d_params_B
        )
        self.d_train_op = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
        ).minimize(
            self.d_cost,
            var_list=self.d_params#_B+self.d_params_A
        )
        
        self.g_train_op_A = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
        ).minimize(
            self.g_cost_A,
            var_list=self.g_params_A#+self.g_params_B
        )

        self.g_train_op_B = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
        ).minimize(
            self.g_cost_B,
            var_list=self.g_params_B#+self.g_params_A
        )
        self.g_train_op = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
        ).minimize(
            self.g_cost,
            var_list=self.g_params#_B+self.g_params_A
        )
        
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
        
        for layer in self.D_A.d_conv_layers:
            layer.set_session(session)

        for layer in self.G_A_to_B.g_enc_conv_layers:
            layer.set_session(session)

        for layer in self.G_A_to_B.g_dec_conv_layers:
            layer.set_session(session)

        for layer in self.D_B.d_conv_layers:
            layer.set_session(session)

        for layer in self.G_B_to_A.g_enc_conv_layers:
            layer.set_session(session)

        for layer in self.G_B_to_A.g_dec_conv_layers:
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

        g_costs=[]
        d_costs=[]

        d_costs_A=[]
        g_GANs_A=[]
        g_cycles_A=[]
        g_costs_A=[]

        d_costs_B=[]
        g_GANs_B=[]
        g_cycles_B=[]
        g_costs_B=[]

        N=len(train_A)
        n_batches = N // self.batch_size

        total_iters=0

        print('\n ****** \n')
        print('Training cycle GAN with pix2pix gen/disc with a total of ' +str(N)+' samples distributed in '+ str((N)//self.batch_size) +' batches of size '+str(self.batch_size)+'\n')
        print('The validation set consists of {0} images'.format(validating_A.shape[0]))
        print('The learning rate is '+str(self.lr)+', and every ' +str(self.save_sample)+ ' batches a generated sample will be saved to '+ self.path)
        print('\n ****** \n')

        for epoch in range(self.epochs):

            seed +=1

            print('Epoch:', epoch)

            batches_A = unsupervised_random_mini_batches(train_A, self.batch_size, seed)
            batches_B = unsupervised_random_mini_batches(train_B, self.batch_size, seed)

            for X_batch_A, X_batch_B in zip(batches_A,batches_B):

                bs = X_batch_A.shape[0]

                t0 = datetime.now()
                
                #optimize generator_A
                g_cost=0
                g_cost_A=0
                g_GAN_A=0
                g_cycle_A=0

                g_cost_B=0
                g_GAN_B=0
                g_cycle_B=0

                for i in range(gen_steps):

                    _, g_cost, g_cost_A, g_cost_B, g_GAN_A, g_GAN_B, g_cycle_A, g_cycle_B =  self.session.run(
                        (self.g_train_op, self.g_cost, self.g_cost_A, self.g_cost_B,
                                          self.g_cost_GAN_A, self.g_cost_GAN_B,
                                          self.g_cost_cycle_A, self.g_cost_cycle_B),

                        feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:bs},
                    )
                    g_cost+=g_cost
                    g_cost_A+=g_cost_A
                    g_cost_B+=g_cost_B

                    g_GAN_A+=g_GAN_A
                    g_GAN_B+=g_GAN_B

                    g_cycle_A+=g_cycle_A
                    g_cycle_B+=g_cycle_B
                
                g_costs.append(g_cost)

                g_costs_A.append(g_cost_A/gen_steps)
                g_costs_B.append(g_cost_B/gen_steps)

                g_GANs_A.append(g_GAN_A/gen_steps)
                g_GANs_B.append(g_GAN_B/gen_steps)

                g_cycles_A.append(self.cycl_weight*g_cycle_A/gen_steps)
                g_cycles_B.append(self.cycl_weight*g_cycle_B/gen_steps)   


                #optimize discriminator_B
                d_cost=0
                d_cost_A=0
                d_cost_B=0
                for i in range(discr_steps):

                    _, d_cost, d_cost_A, d_cost_B, = self.session.run(
                        (self.d_train_op, self.d_cost, self.d_cost_A, self.d_cost_B),
                        feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:bs},
                    )
                    d_cost+=d_cost
                    d_cost_A+=d_cost_A
                    d_cost_B+=d_cost_B

                d_costs.append(d_cost/discr_steps)
                d_costs_B.append(d_cost_B/discr_steps)
                d_costs_A.append(d_cost_A/discr_steps)

                total_iters += 1
                if total_iters % self.save_sample ==0:
                    plt.clf()
                    print("At iter: %d  -  dt: %s " % (total_iters, datetime.now() - t0))
                    print("At iter: %d  -  dt: %s " % (total_iters, datetime.now() - t0))
                    print("Discrimator_A cost {0:.4g}, Generator_B_to_A cost {1:.4g}".format(d_cost_A, g_cost_A))
                    print("Discrimator_B cost {0:.4g}, Generator_A_to_B cost {1:.4g}".format(d_cost_B, g_cost_B))
                    print('Saving a sample...')
                    
                    
                    X_A = validating_A[np.random.randint(validating_size)].reshape(1, self.n_H_A, self.n_W_A, self.n_C)
                    X_B = validating_B[np.random.randint(validating_size)].reshape(1, self.n_H_B, self.n_W_B, self.n_C)

                    test_sample_B=self.get_sample_A_to_B(X_A)
                    test_sample_A=self.get_sample_B_to_A(X_B)

                    if preprocess:

                        test_sample_B_1 = test_sample_B*self.std_B
                        test_sample_B_2 = test_sample_B_1+self.mean_B
                        test_sample_B=test_sample_B_2

                        test_sample_A_1 = test_sample_A*self.std_A
                        test_sample_A_2 = test_sample_A_1+self.mean_A
                        test_sample_A=test_sample_A_2

                    plt.clf()
                    plt.subplot(1,2,1)
                    plt.imshow(X_A.reshape(self.n_H_A,self.n_W_A, self.n_C))
                    plt.axis('off')
                    plt.subplots_adjust(wspace=0.2,hspace=0.2)

                    plt.subplot(1,2,2)
                    plt.imshow(test_sample_B.reshape(self.n_H_A,self.n_W_A, self.n_C))
                    plt.axis('off')
                    plt.subplots_adjust(wspace=0.2,hspace=0.2)
                    plt.savefig(self.path+"/B_to_A_{0}.png".format(total_iters), dpi=80)

                    plt.clf()
                    plt.subplot(1,2,1)
                    plt.imshow(X_B.reshape(self.n_H_B,self.n_W_B, self.n_C))
                    plt.axis('off')
                    plt.subplots_adjust(wspace=0.2,hspace=0.2)

                    plt.subplot(1,2,2)
                    plt.imshow(test_sample_A.reshape(self.n_H_B,self.n_W_B, self.n_C))
                    plt.axis('off')
                    plt.subplots_adjust(wspace=0.2,hspace=0.2)
                    plt.savefig(self.path+"/A_to_B_{0}.png".format(total_iters), dpi=80)

                    plt.clf()
                    plt.subplot(1,2,1)
                    plt.plot(d_costs_A, label='Discriminator A cost')
                    plt.plot(g_GANs_A, label='Generator B to A GAN cost')
                    plt.xlabel('Iteration')
                    plt.ylabel('Cost')
                    plt.legend()
                    
                    plt.subplot(1,2,2)
                    plt.plot(g_costs_A, label='Generator B to A total cost')
                    plt.plot(g_GANs_A, label='Generator B to A GAN cost')
                    plt.plot(g_cycles_B, label='Generators B to A to B cycle cost')
                    plt.xlabel('Iteration')
                    plt.ylabel('Cost')
                    plt.legend()

                    fig = plt.gcf()
                    fig.set_size_inches(15,5)
                    plt.savefig(self.path+'/cost_iteration_gen_disc_B_to_A.png',dpi=150)

                    plt.clf()
                    plt.subplot(1,2,1)
                    plt.plot(d_costs_B, label='Discriminator B cost')
                    plt.plot(g_GANs_B, label='Generator A to B GAN cost')
                    plt.xlabel('Iteration')
                    plt.ylabel('Cost')
                    plt.legend()

                    plt.subplot(1,2,2)
                    plt.plot(g_costs_B, label='Generator A to B total cost')
                    plt.plot(g_GANs_B, label='Generator A to B GAN cost')
                    plt.plot(g_cycles_B, label='Generators B to A to B cycle cost')
                    plt.xlabel('Iteration')
                    plt.ylabel('Cost')
                    plt.legend()

                    fig = plt.gcf()
                    fig.set_size_inches(15,5)
                    plt.savefig(self.path+'/cost_iteration_gen_disc_A_to_B.png',dpi=150)

        return test_sample_A, test_sample_B                          

    def get_sample_A_to_B(self, Z):
        
        one_sample = self.session.run(
            self.sample_images_test_B, 
            feed_dict={self.input_test_A:Z, self.batch_sz: 1})

        return one_sample 

    def get_sample_B_to_A(self, Z):
        
        one_sample = self.session.run(
            self.sample_images_test_A, 
            feed_dict={self.input_test_B:Z, self.batch_sz: 1})

        return one_sample 
    def get_samples_A_to_B(self, Z):
        
        many_samples = self.session.run(
            self.sample_images_test_B, 
            feed_dict={self.input_test_A:Z, self.batch_sz: Z.shape[0]})

        return many_samples 

    def get_samples_B_to_A(self, Z):
        
        many_samples = self.session.run(
            self.sample_images_test_A, 
            feed_dict={self.input_test_B:Z, self.batch_sz: Z.shape[0]})

        return many_samples 




    # def fit(self, X_A, X_B, validating_size):

    #     all_A = X_A
    #     all_B = X_B

    #     gen_steps = self.gen_steps
    #     discr_steps = self.discr_steps

    #     m = X_A.shape[0]
    #     train_A = all_A[0:m-validating_size]
    #     train_B = all_B[0:m-validating_size]

    #     validating_A = all_A[m-validating_size:m]
    #     validating_B = all_B[m-validating_size:m]

    #     seed=self.seed

    #     d_costs_A=[]
    #     g_GANs_A=[]
    #     g_cycles_A=[]
    #     g_costs_A=[]

    #     d_costs_B=[]
    #     g_GANs_B=[]
    #     g_cycles_B=[]
    #     g_costs_B=[]

    #     N=len(train_A)
    #     n_batches = N // self.batch_size

    #     total_iters=0

    #     print('\n ****** \n')
    #     print('Training cycle GAN with pix2pix gen/disc with a total of ' +str(N)+' samples distributed in '+ str((N)//self.batch_size) +' batches of size '+str(self.batch_size)+'\n')
    #     print('The validation set consists of {0} images'.format(validating_A.shape[0]))
    #     print('The learning rate is '+str(self.lr)+', and every ' +str(self.save_sample)+ ' batches a generated sample will be saved to '+ self.path)
    #     print('\n ****** \n')

    #     for epoch in range(self.epochs):

    #         seed +=1

    #         print('Epoch:', epoch)

    #         batches_A = unsupervised_random_mini_batches(train_A, self.batch_size, seed)
    #         batches_B = unsupervised_random_mini_batches(train_B, self.batch_size, seed)

    #         for X_batch_A, X_batch_B in zip(batches_A,batches_B):

    #             bs = X_batch_A.shape[0]

    #             t0 = datetime.now()
                
    #             #optimize generator_A
    #             g_cost_A=0
    #             g_GAN_A=0
    #             g_cycle_A=0
    #             for i in range(gen_steps):

    #                 _, g_cost_A, g_GAN_A, g_cycle_A =  self.session.run(
    #                     (self.g_train_op_A, self.g_cost_A, self.g_cost_GAN_A, self.g_cost_cycle_A),
    #                     feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:bs},
    #                 )
    #                 g_cost_A+=g_cost_A
    #                 g_GAN_A+=g_GAN_A
    #                 g_cycle_A+=g_cycle_A
                
    #             g_costs_A.append(g_cost_A/gen_steps)
    #             g_GANs_A.append(g_GAN_A/gen_steps)
    #             g_cycles_A.append(self.cycl_weight*g_cycle_A/gen_steps)   


    #             #optimize discriminator_B
    #             d_cost_B=0
    #             for i in range(discr_steps):

    #                 _, d_cost_B, = self.session.run(
    #                     (self.d_train_op_B, self.d_cost_B),
    #                     feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:bs},
    #                 )
    #                 d_cost_B+=d_cost_B

    #             d_costs_B.append(d_cost_B/discr_steps)

    #             #optimize generator_B
    #             g_cost_B=0
    #             g_GAN_B=0
    #             g_cycle_B=0
    #             for i in range(gen_steps):
    #                 _, g_cost_B, g_GAN_B, g_cycle_B =  self.session.run(
    #                     (self.g_train_op_B, self.g_cost_B, self.g_cost_GAN_B, self.g_cost_cycle_B),
    #                     feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:bs},
    #                 )
    #                 g_cost_B+=g_cost_B
    #                 g_GAN_B+=g_GAN_B
    #                 g_cycle_B+=g_cycle_B

    #             g_costs_B.append(g_cost_B/gen_steps)
    #             g_GANs_B.append(g_GAN_B/gen_steps)
    #             g_cycles_B.append(self.cycl_weight*g_cycle_B/gen_steps)  

    #             #optimize Discriminator_A 

    #             d_cost_A=0
    #             for i in range(discr_steps):
                    
    #                 _, d_cost_A,  = self.session.run(
    #                     (self.d_train_op_A, self.d_cost_A),
    #                     feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:bs},
    #                 )

    #                 d_cost_A+=d_cost_A

    #             d_costs_A.append(d_cost_A/discr_steps)

    #             total_iters += 1
    #             if total_iters % self.save_sample ==0:
    #                 plt.clf()
    #                 print("At iter: %d  -  dt: %s " % (total_iters, datetime.now() - t0))
    #                 print("At iter: %d  -  dt: %s " % (total_iters, datetime.now() - t0))
    #                 print("Discrimator_A cost {0:.4g}, Generator_B_to_A cost {1:.4g}".format(d_cost_A, g_cost_A))
    #                 print("Discrimator_B cost {0:.4g}, Generator_A_to_B cost {1:.4g}".format(d_cost_B, g_cost_B))
    #                 print('Saving a sample...')
                    
    #                 #A is apples?
    #                 #B is oranges?
                    
    #                 X_A = validating_A[np.random.randint(validating_size)].reshape(1, self.n_H_A, self.n_W_A, self.n_C)
    #                 X_B = validating_B[np.random.randint(validating_size)].reshape(1, self.n_H_B, self.n_W_B, self.n_C)

    #                 test_sample_B=self.get_sample_A_to_B(X_A)
    #                 test_sample_A=self.get_sample_B_to_A(X_B)

    #                 if preprocess:

    #                     test_sample_B_1 = test_sample_B*self.std_B
    #                     test_sample_B_2 = test_sample_B_1+self.mean_B
    #                     test_sample_B=test_sample_B_2

    #                     test_sample_A_1 = test_sample_A*self.std_A
    #                     test_sample_A_2 = test_sample_A_1+self.mean_A
    #                     test_sample_A=test_sample_A_2

    #                 plt.clf()
    #                 plt.subplot(1,2,1)
    #                 plt.imshow(X_A.reshape(self.n_H_A,self.n_W_A, self.n_C))
    #                 plt.axis('off')
    #                 plt.subplots_adjust(wspace=0.2,hspace=0.2)

    #                 plt.subplot(1,2,2)
    #                 plt.imshow(test_sample_B.reshape(self.n_H_A,self.n_W_A, self.n_C))
    #                 plt.axis('off')
    #                 plt.subplots_adjust(wspace=0.2,hspace=0.2)
    #                 plt.savefig(self.path+"/B_to_A_{0}.png".format(total_iters), dpi=80)

    #                 plt.clf()
    #                 plt.subplot(1,2,1)
    #                 plt.imshow(X_B.reshape(self.n_H_B,self.n_W_B, self.n_C))
    #                 plt.axis('off')
    #                 plt.subplots_adjust(wspace=0.2,hspace=0.2)

    #                 plt.subplot(1,2,2)
    #                 plt.imshow(test_sample_A.reshape(self.n_H_B,self.n_W_B, self.n_C))
    #                 plt.axis('off')
    #                 plt.subplots_adjust(wspace=0.2,hspace=0.2)
    #                 plt.savefig(self.path+"/A_to_B_{0}.png".format(total_iters), dpi=80)

    #                 plt.clf()
    #                 plt.subplot(1,2,1)
    #                 plt.plot(d_costs_A, label='Discriminator A cost')
    #                 plt.plot(g_GANs_A, label='Generator B to A GAN cost')
    #                 plt.xlabel('Iteration')
    #                 plt.ylabel('Cost')
    #                 plt.legend()
                    
    #                 plt.subplot(1,2,2)
    #                 plt.plot(g_costs_A, label='Generator B to A total cost')
    #                 plt.plot(g_GANs_A, label='Generator B to A GAN cost')
    #                 plt.plot(g_cycles_B, label='Generators B to A to B cycle cost')
    #                 plt.xlabel('Iteration')
    #                 plt.ylabel('Cost')
    #                 plt.legend()

    #                 fig = plt.gcf()
    #                 fig.set_size_inches(15,5)
    #                 plt.savefig(self.path+'/cost_iteration_gen_disc_B_to_A.png',dpi=150)

    #                 plt.clf()
    #                 plt.subplot(1,2,1)
    #                 plt.plot(d_costs_B, label='Discriminator B cost')
    #                 plt.plot(g_GANs_B, label='Generator A to B GAN cost')
    #                 plt.xlabel('Iteration')
    #                 plt.ylabel('Cost')
    #                 plt.legend()

    #                 plt.subplot(1,2,2)
    #                 plt.plot(g_costs_B, label='Generator A to B total cost')
    #                 plt.plot(g_GANs_B, label='Generator A to B GAN cost')
    #                 plt.plot(g_cycles_B, label='Generators B to A to B cycle cost')
    #                 plt.xlabel('Iteration')
    #                 plt.ylabel('Cost')
    #                 plt.legend()

    #                 fig = plt.gcf()
    #                 fig.set_size_inches(15,5)
    #                 plt.savefig(self.path+'/cost_iteration_gen_disc_A_to_B.png',dpi=150)


    #     #return test_sample_A, test_sample_B      


