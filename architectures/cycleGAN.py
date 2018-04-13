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


class cycleGAN(object):
    
    def __init__(
        self, 
        n_H, n_W, n_C,
        mean_A, std_A, mean_B, std_B,
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
        self.mean_B = mean_B
        self.mean_A = mean_A

        self.std_A = std_A
        self.std_B = std_B

        self.seed=seed
        self.n_W = n_W
        self.n_H = n_H
        self.n_C = n_C
        
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

        D_A = Discriminator(self.input_A, d_sizes_A, 'A')
        D_B = Discriminator(self.input_B, d_sizes_B, 'B')

        G_A_to_B = cycleGenerator(self.input_A, self.n_H, self.n_W, g_sizes_A, 'A_to_B')
        G_B_to_A = cycleGenerator(self.input_B, self.n_H, self.n_W, g_sizes_B, 'B_to_A')
        

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
                   n_H, n_W, n_C),
            name='X_A',
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
                   n_H, n_W, n_C),
            name='X_B',
        )

        with tf.variable_scope('generator_B_to_A') as scope:
            scope.reuse_variables()
            self.sample_images_test_B_to_A = G_B_to_A.g_forward(
                self.input_test_B, reuse=True, is_training=False
            )
            
        #cost building
        
        #Discriminators cost
        #cost is low if real images are predicted as real (1) 
        # d_cost_real_A = tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=logits_A,
        #     labels=tf.ones_like(logits_A)
        # )
        # #cost is low if fake generated images are predicted as fake (0)
        # d_cost_fake_A = tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=sample_logits_A,
        #     labels=tf.zeros_like(sample_logits_A)
        # )
        
        d_cost_real_A=tf.squared_difference(logits_A,1)
        d_cost_fake_A=tf.square(sample_logits_A)

        #discriminator_A cost
        self.d_cost_A = tf.reduce_mean(d_cost_real_A) + tf.reduce_mean(d_cost_fake_A)
        
        #same for discriminator B
        # d_cost_real_B = tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=logits_B,
        #     labels=tf.ones_like(logits_B)
        # )
        
        # d_cost_fake_B = tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=sample_logits_B,
        #     labels=tf.zeros_like(sample_logits_B)
        # )

        d_cost_real_B=tf.squared_difference(logits_B,1)
        d_cost_fake_B=tf.square(sample_logits_B)

        #discriminator_B cost
        self.d_cost_B = tf.reduce_mean(d_cost_real_B) + tf.reduce_mean(d_cost_fake_B)

        #averaging the two discriminators cost
        #self.d_cost = (d_cost_A+d_cost_B)/2

        #Generator cost 
        #cost is low if logits from discriminator A on samples generated by G_B_to_A 
        #are predicted as true (1)
        # g_cost_A = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(
        #         logits=sample_logits_A,
        #         labels=tf.ones_like(sample_logits_A)
        #     )
        # )

        g_cost_A=tf.reduce_mean(tf.squared_difference(sample_logits_A,1))
        

        #cost is low if logits from discriminator B on samples generated by G_A_to_B 
        #are predicted as true (1)
        # g_cost_B = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(
        #         logits=sample_logits_B,
        #         labels=tf.ones_like(sample_logits_B)
        #     )
        # )

        g_cost_B=tf.reduce_mean(tf.squared_difference(sample_logits_B,1))

        #cycle cost is low if cyclic images are similar to input images (in both sets)
        g_cycle_cost_A = tf.reduce_mean(tf.abs(self.input_A-cycl_A)) 
        g_cycle_cost_B = tf.reduce_mean(tf.abs(self.input_B-cycl_B))

        g_cycle_cost= g_cycle_cost_A+g_cycle_cost_B
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
            learning_rate=self.curr_lr,
            beta1=beta1,
        ).minimize(
            self.d_cost_A,
            var_list=self.d_params_A
        )

        self.d_train_op_B = tf.train.AdamOptimizer(
            learning_rate=self.curr_lr,
            beta1=beta1,
        ).minimize(
            self.d_cost_B,
            var_list=self.d_params_B
        )
        
        self.g_train_op_A = tf.train.AdamOptimizer(
            learning_rate=self.curr_lr,
            beta1=beta1,
        ).minimize(
            self.g_cost_A,
            var_list=self.g_params_A
        )

        self.g_train_op_B = tf.train.AdamOptimizer(
            learning_rate=self.curr_lr,
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
        self.sample_images_B=sample_images_B
        self.sample_images_A=sample_images_A
    
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
                    
    def fit(self, X_A, X_B):

        seed = self.seed
        d_costs_A = []
        g_costs_A = []

        d_costs_B = []
        g_costs_B = []

        N = len(X_A)
        n_batches = N // self.batch_size
    
        total_iters=0

        print('\n ****** \n')
        print('Training cycle GAN with a total of ' +str(N)+' samples distributed in batches of size '+str(self.batch_size)+'\n')
        print('The learning rate set for the generator is '+str(self.lr_g)+' while for the discriminator is '+str(self.lr_d)+', and every ' +str(self.save_sample)+ ' epoch a generated sample will be saved to '+ self.path)
        print('\n ****** \n')

        for epoch in range(self.epochs):

            seed +=1

            print('Epoch:', epoch)

            if(epoch < 100) :
                curr_lr = 0.0002
            else:
                curr_lr = 0.0002 - 0.0002*(epoch-100)/100
            
            batches_A = unsupervised_random_mini_batches(X_A, self.batch_size, seed)
            batches_B = unsupervised_random_mini_batches(X_B, self.batch_size, seed)

            for X_batch_A, X_batch_B in zip(batches_A,batches_B):
                
                t0 = datetime.now()
                
                #optimize generator_A

                _, g_cost_A =  self.session.run(
                    (self.g_train_op_A, self.g_cost_A),
                    feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:self.batch_size, self.lr:curr_lr},
                )
                
                # _, g_cost_A2, fake_images_B_temp =  self.session.run(
                #     (self.g_train_op_A, self.g_cost_A, self.sample_images_B),
                #     feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:self.batch_size},
                # )
                # g_cost_A = (g_cost_A1+g_cost_A2)/2
                g_costs_A.append(g_cost_A) # just use the avg    

                #optimize discriminator_B

                _, d_cost_B, d_acc_B = self.session.run(
                    (self.d_train_op_B, self.d_cost_B, self.d_accuracy_B),
                    feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:self.batch_size, self.lr:curr_lr},
                )

                d_costs_B.append(d_cost_B)

                #optimize generator_B

                _, g_cost_B =  self.session.run(
                    (self.g_train_op_B, self.g_cost_B),
                    feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:self.batch_size, self.lr:curr_lr},
                )
                
                # _, g_cost_B2, fake_images_A_temp =  self.session.run(
                #     (self.g_train_op_B, self.g_cost_B, self.sample_images_A),
                #     feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:self.batch_size},
                # )

                # g_cost_B = (g_cost_B1+g_cost_B2)/2
                g_costs_B.append(g_cost_B) 


                #optimize Discriminator_A 
                _, d_cost_A, d_acc_A = self.session.run(
                    (self.d_train_op_A, self.d_cost_A, self.d_accuracy_A),
                    feed_dict={self.input_A:X_batch_A, self.input_B:X_batch_B, self.batch_sz:self.batch_size, self.lr:curr_lr},
                )

                d_costs_A.append(d_cost_A)


                total_iters += 1
                if total_iters % self.save_sample ==0:
                    print("At iter: %d  -  dt: %s - d_acc_A: %.2f" % (total_iters, datetime.now() - t0, d_acc_A))
                    print("At iter: %d  -  dt: %s - d_acc_B: %.2f" % (total_iters, datetime.now() - t0, d_acc_B))
                    print("Discrimator_A cost {0:.4g}, Generator_A_to_B cost {1:.4g}".format(d_cost_A, g_cost_A))
                    print("Discrimator_B cost {0:.4g}, Generator_B_to_A cost {1:.4g}".format(d_cost_B, g_cost_B))
                    print('Saving a sample...')
                    
                    
                    #shape is (1,D,D,color)
                    _, n_H, n_W, n_C = X_batch_A.shape 
                    
                    #for i in range(10):

                    j = np.random.choice(len(X_batch_A))

                    X_batch_A = X_batch_A[j]
                    X_batch_B = X_batch_B[j]

                    #X_batch_A= X_batch_A.reshape(1,n_H,n_W,n_C)
                    
                    sample_B = self.get_sample_A_to_B(X_batch_A.reshape(1,n_H,n_W,n_C))
                    sample_A = self.get_sample_B_to_A(X_batch_B.reshape(1,n_H,n_W,n_C))


                    plt.subplot(2,2,1)
                    X_batch_A=(X_batch_A*self.std_A+self.mean_A).astype(np.int32)

                    X_batch_A[np.where(X_batch_A<0)]=0
                    X_batch_A[np.where(X_batch_A>255)]=255

                    plt.imshow(X_batch_A.astype(np.int32))
                    plt.axis('off')
                    plt.subplots_adjust(wspace=0.2,hspace=0.2)

                    plt.subplot(2,2,2)
                    sample_B=(sample_B*self.std_B+self.mean_B).astype(np.int32)

                    sample_B[np.where(sample_B<0)]=0
                    sample_B[np.where(sample_B>255)]=255

                    plt.imshow(sample_B.reshape(n_H,n_W,n_C).astype(np.int32))
                    plt.axis('off')
                    plt.subplots_adjust(wspace=0.2,hspace=0.2)

                    plt.subplot(2,2,3)
                    X_batch_B=(X_batch_B*self.std_B+self.mean_B).astype(np.int32)

                    X_batch_B[np.where(X_batch_B<0)]=0
                    X_batch_B[np.where(X_batch_B>255)]=255


                    plt.imshow(X_batch_B.astype(np.int32))
                    plt.axis('off')
                    plt.subplots_adjust(wspace=0.2,hspace=0.2)

                    plt.subplot(2,2,4)
                    sample_A=(sample_A*self.std_A+self.mean_A).astype(np.int32)

                    sample_A[np.where(sample_A<0)]=0
                    sample_A[np.where(sample_A>255)]=255

                    plt.imshow(sample_A.reshape(n_H,n_W,n_C).astype(np.int32))
                    plt.axis('off')
                    plt.subplots_adjust(wspace=0.2,hspace=0.2)
                    

                    fig = plt.gcf()
                    fig.set_size_inches(10,8)
                    plt.savefig(self.path+'/sample_at_iter_{0}.png'.format(total_iters),dpi=150)


                    
            plt.clf()
            plt.plot(d_costs_A, label='discriminator cost')
            plt.plot(g_costs_B, label='generator cost')
            plt.legend()
            fig = plt.gcf()
            fig.set_size_inches(8,5)
            plt.savefig(self.path+'/cost vs iteration.png',dpi=150)
    
    # def sample(self, Z):
        
    #     samples = self.session.run(
    #         self.sample_images_test, 
    #         feed_dict={self.Z:Z, self.batch_sz: self.batch_size})

    #     return samples 

    def fake_img_pool(self, num_fakes, fake_img, fake_pool):

        if (num_fakes < pool_size):

            fake_pool[num_fakes] = fake
            return fake

        else:

            p = np.random.random()
            if p > 0.5:
                random_id = np.random.randint(0, pool_size-1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake

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

