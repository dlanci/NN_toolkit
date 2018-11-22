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

#tested on mnist
class DCVAE(object):

    def __init__(self, n_H, n_W, n_C, e_sizes, d_sizes,
        lr=LEARNING_RATE, beta1=BETA1,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        save_sample=SAVE_SAMPLE_PERIOD, path=PATH, seed = SEED):
        
        #size of every layer in the encoder
        #up to the latent layer, decoder
        #will have reverse shape
        self.n_H = n_H
        self.n_W = n_W
        self.n_C = n_C
        self.seed = seed
        self.e_sizes = e_sizes
        self.d_sizes = d_sizes
        self.latent_dims = e_sizes['latent_dims']

        self.x = tf.placeholder(
            tf.float32,
            shape=(None, n_H, n_W, n_C),
            name='x'
        )

        self.z_test = tf.placeholder(
            tf.float32,
            shape=(None, self.latent_dims),
            name='z_test'
        )
        
        self.batch_sz = tf.placeholder(
            tf.int32,
            shape=(),
            name='batch_sz'
        )
        

        E = convEncoder(self.x, e_sizes, 'E')
        
        with tf.variable_scope('encoder_E') as scope:
            z_encoded, z_mu, z_log_sigma = E.e_forward(self.x)

        D = convDecoder(z_encoded, self.n_H, self.n_W, d_sizes, 'D')

        with tf.variable_scope('decoder_D') as scope:
            self.x_hat = D.d_forward(z_encoded)


        with tf.variable_scope('decoder_D') as scope:
            scope.reuse_variables()
            self.x_hat_test = D.d_forward(
                self.z_test, reuse=True, is_training=False
                )

        #Loss:
        #Reconstruction loss
        #minimise the cross-entropy loss
        # H(x, x_hat) = -\Sigma ( x*log(x_hat) + (1-x)*log(1-x_hat) )
        epsilon=1e-10
        
        recon_loss = -tf.reduce_sum(
            -tf.squared_difference(self.x,self.x_hat),
            #self.x*tf.log(epsilon+self.x_hat) + (1-self.x)*tf.log(epsilon + 1 -self.x_hat),
            axis=[1,2,3]
            )
        
        
        self.recon_loss=tf.reduce_mean(recon_loss)

        #KL divergence loss
        # Kullback Leibler divergence: measure the difference between two distributions
        # Here we measure the divergence between the latent distribution and N(0, 1)

        kl_loss= -0.5 * tf.reduce_sum(
            1 + 2*z_log_sigma - (tf.square(z_mu) + tf.exp(2*z_log_sigma)),
            axis=[1]
            )

        self.kl_loss=tf.reduce_mean(kl_loss)

        self.total_loss=tf.reduce_mean(self.kl_loss+10*self.recon_loss)
        
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=beta1,
        ).minimize(self.total_loss)          


        #saving for later
        self.lr = lr
        self.batch_size=batch_size
        self.epochs = epochs
        self.path = path
        self.save_sample = save_sample

        self.E=E
        self.D=D

    def set_session(self, session):
        
        self.session = session
        
        for layer in self.E.e_conv_layers:
            layer.set_session(session)
        for layer in self.E.e_dense_layers:
            layer.set_session(session)
            
        for layer in self.D.d_dense_layers:
            layer.set_session(session) 
        for layer in self.D.d_deconv_layers:
            layer.set_session(session)  
        
    def fit(self, X):

        seed = self.seed

        total_loss = []
        rec_losses=[]
        kl_losses=[]

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

            seed +=1 

            batches = unsupervised_random_mini_batches(X, self.batch_size, seed)

            for X_batch in batches:

                feed_dict = {
                            self.x: X_batch, self.batch_sz: self.batch_size
                            }

                _, l, rec_loss, kl_loss = self.session.run(
                            (self.train_op, self.total_loss, self.recon_loss, self.kl_loss),
                            feed_dict=feed_dict
                    )

                l /= self.batch_size
                rec_loss /= self.batch_size
                kl_loss /= self.batch_size

                total_loss.append(l)
                rec_losses.append(rec_loss)
                kl_losses.append(kl_loss)

                total_iters += 1

                if total_iters % self.save_sample ==0:
                    print("At iteration: %d  -  dt: %s - cost: %.2f" % (total_iters, datetime.now() - t0, l))
                    print('Saving a sample...')


                    z_test= np.random.normal(size=(64, self.latent_dims))
                    probs = self.sample(z_test)  
                    
                    for i in range(64):

                        plt.subplot(8,8,i+1)
                        plt.suptitle('samples' )
                        plt.imshow(probs[i].reshape(28,28), cmap='gray')
                        plt.subplots_adjust(wspace=0.2,hspace=0.2)
                        plt.axis('off')
                        
                    fig = plt.gcf()
                    fig.set_size_inches(4,4)
                    plt.savefig(self.path+'/samples_at_iter_%d.png' % total_iters,dpi=100)

            plt.clf()
            plt.subplot(1,3,1)
            plt.suptitle('learning rate=' + str(self.lr))
            plt.plot(total_loss, label='total_loss')
            plt.ylabel('cost')
            plt.legend()
            
            plt.subplot(1,3,2)
            plt.plot(rec_losses, label='rec loss')
            plt.legend()
            
            plt.subplot(1,3,3)
            plt.plot(kl_losses, label='KL loss')
            
            plt.xlabel('iteration')
            
            plt.legend()
            fig=plt.gcf()

            fig.set_size_inches(10,4)
            plt.savefig(self.path+'/cost_vs_iteration.png',dpi=150)
        
        print('Parameters trained')

    def sample(self, Z):

        n=Z.shape[0]
        samples = self.session.run(
          self.x_hat_test,
          feed_dict={self.z_test: Z, self.batch_sz: n}
        )
        return samples
