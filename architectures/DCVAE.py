#NETWORK ARCHITECTURES

import numpy as np
import os 
import math

import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

from architectures.utils import NN_building_blocks, NN_gen_building_blocks

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

        seed = self.seed

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

            seed +=1 

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