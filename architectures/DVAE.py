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
class DVAE(object):

	"""
	Builds densely connected deep variational autoencoder. Regularization implemented 
	with dropout, no regularization parameter implemented yet. Minimization through
	AdamOptimizer (adaptive learning rate).

	The minimized loss function is the elbo function (explain further)

	Constructor inputs:

	    -Positional arguments:
	        - dim: (int) input features 
	        - e_sizes: (dict)
	        - d_sizes: (dict)
	    
	    -Keyword arguments

	        - an_id: (int) number useful for stacked ae
	        - lr: (float32) learning rate arg for the AdamOptimizer
	        - beta1: (float32) beta1 arg for the AdamOptimizer
	        - batch_size: (int) size of each batch
	        - epochs: (int) number of times the training has to be repeated over all the batches
	        - save_sample: (int) after how many iterations of the training algorithm performs the evaluations in fit function
	        - path: (str) path for saving the session checkpoint

	Class attributes:
	    
	    - X: (tf placeholder) input tensor of shape (batch_size, input features)
	    - Y: (tf placeholder) label tensor of shape (batch_size, n_classes) (one_hot encoding)
	    - Y_hat: (tf tensor) shape=(batch_size, n_classes) predicted class (one_hot)
	    - loss: (tf scalar) reduced mean of cost computed with softmax cross entropy with logits
	    - train_op: gradient descent algorithm with AdamOptimizer


	Class methods:
	    - posterior_predictive_sample:
	    - prior_predictive_sample_with_probs:

	"""

	def __init__(
		self, dim, e_sizes, d_sizes, an_id=0, 
		lr=LEARNING_RATE, beta1=BETA1,
		batch_size=BATCH_SIZE, epochs=EPOCHS,
		save_sample=SAVE_SAMPLE_PERIOD, path=PATH, seed=SEED, img_height=None, img_width=None
		):
		"""
		Positional args


		Keyword args

		"""
		self.dim = dim
		self.e_sizes=e_sizes
		self.d_sizes=d_sizes
		self.img_height=img_height
		self.img_width=img_width
		self.seed = seed

		self.latent_dims=e_sizes['z']
		#self.d_last_act_f = d_sizes['last_act_f']

		self.X = tf.placeholder(
				tf.float32,
				shape=(None, self.dim),
				name='X'
			)
		    
		self.batch_sz = tf.placeholder(
		        tf.float32,
		        shape=(),
		        name='batch_sz'
		    )

		self.E = denseEncoder(self.X, e_sizes, 'A')

		with tf.variable_scope('encoder_A') as scope:

		    self.Z = self.E.encode(self.X)

		self.D = denseDecoder(self.Z, self.latent_dims, dim, d_sizes, 'A')

		with tf.variable_scope('decoder_A') as scope:

		    logits = self.D.decode(self.Z)


		self.X_hat_distribution = Bernoulli(logits=logits)

		# posterior predictive
		# take samples from X_hat

		with tf.variable_scope('encoder_A') as scope:
		    scope.reuse_variables
		    self.Z_dist = self.E.encode(
		        self.X, reuse=True, is_training=False,
		    )#self.X or something on purpose?                                            
		with tf.variable_scope('decoder_A') as scope:
		    scope.reuse_variables()
		    sample_logits = self.D.decode(
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

		Z_std = standard_normal.sample(1, seed=self.seed)

		with tf.variable_scope('decoder_A') as scope:
		    scope.reuse_variables()
		    logits_from_prob = self.D.decode(
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

	def set_session(self, session):

		self.session = session

		for layer in self.D.d_layers:
			layer.set_session(self.session)

		for layer in self.E.e_layers:
			layer.set_session(self.session)

	def fit(self, X):

		"""
		Function is called if the flag is_training is set on TRAIN. If a model already is present
		continues training the one already present, otherwise initialises all params from scratch.

		Performs the training over all the epochs, at when the number of epochs of training
		is a multiple of save_sample, prints the cost at that epoch. When training has gone through all
		the epochs, plots a plot of the cost versus epoch. 

		Positional arguments:

		    - X_train: (ndarray) size=(train set size, input features) training sample set
		    
		"""

		seed = self.seed

		costs = []
		N = len(X)
		n_batches = N // self.batch_size


		print('\n ****** \n')
		print('Training deep VAE with a total of ' +str(N)+' samples distributed in batches of size '+str(self.batch_size)+'\n')
		print('The learning rate set is '+str(self.lr)+', and every ' +str(self.save_sample)+ ' iterations a generated sample will be saved to '+ self.path)
		print('\n ****** \n')
		total_iters=0

		for epoch in range(self.epochs):

			t0 = datetime.now()
			print('Epoch: {0}'.format(epoch))

			seed+=1

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

					probs = []
					for i in range(64):
						probs.append(self.prior_predictive_sample())
						self.seed+=1

					for i in range(64):
						plt.subplot(8,8,i+1)
						plt.imshow(probs[i].reshape(self.img_height,self.img_width), cmap='gray')
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

	def posterior_predictive_sample(self, X):
		# returns a sample from p(x_new | X)
		return self.session.run(self.posterior_predictive_probs, feed_dict={self.X: X, self.batch_sz:self.batch_size})

	def prior_predictive_sample(self):
		# returns a sample from p(x_new | z), z ~ N(0, 1)
		return self.session.run(self.prior_predictive_probs)