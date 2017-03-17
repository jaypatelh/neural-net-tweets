import argparse
import numpy as np
import tensorflow as tf
import pickle
from sequence_util import *

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_glob", required=True) # 'data/*'
parser.add_argument("-lf", "--labels_filename", required=True) # 'labels-03112017.p'
parser.add_argument("-ef", "--embeddings_filename", required=True) # 'word2vec_average.p' # Not implemented, doesn't do anything
parser.add_argument("-mf", "--model_filename", required=True) # 'lstm_sentence'
parser.add_argument("-w", "--warmstart", default=False, type=bool) # Not implemented, doesn't do anything
args = parser.parse_args()

class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    batch_size = 32
    model_filename = args.model_filename
    model = "lstm" # can also choose from "gru" and "rnn" for basic rnn cell
    
    input_size = 200
    num_classes = 18
    # This is the output size of each sequence unit, must be in array or none if no rnn units are needed.
    # Last number must correspond with the classifier if no relu layers are used.
    rnn_layers = [64, 32]
    # This is the output size of each relu unit, must be in array or none if no relu layers are needed.
    # Last number must correspond with the number of classes. The input for the first layer will either
    # be the dimension of the rnn output, or of the input size if no rnn units are used.
    hidden_layers = [18]
    seq_length = 2

    loss = "softmax" # currently the choice is this (sigmoid with l2) or "sofmax" (softmax cross entropy)
    optimizer = "adam" # currently the choice is this or "grad" (gradient descent)
    lr = 1e-4
    dropout_p = 0.0 # Not tested yet, may not work
    learning_rate = 0.1
    clip_gradients = False # Not tested yet, may not work
    max_grad_norm = 5.0 # Not tested yet, may not work
    num_epochs = 10
    clip_gradients = False
    mask = False # Not implemented yet, will not do anything

    tf_random_seed = 59 # None means no random seed set for tensorflow graph
    init_random_seed = 23 # None means no random seed set for weight initialization in ReLu layers
    batch_random_seed = 45 # None means no random seed set for train/dev splits
    test_split = 0.25 # What proportion should be held out for evaluation

class SequenceModel():
	def __init__(self, config):
		self.config = config
		
		# Assign sequence model cell
		if self.config.model == 'rnn': cell_fn = tf.contrib.rnn.BasicRNNCell
		elif self.config.model == 'gru': cell_fn = tf.contrib.rnn.GRUCell
		elif self.config.model == 'lstm': cell_fn = tf.contrib.rnn.BasicLSTMCell
		else: "Model %s is not supported in our code" % config.model

		# Configure RNN and DNN layers
		if self.config.rnn_layers != None and len(self.config.rnn_layers) > 0:
			self.cell = sequence_cells(cell_fn, self.config.rnn_layers, self.config.dropout_p)
		else: self.cell = None
		if self.config.hidden_layers != None and len(self.config.hidden_layers) > 0:
			if self.cell != None: input_dim = self.config.rnn_layers[-1]
			else: self.dnn = input_dim = self.config.input_size*self.config.seq_length
			self.dnn = dnn_layers(input_dim, self.config.hidden_layers, self.config.init_random_seed)
		else: self.dnn = None
		
		self.build()

	def get_batch(self, batch_num):
		embeddings = self.data[batch_num*self.config.batch_size:(batch_num+1)*self.config.batch_size + self.config.seq_length]
		num_series = len(embeddings) - self.config.seq_length + 1
		if num_series < 1: return None
		series_embeddings = [embeddings[i:i+self.config.seq_length] for i in range(num_series-1)]
		labels = self.labels[batch_num*self.config.batch_size+self.config.seq_length:(batch_num+1)*self.config.batch_size+self.config.seq_length]
		return series_embeddings, labels

	def add_placeholders(self):
		self.inputs_placeholder = tf.placeholder(tf.float32, [None, self.config.seq_length, self.config.input_size], 'input_placeholder')
		self.labels_placeholder = tf.placeholder(tf.float32, [None, self.config.num_classes], 'labels_placeholder')

	def create_feed_dict(self, input_batch, labels_batch=None):
		feed_dict = { self.inputs_placeholder: input_batch,}
		if labels_batch is not None: feed_dict[self.labels_placeholder] = labels_batch
		return feed_dict

	def add_prediction_op(self):
		"""Runs an rnn on the input using TensorFlows's
		Returns:
			preds: tf.Tensor of shape (batch_size, 1)
		"""
		if self.cell != None:
			outputs, _ = tf.nn.dynamic_rnn(self.cell, self.inputs_placeholder, dtype=tf.float32)
			# If there was more than one input in the sequence, just take the hidden layer corresponding to the last
			# item of the sequence
			if self.config.seq_length > 1: outputs = outputs[:,-1]
			outputs = tf.reshape(outputs, [-1, self.config.rnn_layers[-1]])
		else:
			outputs = tf.reshape(self.inputs_placeholder, [-1, self.config.input_size * self.config.seq_length])

		if self.dnn != None:
			for i,layer in enumerate(self.dnn):
				outputs = tf.nn.relu_layer(outputs, layer['W'], layer['b'], name='relu_' + str(i))
		# optimizer = tf.train.AdagradOptimizer(learning_rate=self.config.lr)
		# classifier = tf.contrib.learn.DNNClassifier(hidden_units=self.config.hidden_layers,
  #                                           n_classes=self.config.num_classes,
  #                                           optimizer=optimizer)
		return outputs

	def add_loss_op(self, preds):
		"""Adds ops to compute the loss function.
		Args:
			pred: A tensor of shape (batch_size, 1) containing the last
			state of the neural network.
			Returns:
			loss: A 0-d tensor (scalar)
		"""
		if self.config.loss == "l2":
			preds = tf.sigmoid(preds)
			loss_vec = tf.nn.l2_loss(preds-self.labels_placeholder)
		elif self.config.loss == "softmax":
			loss_vec = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=self.labels_placeholder)
			if self.config.mask: loss_vec = tf.boolean_mask(loss_vec, self.mask_placeholder)
		else: raise ValueError("Loss function %s not supported" % self.config.loss)
		return tf.reduce_mean(loss_vec)

	def add_training_op(self, loss):
		"""Sets up the training Ops.

		Creates an optimizer and applies the gradients to all trainable variables.
		The Op returned by this function is what must be passed to the
		`sess.run()` call to cause the model to train. See
		Args:
			loss: Loss tensor.
		Returns:
			train_op: The Op for training.
		"""
		if self.config.optimizer == "adam": optimizer = tf.train.AdamOptimizer(self.config.lr)
		elif self.config.optimizer == "grad": optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.lr)
		else: raise ValueError("Optimizer %s not supported." % self.config.optimizer)
		
		gradients = optimizer.compute_gradients(loss)
		print gradients
		grad_vars = zip(*gradients)
		if self.config.clip_gradients == True:
		    result = tf.clip_by_global_norm(grad_vars[0], self.config.max_grad_norm) 
		    self.grad_norm = tf.global_norm(result[0])
		else:
		    result = grad_vars
		    self.grad_norm = result[0]
		train_op = optimizer.apply_gradients(zip(result[0], grad_vars[1])) 

		assert self.grad_norm is not None, "grad_norm was not set properly!"
		# return optimizer.minimize(loss)
		return train_op

	def predict_on_batch(self, sess, input_batch, mask_batch=None):
		feed = self.create_feed_dict(input_batch=input_batch)
		if mask_batch: feed = self.create_feed_dict(input_batch=input_batch, mask_batch=mask_batch)
		predictions = sess.run(self.pred, feed_dict=feed)
		return predictions

	def train_on_batch(self, sess, input_batch, labels_batch):
		"""
		Perform one step of gradient descent on the provided batch of data.
		This version also returns the norm of gradients.
		"""
		feed = self.create_feed_dict(input_batch, labels_batch=labels_batch)
		_, loss, grad_norm = sess.run([self.train_op, self.loss, self.grad_norm], feed_dict=feed)
		return loss, grad_norm

	def run_epoch(self, sess):
		losses, grad_norms = [], []
		for i in range(int(np.ceil(len(self.data)/float(self.config.batch_size)))):
			input_batch, labels_batch = self.get_batch(i)
			loss, grad_norm = self.train_on_batch(sess, input_batch, labels_batch)
			losses.append(loss)
			grad_norms.append(grad_norm)
		return losses, grad_norms

	def fit(self, sess, X, y):
		self.data, self.labels = X, y
		losses, grad_norms = [], []
		for epoch in range(self.config.num_epochs): print "Epoch %d out of %d" % (epoch + 1, self.config.num_epochs)
		loss, grad_norm = self.run_epoch(sess)
		losses.append(loss)
		grad_norms.append(grad_norm)
		return losses, grad_norms

	def predict(self, sess, inputs, labels):
		total_matches, total_samples = 0, 0
		for i in range(int(np.ceil(len(self.data)/float(self.config.batch_size)))):
			input_batch, labels_batch = self.get_batch(i)
			predictions = self.predict_on_batch(sess, input_batch)
			predicted_classes = tf.argmax(np.array(predictions), axis=1)
			true_classes = tf.argmax(np.array(labels_batch), axis=1)
			total_matches += tf.reduce_sum(tf.to_float(tf.to_int32(tf.equal(true_classes, predicted_classes))))
			total_samples += len(input_batch)
		print "Overall Accuracy: "
		print sess.run(total_matches*100.0 / float(total_samples)), "%"

	def build(self):
		self.add_placeholders()
		self.pred = self.add_prediction_op()
		self.loss = self.add_loss_op(self.pred)
		self.train_op = self.add_training_op(self.loss)


config = Config()
X_train, X_test, y_train, y_test = get_training_data(args.input_glob + "/tweets.p",
	args.input_glob + "/" + args.labels_filename, args.input_glob + "/" + args.embeddings_filename,
	test_split=config.test_split, random_seed = config.batch_random_seed)

with tf.Graph().as_default():
	if config.tf_random_seed != None: tf.set_random_seed(config.tf_random_seed)
	print "Building model..."
	model = SequenceModel(config)
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)
		losses, grad_norms = model.fit(session, X_train, y_train)
		model.predict(session, X_test, y_test)
