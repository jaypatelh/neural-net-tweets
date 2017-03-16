import argparse
import numpy as np
import tensorflow as tf
import glob
import pickle
import tweepy
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_glob", required=True) # 'data/*'
parser.add_argument("-lf", "--labels_filename", required=True) # 'labels-03112017.p'
parser.add_argument("-ef", "--embeddings_filename", required=True) # 'word2vec_average.p'
parser.add_argument("-m", "--model", required=True) # e.g. 'lstm'
parser.add_argument("-mf", "--model_filename", required=True) # 'lstm_sentence'
parser.add_argument("-w", "--warmstart", default=False, type=bool)
parser.add_argument("-es", "--embed_size", default=200, type=int)
parser.add_argument("-nl", "--num_layers", default=1, type=int)
parser.add_argument("-loss", "--loss", default="l2")
parser.add_argument("-mask", "--mask")
parser.add_argument("-ops", "--optimizer", default="adam")
args = parser.parse_args()

def get_tweet_ids_time_ordered():
	sorted_tweet_ids = []
	filenames = glob.glob(args.input_glob + "/tweets.p")
	for filename in filenames:
		print "unloading %s" % filename
		with open(filename, "rb") as input_file:
			tweets, _ = pickle.load(input_file)
			for tweet_id in tweets: sorted_tweet_ids.append([tweet_id, tweets[tweet_id].created_at])
	sorted_tweet_ids.sort(key=lambda x: x[1])
	return [x[0] for x in sorted_tweet_ids]

def get_labels(tweet_ids):
	labels = {}
	filenames = glob.glob(args.input_glob + "/" + args.labels_filename)
	for filename in filenames:
		with open(filename, "rb") as input_file:
			print "unloading %s" % filename
			l = pickle.load(input_file)
			for key in l: labels[key] = l[key]
	return [labels[tweet_id] for tweet_id in tweet_ids if tweet_id in labels]

def get_embeddings(tweet_ids):
	embeddings = {}
	filenames = glob.glob(args.input_glob + "/" + args.embeddings_filename)
	for filename in filenames:
		print "unloading %s" % filename
		with open(filename, "rb") as input_file:
			e = pickle.load(input_file)
			for key in e: embeddings[key] = e[key]
	return [embeddings[tweet_id] for tweet_id in tweet_ids if tweet_id in embeddings]

class SequenceModel():
	def __init__(self, batch_size=32, input_size=200, rnn_size=64, hidden_size=32, num_classes=18, num_layers=1, seq_length=4, dropout_p=0, learning_rate=0.1, clip_gradients=False, num_epochs=10):
		self.filenames = glob.glob(args.input_glob)
		self.output_filename = args.model_filename
		self.batch_size = batch_size
		self.input_size = input_size
		self.rnn_size = rnn_size
		self.hidden_size = hidden_size
		self.num_classes = num_classes
		self.lr = learning_rate
		self.clip_gradients = clip_gradients
		self.seq_length = seq_length
		self.num_epochs = num_epochs
		# Choose a model
		if args.model == 'rnn':
			cell_fn = tf.contrib.rnn.BasicRNNCell
		elif args.model == 'gru':
			cell_fn = tf.contrib.rnn.GRUCell
		elif args.model == 'lstm':
			cell_fn = tf.contrib.rnn.BasicLSTMCell
		else:
			"Model %s is not supported in our code" % args.model
		self.cell = cell_fn(rnn_size)
		if num_layers > 1: self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
		self.build()

	def get_batch(self, batch_num):
		embeddings = self.data[batch_num*self.batch_size:(batch_num+1)*self.batch_size + self.seq_length]
		num_series = len(embeddings) - self.seq_length + 1
		if num_series < 1: return None
		series_embeddings = [embeddings[i:i+self.seq_length] for i in range(num_series-1)]
		labels = self.labels[batch_num*self.batch_size+self.seq_length:(batch_num+1)*self.batch_size+self.seq_length]
		return series_embeddings, labels

	def add_placeholders(self):
		self.inputs_placeholder = tf.placeholder(tf.float32, [None, self.seq_length, self.input_size], 'input_placeholder')
		self.labels_placeholder = tf.placeholder(tf.float32, [None, self.num_classes], 'labels_placeholder')

	def create_feed_dict(self, input_batch, labels_batch=None):
		feed_dict = { self.inputs_placeholder: input_batch,}
		if labels_batch is not None: feed_dict[self.labels_placeholder] = labels_batch
		return feed_dict

	def add_prediction_op(self):
		"""Runs an rnn on the input using TensorFlows's
		@tf.nn.dynamic_rnn function, and returns the final state as a prediction.

		TODO: 
			- Call tf.nn.dynamic_rnn using @cell below. See:
			https://www.tensorflow.org/api_docs/python/nn/recurrent_neural_networks
			- Apply a sigmoid transformation on the final state to
			normalize the inputs between 0 and 1.

		Returns:
			preds: tf.Tensor of shape (batch_size, 1)
		"""
		rnn_outputs, rnn_state = tf.nn.dynamic_rnn(self.cell, self.inputs_placeholder, dtype=tf.float32)
		# rnn_outputs, self.rnn_state = tf.nn.rnn(self.cell, self.inputs_placeholder, initial_state=self.rnn_state)
		
		# rnn_outputs = tf.reshape(rnn_outputs, [-1, self.rnn_size])
    	# logits = tf.matmul(rnn_outputs, W) + b
		if args.model == "lstm": rnn_state = rnn_state[0]
		preds = tf.reshape(tf.sigmoid(rnn_state), [-1, self.rnn_size])
		return preds

	def add_loss_op(self, preds):
		"""Adds ops to compute the loss function.
		Here, we will use a simple l2 loss.

		Tips:
			- You may find the functions tf.reduce_mean and tf.l2_loss
			  useful.

		Args:
			pred: A tensor of shape (batch_size, 1) containing the last
			state of the neural network.
			Returns:
			loss: A 0-d tensor (scalar)
		"""
		if args.loss == "l2":
			loss_vec = tf.nn.l2_loss(preds-self.labels_placeholder)
		elif args.loss == "softmax":
			if args.mask:
				loss_vec = tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(preds, self.labels_placeholder), self.mask_placeholder)
			else:
				loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(preds, self.labels_placeholder)
		else: raise ValueError("Loss function %s not supported" % args.loss)
		return tf.reduce_mean(loss_vec)

	def add_training_op(self, loss):
		"""Sets up the training Ops.

		Creates an optimizer and applies the gradients to all trainable variables.
		The Op returned by this function is what must be passed to the
		`sess.run()` call to cause the model to train. See

		TODO:
			- Get the gradients for the loss from optimizer using
			  optimizer.compute_gradients.
			- if self.clip_gradients is true, clip the global norm of
			  the gradients using tf.clip_by_global_norm to self.config.max_grad_norm
			- Compute the resultant global norm of the gradients using
			  tf.global_norm and save this global norm in self.grad_norm.
			- Finally, actually create the training operation by calling
			  optimizer.apply_gradients.
		See: https://www.tensorflow.org/api_docs/python/train/gradient_clipping
		Args:
			loss: Loss tensor.
		Returns:
			train_op: The Op for training.
		"""
		if args.optimizer == "adam":
			optimizer = tf.train.AdamOptimizer(self.lr)
		elif args.optimizer == "grad":
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
		else: raise ValueError("Optimizer %s not supported." % args.optimizer)
		gradients = optimizer.compute_gradients(loss) 
		grad_vars = zip(*gradients)
		if self.clip_gradients == True:
		    result = tf.clip_by_global_norm(grad_vars[0], 5.0) 
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
		"""Perform one step of gradient descent on the provided batch of data.
		This version also returns the norm of gradients. """
		feed = self.create_feed_dict(input_batch, labels_batch=labels_batch)
		_, loss, grad_norm = sess.run([self.train_op, self.loss, self.grad_norm], feed_dict=feed)
		return loss, grad_norm

	def run_epoch(self, sess):
		losses, grad_norms = [], []
		for i in range(int(np.ceil(len(self.data)/float(self.batch_size)))):
			input_batch, labels_batch = self.get_batch(i)
			loss, grad_norm = self.train_on_batch(sess, input_batch, labels_batch)
			losses.append(loss)
			grad_norms.append(grad_norm)
		return losses, grad_norms

	def fit(self, sess, X, y):
		self.data = X
		self.labels = y
		losses, grad_norms = [], []
		for epoch in range(self.num_epochs): print "Epoch %d out of %d" % (epoch + 1, self.num_epochs)
		loss, grad_norm = self.run_epoch(sess)
		losses.append(loss)
		grad_norms.append(grad_norm)
		return losses, grad_norms

	def predict(self, sess, inputs, labels):
		total_matches, total_samples = 0, 0
		for i in range(int(np.ceil(len(self.data)/float(self.batch_size)))):
			input_batch, labels_batch = self.get_batch(i)
			predictions = self.predict_on_batch(sess, input_batch)
			predicted_classes = tf.argmax(np.array(predictions), axis=1)
			true_classes = tf.argmax(np.array(labels_batch), axis=1)
			total_matches += tf.reduce_sum(tf.to_float(tf.to_int32(tf.equal(true_classes, predicted_classes))))
			total_samples += self.batch_size
		print "Overall Accuracy: "
		print sess.run(total_matches*100.0 / float(total_samples)), "%"

	def build(self):
		self.add_placeholders()
		self.pred = self.add_prediction_op()
		self.loss = self.add_loss_op(self.pred)
		self.train_op = self.add_training_op(self.loss)

# Read in all data and sort in accordance to time sequence
filenames = glob.glob(args.input_glob)
print "found %d input files" % len(filenames)
tweet_ids = get_tweet_ids_time_ordered()
labels = get_labels(tweet_ids)
assert(len(labels)==len(tweet_ids)), "Tweet ids in pickle files do not map fully to labels"
embeddings = get_embeddings(tweet_ids)
assert(len(embeddings)==len(tweet_ids)), "Tweet ids in pickle files do not map fully to embeddings"

X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.25, random_state=42)

with tf.Graph().as_default():
	tf.set_random_seed(59)
	print "Building model..."
	model = SequenceModel(seq_length=1, rnn_size=18, num_classes=18)
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)
		losses, grad_norms = model.fit(session, X_train, y_train)
		model.predict(session, X_test, y_test)
