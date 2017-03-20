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
parser.add_argument("-eo", "--english_only", default=True)
parser.add_argument("-w", "--warmstart", default=False, type=bool) # Not implemented, doesn't do anything
args = parser.parse_args()

class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    model_filename = args.model_filename
    model = "lstm" # can also choose from "gru" and "rnn" for basic rnn cell
    input_size = 200
    batch_size = 32
    
    # In the example of having words in a sentence, each word is a separate feature. Here, it is treated
    # as the maximum number of features, and padding is used to fill the sentence
    num_classes = 18
    
    # This is the output size of each sequence unit, must be in array or none if no rnn units are needed.
    # Last number must correspond with the classifier if no relu layers are used.
    rnn_layers = [64]
    
    # This is the output size of each relu unit, must be in array or none if no relu layers are needed.
    # Last number must correspond with the number of classes. The input for the first layer will either
    # be the dimension of the rnn output, or of the input size if no rnn units are used.
    hidden_layers = [32, 18]
    
    # Augment the inputs by a dimension, and take windows of inputs in a sequence of seq_length
    # NOTE: sequence batch will not work if embeddings is non null, since that indicates the sequence will
    # be the words and not time windows
    sequence_batch = True
    
    # Length of the input sequence for RNN, if only one input used at a time, length is still 1
    seq_length = 3
    max_words = 10

    loss = "softmax" # currently the choice is this (sigmoid with l2) or "sofmax" (softmax cross entropy)
    optimizer = "adam" # currently the choice is this or "grad" (gradient descent)
    lr = 0.1
    dropout_p = 0.0 # Not tested yet, may not work
    clip_gradients = False # Not tested yet, may not work
    max_grad_norm = 5.0 # Not tested yet, may not work
    num_epochs = 2

    tf_random_seed = 59 # None means no random seed set for tensorflow graph
    init_random_seed = 23 # None means no random seed set for weight initialization in ReLu layers
    batch_random_seed = 45 # None means no random seed set for train/dev splits
    test_split = 0.25 # What proportion should be held out for evaluation

    # If input sequences vary in length, seq length is interpereted as the maximum number of components
    # in any input sequence, so preprocessing can truncate or pad sequences to this length
    # This implies a mask input will be fed in the feed dictionary
    pad_sequences = False
    embedding_lookup = None
    sentence_embeddings = None

    # figure_title = "%s with %s and %s" % (model, loss, optimizer)

class TweetSequenceLookupEmbeddingSequenceConfig(Config):
	pad_sequences = True
	embedding_lookup = "pretrained"
	sequence_batch = True
	sentence_embeddings = "average"
	seq_length = 3
	max_words = 10

class TweetSequenceLookupEmbeddingConfig(Config):
	pad_sequences = True
	embedding_lookup = "pretrained"
	sequence_batch = False
	sentence_embeddings = None
	max_words = 10

####################################################################
# CONFIG IS MOVED UP HERE FOR EASE OF USE
# READ THE CORRESPONDING TEXT IF YOU WANT TO RUN SPECIAL CONFIGS
# THAT ENABLE YOU TO LOOK UP EMBEDDINGS OR USE WORD TO WORD INSTEAD
# OF TWEET TO TWEET
####################################################################
# This is the default config. It will take in the raw input embeddings
# for a rnn sequence model with the specifications above. The rnn 
# sequence of model takes windows of inputs and predicts the label of
# the last input.
# The embedding file in this case should be something
# that maps tweet ids to the input embeddings for that id, for example
# word2vec_average.p or word2vec_minmax.p

config = Config()

# This is the lookup config. It does the same thing as the default,
# with the exception that it looks up the embeddings instead of taking
# them in as raw input. The difference this makes is that the gradients
# will backpropogate to the embeddings themselves.
# The embedding file in this case should be the model file, for example
# word2vec_model
# Look at the class for specific things you should modify with this one.

# config = TweetSequenceLookupEmbeddingSequenceConfig()

# This is another lookup config (similar to the previous one), except
# that instead of averaging or aggregating the words before feeding it
# to the RNN, each word is an input and the last word is the output
# that has a label predicted on.
# The embedding file in this case should be the model file, for example
# word2vec_model

# config = TweetSequenceLookupEmbeddingConfig()
####################################################################

class SequenceModel():
	def __init__(self, config, pretrained_embeddings=None, vocab=None):
		self.config = config

		# Assign sequence model cell
		if self.config.model == 'rnn': cell_fn = tf.contrib.rnn.BasicRNNCell
		elif self.config.model == 'gru': cell_fn = tf.contrib.rnn.GRUCell
		elif self.config.model == 'lstm': cell_fn = tf.contrib.rnn.BasicLSTMCell
		else: "Model %s is not supported in our code" % config.model

		# Configure RNN and DNN layers
		if self.config.rnn_layers is not None and len(self.config.rnn_layers) > 0:
			self.cell = sequence_cells(cell_fn, self.config.rnn_layers, self.config.dropout_p)
		else: self.cell = None
		if self.config.hidden_layers is not None and len(self.config.hidden_layers) > 0:
			if self.cell is not None: input_dim = self.config.rnn_layers[-1]
			else: self.dnn = input_dim = self.config.input_size*self.config.seq_length
			self.dnn = dnn_layers(input_dim, self.config.hidden_layers, self.config.init_random_seed)
		else: self.dnn = None

		# If there are pretrained embeddings, set them
		if self.config.embedding_lookup == "pretrained" and pretrained_embeddings is not None:
			self.embeddings = tf.Variable(pretrained_embeddings, "embedding", dtype=tf.float32)
		else: self.embeddings = None
		
		self.build()

	def get_sequence_batch(self, batch_num):
		# Inputs are (sequences) of either embeddings or indices in to embeddings
		mask_series = None
		inputs = self.data[batch_num*self.config.batch_size:(batch_num+1)*self.config.batch_size + self.config.seq_length]
		num_series = len(inputs) - self.config.seq_length + 1
		if num_series < 1: return None
		series_inputs = [inputs[i:i + self.config.seq_length] for i in range(num_series - 1)]
		labels = self.labels[batch_num*self.config.batch_size + self.config.seq_length:(batch_num + 1)*self.config.batch_size+self.config.seq_length]
		mask_series = None
		if self.config.pad_sequences and self.mask is not None:
			mask = self.mask[batch_num*self.config.batch_size:(batch_num+1)*self.config.batch_size + self.config.seq_length]
			mask_series = [mask[i:i + self.config.seq_length] for i in range(num_series - 1)]
		return series_inputs, labels, mask_series

	def get_batch(self, batch_num):
		masks = None
		inputs = self.data[batch_num*self.config.batch_size:(batch_num+1)*self.config.batch_size]
		labels = self.labels[batch_num*self.config.batch_size:(batch_num + 1)*self.config.batch_size]
		if self.config.pad_sequences and self.mask is not None:
			masks = self.mask[batch_num*self.config.batch_size:(batch_num+1)*self.config.batch_size]
		return inputs, labels, masks

	def add_placeholders(self):
		if self.config.embedding_lookup is not None and not self.config.sequence_batch:
			self.inputs_placeholder = tf.placeholder(tf.float32, [None, self.config.max_words], 'input_placeholder')
		elif self.config.embedding_lookup is not None and self.config.sequence_batch:
			assert(self.config.sentence_embeddings is not None), "If both sequence expansion and word sequence is being used, the word sequence must be aggregated to a sentence level to avoid too many dimensions"
			self.inputs_placeholder = tf.placeholder(tf.float32, [None, self.config.seq_length, self.config.max_words], 'input_placeholder')
		else:
			self.inputs_placeholder = tf.placeholder(tf.float32, [None, self.config.seq_length, self.config.input_size], 'input_placeholder')
		self.labels_placeholder = tf.placeholder(tf.float32, [None, self.config.num_classes], 'labels_placeholder')
		if self.config.pad_sequences:
			self.mask_placeholder = tf.placeholder(tf.bool, self.inputs_placeholder.get_shape())
			print "Added mask placeholder: ", self.mask_placeholder
		print "Added input placeholder: ", self.inputs_placeholder

	def create_feed_dict(self, input_batch, labels_batch = None, mask_batch = None):
		feed_dict = { self.inputs_placeholder: input_batch,}
		if labels_batch is not None: feed_dict[self.labels_placeholder] = labels_batch
		if mask_batch is not None: feed_dict[self.mask_placeholder] = mask_batch
		return feed_dict

	def add_embedding(self):
		"""Adds an embedding layer that maps from input tokens (integers) to vectors and then
		concatenates those vectors:
		- Creates an embedding tensor and initializes it with self.pretrained_embeddings.
		- Uses the input_placeholder to index into the embeddings tensor
		"""
		embeddings = tf.nn.embedding_lookup(self.embeddings, tf.cast(self.inputs_placeholder, tf.int64))
		return embeddings

	def add_prediction_op(self):
		"""Runs an rnn on the input using TensorFlows's
		Returns:
			preds: tf.Tensor of shape (batch_size, 1)
		"""
		x = self.inputs_placeholder
		
		if self.embeddings is not None:
			if self.config.embedding_lookup == "pretrained":
				x = self.add_embedding()
			print "After embeddings look ups ", x
		
		# if self.config.sentence_embeddings is not None and self.config.pad_sequences and self.mask_placeholder is not None:
		# 	x = tf.boolean_mask(x, self.mask_placeholder)
		# 	print "After mask, dimensions are ", x
		if self.config.sentence_embeddings == "average":
			# the last axis represents the different words, reduce a dimension for input for compatibility w/ RNN
			x = tf.reduce_mean(x, axis=-2)
		elif self.config.sentence_embeddings == "minmax":
			x = tf.concat(tf.reduce_min(x, axis=-2), tf.reduce_max(x, axis=-2), axis=-2)
		elif self.config.sentence_embeddings is not None:
			raise ValueError("Sentence embeddings has unsupported value %s" % self.config.sentence_embeddings)
		
		if self.config.sentence_embeddings is not None:
			print "After sentence embeddings", x

		if self.cell is not None:
			outputs, _ = tf.nn.dynamic_rnn(self.cell, x, dtype=tf.float32)
			# If there was more than one input in the sequence, just take the hidden layer corresponding to the last
			# item of the sequence
			rnn_seq_length = self.config.seq_length
			# Check if word representations are fed as a sequence instead of windows of sentence representations
			if self.config.sentence_embeddings is None and not self.config.sequence_batch:
				rnn_seq_length = self.config.max_words
			if rnn_seq_length > 1: outputs = outputs[:,-1]
			outputs = tf.reshape(outputs, [-1, self.config.rnn_layers[-1]])
			print "RNN output is ", outputs
		else:
			outputs = tf.reshape(self.inputs_placeholder, [-1, self.config.input_size * rnn_seq_length])
			"Skipping RNN ... output is ", outputs

		if self.dnn is not None:
			for i,layer in enumerate(self.dnn):
				outputs = tf.nn.relu_layer(outputs, layer['W'], layer['b'], name='relu_' + str(i))
			"After DNN, outputs is ", outputs

		return outputs

	def add_loss_op(self, preds):
		"""Adds ops to compute the loss function.
		Args:
			pred: A tensor of shape (batch_size, 1) containing the last
			state of the neural network.
			Returns:
			loss: A 0-d tensor (scalar)
		"""
		print "Logit input is ", preds
		if self.config.loss == "l2":
			preds = tf.sigmoid(preds)
			print "Pred vector is ", preds
			loss_vec = tf.nn.l2_loss(preds-self.labels_placeholder)
		elif self.config.loss == "softmax":
			loss_vec = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=self.labels_placeholder)
		else: raise ValueError("Loss function %s not supported" % self.config.loss)
		print "Loss vector is ", loss_vec
		result = tf.reduce_mean(loss_vec)
		print "Loss is ", result
		return result

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
		print "Gradients are: ", gradients
		grad_vars = zip(*gradients)
		if self.config.clip_gradients:
		    result = tf.clip_by_global_norm(grad_vars[0], self.config.max_grad_norm) 
		    self.grad_norm = tf.global_norm(result[0])
		else:
		    result = grad_vars
		    self.grad_norm = result[0]
		train_op = optimizer.apply_gradients(zip(result[0], grad_vars[1])) 

		assert self.grad_norm is not None, "grad_norm was not set properly!"
		return train_op

	def predict_on_batch(self, sess, input_batch, mask_batch=None):
		feed = self.create_feed_dict(input_batch=input_batch, mask_batch=mask_batch)
		predictions = sess.run(self.pred, feed_dict=feed)
		return predictions

	def train_on_batch(self, sess, input_batch, labels_batch, mask_batch=None):
		"""
		Perform one step of gradient descent on the provided batch of data.
		This version also returns the norm of gradients.
		"""
		feed = self.create_feed_dict(input_batch, labels_batch=labels_batch, mask_batch=mask_batch)
		_, loss, grad_norm = sess.run([self.train_op, self.loss, self.grad_norm], feed_dict=feed)
		return loss, grad_norm

	def run_epoch(self, sess):
		losses, grad_norms = [], []
		for i in range(int(np.ceil(len(self.data)/float(self.config.batch_size)))):
			if self.config.sequence_batch: batches = self.get_sequence_batch(i)
			else: batches = self.get_batch(i)
			if batches is None: continue
			input_batch, labels_batch, mask_batch = batches
			if len(input_batch) == 0: continue
			result = self.train_on_batch(sess, input_batch, labels_batch, mask_batch)
			loss, grad_norm = result
			losses.append(loss)
			grad_norms.append(grad_norm)
		return losses, grad_norms

	def fit(self, sess, X, y, mask=None):
		self.data, self.labels, self.mask = X, y, mask
		losses, grad_norms = [], []
		for epoch in range(self.config.num_epochs):
			print "Epoch %d out of %d" % (epoch + 1, self.config.num_epochs)
			loss, grad_norm = self.run_epoch(sess)
			losses.append(loss)
			grad_norms.append(grad_norm)
		return losses, grad_norms

	def predict(self, sess, inputs, labels, mask_batch=None):
		self.data, self.labels, self.mask = inputs, labels, mask_batch
		total_matches, total_samples = 0, 0
		for i in range(int(np.ceil(len(self.data)/float(self.config.batch_size)))):
			
			# Get data, labels
			if self.config.sequence_batch: batches = self.get_sequence_batch(i)
			else: batches = self.get_batch(i)
			if batches is None: continue
			input_batch, labels_batch, mask_batch = batches
			if len(input_batch) == 0: continue

			# Get predictions
			predictions = self.predict_on_batch(sess, input_batch, mask_batch=mask_batch)
			predicted_classes = tf.argmax(np.array(predictions), axis=1)
			true_classes = tf.argmax(np.array(labels_batch), axis=1)

			# Add to correct and running totals
			total_matches += tf.reduce_sum(tf.to_float(tf.to_int32(tf.equal(true_classes, predicted_classes))))
			total_samples += len(input_batch)

		return sess.run(total_matches*100.0 / float(total_samples))

	def build(self):
		self.add_placeholders()
		self.pred = self.add_prediction_op()
		self.loss = self.add_loss_op(self.pred)
		self.train_op = self.add_training_op(self.loss)


embeddings, vocab = None, None
input_file_directories = glob.glob(args.input_glob)
if args.english_only: input_file_directories = [d for d in input_file_directories if "non-english" not in d]
print "Input directories included are: ", input_file_directories

if config.embedding_lookup is not None:
	print "Embedding lookup ... finding vocab and training blob based on vocab index"
	vocab, training_blob = load_indexed_data_with_vocab(
		[d + "/cleaned_tweets.p" for d in input_file_directories],
		[d + "/" + args.labels_filename for d in input_file_directories],
		test_split=config.test_split,
		random_seed= config.batch_random_seed)
	X_train, X_test, y_train, y_test = training_blob
	print "Vocabulary is %d words" % len(vocab)
	if config.embedding_lookup == "pretrained":
		print "Pretrained embeddings ... loading embeddings"
		embeddings = word_embedding_features(args.embeddings_filename, vocab)
		assert(len(embeddings) == len(vocab)), "Embeddings and vocabulary not the same size!"
		print "Start of embedding for %s: " % vocab[0], embeddings[0][:5]
else:
	print "No embedding lookup ... finding data"
	X_train, X_test, y_train, y_test = load_data(
		[d + "/cleaned_tweets.p" for d in input_file_directories],
		[d + "/" + args.labels_filename for d in input_file_directories],
		[d + "/" + args.embeddings_filename for d in input_file_directories],
		test_split=config.test_split,
		random_seed = config.batch_random_seed)

train_mask, test_mask = None, None
# Use the pad_sequences functions to get the altered training/test data as well
# as a corresponding mask
if config.embedding_lookup is not None and config.pad_sequences:
	print "Embedding lookup with padded sequences ... doing padding modifications"
	assert(vocab is not None), "Vocabulary did not build!"
	# This is the padding word that we add to our vocab
	vocab.append("<NULL_TOKEN>")
	# The last index is for the null word and what we pass in to the pad_sequences
	# function
	zero_input_idx = len(vocab)-1 
	# If we are providing the embeddings, make sure we add an extra embedding to 
	# correspond to the null word, in this case it is just a zero vector
	if config.embedding_lookup == "pretrained" and embeddings is not None:
		zero_input = [0.0]*config.input_size
		embeddings.append(np.array(zero_input))
		embeddings = np.array(embeddings)
		print "Embeddings is shape", embeddings.shape
		print "Start of embeddings corresponding with zero input is" % embeddings[zero_input_idx, :5]
	X_train, train_mask = pad_sequences(X_train, zero_input_idx, config.max_words)
	X_test, test_mask = pad_sequences(X_test, zero_input_idx, config.max_words)
	print "Example padded sequence in training is ", X_train[0]
	print "Corresponding mask is ", train_mask[0]

with tf.Graph().as_default():
	if config.tf_random_seed is not None: tf.set_random_seed(config.tf_random_seed)
	print "Building model..."
	model = SequenceModel(config, pretrained_embeddings=embeddings, vocab=vocab)
	init = tf.global_variables_initializer()
	print "Running session..."
	with tf.Session() as session:
		session.run(init)
		losses, grad_norms = model.fit(session, X_train, y_train, mask=train_mask)
		# make_prediction_plot(config.figure_title, losses, grad_norms)
		print "Losses are" % np.sum(losses, axis=1)
		train_accuracy = model.predict(session, X_train, y_train, mask_batch=train_mask)
		print "Training accuracy is %.6f" % train_accuracy
		test_accuracy = model.predict(session, X_test, y_test, mask_batch=test_mask)
		print "Test accuracy is %.6f" % test_accuracy
