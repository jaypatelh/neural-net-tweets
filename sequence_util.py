import numpy as np
import tensorflow as tf
from gensim.models import word2vec
import glob
import pickle
import tweepy
from sklearn.model_selection import train_test_split
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt

def get_tweet_ids_time_ordered(filenames, include_text=False):
	sorted_tweet_ids = []
	for filename in filenames:
		assert(len(glob.glob(filename)) == 1), "File %s not found!" % filename
		with open(filename, "rb") as input_file:
			print "unloading %s" % filename
			tweets = pickle.load(input_file)
			for tweet_id in tweets:
				if tweets[tweet_id].text == []: continue
				result = [tweet_id, tweets[tweet_id].created_at]
				if include_text: result.append(tweets[tweet_id].text)
				sorted_tweet_ids.append(result)
	sorted_tweet_ids.sort(key=lambda x: x[1])
	if include_text: return [(x[0], x[2]) for x in sorted_tweet_ids]
	return [x[0] for x in sorted_tweet_ids]

def get_labels(filenames, tweet_ids):
	labels = {}
	for filename in filenames:
		assert(len(glob.glob(filename)) == 1), "File %s not found!" % filename
		with open(filename, "rb") as input_file:
			print "unloading %s" % filename
			l = pickle.load(input_file)
			for key in l: labels[key] = l[key]
	tweet_ids = [tweet_id for tweet_id in tweet_ids if tweet_id in labels]
	labels = [labels[str(tweet_id)] for tweet_id in tweet_ids]
	return tweet_ids, labels

def get_embeddings(filenames, tweet_ids):
	embeddings = {}
	for filename in filenames:
		assert(len(glob.glob(filename)) == 1), "File %s not found!" % filename
		with open(filename, "rb") as input_file:
			print "unloading %s" % filename
			e = pickle.load(input_file)
			for key in e: embeddings[key] = e[key]
	return [embeddings[tweet_id] for tweet_id in tweet_ids if tweet_id in embeddings]

def load_data(input_files, label_files, embedding_files, test_split=None, random_seed=None):
	# Read in all data and sort in accordance to time sequence
	tweet_ids = get_tweet_ids_time_ordered(input_files)
	num_tweets = len(tweet_ids)
	tweet_ids, labels = get_labels(label_files, tweet_ids)
	if len(tweet_ids) != num_tweets: print "Discrepancy: labels file has %d tweets and there are %d tweet ids" % (len(tweet_ids), num_tweets)
	embeddings = get_embeddings(embedding_files, tweet_ids)
	assert(len(embeddings)==len(tweet_ids)), "Tweet ids in pickle files do not map fully to labels: %d tweet ids and %d labels" % (num_tweets, len(embeddings))
	print "Found %d total tweets with matching labels" % len(tweet_ids)
	return train_test_split(embeddings, labels, test_size=test_split, random_state=random_seed)

def load_indexed_data_with_vocab(input_files, label_files, test_split=None, random_seed=None):
	# Read in all data and sort in accordance to time sequence
	tweets = get_tweet_ids_time_ordered(input_files, include_text=True)
	tweet_ids = [pair[0] for pair in tweets]
	num_tweets = len(tweet_ids)
	tweet_ids, labels = get_labels(label_files, tweet_ids)
	if len(tweet_ids) != num_tweets:
		print "Discrepancy: labels file has %d tweets and there are %d tweet ids" % (len(tweet_ids), num_tweets)
	print "Found %d total tweets with matching labels" % len(tweet_ids)
	sentences, words = [], []
	word_to_idx_map = {}
	num_words = 0
	for pair in tweets:
		tweet_id, sentence = pair
		if tweet_id not in tweet_ids: continue
		word_indices = []
		for word in sentence:
			if word not in word_to_idx_map:
				words.append(word)
				word_to_idx_map[word] = num_words
				num_words += 1
			word_indices.append(word_to_idx_map[word])
		sentences.append(word_indices)
	return words, train_test_split(sentences, labels, test_size=test_split, random_state=random_seed)

def sequence_cells(cell_fn, layers, dropout_p=0.0, attention=None):
	if len(layers) == 1:
		cell = cell_fn(layers[0])
		if dropout_p > 0:
			cell = tf.contrib.rnn.DropoutWrapper(cell, dropout_p)
			print "Added dropout to cell ", cell
		if attention is not None:
			attn_length, attn_size, attn_vec_size = attention
			cell = tf.contrib.rnn.AttentionCellWrapper(
				cell, attn_length=attn_length, attn_size=attn_size,
            	attn_vec_size=attn_vec_size, state_is_tuple=True)
			print "Added attention to cell ", cell
		return cell
	
	# For multi layer cells
	rnn_layers = [cell_fn(layer_dim) for layer_dim in layers]
	if attention is not None:
		attn_length, attn_size, attn_vec_size = attention
        rnn_layers = [tf.contrib.rnn.AttentionCellWrapper(
            rnn_cell, attn_length=attn_length, attn_size=attn_size,
            attn_vec_size=attn_vec_size, state_is_tuple=True) for rnn_cell in rnn_layers]
        print "Added attention to each layer ", rnn_layers
	if dropout_p > 0:
		rnn_layers = [tf.contrib.rnn.DropoutWrapper(rnn_cell, dropout_p) for rnn_cell in rnn_layers]
		print "Added dropout to each layer ", rnn_layers
	return tf.contrib.rnn.MultiRNNCell(rnn_layers, state_is_tuple=True)
 
def dnn_layers(input_size, hidden_layers, random_seed=None):
	if random_seed != None: np.random.seed(random_seed)
	dnn_layers = []
	hidden_layers.insert(0, input_size)
	for i in range(1, len(hidden_layers)):
		W = tf.Variable(np.random.rand(hidden_layers[i-1], hidden_layers[i]), dtype=tf.float32)
		b = tf.Variable(np.zeros(hidden_layers[i]), dtype=tf.float32)
		dnn_layers.append({'W': W, 'b': b})
	return dnn_layers

# Creates features for each input sentence (each sentence is a list of words) based on averaging the words in the vector  
def word_embedding_features(model_filename, input_list, aggregation=None):
	model = word2vec.Word2Vec.load(model_filename)
	embeddings = []
	for word in input_list:
		if word not in model: print "Could not find %s in the model file %s" % (word, model_filename)
		else: embeddings.append(model[word])
	return embeddings

def pad_sequences(data, zero_input, max_length):
    """Ensures each input-output seqeunce pair in @data is of length
    @max_length by padding it with zeros and truncating the rest of the
    sequence.

    The zero label is created in this case as an addition to the one hot
    label vectors.

    Example: for the (sentence, labels) pair: [[4,1], [6,0], [7,0]], [1,
    0, 0], and max_length = 5, we would construct
        - a new sentence: [[4,1], [6,0], [7,0], [0,0], [0,0]]
        - a new label seqeunce: [1, 0, 0, 4, 4], and
        - a masking seqeunce: [True, True, True, False, False].
    """
    new_data = []
    lengths = []

    for i in range(len(data)):
        sentence = data[i]
        sentence_len = len(sentence)
        len_dif = max_length - sentence_len
        if (len_dif < 0):
            newS = sentence[:max_length]
            mSeq = [True]*max_length
            sentence_len = max_length
        else:
            newS = list(sentence)
            for i in range(0,len_dif):
                newS.append(zero_input)
        new_data.append(newS)
        lengths.append(sentence_len)

    return new_data, lengths

def load_data_with_embedding_lookup(config, input_file_directories, labels_filename, embeddings_filename):
	print "Embedding lookup ... finding vocab and training blob based on vocab index"
	vocab, training_blob = load_indexed_data_with_vocab(
		[d + "/cleaned_tweets.p" for d in input_file_directories],
		[d + "/" + labels_filename for d in input_file_directories],
		test_split=config.test_split,
		random_seed= config.batch_random_seed)
	X_train, X_test, y_train, y_test = training_blob
	print "Vocabulary is %d words" % len(vocab)

	if config.embedding_lookup == "pretrained":
		print "Pretrained embeddings ... loading embeddings"
		embeddings = word_embedding_features(embeddings_filename, vocab)
		assert(len(embeddings) == len(vocab)), "Embeddings and vocabulary not the same size!"
		print "Start of embedding for %s: " % vocab[0], embeddings[0][:5]
	# If embeddings are trained from scratch, just initialize an array of the correct dimension to be the pretrained
	# embedding
	elif config.embedding_lookup == "create": embeddings = np.random.rand(len(vocab), config.input_size)
	else: raise ValueError("%s not recognized as a supported embedding lookup!" % config.embedding_lookup)

	train_lengths, test_lengths = None, None
	print "Embedding lookup with padded sequences ... doing padding modifications"
	assert(vocab is not None), "Vocabulary did not build!"
	# This is the padding word that we add to our vocab
	vocab.append("<NULL_TOKEN>")
	# The last index is for the null word and what we pass in to the pad_sequences
	# function
	zero_input_idx = len(vocab)-1 
	# If we are providing the embeddings, make sure we add an extra embedding to 
	# correspond to the null word, in this case it is just a zero vector
	assert(embeddings is not None), "Embeddings were not created!"

	# Pad sequences with zero input and create a corresponding zero input vector in the embeddings
	zero_input = [0.0]*config.input_size
	embeddings.append(np.array(zero_input))
	embeddings = np.array(embeddings)
	print "Embeddings is shape", embeddings.shape
	print "Start of embeddings corresponding with zero input is" % embeddings[zero_input_idx, :5]
	X_train, train_lengths = pad_sequences(X_train, zero_input_idx, config.max_words)
	X_test, test_lengths = pad_sequences(X_test, zero_input_idx, config.max_words)
	print "Example padded sequence in training is ", X_train[0]
	print "Corresponding lengths are ", train_lengths[0]

	return embeddings, X_train, X_test, y_train, y_test, train_lengths, test_lengths

def make_prediction_plot(title, losses, grad_norms):
	pass
	# losses, grad_norms = np.array(losses), np.sum(np.array(grad_norms), axis=0)
	# plt.subplot(2, 1, 1)
	# plt.title(title)
	# plt.plot(np.arange(losses.shape[0]), np.sum(losses, axis=1), label="Loss")
	# plt.ylabel("Loss")

	# plt.subplot(2, 1, 2)
	# plt.plot(np.arange(grad_norms.shape[0]), np.sum(grad_norms, axis=1), label="Gradients")
	# plt.ylabel("Gradients")
	# plt.xlabel("Minibatch")