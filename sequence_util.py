import numpy as np
import tensorflow as tf
import glob
import pickle
import tweepy
from sklearn.model_selection import train_test_split

def get_tweet_ids_time_ordered(tweets_file_pattern):
	sorted_tweet_ids = []
	filenames = glob.glob(tweets_file_pattern)
	print "found %d tweet files" % len(filenames)
	for filename in filenames:
		print "unloading %s" % filename
		with open(filename, "rb") as input_file:
			tweets, _ = pickle.load(input_file)
			for tweet_id in tweets: sorted_tweet_ids.append([tweet_id, tweets[tweet_id].created_at])
	sorted_tweet_ids.sort(key=lambda x: x[1])
	return [x[0] for x in sorted_tweet_ids]

def get_labels(labels_file_pattern, tweet_ids):
	labels = {}
	filenames = glob.glob(labels_file_pattern)
	print "found %d label files" % len(filenames)
	for filename in filenames:
		with open(filename, "rb") as input_file:
			print "unloading %s" % filename
			l = pickle.load(input_file)
			for key in l: labels[key] = l[key]
	return [labels[tweet_id] for tweet_id in tweet_ids if tweet_id in labels]

def get_embeddings(embeddings_file_pattern, tweet_ids):
	embeddings = {}
	filenames = glob.glob(embeddings_file_pattern)
	print "found %d embedding files" % len(filenames)
	for filename in filenames:
		print "unloading %s" % filename
		with open(filename, "rb") as input_file:
			e = pickle.load(input_file)
			for key in e: embeddings[key] = e[key]
	return [embeddings[tweet_id] for tweet_id in tweet_ids if tweet_id in embeddings]

def get_training_data(input_glob, label_glob, embedding_glob, test_split=0.3, embedding_map=True, random_seed=None):
	# Read in all data and sort in accordance to time sequence
	tweet_ids = get_tweet_ids_time_ordered(input_glob)
	labels = get_labels(label_glob, tweet_ids)
	assert(len(labels)==len(tweet_ids)), "Tweet ids in pickle files do not map fully to labels"
	embeddings = get_embeddings(embedding_glob, tweet_ids)
	assert(len(embeddings)==len(tweet_ids)), "Tweet ids in pickle files do not map fully to embeddings"
	return train_test_split(embeddings, labels, test_size=test_split, random_state=random_seed)

def sequence_cells(cell_fn, layers, dropout_p):
	if len(layers) == 1:
		if dropout_p == 0: return cell_fn(layers[0])
		else: return tf.contrib.rnn.DropoutWrapper(cell_fn(layers[0]), dropout_p)
	# For multi layer cells
	rnn_layers = [cell_fn(layer_dim) for layer_dim in layers]
	if dropout_p > 0: [tf.contrib.rnn.DropoutWrapper(cell_fn(layer_dim), dropout_p) for layer_dim in layers]
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
