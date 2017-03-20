import numpy as np
import tensorflow as tf
from gensim.models import word2vec
import glob
import pickle
import tweepy
from sklearn.model_selection import train_test_split

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

def sequence_cells(cell_fn, layers, dropout_p):
	if len(layers) == 1:
		if dropout_p == 0: return cell_fn(layers[0])
		else: return tf.contrib.rnn.DropoutWrapper(cell_fn(layers[0]), dropout_p)
	# For multi layer cells
	rnn_layers = [cell_fn(layer_dim) for layer_dim in layers]
	if dropout_p > 0: [tf.contrib.rnn.DropoutWrapper(cell_fn(layer_dim), dropout_p) for layer_dim in layers]
	return tf.nn.rnn_cell.MultiRNNCell(rnn_layers, state_is_tuple=True)
 
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
    masks = []

    for i in range(len(data)):
        sentence = data[i]
        len_dif = max_length - len(sentence)
        if (len_dif < 0):
            newS = sentence[:max_length]
            mSeq = [True]*max_length
        else:
            newS = list(sentence)
            mSeq = [True]*len(sentence)
            for i in range(0,len_dif):
                newS.append(zero_input)
                mSeq.append(False)
        new_data.append(newS)
        masks.append(mSeq)

    return new_data, masks

# def make_prediction_plot(title, losses, grad_norms):
#     plt.subplot(2, 1, 1)
#     plt.title(title)
#     plt.plot(np.arange(losses.size), losses.flatten(), label="Loss")
#     plt.ylabel("Loss")

#     plt.subplot(2, 1, 2)
#     plt.plot(np.arange(grad_norms.size), grad_norms.flatten(), label="Gradients")
#     plt.ylabel("Gradients")
#     plt.xlabel("Minibatch")