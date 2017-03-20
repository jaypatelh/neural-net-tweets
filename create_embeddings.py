import argparse
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import word2vec
import numpy as np
import glob
import pickle

# NOTE: Files will be saved in the path <one directory before the input file>/<model_name>/<filename_without_extension>.p
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_glob", required=True) # e.g. */cleaned_tweets.p
parser.add_argument("-m", "--model_filename", required=True) # e.g. bigram_model
parser.add_argument("-t", "--model", required=True) # e.g. "bigram" or "word2vec_average"
parser.add_argument("-o", "--output_filename", default="embedding") #e.g. 
parser.add_argument("-e", "--sentence_embedding", required=True)
args = parser.parse_args()

# Creates features for each input sentence (each sentence is a list of words) based on averaging the words in the vector  
def word_embedding_features(model_filename, input_list, aggregation=None):
	model = word2vec.Word2Vec.load(model_filename)
	# print model.wv.similarity('sunny', 'end')
	# print model.wv.most_similar(positive=['landslide', 'emergency'], negative=['aid'])
	embedding_list = []
	for sentence in input_list:
		if sentence == []:
			continue
		embeddings = [model[word] for word in sentence]
		if aggregation == "average":
			sentence_embedding = np.mean(embeddings, axis=0)
		elif aggregation== "minmax":
			sentence_embedding = np.append(np.min(embeddings, axis=0), np.max(embeddings, axis=0), axis=0)
		elif aggregation == None:
			embedding_list.append(sentence_embedding)
		else: raise ValueError("aggregation value is not recognized.")
	return embedding_list

#creates features for each input sentence (each sentence is a string) based on bigrams
def bigram_features(input_list):
	input_text = [' '.join(sentence) for sentence in input_list]
	with open(args.model_filename, 'rb') as input_file:
		bigram_vectorizer = pickle.load(input_file)
		feature_vectors = bigram_vectorizer.fit_transform(input_text).toarray()
		return feature_vectors

# Ensure that some model file exists
assert(len(glob.glob(args.model_filename)) > 0), "Model file does not seem to exist."
input_filenames = glob.glob(args.input_glob)

# Iterate through each input file matching the input glob
for filename in input_filenames:
	output_filename = "%s/%s" % ('/'.join(filename.split('/')[:-1]), args.output_filename)
	print "Saving output to", output_filename

	# Input all text from the input file in to the corresponding model featurizer, and dump a new
	# dictionary mapping tweet ids to the features in to the corresponding output file
	with open(filename, 'rb') as input_file:
		tweets = pickle.load(input_file)
		embeddings = []
		keys, values = tweets.keys(), tweets.values()
		tweet_text = [tweet.text for tweet in values]
		tweet_text = filter(lambda a: a != [], tweet_text)
		if args.model == "word2vec":
			embeddings = word_embedding_features(args.model_filename, tweet_text, args.sentence_embedding)
		elif args.model == "bigram":
			embeddings = bigram_features(tweet_text)
		else:
			print "Model %s not recognized." % args.model
			break
		assert(len(embeddings) == len(tweet_text)), "Returned list of embeddings is %d while the number of inputs was %d" % (len(embeddings), len(tweet_text))
		
		# Replace each tweet id entry with its corresponding feature value
		i = 0
		for key in keys: 
			if tweets[key].text != []:
				tweets[key] = embeddings[i]
				i += 1
		with open(output_filename, "wb") as output_file:
			pickle.dump(tweets, output_file)