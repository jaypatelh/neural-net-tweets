import argparse
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import word2vec
import numpy as np
import glob
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_glob", required=True) # */tweets_cleaned.p
parser.add_argument("-m", "--model_filename", required=True) # e.g. 'word2vec_embedding_model'
parser.add_argument("-w", "--warmstart", default=False, type=bool)
parser.add_argument("-b", "--batch_size", default=256, type=int)
parser.add_argument("-e", "--embed_size", default=200, type=int)
parser.add_argument("-t", "--model", required=True) # e.g. "bigram" or "word2vec"
args = parser.parse_args()

# Creates a word2vec model based on provided list of sentences (each sentence is a list of words)
# Filename is the name that you want to save the model as
def train_word_embedding_model(input_list, filename, model):
	if model == None:
		model = word2vec.Word2Vec(input_list, size=args.embed_size, min_count=1)
	else:
		model.train(input_list)
	# Save model so that if our program quits in the middle, we still have checkpoints from after each input file
	model.save(filename)

# Create a word2vec model with the given arguments
def word_embedding_model():
	# If warmstart is true and viable, load the model from the given model filename
	model = None
	if args.warmstart and glob.glob(args.model_filename) > 0:
		print "Warmstarting model from %s" % args.output_model_filename
		model = word2vec.Word2Vec.load(filename)

	# Get all input files to train on, and pass them in to the model, saving the model to the given filename after
	# each input file is processed
	input_filenames = glob.glob(args.input_glob)
	print "Found %d input files for training" % len(input_filenames)
	X = []
	for filename in input_filenames:
		with open(filename, "rb") as input_file:
			tweets = pickle.load(input_file)
			tweet_texts = [tweet.text for tweet in tweets.values()]
			X.extend(tweet_texts)
	train_word_embedding_model(X, args.model_filename, model)

# Create a bigram model with the given arguments, save a bigram vectorizer object
# http://scikit-learn.org/stable/modules/feature_extraction.html
def bigram_model():
	input_filenames = glob.glob(args.input_glob)
	print "Found %d input files for training" % len(input_filenames)
	bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
	for filename in input_filenames:
		with open(filename, "rb") as input_file:
			tweets = pickle.load(input_file)
			tweet_texts = [tweet.text for tweet in tweets.values()]
			sentences = [' '.join(sentence) for sentence in tweet_texts]
			bigram_vectorizer.fit_transform(sentences)
	with open(args.model_filename, 'wb') as output_file:
		pickle.dump(bigram_vectorizer, output_file)

if args.model == "bigram": bigram_model()
elif args.model == "word2vec": word_embedding_model()
else: print "Model %s not recognized." % args.model