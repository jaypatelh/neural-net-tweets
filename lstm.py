import argparse
import numpy as np
import tensorflow as tf
import glob
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_glob", required=True) # data/*/word2vec_average.p
parser.add_argument("-m", "--model_filename", required=True) # e.g. 'lstm_model'
parser.add_argument("-w", "--warmstart", default=False, type=bool)
parser.add_argument("-b", "--batch_size", default=256, type=int)
parser.add_argument("-e", "--embed_size", default=200, type=int)
parser.add_argument("-l", "--learning_rate", default=0.001, type=float)
parser.add_argument("-d", "--decay", default=0.99, type=int)
parser.add_argument("-s", "--steps", default=1000, type=int)
parser.add_argument("-e", "--epochs", default=50, type=int)
parser.add_argument("-h", "--hidden_size", default=20, type=int)
parser.add_argument("-d", "--dropout_p", default=0.1, type=float)
args = parser.parse_args()

def get_embeddings(offset, batch_size, filename):
	with open(filename, "rb") as input_file:
		tweets = pickle.load(input_file)
		tweet_texts = [tweet.text for tweet in tweets.values()[offset:offset+bath_size]]

class TweetLSTM():

	def __init__():
		self.filenames = glob.glob(args.input_glob)
		print "found %d input files" % len(self.filenames)
		self.output_filename = args.model_filename
		self.batch_size = args.batch_size

