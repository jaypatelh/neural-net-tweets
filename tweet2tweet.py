import argparse
import numpy as np
import tensorflow as tf
import pickle
from sequence_util import *
from sequence import TweetSequenceLookupEmbeddingSequenceConfig
from sequence import SequenceModel

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_glob", required=True) # 'data/*'
parser.add_argument("-lf", "--labels_filename", required=True) # 'labels-03112017.p'
parser.add_argument("-ef", "--embeddings_filename", required=True) # 'word2vec_model'
parser.add_argument("-eo", "--english_only", default=True)
args = parser.parse_args()


baseline_config = TweetSequenceLookupEmbeddingSequenceConfig()
configs = [TweetSequenceLookupEmbeddingSequenceConfig() for _ in range(1)]

def run_graph(config, X_train, X_test, y_train, y_test, embeddings=None, train_lengths=None, test_lengths=None):
	with tf.Graph().as_default():
		if config.tf_random_seed is not None: tf.set_random_seed(config.tf_random_seed)
		print "Building model..."
		model = SequenceModel(config, pretrained_embeddings=embeddings)
		init = tf.global_variables_initializer()
		print "Running session..."
		with tf.Session() as session:
			session.run(init)
			losses, grad_norms = model.fit(session, X_train, y_train, lengths=train_lengths)
			make_prediction_plot(config.figure_title, losses, grad_norms)
			print "Losses are", np.sum(losses, axis=1)
			train_accuracy = model.predict(session, X_train, y_train, lengths_batch=train_lengths)
			print "Training accuracy is %.6f" % train_accuracy
			test_accuracy = model.predict(session, X_test, y_test, lengths_batch=test_lengths)
			print "Test accuracy is %.6f" % test_accuracy

input_file_directories = glob.glob(args.input_glob)
if args.english_only: input_file_directories = [d for d in input_file_directories if "non-english" not in d]
print "Input directories included are: ", input_file_directories

embeddings, X_train, X_test, y_train, y_test, train_lengths, test_lengths = load_data_with_embedding_lookup(baseline_config, input_file_directories, args.labels_filename, args.embeddings_filename)

for i, config in enumerate(configs):
	print "Running config with %d max words" % config.max_words
	run_graph(config, X_train, X_test, y_train, y_test, embeddings=embeddings, train_lengths=train_lengths, test_lengths=test_lengths)



