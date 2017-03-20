import argparse
import numpy as np
import tensorflow as tf
import pickle
from sequence_util import *
from sequence import Config
from sequence import SequenceModel
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_glob", required=True) # 'data/*'
parser.add_argument("-lf", "--labels_filename", required=True) # 'labels-03112017.p'
parser.add_argument("-ef", "--embeddings_filename", required=True) # 'word2vec_average.p'
parser.add_argument("-eo", "--english_only", default=True)
args = parser.parse_args()

baseline_config = Config()
configs = [Config() for _ in range(1)]

def run_graph(config, X_train, X_test, y_train, y_test, embeddings=None, train_mask=None, test_mask=None):
	with tf.Graph().as_default():
		if config.tf_random_seed is not None: tf.set_random_seed(config.tf_random_seed)
		print "Building model..."
		model = SequenceModel(config, pretrained_embeddings=embeddings)
		init = tf.global_variables_initializer()
		print "Running session..."
		with tf.Session() as session:
			session.run(init)
			losses, grad_norms = model.fit(session, X_train, y_train, mask=train_mask)
			make_prediction_plot(config.figure_title, losses, grad_norms)
			print "Losses are", np.sum(losses, axis=1)
			train_accuracy = model.predict(session, X_train, y_train, mask_batch=train_mask)
			print "Training accuracy is %.6f" % train_accuracy
			test_accuracy = model.predict(session, X_test, y_test, mask_batch=test_mask)
			print "Test accuracy is %.6f" % test_accuracy

input_file_directories = glob.glob(args.input_glob)
if args.english_only: input_file_directories = [d for d in input_file_directories if "non-english" not in d]
print "Input directories included are: ", input_file_directories

X_train, X_test, y_train, y_test = load_data(
	[d + "/cleaned_tweets.p" for d in input_file_directories],
	[d + "/" + args.labels_filename for d in input_file_directories],
	[d + "/" + args.embeddings_filename for d in input_file_directories],
	test_split=baseline_config.test_split,
	random_seed = baseline_config.batch_random_seed)

for i, config in enumerate(configs):
	print "Running config %d" % (i+1) 
	run_graph(config, X_train, X_test, y_train, y_test)

