#takes in two dictionaries -- one is tweet IDs --> feature vectors and one is tweet IDs --> labels 
#Creates feature vector --> labels mapping 
#Separates into two lists
#Separates 80:20 for train and test 
#Returns train_data, train_labels, test_data, test_labels
#Saves these to output pickle files 

import argparse
import numpy as np
import glob
import pickle
from baseline_models import *
import os

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--tweet_glob", required=True) #set of folders for each event 
parser.add_argument("-f", "--feat_vec", required=True)
parser.add_argument("-m", "--model_type", required=True)
args = parser.parse_args()

method = args.feat_vec


training_data = []
test_data = []
training_labels = []
test_labels = []
parent_dirs = glob.glob(args.tweet_glob)
for directory in parent_dirs:
	feature_vectors = {}
	feat_vec_list = os.listdir(directory+"/"+method)
	feat_vec_list = [filename for filename in feat_vec_list if filename[0] != '.']
	with open(directory+"/labels.p", "rb") as labels_file:
		labels = pickle.load(labels_file)
	for filename in feat_vec_list:
		filepath = directory+"/"+method+"/"+filename
		with open(filepath, "rb") as input_file:
			feature_vectors.update(pickle.load(input_file))

	keys = [tweet_id for tweet_id in labels if str(tweet_id) in feature_vectors and 'unknown' not in labels.values()]
	x_val = [feature_vectors[tweet_id] for tweet_id in keys]
	y_val = []
	for tweet_id in keys:
		if "paid" in labels[tweet_id]: y_val.append(labels[tweet_id])["paid"]
		elif "volunteer" in labels[tweet_id]: y_val.append(labels[tweet_id]["volunteer"])

	num_train = int(.8*(len(x_val)))
	training_data.extend(x_val[:num_train])
	test_data.extend(x_val[num_train:])
	training_labels.extend(y_val[:num_train])
	test_labels.extend(y_val[num_train:])

clf = train_model(training_data, training_labels, glob.glob(args.model_type))
train_score = calculate_score(training_data, training_labels, clf)
test_score = calculate_score(test_data, test_labels, clf)
predicted_labels = predict_labels(test_data, clf)
print train_score
print test_score
print test_labels
print predicted_labels

