import argparse
import csv
import glob
import pickle
import re

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--labels_blob", required=True) # e.g. */*_labels.csv
parser.add_argument("-d", "--debug_every", default=100, type=int)
args = parser.parse_args()

LABEL_KEY_LIST = ['choose_one_category','label']
labels_filenames = glob.glob(args.labels_blob)
print "Found %d input files" % len(labels_filenames)

for filename in labels_filenames:
	labels = {}
	output_filename = filename.split('.')[0] + ".p"
	print "Saving output at ", output_filename
	# Initialize objects and csv reader
	with open(filename, 'rb') as f:
		rows = f.readlines()[0].split('\r')
		key = rows[0].split(',')
		tweet_id_idx = key.index('tweet_id')
		for label_key in LABEL_KEY_LIST:
			if label_key in key: label_idx = key.index(label_key)
		for row in rows[1:]:
			try:
				row = row.split(',')
				c, tweet_id = row[label_idx], row[tweet_id_idx]
				labels[re.sub("[^0-9]", "", tweet_id)] = c
			except:
				print "Failed label parse: %s" % row
	with open(output_filename, "wb") as output_file: pickle.dump(labels, output_file)