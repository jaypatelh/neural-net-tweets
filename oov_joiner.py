import argparse
import csv
import glob
import pickle
import re

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--oov_file", default="OOV_Dict/OOV_Dictionary_V1.0.tsv")
parser.add_argument("-d", "--debug_every", default=100, type=int)
parser.add_argument("-w", "--warmstart", default=True, type=bool)
parser.add_argument("-i", "--input_glob", required=True) #california_earthquake/tweets/*
parser.add_argument("-o", "--output_dir", required=True) #california_earthquake/tweets_cleaned/*
args = parser.parse_args()

URL_REGEX_PATTERN = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

# Parse the tsv file in to a dictionary mapping out of vocabulary words to in vocabulary words
def build_oov_dict(oov_filename):
	oov_dict = {}
	with open(oov_filename, 'rb') as tsvfile:
		lines = []
		for c in tsvfile.readlines(): lines.extend(c.split('\r'))
		for line in lines:
			key, val = line.split('\t')
			oov_dict[key] = val
	return oov_dict

def clean_and_format_text(tweet_text):
	# First, remove all user handlers - i.e. any word beginning with the '@' sign
	text = re.sub(r'@\S+','',tweet_text)
	# Remove all URL formats
	text = re.sub(URL_REGEX_PATTERN, '', text)
	# Then remove all characters that are not in the alphabet, a digit, whitespace,
	# in addition to the appostrophe which is still used in contractions, periods,
	# and colons, which will be dealt with in the next steps
	text = re.sub(r'[^\w\s\'\.\:]', '', text)
	# Keep only colons in between two numbers (for a time)
	text = re.sub(r'(\D)\:(\D)',r'\1\2', text)
	# Keep only periods that are followed by a number and preceded by either a number
	# or a whitespace
	text = re.sub(r'\.(\D)', r'\1', text)
	text = re.sub(r'([^\s\d])\.', r'\1', text)
	# Keep only apostrophes that follow an alphabet character and are followed by either
	# an alphabet character or a whitespace
	text = re.sub(r'([^a-zA-Z])\'', r'\1', text)
	text = re.sub(r'\'([^a-zA-Z\s])', r'\1', text)
	# Return word vector, excluding start marker 'RT'
	return [word for word in text.split() if word != 'RT']

def replace_oov_words(oov_dict, words):
	result = []
	for word in words:
		# If the word is in the OOV dictionary, clean the resulting in vocabulary text
		# and add it to the result. Otherwise, just add the original word
		if word in oov_dict: result.extend(clean_and_format_text(oov_dict[word]))
		else: result.append(word)
	return result


oov_dict = build_oov_dict(args.oov_file)
tsv_filenames = glob.glob(args.input_glob)
print "Found %d input files" % len(tsv_filenames)

for filename in tsv_filenames:
	output_filename = filename.split('/')[-1]
	output_filename = args.output_dir + "/" + output_filename.split('.')[0] + ".p"
	print "Saving output at ", output_filename

	with open(filename, 'rb') as input_file:
		tweet_text, _ = pickle.load(input_file)
		for tweet_id in tweet_text:
			cleaned_text = clean_and_format_text(tweet_text[tweet_id])
			tweet_text[tweet_id] = replace_oov_words(oov_dict, cleaned_text)

	with open(output_filename, "wb") as output_file:
		pickle.dump(tweet_text, output_file)
