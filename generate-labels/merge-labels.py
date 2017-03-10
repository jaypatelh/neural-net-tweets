import argparse
import pickle
import csv
import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-pl", "--paid")
parser.add_argument("-vl", "--volunteer")
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

matching_labels = {
    'other_relevant_information': 'other_useful_information',
    'infrastructure_and_utilities': 'infrastructure_and_utilities_damage',
    'money': 'donation_needs_or_offers_or_volunteering_services'
}

print "Loading labels..."
labels = dict()
unique = set()

if args.paid is not None:
    with open(args.paid, "rU") as labels_file:
        print "loading paid data"
        reader = csv.reader(labels_file)
        next(reader, None)
        for row in reader:
            tweetid = str(row[1][1:-1])
            if tweetid not in labels:
                labels[tweetid] = {}
            labels[tweetid]["paid"] = row[0]
            unique.add(row[0])

if args.volunteer is not None:
    with open(args.volunteer, "rU") as labels_file:
        print "loading volunteer data"
        reader = csv.reader(labels_file)
        next(reader, None)
        for row in reader:
            tweetid = str(row[0][1:-1])
            if tweetid not in labels:
                labels[tweetid] = {}
            label = '_'.join(row[2].replace(',', '').split()).lower()
            labels[tweetid]["volunteer"] = label if label not in matching_labels else matching_labels[label]
            unique.add(labels[tweetid]["volunteer"])

print "Done loading labels."

print "Converting labels into one-hot vectors"
unique = list(unique)
num_unique = len(unique)
labels_vectors = dict()
for tweetid in labels:
    vec = np.zeros(num_unique)
    label = labels[tweetid]
    if "paid" in label:
        label = label["paid"]
    else:
        label = label["volunteer"]
    vec[unique.index(label)] = 1.0
    labels_vectors[tweetid] = vec

with open(args.output, 'wb') as output_file:
    pickle.dump(labels_vectors, output_file, protocol=pickle.HIGHEST_PROTOCOL)

print "Wrote labels out to", args.output
