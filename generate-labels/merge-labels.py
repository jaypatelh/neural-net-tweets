import argparse
import pickle
import csv
import glob

parser = argparse.ArgumentParser()
parser.add_argument("-pl", "--paid")
parser.add_argument("-vl", "--volunteer")
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

matching_labels = {
    'other_relevant_information': 'other_useful_information',
    'infrastructure_and_utilities': 'infrastructure_and_utilities_damage'
}

print "Loading labels..."
labels = dict()

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

print "Done loading labels."

with open(args.output, 'wb') as output_file:
    pickle.dump(labels, output_file, protocol=pickle.HIGHEST_PROTOCOL)

print "Wrote labels out to", args.output
