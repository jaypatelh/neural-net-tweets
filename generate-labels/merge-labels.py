import os
import argparse
import pickle
import csv
import glob
import numpy as np

def find_csv_file_path(quake_dir):
    try:
        files = os.listdir(quake_dir)
        for file in files:
            if file.endswith(".csv"):
                return quake_dir + "/" + file
    except OSError as e:
        return None

def remove_bad_files(files, to_filter):
    new_files = []
    for file in files:
        if file in to_filter:
            continue
        new_files.append(file)
    return new_files

def merge_labels_into_one_hot(paid_csv, volunteer_csv, output_path):
    labels = dict()
    unique = set()
    
    if paid_csv is not None:
        with open(paid_csv, "rU") as labels_file:
            print "loading paid data"
            reader = csv.reader(labels_file)
            next(reader, None)
            for row in reader:
                tweetid = str(row[1][1:-1])
                if tweetid not in labels:
                    labels[tweetid] = {}
                labels[tweetid]["paid"] = row[0]
                unique.add(row[0])

    if volunteer_csv is not None:
        with open(volunteer_csv, "rU") as labels_file:
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

    with open(output_path + ".p", 'wb') as output_file:
        pickle.dump(labels_vectors, output_file, protocol=pickle.HIGHEST_PROTOCOL)

    print "Wrote %d labels out to %s" % (len(labels_vectors), output_path)
    return len(labels_vectors)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pl", "--paid")
    parser.add_argument("-vl", "--volunteer")
    parser.add_argument("-o", "--output", required=True)\
    args = parser.parse_args()

    matching_labels = {
        'other_relevant_information': 'other_useful_information',
        'infrastructure_and_utilities': 'infrastructure_and_utilities_damage',
        'money': 'donation_needs_or_offers_or_volunteering_services'
    }

    filter_files = ['Terms of use.txt', '.DS_Store', 'README.txt', '']

    print "Loading labels..."
    total_labels = 0

    paid_dirs = remove_bad_files(list(os.listdir(args.paid)), filter_files)
    volunteer_dirs = remove_bad_files(list(os.listdir(args.volunteer)), filter_files)

    for quake_dir in paid_dirs:
        paid_csv = find_csv_file_path(args.paid + "/" + quake_dir)
        volunteer_csv = find_csv_file_path(args.volunteer + "/" + quake_dir)

        if quake_dir in volunteer_dirs:
            volunteer_dirs.remove(quake_dir)

        total_labels += merge_labels_into_one_hot(paid_csv, volunteer_csv, quake_dir)

    for quake_dir in volunteer_dirs:
        volunteer_csv = find_csv_file_path(args.volunteer + "/" + quake_dir)
        total_labels += merge_labels_into_one_hot(None, volunteer_csv, quake_dir)

    print total_labels