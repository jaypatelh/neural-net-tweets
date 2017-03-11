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
    
    if paid_csv is not None:
        with open(paid_csv, "rU") as labels_file:
            print "loading paid data"
            reader = csv.reader(labels_file)
            next(reader, None)
            for row in reader:
                tweetid = str(row[1][1:-1])
                if tweetid not in labels:
                    labels[tweetid] = {}
                label = row[0]
                if label not in matching_labels and label not in labels_indices:
                    print "NEW UNIQUE: ", label
                labels[tweetid]["paid"] = label if label not in matching_labels else matching_labels[label]

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
                if label not in matching_labels and label not in labels_indices:
                    print "NEW UNIQUE: ", label
                labels[tweetid]["volunteer"] = label if label not in matching_labels else matching_labels[label]

    print "Done loading labels."

    print "Converting labels into one-hot vectors"
    labels_vectors = dict()
    for tweetid in labels:
        vec = np.zeros(len(labels_indices.keys()))
        label = labels[tweetid]
        if "paid" in label:
            label = label["paid"]
        else:
            label = label["volunteer"]
        if label in weird_labels:
            continue
        vec[labels_indices[label]] = 1.0
        labels_vectors[tweetid] = vec

    with open(output_path + "-labels-03112017.p", 'wb') as output_file:
        pickle.dump(labels_vectors, output_file, protocol=pickle.HIGHEST_PROTOCOL)

    print "Wrote %d labels out to %s" % (len(labels_vectors), output_path)
    return len(labels_vectors)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pl", "--paid")
    parser.add_argument("-vl", "--volunteer")
    parser.add_argument("-o", "--outputdir")
    args = parser.parse_args()

    weird_labels = ['yes', 'no']

    matching_labels = {
        'other_relevant_information': 'other_useful_information',
        'infrastructure_and_utilities': 'infrastructure_and_utilities_damage',
        'money': 'donation_needs_or_offers_or_volunteering_services',
        'volunteer_or_professional_services': 'donation_needs_or_offers_or_volunteering_services',
        'shelter_and_supplies': 'donation_needs_or_offers_or_volunteering_services',
        'infrastructure_damage': 'infrastructure_and_utilities_damage',
        'not_relevant': 'not_related_or_irrelevant',
        'urgent_needs': 'donation_needs_or_offers_or_volunteering_services',
        'other_relevant': 'other_useful_information',
        'displaced_people': 'displaced_people_and_evacuations',
        'personal_updates': 'personal',
        'infrastructure': 'infrastructure_and_utilities_damage',
        'personal_updates_sympathy_support': 'sympathy_and_emotional_support',
        'donations_of_supplies_and/or_volunteer_work': 'donation_needs_or_offers_or_volunteering_services',
        'needs_of_those_affected': 'donation_needs_or_offers_or_volunteering_services',
        'injured_and_dead': 'injured_or_dead_people',
        'donations_of_money': 'donation_needs_or_offers_or_volunteering_services',
        'people_missing_or_found': 'missing_trapped_or_found_people',
        'disease_signs_or_symptoms': 'diseases',
        'treatment': 'diseases',
        'disease_transmission': 'diseases',
        'prevention': 'diseases',
        'affected_people': 'infected_people',
        'deaths_reports': 'injured_or_dead_people',
        'not_related_to_crisis': 'not_related_or_irrelevant',
        'informative': 'other_useful_information',
        'personal_only': 'personal',
        'not_informative': 'not_related_or_irrelevant',
        'humanitarian_aid_provided': 'donation_needs_or_offers_or_volunteering_services',
        'requests_for_help/needs': 'donation_needs_or_offers_or_volunteering_services',
        'praying': 'sympathy_and_emotional_support',
    }

    labels_indices = {
        'not_related_or_irrelevant': 0,
        'donation_needs_or_offers_or_volunteering_services': 1,
        'displaced_people_and_evacuations': 2,
        'animal_management': 3,
        'other_useful_information': 4,
        'response_efforts': 5,
        'caution_and_advice': 6,
        'sympathy_and_emotional_support': 7,
        'injured_or_dead_people': 8,
        'missing_trapped_or_found_people': 9,
        'infrastructure_and_utilities_damage': 10,
        'diseases': 11,
        'infected_people': 12,
        'personal': 13,
        'non-government': 14,
        'physical_landslide': 15,
        'not_physical_landslide': 16,
        'traditional_media': 17,
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

        total_labels += merge_labels_into_one_hot(paid_csv, volunteer_csv, args.outputdir + "/" + quake_dir)

    for quake_dir in volunteer_dirs:
        volunteer_csv = find_csv_file_path(args.volunteer + "/" + quake_dir)
        total_labels += merge_labels_into_one_hot(None, volunteer_csv, args.outputdir + "/" + quake_dir)

    print total_labels
