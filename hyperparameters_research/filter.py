"""Prepare data for training, validation and testing."""

import os
import fnmatch
import extract_data


def filter_function(data_directory, n):

    # Dictionary that will count how many times each label appears
    count_labels = dict()

    # Loop over data_set directory
    files = [f for f in os.listdir(data_directory) if fnmatch.fnmatch(f, "*_label.txt")]
    for file in files:

        # Get label
        label = extract_data.extract_label_from_txt(data_directory + file)[1]

        # Increment count_label
        if label not in count_labels:
            count_labels[label] = 1
        else:
            count_labels[label] += 1

    for file in files:

        # Get label
        label = extract_data.extract_label_from_txt(data_directory + file)[1]

        # Delete file if count < n
        if count_labels[label] < n:
            os.remove(data_directory + file)

    print(count_labels)


filter_function("data_set/", 10)
