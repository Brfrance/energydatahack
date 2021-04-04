"""
Lit les fichiers .txt et extrait les données.
"""

import os
import fnmatch
import numpy as np
import extract_data


def create_samples(time_series, n_steps):
    """
    Extract features from samples. (Min, Max, Mean, STD)
    """

    # Split a univariable sequence into samples
    X = list()
    n = time_series.shape[0]
    for i in range(n - n_steps):
        sample = time_series[i:i+n_steps]
        minimum, maximum = min(sample), max(sample)
        mean, std = np.mean(sample), np.std(sample)
        features = np.array([minimum, maximum, mean, std])
        X.append(features)
    return X


def get_data_sets(cnn_n_input, data_directory="data_set/", n_data_sets=5):
    """
    Retrieve data and partition it into n_data_sets for cross-validation.
    """

    print("Partitioning data into", str(n_data_sets), "splits.")

    # Get list of labels
    list_labels = extract_data.get_labels(data_directory + "labels.txt")
    n_labels = len(list_labels)

    # Dictionary that gives labels ID
    label_to_int = dict()
    for i in range(n_labels):
        label_to_int[list_labels[i]] = i

    # Dictionary that will count how many times each label appears
    count_labels = dict()

    # Data partitions : (time series, labels)
    data_partition = [(list(), list()) for _ in range(n_data_sets)]

    # Loop over data_set directory
    files = [f for f in os.listdir(data_directory) if fnmatch.fnmatch(f, "*_label.txt")]
    for file in files:

        # Get label
        label = extract_data.extract_label_from_txt(data_directory + file)[1]
        # Increment label count
        if label in count_labels:
            count_labels[label] += 1
        else:
            count_labels[label] = 1
        # Label_id
        label_id = label_to_int[label]

        # Get time series (data)
        data = extract_data.extract_data_from_txt(data_directory + "MIN " + file.split('_')[0] + ".txt")\
            .Value.values.astype(dtype="uint16", copy=False)
        # Split data into samples
        data = create_samples(data, cnn_n_input)
        # Create labels
        labels = [label_id] * len(data)

        # Append to partition
        data_partition[count_labels[label] % n_data_sets][0].extend(data)            # Add data
        data_partition[count_labels[label] % n_data_sets][1].extend(labels)          # Add labels

    print("--\nBuilding types inventory :")
    print(count_labels)

    print("--\nNumber of samples in each split :")
    for x in data_partition:
        print('\t' + str(len(x[0])))

    return data_partition
