"""
Lit les fichiers .txt et extrait les données.
"""

import os
import fnmatch
import extract_data
from numpy import array


def create_samples(time_series, n_steps):
    """
    Split a time series into samples of size n_steps.
    Example :
        time_series = [1, 2, 3, 4]
        n_steps = 2
        create_samples(time_series, n_steps) = [ [1, 2], [2, 3], [3, 4] ]
    """

    # Split a univariable sequence into samples
    X = list()
    n = len(time_series)
    for i in range(n):
        # Find the end of this pattern
        end_ix = i + n_steps
        # Check if we are beyond the sequence
        if end_ix > n - 1:
            break
        # Gather input and output parts of the pattern
        X.append(time_series[i:end_ix])
    return array(X, dtype="uint16")


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
