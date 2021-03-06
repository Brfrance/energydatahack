"""Prepare data for training, validation and testing."""

import os
import fnmatch
import extract_data
from numpy import array, concatenate, mean, split
from keras.utils import to_categorical


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


def split_samples(time_series, n_steps):
    ret = split(time_series, n_steps)
    if ret[-1].shape[0] < n_steps:
        return array(ret[:-1], dtype="uint16")
    return array(ret, dtype="uint16")


def get_data_sets(cnn_n_input):
    """Prepare training, validation and testing sets."""

    # Get list of labels
    data_directory = "data_set/"
    list_labels = extract_data.get_labels(data_directory + "labels.txt")
    n_labels = len(list_labels)

    # Dictionary that gives labels ID
    label_to_int = dict()
    for i in range(n_labels):
        label_to_int[list_labels[i]] = i

    # Dictionary that will count how many times each label appears
    count_labels, count_labels2 = dict(), dict()

    # Train/Validation/Test
    trainX, trainy = list(), list()
    validX, validy = list(), list()
    testX, testy = list(), list()

    # Loop over data_set directory
    files = [f for f in os.listdir(data_directory) if fnmatch.fnmatch(f, "*_label.txt")]
    for file in files:

        # Get chorus code
        chorus = file.split('_')[0]

        # Get time series (data)
        input_data = extract_data.extract_data_from_txt(data_directory + "MIN " + chorus + ".txt").Value.values\
            .astype(dtype="uint16", copy=False)
        # input_data = mean(input_data.reshape(-1, 3), 1)

        # Get respective label
        label = extract_data.extract_label_from_txt(data_directory + file)

        # Increment label count
        if label[0] in count_labels:
            count_labels[label[0]] += 1
        else:
            count_labels[label[0]] = 1
        if label[1] in count_labels2:
            count_labels2[label[1]] += 1
        else:
            count_labels2[label[1]] = 1

        # Decide whether these data should be used for training/validation/testing
        label_id = label_to_int[label[0]]
        # Split data into samples
        X = split_samples(input_data, cnn_n_input)
        X = X.reshape(X.shape[1], X.shape[0], 1)
        # Create respective Y values
        Y = to_categorical([[label_id] for _ in X], dtype="uint8", num_classes=n_labels)

        if count_labels[label[0]] % 5 == 7:             # 20% of data is for testing
            testX.append(X)
            testy.append(Y)
        elif count_labels[label[0]] % 5 == 3:           # 20% of data is for validation
            # Append validation samples
            validX.append(X)
            validy.append(Y)
        else:                                           # 60% of data is for training
            # Append training samples
            trainX.append(X)
            trainy.append(Y)

    print("--\nInventaire des donn??es globales :")
    print(count_labels)
    print(count_labels2)

    # Concatenate all training and validation samples to get the final sets
    TrainX = concatenate([x for x in trainX])
    Trainy = concatenate([y for y in trainy])
    ValidX = concatenate([x for x in validX])
    Validy = concatenate([y for y in validy])
    # TestX = concatenate([x for x in testX])
    # Testy = concatenate([y for y in testy])

    print("Training set:\n\t", TrainX.shape)
    print("Validation set:\n\t", ValidX.shape)
    # print("Test set:\n\t", TestX.shape)

    return TrainX, Trainy, ValidX, Validy, None, None
