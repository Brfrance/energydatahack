"""Define CNN's architecture and functions (training, prediction)."""

import os
import numpy as np
from seaborn import heatmap
import matplotlib.pyplot as plt
from keras import backend as bkd
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, load_model
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dropout, Dense

# Training variables
n_epochs = 10
batch_size = 256
verbose = 1
learning_rate = 0.001
dropout = 0.4
optimizer = "Adam"

# Hyperparameters
n_filters1 = 7
n_filters2 = 7
kernel_size = 7
pool_size = 2
dense1 = 7
dense2 = 4
output = 2                  # Number of classes
input_length = 72           # CNN input length


def compile_fit_test_without_valid(trainX, trainy, testX, testy, info):
    """
    Build, train and test the model.
    Load the model if already trained.
    Save training info and model weights.
    """

    # Load model if already trained
    if os.path.exists("model_backup/CNN_" + info):
        model = load_model("model_backup/CNN_" + info)
    else:
        # Build new model
        model = Sequential()
        # Convolution layer (input layer)
        model.add(Conv1D(filters=n_filters1, kernel_size=kernel_size, padding="same",
                         activation="relu", input_shape=(input_length, 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size, padding="same"))
        # Convolution layer
        model.add(Conv1D(filters=n_filters2, kernel_size=kernel_size, padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size, padding="same"))
        model.add(Flatten())
        # Dense layers
        model.add(Dense(dense1, activation="tanh"))
        model.add(Dropout(dropout))
        model.add(Dense(dense2, activation="tanh"))
        model.add(Dropout(dropout))
        # Dense layer (output layer)
        model.add(Dense(output, activation="softmax"))

        # Train network
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        bkd.set_value(model.optimizer.learning_rate, learning_rate)
        history = model.fit(trainX, trainy, epochs=n_epochs, batch_size=batch_size,
                            verbose=verbose)
        model.save("model_backup/CNN_" + info)

        # Plot accuracy
        plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig('plots/acc_' + info)
        plt.clf()

        # Plot loss
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('plots/loss_' + info)
        plt.clf()

    # Evaluate model on testing set
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=verbose)
    matrix_confusion(model, testX, testy, info)

    return accuracy


def compile_fit_test(trainX, trainy, validX, validy, testX, testy, info):
    """
    Build, train and test the model.
    Load the model if already trained.
    Save training info and model weights.
    """

    # Load model if already trained
    if os.path.exists("model_backup/CNN_" + info):
        model = load_model("model_backup/CNN_" + info)
    else:
        # Build new model
        model = Sequential()
        # Convolution layer (input layer)
        model.add(Conv1D(filters=n_filters1, kernel_size=kernel_size, padding="same",
                         activation="relu", input_shape=(input_length, 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size, padding="same"))
        # Convolution layer
        model.add(Conv1D(filters=n_filters2, kernel_size=kernel_size, padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size, padding="same"))
        model.add(Flatten())
        # Dense layers
        model.add(Dense(dense1, activation="tanh"))
        model.add(Dropout(dropout))
        model.add(Dense(dense2, activation="tanh"))
        model.add(Dropout(dropout))
        # Dense layer (output layer)
        model.add(Dense(output, activation="softmax"))

        # Train network
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        bkd.set_value(model.optimizer.learning_rate, learning_rate)
        history = model.fit(trainX, trainy, validation_data=(validX, validy), epochs=n_epochs, batch_size=batch_size,
                            verbose=verbose)
        model.save("model_backup/CNN_" + info)

        # Plot accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('plots/acc_' + info)
        plt.clf()

        # Plot loss
        plt.plot(history.history['val_loss'])
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['val_loss', 'loss'], loc='upper left')
        plt.savefig('plots/loss_' + info)
        plt.clf()

    # Evaluate model on testing set
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=verbose)
    matrix_confusion(model, testX, testy, info)

    return accuracy


def cross_validation(data_partition, info):
    """
    Return model's accuracy using cross-validation.
    Each split of the partition is used 1 time for testing.
    The validation split is the one next to the testing split.
    The rest is used for training.

    Example :
        data_partition = [1, 2, 3, 4]
        test1 = 1, valid1 = 2, train1 = [3, 4]
        test2 = 2, valid2 = 3, train2 = [1, 4]
        test3 = 3, valid3 = 4, train3 = [1, 2]
        test4 = 4, valid4 = 1, train4 = [2, 3]
    """

    list_acc = list()
    n_partitions = len(data_partition)
    for i in range(n_partitions):

        print("[ New Model ]\n\tTraining : ", end='\0')

        # Training set
        trainx, trainy = list(), list()
        for j in range(n_partitions):
            if j == i or j == i + 1 or (i + 1 == n_partitions and j == 0):
                continue
            trainx.extend(data_partition[j][0])
            trainy.extend(data_partition[j][1])
            print(j, end=' ')

        # Validation set
        if i + 1 == n_partitions:
            validx = data_partition[0][0]
            validy = data_partition[0][1]
            print("\n\tValidation : 0", end="\n\t")
        else:
            validx = data_partition[i + 1][0]
            validy = data_partition[i + 1][1]
            print("\n\tValidation :", str(i + 1), end="\n\t")

        # Testing set
        testx = data_partition[i][0]
        testy = data_partition[i][1]
        print("Testing :", str(i))

        list_acc.append(compile_fit_test(np.array(trainx), np.array(trainy), np.array(validx), np.array(validy),
                                         np.array(testx), np.array(testy), info + "-model_" + str(i)))

    return list_acc


def cross_validation_without_valid(data_partition, info):

    list_acc = list()
    n_partitions = len(data_partition)
    for i in range(n_partitions):

        print("[ New Model ]\n\tTraining : ", end='\0')

        # Training set
        trainx, trainy = list(), list()
        for j in range(n_partitions):
            if j == i:
                continue
            trainx.extend(data_partition[j][0])
            trainy.extend(data_partition[j][1])
            print(j, end=' ')

        # Testing set
        testx = data_partition[i][0]
        testy = data_partition[i][1]
        print("Testing :", str(i))

        list_acc.append(compile_fit_test_without_valid(np.array(trainx), np.array(trainy),
                                                       np.array(testx), np.array(testy), info + "-model_" + str(i)))

    return list_acc


def loop_hyperparameters(data_partition):
    """
    Get accuracies of models with different hyperparameters.
    Write results in results.txt.
    """

    global n_filters1, n_filters2, dense1, dense2

    # Model information
    info = str(n_filters1) + 'c_' + str(n_filters2) + 'c_' + str(dense1) + 'd_' + str(dense2) + 'd'
    print("[ " + info + " ]")

    # Get cross validation accuracies
    list_acc = cross_validation_without_valid(data_partition, info)

    # Register model's performances
    f = open("plots/results.txt", "a")
    f.write("[ " + info + " ]\n")
    f.write("\tAccuracies list :\t" + str(list_acc) + '\n')
    f.write("\tAccuracy mean :\t" + str(round(np.mean(list_acc), 3)) + '\n')
    f.write("\tAccuracy std :\t" + str(round(np.std(list_acc), 3)) + '\n')
    f.close()

    print("Model score :", str(round(np.mean(list_acc), 3)))

    print("Computation terminated.")


def matrix_confusion(model, testx, testy, info):
    predictions = model.predict(testx)
    predictions = np.array([np.argmax(pred) for pred in predictions])
    mc = confusion_matrix(predictions, np.argmax(testy, axis=-1), normalize="true")

    ax = plt.subplot()
    heatmap(mc, annot=True, ax=ax)
    ax.set_title("Confusion Matrix " + info)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(["Immeuble de bureau", "Commissariat"])
    ax.yaxis.set_ticklabels(["IdB", "C"])
    plt.savefig('plots/confusion_maxtrix_' + info)
    plt.clf()
