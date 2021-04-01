"""Define CNN's architecture and functions (training, prediction)."""

import os
import matplotlib.pyplot as plt
from keras import backend as bkd
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dropout, Dense

# Training variables
n_epochs = 20
batch_size = 64
verbose = True
learning_rate = 0.00001
dropout = 0.4

# Hyperparameters
n_filters1 = 4
n_filters2 = 2
kernel_size = 2
pool_size = 2
dense1 = 4
dense2 = 2
output = 3                # Number of classes
input_length = 48         # CNN input length


def compile_and_fit(trainX, trainy, validX, validy, testX, testy):
    """Build and train the model. Load it if it has already been trained."""

    # Load the model if already trained
    backup_file = "model_backup/" + str(n_filters1) + 'c_' + str(n_filters2) + 'c_' + str(dense1) + 'd_' + str(dense2) \
                  + 'd_' + str(n_epochs) + "epochs_" + str(input_length) + "input_length.cnn"
    if os.path.exists(backup_file):
        return load_model(backup_file)

    # Else, build new model
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
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    bkd.set_value(model.optimizer.learning_rate, learning_rate)
    history = model.fit(trainX, trainy, validation_data=(validX, validy), epochs=n_epochs, batch_size=batch_size,
                        verbose=verbose)
    model.save(backup_file)

    # Plot accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # Plot loss
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['val_loss', 'loss'], loc='upper left')
    plt.show()

    # Evaluate model on testing set
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=verbose)

    return model, accuracy
