"""Define CNN's architecture and functions (training, prediction)."""

import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dropout, Dense

# Variables d'entraînement
n_epochs = 5
batch_size = 64
verbose = True

# Hyperparamètres
n_filters1 = 128
n_filters2 = 64
kernel_size = 3
pool_size = 2
dense1 = 128
dense2 = 64
dense3 = 32
output = 3               # Le nombre de classes
input_length = 1008      # Longueur des entrées à donner au CNN


def compile_and_fit(trainX, trainy, validX, validy, testX, testy):
    """Build and train the model. Load it if it has already been trained."""

    # Load the model if already trained
    backup_file = "model_backup/" + str(n_epochs) + "epochs_" + str(input_length) + "input_length.cnn"
    if os.path.exists(backup_file):
        return load_model(backup_file)

    # Else, build new model
    model = Sequential()

    # Convolution layer 1 (input layer)
    model.add(Conv1D(filters=n_filters1, kernel_size=kernel_size, padding="same",
                     activation="relu", input_shape=(input_length, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size, padding="same"))

    # Convolution layer 2
    model.add(Conv1D(filters=n_filters2, kernel_size=kernel_size, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size, padding="same"))
    model.add(Flatten())

    # 3 Dense layers
    model.add(Dense(dense1, activation="tanh"))
    model.add(Dropout(0.2))
    model.add(Dense(dense2, activation="tanh"))
    model.add(Dropout(0.2))
    model.add(Dense(dense3, activation="relu"))
    model.add(Dropout(0.2))

    # Dense layer (output layer)
    model.add(Dense(output, activation="softmax"))

    # Train network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
