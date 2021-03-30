"""Define CNN's architecture and functions (training, prediction)."""

import os
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dropout, Dense

# Variables d'entraînement
n_epochs = 1
batch_size = 64
verbose = True

# Hyperparamètres
n_filters = 64
kernel_size = 3
pool_size = 2
dense1 = 64
dense2 = 32
dense3 = 16
output = 3              # Le nombre de classes
input_length = 100      # Longueur des entrées à donner au CNN


def compile_and_fit(trainX, trainy, validX, validy):
    """Build and train the model. Load it if it has already been trained."""

    # Load the model if already trained
    backup_file = "model_backup/" + str(n_epochs) + "epochs_" + str(input_length) + "input_length.cnn"
    if os.path.exists(backup_file):
        return load_model(backup_file)

    # Else, build new model
    model = Sequential()

    # Convolution layer 1 (input layer)
    model.add(Conv1D(filters=n_filters, kernel_size=kernel_size, padding="same",
                     activation="relu", input_shape=(input_length, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=pool_size))

    # Convolution layer 2
    model.add(Conv1D(filters=n_filters, kernel_size=kernel_size, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=pool_size))
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
    model.fit(trainX, trainy, validation_data=(validX, validy), epochs=n_epochs, batch_size=batch_size, verbose=verbose)
    model.save(backup_file)

    return model


def predict(CNN, time_series):
    """Utilise le CNN pour prédire le type de bâtiment à partir d'une série temporelle."""
    return CNN.predict(time_series)
