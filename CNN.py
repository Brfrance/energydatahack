# CNN : Dans ce fichier, on définit le modèle et les fonctions qui permettent de l'entraîner et de l'utiliser.

import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dropout, Dense

# Variables d'entraînement
n_epochs = 10
batch_size = 64
verbose = 1

# Hyperparamètres
n_filters = 64
kernel_size = 3
pool_size = 2
Dense1 = 64
Dense2 = 32
Dense3 = 16
dropout = 0.2
padding = "same"
input_shape = (2016, 1)    # Série temporelle univariable constituée de 100 valeurs
output = 42                # Le nombre de classes


# Crée et entraîne le modèle
def compile_and_fit(trainX, trainy, validX, validy):
    model = Sequential()
    # Couche de convolution 1
    model.add(
        Conv1D(filters=n_filters, kernel_size=kernel_size, padding=padding, activation="relu", input_shape=input_shape))
    # ???
    model.add(BatchNormalization())
    #
    model.add(MaxPooling1D(pool_size=pool_size))
    # Couche de convolution 2
    model.add(Conv1D(filters=n_filters, kernel_size=kernel_size, padding=padding, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())

    model.add(Dense(Dense1, activation="tanh"))
    model.add(Dropout(dropout))
    model.add(Dense(Dense2, activation="tanh"))
    model.add(Dropout(dropout))
    model.add(Dense(Dense3, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(output, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=n_epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)

    return accuracy

# Utilise le CNN pour prédire le type de bâtiment à partir de la série temporelle
def predict(CNN, time_series):
    return CNN.predict(times_series)