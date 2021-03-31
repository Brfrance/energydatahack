"""Main. Point d'entrer du programme."""

import CNN
import tensorflow as tf
from CNN_data import get_data_sets

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

TrainX, Trainy, ValidX, Validy, TestX, Testy = get_data_sets(CNN.input_length)

cnn = CNN.compile_and_fit(TrainX, Trainy, ValidX, Validy, TestX, Testy)
print(cnn)

"""
TODO:
    - Optimiser les hyperparamètres (nombres de filtres pour les couches de Convolution,
      nombre de neurones pour les couches Dense, taille des séries temporelles à donner en entrer au CNN, etc.)
    - Evaluer le modèle avec Test
"""
