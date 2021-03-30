"""Main. Point d'entrer du programme."""

import CNN
from CNN_data import get_data_sets

TrainX, Trainy, ValidX, Validy, Test = get_data_sets(CNN.input_length)

cnn = CNN.compile_and_fit(TrainX, Trainy, ValidX, Validy)
cnn.summary()


"""
TODO:
    - Optimiser les hyperparamètres (nombres de filtres pour les couches de Convolution,
      nombre de neurones pour les couches Dense, taille des séries temporelles à donner en entrer au CNN, etc.)
    - Evaluer le modèle avec Test
"""
