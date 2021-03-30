# CNN data : Dans ce fichier, on crée les sets de données qui vont servir à entraîner, valider et tester le CNN.

import numpy as np

# Return l'integer associé au label
def label_to_int(label):
    return 0

# Return le label associé à l'integer
def int_to_label(integer):
    return 0

# Split une série temporelle en échantillons d'entraînement
def create_samples(numpy_array, n_steps):
    # Split a univariate sequence into samples
    X = list()
    n = len(numpy_array)
    for i in range(n):
        # Find the end of this pattern
        end_ix = i + n_steps
        # Check if we are beyond the sequence
        if end_ix > n - 1:
            break
        # Gather input and output parts of the pattern
        X.append(numpy_array[i:end_ix])
    return np.array(X)
