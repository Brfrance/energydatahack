"""Main. Point d'entrer du programme."""

import CNN
import tensorflow as tf
from CNN_data import get_data_sets

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data_partition = get_data_sets(CNN.input_length)
CNN.loop_hyperparameters(data_partition)
