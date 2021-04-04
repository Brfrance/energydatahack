"""
Main. Point d'entrer du programme de cross validation de la fôret aléatoire.

1) Crée n_data_sets ensembles de données distincts.
    partition = [ set_1, set_2, set_3, set_4 ]

2) Crée et entraîne le Random Forest sur n_data_sets - 1 de ces ensembles. Evalue le Random Forest sur le dernier.
    training_set = set_1, set_2, set_3
    testing_set = set_4

3) Répète n_repeat fois l'étape 2.

4) Réitère n_data_sets fois l'étape 3 en changeant le testing_set, de sorte à ce que chaque set_i serve une fois de set
   de testing.

5) Affiche la moyenne des scores f1 de tous les modèles initialisés durant l'exécution du programme.

La cross validation donne de la crédibilité aux résultats en permettant d'éviter certains biais (par exemple :
toujours entraîner le modèle sur les mêmes training et testing sets) et réduire l'impact du facteur aléatoire.
"""

import random_forest
from numpy import mean, std
from forest_data import get_data_sets

data_partition = get_data_sets(random_forest.input_length, data_directory="data_set/", n_data_sets=5)
list_acc = random_forest.cross_validation(data_partition, n_repeat=1)

f = open("RandomForest-results.txt", "a")
f.write("\n\nList f1-scores : (" + str(random_forest.n_estimators)
        + " trees, input=" + str(random_forest.input_length) + ")\n\t")
f.write(str(list_acc))
f.write("\nMean & std :\n\t")
f.write(str(round(mean(list_acc), 3)) + ' ' + str(round(std(list_acc), 3)))
f.close()
