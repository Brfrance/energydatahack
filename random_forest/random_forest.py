"""
Fonctions qui concernent le modèle de classification : Random Forest.
On évalue par Cross Validation les performances du modèle.
Les matrices de confusions de chaque modèle sont sauvegardées dans le dossier "plots/".
La liste des performances de chaque modèle créé est ajouté au fichier "RandomForest-results.txt".
Le score utilisé est le f1_score.
"""

import numpy as np
from seaborn import heatmap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

n_estimators = 10           # Number of trees
n_jobs = 6                  # Number of jobs to run in parallel to fit Random Forest model
output = 2                  # Number of classes
input_length = 60           # CNN input length


def matrix_confusion(model, testx, testy, info):
    """Return confusion matrix for a model on a given testing set. Save the plot."""

    # Predict
    predictions = model.predict(testx)
    predictions = [round(pred) for pred in predictions]

    # Matrix
    mc = confusion_matrix(predictions, testy, normalize="all")

    # Plot
    ax = plt.subplot()
    heatmap(mc, annot=True, ax=ax)
    ax.set_title("Confusion Matrix " + info)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(["Immeuble de bureau", "Commissariat"])
    ax.yaxis.set_ticklabels(["IdB", "C"])
    plt.savefig('plots/confusion_matrix_' + info)
    plt.clf()

    # Compute F1-score
    TP = mc[1][1]
    FP = mc[0][1]
    FN = mc[1][0]

    return TP / (TP + (FP + FN) / 2)


def compile_fit_test(trainX, trainy, testX, testy, info, n_repeat):
    """Make Random Forest model. Compute confusion matrix."""

    list_f1_score = list()
    for i_repetition in range(n_repeat):
        # Build and fit model
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=i_repetition, verbose=2, n_jobs=6)
        model.fit(trainX, trainy)

        # Evaluate model on testing set
        f1_score = matrix_confusion(model, testX, testy, info + '_' + str(i_repetition))
        list_f1_score.append(f1_score)

    return list_f1_score


def cross_validation(data_partition, n_repeat=5):
    """Cross validation for Random Forest."""
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

        list_acc.extend(compile_fit_test(np.array(trainx), np.array(trainy), np.array(testx), np.array(testy),
                                         "RandomForest_" + str(i), n_repeat))

    return list_acc
