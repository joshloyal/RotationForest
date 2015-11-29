#!/usr/bin/python

import os
import time
import random
import numpy as np

from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from rotation_forest import RotationForestClassifier

def get_uci_path():
    """
    Returns the path to the UCI datasets, depending on the
    host name.

    :return: string
    """
    path = 'uci-datasets'

    return path


def read_uci_dataset(path):
    """
    This function returns the path to the UCI dataset

    :param base_dir: string
    The path to where the UCI datasets folder are.

    :param dataset_idx: int
    The number of the dataset

    :return: 2-tuple of ndarray
    The dataset in a Numpy array,
    the first is the data samples and the second, the labels
    """

    #full_path = os.path.join(base_dir, 'uci-datasets')

    # Load the data
    data = np.genfromtxt(path, delimiter=",")
    rows, cols = data.shape

    # Delete the first column of labels
    Y = data[:, 0]
    X = data[:, 1:rows]

    return X, Y

if __name__ == '__main__':

    dataset_dir = './datasets/'
    for dataset in os.listdir(dataset_dir):
        name = dataset.split('.')[0]

        X, Y = read_uci_dataset(dataset_dir+dataset)

        K = 10
        accuracy = []
        cv = StratifiedKFold(Y, K)

        print name
        for clf in [RotationForestClassifier(), RandomForestClassifier(n_estimators=10)]:
            for train, test in cv:

                x_train, x_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]

                clf = clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                accuracy.append(accuracy_score(y_test, y_pred))

            bd_std = np.std(accuracy)
            bd_acc = np.mean(accuracy)
            print '{0}: {1:2f} +/- {2:2f}'.format(clf.__class__.__name__, bd_acc*100, bd_std*100)
        print ""
