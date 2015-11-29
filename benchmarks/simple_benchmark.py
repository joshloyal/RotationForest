import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from rotation_forest import RotationTreeClassifier, RotationForestClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA


def classification_data():
    n_features = 30
    n_redundant = int(0.1 * n_features)
    n_informative = int(0.6 * n_features)
    n_repeated = int(0.1 * n_features)

    X, y = make_classification(n_samples=500, n_features=n_features, flip_y=0.03,
                               n_informative=n_informative, n_redundant=n_redundant,
                               n_repeated=n_repeated, random_state=9)

    return X, y

def test_toy_data(name, clf):
    X, y = classification_data()
    k_folds = 5
    cv = StratifiedKFold(y, k_folds, random_state=1234)

    acc, auc = [], []
    for train, test in cv:
        xt, xv, yt, yv = X[train, :], X[test, :], y[train], y[test]
        clf.fit(xt, yt)
        yhat = clf.predict(xv)
        proba = clf.predict_proba(xv)[:, 1]
        acc.append(np.mean(yhat == yv))
        auc.append(roc_auc_score(yv, proba))

    acc_mean, acc_std = np.mean(acc), np.std(acc)
    auc_mean, auc_std = np.mean(auc), np.std(auc)
    print name
    print 'accuracy: {0:.3f} +/- {1:.3f}'.format(acc_mean, acc_std)
    print 'auc: {0:.3f} +/- {1:.3f}'.format(auc_mean, auc_std)
    print '-'*80

classifiers = [('Random Forest',
               RandomForestClassifier(random_state=12, n_estimators=25)),
              ('PCA + Random Forest',
               make_pipeline(PCA(), RandomForestClassifier(random_state=12,
                                                           n_estimators=25))),
              ('Rotation Tree',
               RotationTreeClassifier(random_state=12,
                                      n_features_per_subset=3)),
              ('Rotation Forest (PCA)',
               RotationForestClassifier(random_state=12,
                                        n_estimators=25,
                                        n_features_per_subset=3)),
              ('Rotation Forest (Randomized PCA)',
               RotationForestClassifier(random_state=12,
                                        n_estimators=25,
                                        n_features_per_subset=3,
                                        rotation_algo='randomized')),
              ('Adaboost (Rotation Tree)',
               AdaBoostClassifier(RotationTreeClassifier(n_features_per_subset=3,
                                                         random_state=12,
                                                         max_depth=3),
                                  n_estimators=25,
                                  random_state=12)
              ),
              ('Adaboost (Decision Tree)',
               AdaBoostClassifier(DecisionTreeClassifier(random_state=12,
                                                         max_depth=3),
                                  n_estimators=25,
                                  random_state=12)
              )]



for name, clf in classifiers:
    test_toy_data(name, clf)
