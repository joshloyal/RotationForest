import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from rotation_forest import RotationTreeClassifier, RotationForestClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

def classification_data():
    n_features = 30
    n_redundant = int(0.1 * n_features)
    n_informative = int(0.6 * n_features)
    n_repeated = int(0.1 * n_features)

    X, y = make_classification(
        n_samples=500, n_features=n_features, flip_y=0.03,
        n_informative=n_informative, n_redundant=n_redundant,
        n_repeated=n_repeated, random_state=9
    )

    return X, y

def test_toy_data(name, clf):
    X, y = classification_data()
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=1234)

    acc, auc = [], []
    for train_index, test_index in skf.split(X, y):
        xt, xv = X[train_index], X[test_index]
        yt, yv = y[train_index], y[test_index]
        clf.fit(xt, yt)
        yhat = clf.predict(xv)
        proba = clf.predict_proba(xv)[:, 1]
        acc.append(np.mean(yhat == yv))
        auc.append(roc_auc_score(yv, proba))

    acc_mean, acc_std = np.mean(acc), np.std(acc)
    auc_mean, auc_std = np.mean(auc), np.std(auc)
    print(name)
    print('accuracy: {0:.3f} +/- {1:.3f}'.format(acc_mean, acc_std))
    print('auc: {0:.3f} +/- {1:.3f}'.format(auc_mean, auc_std))
    print('-'*80)
    return {'name': name,
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'auc_mean': auc_mean,
            'auc_std': auc_std}

classifiers = [('Random Forest',
               RandomForestClassifier(random_state=12, n_estimators=25)),
              ('PCA + Random Forest',
               make_pipeline(PCA(), RandomForestClassifier(
                   random_state=12, n_estimators=25))),
              ('Rotation Tree',
               RotationTreeClassifier(random_state=12,
                                      n_features_per_subset=3)),
              ('Decision Tree',
               DecisionTreeClassifier(random_state=12)),
              ('Rotation Forest\n(PCA)',
               RotationForestClassifier(random_state=12,
                                        n_estimators=25,
                                        n_features_per_subset=3)),
              ('Rotation Forest\n(Randomized PCA)',
               RotationForestClassifier(random_state=12,
                                        n_estimators=25,
                                        n_features_per_subset=3,
                                        rotation_algo='randomized')),
              ('Adaboost\n(Rotation Tree)',
               AdaBoostClassifier(RotationTreeClassifier(
                   n_features_per_subset=3, random_state=12, max_depth=3),
                                  n_estimators=25,
                                  random_state=12)
              ),
              ('Adaboost\n(Decision Tree)',
               AdaBoostClassifier(DecisionTreeClassifier(random_state=12,
                                                         max_depth=3),
                                  n_estimators=25,
                                  random_state=12)
              )]



if __name__ == '__main__':
    results = []
    for name, clf in classifiers:
        results.append(test_toy_data(name, clf))
    df = pd.DataFrame(results)
    df = df.sort_values('auc_mean')
    fig = plt.figure(figsize=(15, 15))
    n = len(df.index)
    cm = plt.cm.get_cmap('rainbow', n)
    bcolors = [cm(i) for i in range(n)]
    ax = fig.add_subplot(111)
    ax.bar(
        x=(np.arange(n) - 0.15), 
        height=df['auc_mean'],
        bottom=0,
        align='center',
        color=bcolors,
        yerr=df['auc_std'],
        width=0.25
    )
    ax.bar(
        x=(np.arange(n) + 0.15), 
        height=df['acc_mean'],
        bottom=0,
        align='center',
        color=bcolors,
        yerr=df['acc_std'],
        width=0.25
    )
    ax.set_title('Classification Benchmark')
    ax.set_xlabel('Classifier')
    ax.set_ylabel('AUC / Accuracy')
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(df['name'])
    ax.set_ylim(bottom=0.5, top=1.0)
    ax.tick_params(axis="x", rotation=45)
    plt.savefig('simple_benchmark_classification.png')
