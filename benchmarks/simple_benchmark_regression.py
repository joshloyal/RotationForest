import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor

from rotation_forest.rotation_forest import RotationTreeRegressor, RotationForestRegressor


def regression_data():
    n_features = 30
    n_informative = int(0.6 * n_features)

    X, y = make_regression(n_samples=500, n_features=n_features,
                           n_informative=n_informative, random_state=9)

    return X, y


def _test_toy_data(name, clf):
    X, y = regression_data()
    k_folds = 5
    cv = KFold(k_folds, random_state=1234)

    mse, mae = [], []
    for train, test in cv.split(X, y):
        xt, xv, yt, yv = X[train, :], X[test, :], y[train], y[test]
        clf.fit(xt, yt)
        yhat = clf.predict(xv)
        mse.append(np.mean((yhat - yv) ** 2))
        mae.append(np.mean(np.abs((yhat - yv))))

    mse_mean, mse_std = np.mean(mse), np.std(mse)
    mae_mean, mae_std = np.mean(mae), np.std(mae)
    print(name)
    print('mse: {0:.3f} +/- {1:.3f}'.format(mse_mean, mse_std))
    print('mae: {0:.3f} +/- {1:.3f}'.format(mae_mean, mae_std))
    print('-' * 80)
    return {'name': name,
            'mse_mean': mse_mean,
            'mse_std': mse_std,
            'mae_mean': mae_mean,
            'mae_std': mae_std}


classifiers = [('Random Forest',
                RandomForestRegressor(random_state=12, n_estimators=25)),
               ('PCA + Random Forest',
                make_pipeline(PCA(), RandomForestRegressor(random_state=12,
                                                           n_estimators=25))),
               ('Rotation Tree',
                RotationTreeRegressor(random_state=12,
                                      n_features_per_subset=3)),
               ('Decision Tree',
                DecisionTreeRegressor(random_state=12)),
               ('Rotation Forest (PCA)',
                RotationForestRegressor(random_state=12,
                                        n_estimators=25,
                                        n_features_per_subset=3)),
               ('Rotation Forest (Randomized PCA)',
                RotationForestRegressor(random_state=12,
                                        n_estimators=25,
                                        n_features_per_subset=3,
                                        rotation_algo='randomized')),
               ('Adaboost (Rotation Tree)',
                AdaBoostRegressor(RotationTreeRegressor(n_features_per_subset=3,
                                                        random_state=12,
                                                        max_depth=3),
                                  n_estimators=25,
                                  random_state=12)
                ),
               ('Adaboost (Decision Tree)',
                AdaBoostRegressor(DecisionTreeRegressor(random_state=12,
                                                        max_depth=3),
                                  n_estimators=25,
                                  random_state=12)
                )]

if __name__ == '__main__':
    results = []
    for name, clf in classifiers:
        results.append(_test_toy_data(name, clf))
    df = pd.DataFrame(results)
    plt.figure(figsize=(15, 12))
    sns.barplot(x='name', y='mse_mean', data=df.sort_values('mse_mean'),
                xerr=df['mse_std'].values)
    plt.xticks(rotation=45)
    plt.xlabel('Regressor')
    plt.ylabel('MSE')
    plt.savefig('simple_benchmark.png')
