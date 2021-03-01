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
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=1234)

    mse, mae = [], []
    for train_index, test_index in kf.split(X, y):
        xt, xv = X[train_index], X[test_index]
        yt, yv = y[train_index], y[test_index]
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
               ('Rotation Forest\n(PCA)',
                RotationForestRegressor(random_state=12,
                                        n_estimators=25,
                                        n_features_per_subset=3)),
               ('Rotation Forest\n(Randomized PCA)',
                RotationForestRegressor(random_state=12,
                                        n_estimators=25,
                                        n_features_per_subset=3,
                                        rotation_algo='randomized')),
               ('Adaboost\n(Rotation Tree)',
                AdaBoostRegressor(RotationTreeRegressor(n_features_per_subset=3,
                                                        random_state=12,
                                                        max_depth=3),
                                  n_estimators=25,
                                  random_state=12)
                ),
               ('Adaboost\n(Decision Tree)',
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
    df = df.sort_values('mse_mean')
    fig = plt.figure(figsize=(20, 15))
    n = len(df.index)
    cm = plt.cm.get_cmap('rainbow', n)
    bcolors = [cm(i) for i in range(n)]
    
    ax1 = fig.add_subplot(121)
    ax1.bar(
        x=np.arange(n), 
        height=df['mse_mean'],
        bottom=0,
        align='center',
        color=bcolors,
        yerr=df['mse_std'],
        width=0.25
    )
    ax1.set_xlabel('Classifier')
    ax1.set_ylabel('MSE')
    ax1.set_xticks(np.arange(n))
    ax1.set_xticklabels(df['name'])
    ax1.tick_params(axis="x", rotation=45)
    
    ax2 = fig.add_subplot(122)
    ax2.bar(
        x=np.arange(n), 
        height=df['mae_mean'],
        bottom=0,
        align='center',
        color=bcolors,
        yerr=df['mae_std'],
        width=0.25
    )
    ax2.set_xlabel('Classifier')
    ax2.set_ylabel('MAE')
    ax2.set_xticks(np.arange(n))
    ax2.set_xticklabels(df['name'])
    ax2.tick_params(axis="x", rotation=45)
    
    fig.suptitle('Regression Benchmark')
    plt.savefig('simple_benchmark_regression.png')

    #plt.figure(figsize=(15, 15))
    #sns.barplot(x='name', y='mse_mean', data=df.sort_values('mse_mean'),
                #xerr=df['mse_std'].values)
    #plt.xticks(rotation=45)
    #plt.xlabel('Regressor')
    #plt.ylabel('MSE')
    #plt.savefig('simple_benchmark_regression.png')
