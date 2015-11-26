import numpy as np
from sklearn.datasets import make_classification
from rotation_forest import RotationForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report

def classification_data():
    n_features = 30
    n_redundant = int(0.1 * n_features)
    n_informative = int(0.6 * n_features)
    n_repeated = int(0.1 * n_features)

    X, y = make_classification(n_samples=500, n_features=n_features, flip_y=0.03,
                               n_informative=n_informative, n_redundant=n_redundant,
                               n_repeated=n_repeated, random_state=9)

    return X, y

X, y = classification_data()
xt,xv,yt,yv = train_test_split(X, y, test_size=.3, random_state=7)

clf = RandomForestClassifier(random_state=12, n_estimators=25)
clf.fit(xt, yt)
yhat = clf.predict(xv)
print np.mean(yhat == yv), log_loss(yv, yhat)
print classification_report(yv, yhat)

clf = RotationForestClassifier(random_state=12, n_estimators=25, n_subsets=10, max_features=X.shape[1])
clf.fit(xt,yt)
yhat = clf.predict(xv)
print np.mean(yhat == yv), log_loss(yv, yhat)
print classification_report(yv, yhat)
