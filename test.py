import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from rotation_forest import RotationTreeClassifier, RotationForestClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from blue.preprocessing import binarize_ova

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
#df = pd.read_csv('datasets/encoded_annealing.csv') # should make binary (3 and not 3. this is very unbalanced)
#df['classes'] = np.where(df['classes'] == 2, 1, 0)
#y = df.pop('classes').values
#X = df.values
xt,xv,yt,yv = train_test_split(X, y, test_size=.3, random_state=7)

clf = RandomForestClassifier(random_state=12, n_estimators=25)
print clf.__class__.__name__
clf.fit(xt, yt)
yhat = clf.predict(xv)
proba = clf.predict_proba(xv)
print np.mean(yhat == yv), log_loss(yv, proba)
print ""

clf = make_pipeline(PCA(), RandomForestClassifier(random_state=12, n_estimators=25))
print clf.__class__.__name__
clf.fit(xt, yt)
yhat = clf.predict(xv)
proba = clf.predict_proba(xv)
print np.mean(yhat == yv), log_loss(yv, proba)
print ""

clf = RotationTreeClassifier(random_state=12,  n_features_per_subset=3)
print clf.__class__.__name__
clf.fit(xt,yt)
yhat = clf.predict(xv)
proba = clf.predict_proba(xv)
print np.mean(yhat == yv), log_loss(yv, proba)
print ""

clf = RotationForestClassifier(random_state=12, n_estimators=25, n_features_per_subset=3)
print clf.__class__.__name__ + " (PCA)"
clf.fit(xt,yt)
yhat = clf.predict(xv)
proba = clf.predict_proba(xv)
print np.mean(yhat == yv), log_loss(yv, proba)
print ""

clf = RotationForestClassifier(random_state=12, n_estimators=25, n_features_per_subset=3, rotation_algo='randomized')
print clf.__class__.__name__ + " (RandomizedPCA)"
clf.fit(xt,yt)
yhat = clf.predict(xv)
proba = clf.predict_proba(xv)
print np.mean(yhat == yv), log_loss(yv, proba)
print ""

base = RotationTreeClassifier(n_features_per_subset=3, random_state=12)
clf = BaggingClassifier(base, n_estimators=25, random_state=12)
print clf.__class__.__name__
clf.fit(xt,yt)
yhat = clf.predict(xv)
proba = clf.predict_proba(xv)
print np.mean(yhat == yv), log_loss(yv, proba)
print ""

base = RotationTreeClassifier(n_features_per_subset=3, random_state=12, max_depth=3)
clf = AdaBoostClassifier(base, n_estimators=25, random_state=12)
print clf.__class__.__name__ + ' (RotationTree)'
clf.fit(xt,yt)
yhat = clf.predict(xv)
proba = clf.predict_proba(xv)
print np.mean(yhat == yv), log_loss(yv, proba)
print ""

base = DecisionTreeClassifier(random_state=12, max_depth=3)
clf = AdaBoostClassifier(base, n_estimators=25, random_state=12)
print clf.__class__.__name__ + ' (DecisionTree)'
clf.fit(xt,yt)
yhat = clf.predict(xv)
proba = clf.predict_proba(xv)
print np.mean(yhat == yv), log_loss(yv, proba)
print ""
