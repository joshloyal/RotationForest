import pytest
#import copy

import numpy as np
import numpy.testing as npt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from rotation_forest import random_feature_subsets
from rotation_forest import RotationTreeClassifier, RotationForestClassifier


def classification_data(n_samples=500, n_features=30, redundant_size=0.1, informative_size=0.6, repeated_size=0.1):
    assert (redundant_size + informative_size + repeated_size) < 1
    n_redundant = int(redundant_size * n_features)
    n_informative = int(informative_size * n_features)
    n_repeated = int(repeated_size * n_features)

    X, y = make_classification(n_samples=n_samples, n_features=n_features, flip_y=0.03,
                               n_informative=n_informative, n_redundant=n_redundant,
                               n_repeated=n_repeated, random_state=9)

    return X, y


class TestRotationTreeClassifier(object):
    """ Test Suite for RotationTreeClassifier """
    def test_rotation_tree(self):
        """ Smoke test for rotation tree """
        X, y = classification_data()
        xt, xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=77)
        clf = RotationTreeClassifier(random_state=1234)
        clf.fit(xt, yt)

        proba = clf.predict_proba(xv)
        assert proba.shape[0] == xv.shape[0]
        assert np.all(proba <= 1)
        assert np.all(proba >= 0)

        yhat = clf.predict(xv)
        assert yhat.shape[0] == xv.shape[0]
        assert np.unique(yhat).tolist() == [0, 1]

    def test_random_feature_subsets(self):
        """ check we generate disjoint feature subsets that cover all features. """
        array = np.arange(6*6).reshape(6, 6)
        subsets = list(random_feature_subsets(array, batch_size=3))
        assert len(subsets) == 2
        for subset in subsets:
            assert set(subset).issubset(range(6))

        assert set(subsets[0]).intersection(set(subsets[1])) == set()

    def test_random_feature_subsets_batch_size_not_even(self):
        array = np.arange(6*6).reshape(6, 6)
        subsets = list(random_feature_subsets(array, batch_size=5))
        assert len(subsets) == 2
        assert (len(subsets[0]) + len(subsets[1])) == 6
        for subset in subsets:
            assert set(subset).issubset(range(6))

        assert set(subsets[0]).intersection(set(subsets[1])) == set()

    def test_random_feature_subsets_batch_size_too_big(self):
        """ request of a too large subset gives the full set """
        array = np.arange(6*6).reshape(6, 6)
        subsets = list(random_feature_subsets(array, batch_size=8))
        assert len(subsets) == 1
        assert sorted(subsets[0]) == list(range(6))

    def test_rotation_matrix(self):
        """ Smoke test for rotation forest """
        X, y = classification_data(n_features=6)
        tree = RotationTreeClassifier(random_state=1234)
        tree.fit(X, y)
        assert tree.rotation_matrix.shape == (6, 6)

        # note that this random state generates the following subsets:
        subset1 = np.array([2, 1, 5])
        subset2 = np.array([0, 4, 3])

        # make sure the loadings are input in the proper order
        for feature in subset1:
            assert np.any(tree.rotation_matrix[:, feature][subset1] != 0)
            assert np.any(tree.rotation_matrix[:, feature][subset2] == 0)

        for feature in subset2:
            assert np.any(tree.rotation_matrix[:, feature][subset1] == 0)
            assert np.any(tree.rotation_matrix[:, feature][subset2] != 0)


class TestRotationForestClassifier(object):
    """ Test suite for RotationForestClassifier """
    def test_rotation_forest(self):
        """ Smoke test for rotation forest """
        X, y = classification_data()
        xt, xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=77)
        clf = RotationForestClassifier(random_state=1234)
        clf.fit(xt, yt)

        proba = clf.predict_proba(xv)
        assert proba.shape[0] == xv.shape[0]
        assert np.all(proba <= 1)
        assert np.all(proba >= 0)

        yhat = clf.predict(xv)
        assert yhat.shape[0] == xv.shape[0]
        assert np.unique(yhat).tolist() == [0, 1]

    def test_randomized_pca(self):
        """ smoke test for randomized pca """
        X, y = classification_data()
        xt, xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=77)
        clf = RotationForestClassifier(random_state=1234, rotation_algo='randomized')
        clf.fit(xt, yt)

        proba = clf.predict_proba(xv)
        assert proba.shape[0] == xv.shape[0]
        assert np.all(proba <= 1)
        assert np.all(proba >= 0)

        yhat = clf.predict(xv)
        assert yhat.shape[0] == xv.shape[0]
        assert np.unique(yhat).tolist() == [0, 1]

    def test_error_unkown_algo(self):
        """ Make sure we throw an error when selecting an unknown algorithm """
        X, y = classification_data()
        clf = RotationForestClassifier(random_state=1234, rotation_algo='cat')
        with pytest.raises(ValueError):
            clf.fit(X, y)

    def test_warm_start(self):
        """ Test if fitting incrementally with warm start gives a forest of the right
            size and the same results as a normal fit.
        """
        X, y = classification_data()
        clf_ws = None
        for n_estimators in [5, 10]:
            if clf_ws is None:
                clf_ws = RotationForestClassifier(n_estimators=n_estimators,
                                                  random_state=1234,
                                                  warm_start=True)
            else:
                clf_ws.set_params(n_estimators=n_estimators)
            clf_ws.fit(X, y)
            assert len(clf_ws) == n_estimators

        clf_no_ws = RotationForestClassifier(n_estimators=10,
                                             random_state=1234,
                                             warm_start=False)
        clf_no_ws.fit(X, y)
        assert set([tree.random_state for tree in clf_ws]) == set([tree.random_state for tree in clf_no_ws])

        npt.assert_array_equal(clf_ws.apply(X), clf_no_ws.apply(X))
