"""Test module for the FPBoost class."""

import numpy as np
from sklearn.model_selection import (ShuffleSplit, cross_val_score,
                                     train_test_split)
from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder

from fpboost.models import FPBoost


def test_fpboost_single_training():
    """Test the FPBoost class."""
    random_state = 1234

    np.random.seed(random_state)

    X, y = load_gbsg2()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.75, random_state=random_state
    )
    X_train = OneHotEncoder().fit_transform(X_train)
    X_test = OneHotEncoder().fit_transform(X_test)

    fpboost = FPBoost(random_state=random_state)
    fpboost.fit(X_train, y_train)

    score = fpboost.score(X_test, y_test)
    risk = fpboost.predict(X_test).mean()

    assert np.isclose(score, 0.6713695357396526), score
    assert np.isclose(risk, -0.6589556059563442), risk


def test_fpboost_cross_validation():
    """Test the FPBoost class with cross-validation."""
    random_state = 1234

    np.random.seed(random_state)

    X, y = load_gbsg2()
    X = OneHotEncoder().fit_transform(X)

    fpboost = FPBoost(random_state=random_state)
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=random_state)
    scores = cross_val_score(fpboost, X, y, cv=cv)

    assert np.isclose(scores[0], 0.651973628602823), scores[0]
    assert np.isclose(scores[1], 0.6941291744917123), scores[1]
    assert np.isclose(scores[2], 0.6940556628056628), scores[2]


def test_fpboost_survival_fn():
    """Test the FPBoost class with survival function."""
    random_state = 1234

    np.random.seed(random_state)

    X, y = load_gbsg2()
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.75, random_state=random_state)
    X_train = OneHotEncoder().fit_transform(X_train)
    X_test = OneHotEncoder().fit_transform(X_test)

    fpboost = FPBoost(random_state=random_state)
    fpboost.fit(X_train, y_train)

    survival_fns = fpboost.predict_survival_function(X_test)
    survival_fn = survival_fns[0]

    assert np.isclose(survival_fn(0), 1.0), survival_fn(0)
    assert np.isclose(survival_fn(50), 0.9946023789232016), survival_fn(50)
    assert np.isclose(survival_fn(100), 0.9802410014332007), survival_fn(100)
    assert np.isclose(survival_fn(1000), 0.6405448045963135), survival_fn(1000)
