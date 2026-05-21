import numpy as np
import pytest
from scipy.stats import qmc

from surrogatelab.surrogates import KrigingSurrogate


def _smooth_target(X: np.ndarray) -> np.ndarray:
    return 2.0 + np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1])


def test_kriging_fits_known_function():
    rng_train = qmc.LatinHypercube(d=2, scramble=True, rng=0).random(30)
    rng_test = np.random.default_rng(1).uniform(0.0, 1.0, (100, 2))
    y_train = _smooth_target(rng_train)
    y_test = _smooth_target(rng_test)

    model = KrigingSurrogate(log_transform=True, n_restarts=3).fit(rng_train, y_train)
    y_pred = model.predict(rng_test)
    rss = float(np.sum((y_test - y_pred) ** 2))
    tss = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = 1.0 - rss / tss
    assert r2 > 0.95


def test_kriging_interpolation_property():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, (15, 2))
    y = _smooth_target(X)
    model = KrigingSurrogate(log_transform=False, nugget=1e-12, n_restarts=3).fit(X, y)
    y_pred_train = model.predict(X)
    assert np.allclose(y_pred_train, y, atol=1e-4)


def test_kriging_log_transform_positive_y():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, (10, 2))
    y_bad = np.array([1.0, 2.0, -0.5, 3.0, 1.5, 0.0, 2.5, 1.2, 3.3, 4.1])
    with pytest.raises(ValueError):
        KrigingSurrogate(log_transform=True).fit(X, y_bad)
