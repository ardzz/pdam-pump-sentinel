import pickle
from importlib import import_module

import numpy as np
import pytest


def _sklearn_clone(estimator):
    return import_module('sklearn.base').clone(estimator)


def _not_fitted_error():
    return import_module('sklearn.exceptions').NotFittedError


def _detector_class():
    return import_module('ml.training.pca_detector').PcaT2QDetector


def _training_matrix(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(160, 2))
    noise = rng.normal(scale=0.05, size=(160, 4))
    return np.column_stack(
        (
            latent[:, 0],
            latent[:, 1],
            0.7 * latent[:, 0] + 0.2 * latent[:, 1],
            -0.4 * latent[:, 0] + 0.8 * latent[:, 1],
        )
    ) + noise


def test_fit_learns_positive_thresholds_and_finite_scores():
    detector = _detector_class()()
    X = _training_matrix()

    fitted = detector.fit(X)
    scores = detector.score_samples(X)
    statistics = detector.transform(X)

    assert fitted is detector
    assert detector.t2_threshold_ > 0
    assert detector.q_threshold_ > 0
    assert scores.shape == (X.shape[0],)
    assert statistics.shape == (X.shape[0], 2)
    assert np.all(np.isfinite(scores))
    assert np.all(np.isfinite(statistics))


def test_detector_defaults_to_robust_scaler():
    detector = _detector_class()().fit(_training_matrix())

    assert detector.scaler == 'robust'
    assert detector.scaler_.__class__.__name__ == 'RobustScaler'


def test_predict_flags_shifted_sample_and_keeps_training_samples_mostly_normal():
    detector = _detector_class()().fit(_training_matrix())
    X = _training_matrix()
    shifted = X[:1] + np.array([[7.0, -7.0, 5.0, -5.0]])

    training_labels = detector.predict(X)

    assert set(np.unique(training_labels)).issubset({0, 1})
    assert training_labels.dtype.kind in {'i', 'u'}
    assert training_labels.mean() <= 0.15
    assert detector.predict(shifted).tolist() == [1]


def test_detector_is_cloneable_and_pickle_serializable_after_fit():
    detector_class = _detector_class()
    X = _training_matrix()
    detector = detector_class().fit(X)

    cloned = _sklearn_clone(detector_class(threshold_quantile=0.9)).fit(X)
    round_tripped = pickle.loads(pickle.dumps(detector))

    assert cloned.predict(X[:3]).shape == (3,)
    np.testing.assert_array_equal(round_tripped.predict(X[:5]), detector.predict(X[:5]))
    np.testing.assert_allclose(round_tripped.score_samples(X[:5]), detector.score_samples(X[:5]))


def test_calibrate_thresholds_updates_from_validation_without_refitting():
    detector_class = _detector_class()
    rng = np.random.default_rng(42)
    X_train = _training_matrix(seed=7)
    X_val = _training_matrix(seed=99) + rng.normal(scale=0.02, size=X_train.shape)

    detector = detector_class(threshold_quantile=0.95).fit(X_train)
    old_t2_thresh = float(detector.t2_threshold_)
    old_q_thresh = float(detector.q_threshold_)
    old_pca_components = detector.pca_.components_.copy()
    old_scaler_scale = detector.scaler_.scale_.copy() if detector.scaler_ is not None else None

    detector.calibrate_thresholds(X_val)

    assert detector.t2_threshold_ != old_t2_thresh
    assert detector.q_threshold_ != old_q_thresh
    np.testing.assert_array_equal(detector.pca_.components_, old_pca_components)
    if old_scaler_scale is not None:
        np.testing.assert_array_equal(detector.scaler_.scale_, old_scaler_scale)

    scores = detector.score_samples(X_val)
    labels = detector.predict(X_val)
    assert scores.shape == (X_val.shape[0],)
    assert labels.shape == (X_val.shape[0],)
    assert set(np.unique(labels)).issubset({0, 1})


def test_calibrate_thresholds_uses_quantile_on_validation_scores():
    detector_class = _detector_class()
    rng = np.random.default_rng(42)
    X_train = _training_matrix(seed=7)
    X_val = _training_matrix(seed=99) + rng.normal(scale=0.02, size=X_train.shape)

    detector = detector_class(threshold_quantile=0.90).fit(X_train)
    detector.calibrate_thresholds(X_val)

    t2_val, q_val = detector.transform(X_val).T
    expected_t2 = max(float(np.quantile(t2_val, 0.90)), detector.eigenvalue_floor)
    expected_q = max(float(np.quantile(q_val, 0.90)), detector.eigenvalue_floor)

    np.testing.assert_allclose(detector.t2_threshold_, expected_t2, rtol=1e-12)
    np.testing.assert_allclose(detector.q_threshold_, expected_q, rtol=1e-12)


def test_calibrate_thresholds_before_fit_raises():
    detector_class = _detector_class()
    X = _training_matrix()
    detector = detector_class()
    with pytest.raises(_not_fitted_error()):
        detector.calibrate_thresholds(X)


def test_invalid_input_behavior_is_explicit():
    detector_class = _detector_class()
    X = _training_matrix()

    with pytest.raises(_not_fitted_error()):
        detector_class().predict(X)
    with pytest.raises(ValueError, match='threshold_quantile'):
        detector_class(threshold_quantile=1.0).fit(X)
    with pytest.raises(ValueError, match='scaler'):
        detector_class(scaler='minmax').fit(X)
    with pytest.raises(ValueError):
        detector_class().fit(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        detector_class().fit(np.array([[1.0, np.nan], [2.0, 3.0]]))

    fitted = detector_class().fit(X)
    with pytest.raises(ValueError, match='features'):
        fitted.predict(X[:, :3])
