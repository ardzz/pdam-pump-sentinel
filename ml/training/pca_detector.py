from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

_sklearn_base = import_module('sklearn.base')
_sklearn_decomposition = import_module('sklearn.decomposition')
_sklearn_preprocessing = import_module('sklearn.preprocessing')
_sklearn_validation = import_module('sklearn.utils.validation')

BaseEstimator = _sklearn_base.BaseEstimator
clone = _sklearn_base.clone
PCA = _sklearn_decomposition.PCA
RobustScaler = _sklearn_preprocessing.RobustScaler
StandardScaler = _sklearn_preprocessing.StandardScaler
check_array = _sklearn_validation.check_array
check_is_fitted = _sklearn_validation.check_is_fitted


class PcaT2QDetector(BaseEstimator):
    """PCA Hotelling T² and SPE/Q anomaly detector.

    The detector follows the scikit-learn estimator contract so it can be cloned,
    pickled, and logged through MLflow's sklearn flavor. Higher scores indicate
    samples farther from the fitted normal operating region.
    """

    def __init__(
        self,
        n_components: int | float | None = 0.9,
        threshold_quantile: float = 0.95,
        scaler: str | Any | None = 'robust',
        eigenvalue_floor: float = 1e-12,
    ):
        self.n_components = n_components
        self.threshold_quantile = threshold_quantile
        self.scaler = scaler
        self.eigenvalue_floor = eigenvalue_floor

    def fit(self, X, y=None):
        """Fit the scaler, PCA model, and empirical T²/Q thresholds."""
        del y

        quantile = self._validated_threshold_quantile()
        eigenvalue_floor = self._validated_eigenvalue_floor()
        X_checked = check_array(X, dtype=np.float64, ensure_2d=True, estimator=self)

        self.n_features_in_ = X_checked.shape[1]
        self._set_feature_names_in(X)
        self.scaler_ = self._make_scaler()
        X_scaled = self._fit_transform_scaler(X_checked)

        self.pca_ = PCA(n_components=self.n_components, svd_solver='full')
        self.pca_.fit(X_scaled)
        self.explained_variance_ = np.asarray(self.pca_.explained_variance_, dtype=np.float64)
        self.safe_explained_variance_ = np.maximum(self.explained_variance_, eigenvalue_floor)

        t2, q = self._statistics_from_scaled(X_scaled)
        self.training_t2_ = t2
        self.training_q_ = q
        self.t2_threshold_ = max(float(np.quantile(t2, quantile)), eigenvalue_floor)
        self.q_threshold_ = max(float(np.quantile(q, quantile)), eigenvalue_floor)
        return self

    def calibrate_thresholds(self, X) -> 'PcaT2QDetector':
        """Recalibrate thresholds from validation-normal features without refitting."""
        check_is_fitted(self, ('pca_', 'scaler_', 't2_threshold_', 'q_threshold_'))
        quantile = self._validated_threshold_quantile()
        eigenvalue_floor = self._validated_eigenvalue_floor()
        X_checked = check_array(X, dtype=np.float64, ensure_2d=True, estimator=self)
        if X_checked.shape[1] != self.n_features_in_:
            msg = f'X has {X_checked.shape[1]} features, but PcaT2QDetector is fitted with {self.n_features_in_}'
            raise ValueError(msg)

        t2, q = self._statistics_from_scaled(self._transform_scaler(X_checked))
        self.t2_threshold_ = max(float(np.quantile(t2, quantile)), eigenvalue_floor)
        self.q_threshold_ = max(float(np.quantile(q, quantile)), eigenvalue_floor)
        return self

    def score_samples(self, X) -> np.ndarray:
        """Return a normalized anomaly score for each sample.

        The returned score is max(T² / T²_threshold, Q / Q_threshold), so values
        above 1.0 are predicted anomalous by the detector decision rule.
        """
        t2, q = self._statistics(X)
        return np.maximum(t2 / self.t2_threshold_, q / self.q_threshold_)

    def predict(self, X) -> np.ndarray:
        """Return integer labels where 1 means anomaly and 0 means normal."""
        t2, q = self._statistics(X)
        return ((t2 > self.t2_threshold_) | (q > self.q_threshold_)).astype(np.int_)

    def transform(self, X) -> np.ndarray:
        """Return per-sample [T², Q] statistics."""
        t2, q = self._statistics(X)
        return np.column_stack((t2, q))

    def _statistics(self, X) -> tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ('pca_', 'scaler_', 't2_threshold_', 'q_threshold_'))
        X_checked = check_array(X, dtype=np.float64, ensure_2d=True, estimator=self)
        if X_checked.shape[1] != self.n_features_in_:
            msg = f'X has {X_checked.shape[1]} features, but PcaT2QDetector is fitted with {self.n_features_in_}'
            raise ValueError(msg)
        return self._statistics_from_scaled(self._transform_scaler(X_checked))

    def _statistics_from_scaled(self, X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = self.pca_.transform(X_scaled)
        t2 = np.sum((scores**2) / self.safe_explained_variance_, axis=1)
        reconstructed = self.pca_.inverse_transform(scores)
        q = np.sum((X_scaled - reconstructed) ** 2, axis=1)
        return t2, q

    def _fit_transform_scaler(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_ is None:
            return X.copy()
        return self.scaler_.fit_transform(X)

    def _transform_scaler(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_ is None:
            return X.copy()
        return self.scaler_.transform(X)

    def _make_scaler(self):
        if self.scaler is None:
            return None
        if isinstance(self.scaler, str):
            scaler_name = self.scaler.lower()
            if scaler_name == 'standard':
                return StandardScaler()
            if scaler_name == 'robust':
                return RobustScaler()
            if scaler_name == 'none':
                return None
            msg = "scaler must be one of 'standard', 'robust', 'none', None, or a sklearn-compatible transformer"
            raise ValueError(msg)
        if hasattr(self.scaler, 'fit_transform') and hasattr(self.scaler, 'transform'):
            return clone(self.scaler)
        msg = "scaler must be one of 'standard', 'robust', 'none', None, or a sklearn-compatible transformer"
        raise ValueError(msg)

    def _validated_threshold_quantile(self) -> float:
        quantile = float(self.threshold_quantile)
        if not 0.0 < quantile < 1.0:
            msg = 'threshold_quantile must be between 0 and 1, exclusive'
            raise ValueError(msg)
        return quantile

    def _validated_eigenvalue_floor(self) -> float:
        eigenvalue_floor = float(self.eigenvalue_floor)
        if eigenvalue_floor <= 0.0:
            msg = 'eigenvalue_floor must be positive'
            raise ValueError(msg)
        return eigenvalue_floor

    def _set_feature_names_in(self, X) -> None:
        columns = getattr(X, 'columns', None)
        if columns is not None and all(isinstance(column, str) for column in columns):
            self.feature_names_in_ = np.asarray(columns, dtype=object)


__all__ = ['PcaT2QDetector']
