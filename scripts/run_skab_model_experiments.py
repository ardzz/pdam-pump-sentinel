from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import import_module
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.datasets.skab_manifest import load_skab_split_manifest  # noqa: E402
from ml.evaluation.metrics import evaluate_split  # noqa: E402
from ml.training import train_pca as pca_training  # noqa: E402
from ml.training.pca_detector import PcaT2QDetector  # noqa: E402
from ml.training.train_distance_profile import (  # noqa: E402
    DistanceProfileTrainingConfig,
    train_distance_profile_from_skab,
)
from ml.training.train_forecasting_residual import (  # noqa: E402
    ForecastingResidualTrainingConfig,
    train_forecasting_residual_from_skab,
)
from ml.training.train_isoforest import IsoForestTrainingConfig, train_isoforest_from_skab  # noqa: E402
from ml.training.train_lstm_ae import LstmAeTrainingConfig, train_lstm_ae_from_skab  # noqa: E402
from ml.training.train_oneclass_svm import OneClassSvmTrainingConfig, train_oneclass_svm_from_skab  # noqa: E402
from ml.training.train_pca import PcaTrainingConfig, train_pca_from_skab  # noqa: E402
from ml.training.train_supervised import SupervisedTrainingConfig, train_supervised_from_skab  # noqa: E402

DEFAULT_MANIFEST_PATH = _PROJECT_ROOT / 'data' / 'skab_split_manifest.json'
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / 'artifacts' / 'skab-model-experiments'
DEFAULT_MLFLOW_EXPERIMENT_NAME = 'skab_model_experiments'
SPLIT_PROTOCOL = 'manifest_train_validation_test_no_point_adjustment'
SUPERVISED_LABEL_WARNING = (
    'XGBoost/LightGBM are supervised upper-bound experiments only; they require labeled anomaly examples in '
    'the training split and must not be described as deployment-safe novel-fault detectors.'
)
DEPLOYMENT_COMPARISON_KEYS = frozenset(
    {
        '01_pca_raw',
        '02_pca_spectral',
        '03_pca_feature_bagging_ensemble',
        '04_lstm_ae',
        '08_isolation_forest_spectral',
        '09_oneclass_svm_spectral',
        '11_forecasting_residual_naive',
        '12_distance_profile_nearest_normal',
    }
)
REPORTING_EXPERIMENT_FAMILY_NOTES = [
    {
        'key': '08_isolation_forest_spectral',
        'role': 'normal-only comparative baseline',
        'note': (
            'Isolation Forest on spectral features, fit on train-normal windows and calibrated on '
            'validation-normal windows.'
        ),
    },
    {
        'key': '09_oneclass_svm_spectral',
        'role': 'normal-only comparative baseline',
        'note': 'One-Class SVM on spectral features, scored as negative decision_function so higher means more abnormal.',
    },
    {
        'key': '10_isolation_forest_conformal',
        'role': 'threshold calibration variant',
        'note': 'Conformal wrapper around Isolation Forest scores, not a separately trained model family.',
    },
    {
        'key': '11_forecasting_residual_naive',
        'role': 'normal-only comparative baseline',
        'note': 'Naive lag-one residual score with no future rows used for prediction.',
    },
    {
        'key': '12_distance_profile_nearest_normal',
        'role': 'bounded optional comparative baseline',
        'note': 'Nearest train-normal window distance with runtime caps and no extra matrix-profile dependency.',
    },
]
REPORTING_CAVEATS = [
    'Train-normal fitting, validation-normal calibration, and held-out test metrics are separate protocol steps.',
    'No point-adjustment is applied to thresholded metrics.',
    SUPERVISED_LABEL_WARNING,
    (
        'New unsupervised, calibration, and optional distance-profile rows are comparison evidence, not automatic '
        'deployment replacements.'
    ),
]


@dataclass(frozen=True)
class HarnessConfig:
    manifest_path: Path
    output_dir: Path
    window_size: int
    stride: int
    seed: int
    lstm_epochs: int
    lstm_batch_size: int
    lstm_patience: int
    supervised_estimators: int
    isoforest_estimators: int
    oneclass_svm_nu: float
    oneclass_svm_max_iter: int
    distance_profile_max_reference_windows: int
    distance_profile_max_distance_comparisons: int
    pca_ensemble_size: int
    pca_ensemble_subset_ratio: float
    pca_variant_feature_mode: str
    skip_lstm: bool
    smoke: bool
    log_mlflow: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT_NAME


@dataclass(frozen=True)
class ExperimentRecord:
    key: str
    display_name: str
    model_family: str
    feature_mode: str
    split_protocol: str
    status: str
    output_dir: Path
    metrics: dict[str, Any]
    artifact_paths: dict[str, Path]
    notes: str = ''
    metadata: dict[str, Any] | None = None
    params: dict[str, Any] | None = None


@dataclass(frozen=True)
class DbscanThresholdRule:
    threshold: float
    fallback_threshold: float
    fallback_used: bool
    reason: str
    eps: float | None
    min_samples: int | None
    cluster_count: int
    noise_count: int
    high_error_cluster: int | None

    def to_metadata(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class ConformalThresholdRule:
    threshold: float
    alpha: float
    calibration_count: int
    corrected_quantile_rank: int
    corrected_quantile_level: float
    threshold_is_infinite: bool
    corrected_quantile_formula: str
    prediction_rule: str
    reason: str

    def to_metadata(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def main(argv: Sequence[str] | None = None) -> list[ExperimentRecord]:
    config = _parse_args(argv)
    records = run_experiments(config)
    payload = {
        'output_dir': str(config.output_dir),
        'summary_csv': str(config.output_dir / 'summary.csv'),
        'summary_md': str(config.output_dir / 'summary.md'),
        'metadata_json': str(config.output_dir / 'metadata.json'),
        'experiments': [record.key for record in records],
    }
    print(json.dumps(payload, sort_keys=True))
    return records


def run_experiments(config: HarnessConfig) -> list[ExperimentRecord]:
    _validate_config(config)
    manifest = load_skab_split_manifest(config.manifest_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[ExperimentRecord] = []
    records.append(_run_pca_experiment(config, '01_pca_raw', 'PCA T²/Q baseline', 'raw'))
    records.append(
        _run_pca_experiment(
            config,
            f'02_pca_{config.pca_variant_feature_mode}',
            f'PCA {config.pca_variant_feature_mode} variant',
            config.pca_variant_feature_mode,
        )
    )
    records.append(_run_pca_ensemble(config, manifest))

    if config.skip_lstm:
        lstm_baseline = _skipped_record(
            config,
            key='04_lstm_ae',
            display_name='LSTM-AE threshold baseline',
            model_family='lstm_ae',
            feature_mode='raw',
            reason='skipped by --skip-lstm',
        )
    else:
        lstm_baseline = _run_lstm_experiment(config)
    records.append(lstm_baseline)
    records.append(_run_lstm_dbscan_experiment(config, lstm_baseline))
    records.append(_run_supervised_experiment(config, '06_xgboost', 'XGBoost supervised upper bound', 'xgboost'))
    records.append(_run_supervised_experiment(config, '07_lightgbm', 'LightGBM supervised upper bound', 'lightgbm'))
    isoforest_record = _run_isoforest_experiment(config)
    records.append(isoforest_record)
    records.append(_run_oneclass_svm_experiment(config))
    records.append(
        _run_conformal_threshold_experiment(
            config,
            isoforest_record,
            key='10_isolation_forest_conformal',
            display_name='Isolation Forest conformal threshold wrapper',
            model_family='isolation_forest_conformal_threshold',
            feature_mode='spectral',
            alpha=0.1,
        )
    )
    records.append(_run_forecasting_residual_experiment(config))
    records.append(_run_distance_profile_experiment(config))

    _write_root_artifacts(config, records)
    _log_records_to_mlflow(config, records)
    return records


def _parse_args(argv: Sequence[str] | None) -> HarnessConfig:
    parser = argparse.ArgumentParser(
        description='Run the approved SKAB model experiment sequence without random row/window splits.',
    )
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--window-size', type=int, default=None)
    parser.add_argument('--stride', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lstm-epochs', type=int, default=None)
    parser.add_argument('--lstm-batch-size', type=int, default=None)
    parser.add_argument('--lstm-patience', type=int, default=None)
    parser.add_argument('--supervised-estimators', type=int, default=None)
    parser.add_argument('--isoforest-estimators', type=int, default=None)
    parser.add_argument('--oneclass-svm-nu', type=float, default=0.05)
    parser.add_argument('--oneclass-svm-max-iter', type=int, default=None)
    parser.add_argument('--distance-profile-max-reference-windows', type=int, default=None)
    parser.add_argument('--distance-profile-max-distance-comparisons', type=int, default=None)
    parser.add_argument('--pca-ensemble-size', type=int, default=None)
    parser.add_argument('--pca-ensemble-subset-ratio', type=float, default=0.7)
    parser.add_argument('--pca-variant-feature-mode', choices=('spectral', 'enriched'), default='spectral')
    parser.add_argument('--skip-lstm', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Use faster smoke defaults unless explicit overrides are provided.')
    parser.add_argument('--log-mlflow', action='store_true', help='Opt in to harness-level MLflow logging after training completes.')
    parser.add_argument('--mlflow-tracking-uri', default=None)
    parser.add_argument(
        '--mlflow-experiment-name',
        '--experiment-name',
        dest='mlflow_experiment_name',
        default=DEFAULT_MLFLOW_EXPERIMENT_NAME,
    )
    args = parser.parse_args(argv)

    return HarnessConfig(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        window_size=args.window_size if args.window_size is not None else 60,
        stride=args.stride if args.stride is not None else (30 if args.smoke else 1),
        seed=args.seed,
        lstm_epochs=args.lstm_epochs if args.lstm_epochs is not None else (1 if args.smoke else 30),
        lstm_batch_size=args.lstm_batch_size if args.lstm_batch_size is not None else (16 if args.smoke else 32),
        lstm_patience=args.lstm_patience if args.lstm_patience is not None else (1 if args.smoke else 5),
        supervised_estimators=args.supervised_estimators if args.supervised_estimators is not None else (8 if args.smoke else 200),
        isoforest_estimators=args.isoforest_estimators if args.isoforest_estimators is not None else (8 if args.smoke else 200),
        oneclass_svm_nu=args.oneclass_svm_nu,
        oneclass_svm_max_iter=args.oneclass_svm_max_iter if args.oneclass_svm_max_iter is not None else (200 if args.smoke else 1000),
        distance_profile_max_reference_windows=(
            args.distance_profile_max_reference_windows
            if args.distance_profile_max_reference_windows is not None
            else (32 if args.smoke else 512)
        ),
        distance_profile_max_distance_comparisons=(
            args.distance_profile_max_distance_comparisons
            if args.distance_profile_max_distance_comparisons is not None
            else (100_000 if args.smoke else 2_000_000)
        ),
        pca_ensemble_size=args.pca_ensemble_size if args.pca_ensemble_size is not None else (2 if args.smoke else 8),
        pca_ensemble_subset_ratio=args.pca_ensemble_subset_ratio,
        pca_variant_feature_mode=args.pca_variant_feature_mode,
        skip_lstm=args.skip_lstm,
        smoke=args.smoke,
        log_mlflow=args.log_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
    )


def _validate_config(config: HarnessConfig) -> None:
    for name in (
        'window_size',
        'stride',
        'lstm_epochs',
        'lstm_batch_size',
        'lstm_patience',
        'supervised_estimators',
        'isoforest_estimators',
        'oneclass_svm_max_iter',
        'distance_profile_max_reference_windows',
        'distance_profile_max_distance_comparisons',
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if not 0.0 < float(config.oneclass_svm_nu) <= 1.0:
        raise ValueError('oneclass_svm_nu must be in the interval (0, 1]')
    if isinstance(config.pca_ensemble_size, bool) or config.pca_ensemble_size <= 0:
        raise ValueError('pca_ensemble_size must be a positive integer')
    if not 0.0 < float(config.pca_ensemble_subset_ratio) <= 1.0:
        raise ValueError('pca_ensemble_subset_ratio must be in the interval (0, 1]')


def _run_pca_experiment(config: HarnessConfig, key: str, display_name: str, feature_mode: str) -> ExperimentRecord:
    output_dir = config.output_dir / key
    result = train_pca_from_skab(
        PcaTrainingConfig(
            input_path=output_dir,
            output_dir=output_dir,
            split_manifest_path=config.manifest_path,
            window_size=config.window_size,
            stride=config.stride,
            threshold_quantile=0.95,
            scaler='robust',
            feature_mode=feature_mode,
            log_mlflow=False,
        )
    )
    return _record_from_training_result(
        key=key,
        display_name=display_name,
        model_family='pca',
        feature_mode=feature_mode,
        result=result,
    )


def _run_lstm_experiment(config: HarnessConfig) -> ExperimentRecord:
    key = '04_lstm_ae'
    output_dir = config.output_dir / key
    result = train_lstm_ae_from_skab(
        LstmAeTrainingConfig(
            input_path=output_dir,
            output_dir=output_dir,
            split_manifest_path=config.manifest_path,
            window_size=config.window_size,
            stride=config.stride,
            threshold_quantile=0.99,
            epochs=config.lstm_epochs,
            batch_size=config.lstm_batch_size,
            patience=config.lstm_patience,
            seed=config.seed,
            log_mlflow=False,
        )
    )
    return _record_from_training_result(
        key=key,
        display_name='LSTM-AE threshold baseline',
        model_family='lstm_ae',
        feature_mode='raw',
        result=result,
    )


def _run_supervised_experiment(config: HarnessConfig, key: str, display_name: str, model_type: str) -> ExperimentRecord:
    output_dir = config.output_dir / key
    try:
        result = train_supervised_from_skab(
            SupervisedTrainingConfig(
                input_path=output_dir,
                output_dir=output_dir,
                split_manifest_path=config.manifest_path,
                window_size=config.window_size,
                stride=config.stride,
                model_type=model_type,
                n_estimators=config.supervised_estimators,
                early_stopping_rounds=max(1, min(30, config.supervised_estimators // 4)),
                seed=config.seed,
                log_mlflow=False,
            )
        )
    except ValueError as exc:
        reason = str(exc)
        if 'training split must contain both binary classes' not in reason:
            raise
        return _skipped_record(
            config,
            key=key,
            display_name=display_name,
            model_family=model_type,
            feature_mode='enriched',
            reason=f'{reason}; supervised models require labels in manifest train split',
        )
    return _record_from_training_result(
        key=key,
        display_name=display_name,
        model_family=model_type,
        feature_mode='enriched',
        result=result,
        notes=SUPERVISED_LABEL_WARNING,
    )


def _run_isoforest_experiment(config: HarnessConfig) -> ExperimentRecord:
    key = '08_isolation_forest_spectral'
    output_dir = config.output_dir / key
    result = train_isoforest_from_skab(
        IsoForestTrainingConfig(
            input_path=output_dir,
            output_dir=output_dir,
            split_manifest_path=config.manifest_path,
            window_size=config.window_size,
            stride=config.stride,
            threshold_quantile=0.95,
            scaler='standard',
            feature_mode='spectral',
            log_mlflow=False,
            n_estimators=config.isoforest_estimators,
            seed=config.seed,
        )
    )
    return _record_from_isoforest_training_result(
        key=key,
        display_name='Isolation Forest spectral baseline',
        result=result,
    )


def _record_from_isoforest_training_result(*, key: str, display_name: str, result: Any) -> ExperimentRecord:
    metadata_path = Path(getattr(result, 'artifact_paths')['metadata'])
    metadata = _read_json(metadata_path)
    protocol_metadata = {
        'key': key,
        'display_name': display_name,
        'status': 'completed',
        'split_protocol': SPLIT_PROTOCOL,
        'feature_mode': 'spectral',
        'threshold_method': 'validation_normal_quantile',
        'calibration_split': 'validation_normal_windows',
        'threshold_calibration': 'threshold calibrated on validation windows with anomaly label 0 only',
        'train_label_policy': 'fit Isolation Forest on manifest train windows with anomaly label 0 only',
        'test_metrics_policy': 'held-out test split used only for final test_ metrics after calibration',
        'test_labels_used_for_training_or_threshold': False,
    }
    metadata.update(protocol_metadata)
    _write_json(metadata_path, metadata)
    return _record_from_training_result(
        key=key,
        display_name=display_name,
        model_family='isolation_forest',
        feature_mode='spectral',
        result=result,
        notes='Normal-only train; threshold calibrated on validation-normal windows; held-out test metrics only.',
        metadata=metadata,
        experiment_metadata_extra=protocol_metadata,
    )


def _run_oneclass_svm_experiment(config: HarnessConfig) -> ExperimentRecord:
    key = '09_oneclass_svm_spectral'
    output_dir = config.output_dir / key
    result = train_oneclass_svm_from_skab(
        OneClassSvmTrainingConfig(
            input_path=output_dir,
            output_dir=output_dir,
            split_manifest_path=config.manifest_path,
            window_size=config.window_size,
            stride=config.stride,
            threshold_quantile=0.95,
            scaler='standard',
            feature_mode='spectral',
            log_mlflow=False,
            kernel='rbf',
            nu=config.oneclass_svm_nu,
            gamma='scale',
            max_iter=config.oneclass_svm_max_iter,
        )
    )
    return _record_from_oneclass_svm_training_result(
        key=key,
        display_name='One-Class SVM spectral baseline',
        result=result,
    )


def _record_from_oneclass_svm_training_result(*, key: str, display_name: str, result: Any) -> ExperimentRecord:
    metadata_path = Path(getattr(result, 'artifact_paths')['metadata'])
    metadata = _read_json(metadata_path)
    protocol_metadata = {
        'key': key,
        'display_name': display_name,
        'status': 'completed',
        'split_protocol': SPLIT_PROTOCOL,
        'feature_mode': 'spectral',
        'threshold_method': 'validation_normal_quantile',
        'calibration_split': 'validation_normal_windows',
        'threshold_calibration': 'threshold calibrated on validation windows with anomaly label 0 only',
        'score_orientation': 'higher_is_more_anomalous:-decision_function',
        'train_label_policy': 'fit One-Class SVM on manifest train windows with anomaly label 0 only',
        'test_metrics_policy': 'held-out test split used only for final test_ metrics after calibration',
        'test_labels_used_for_training_or_threshold': False,
    }
    metadata.update(protocol_metadata)
    _write_json(metadata_path, metadata)
    return _record_from_training_result(
        key=key,
        display_name=display_name,
        model_family='oneclass_svm',
        feature_mode='spectral',
        result=result,
        notes='Normal-only train; score=-decision_function; threshold calibrated on validation-normal windows; held-out test metrics only.',
        metadata=metadata,
        experiment_metadata_extra=protocol_metadata,
    )


def _run_forecasting_residual_experiment(config: HarnessConfig) -> ExperimentRecord:
    key = '11_forecasting_residual_naive'
    output_dir = config.output_dir / key
    result = train_forecasting_residual_from_skab(
        ForecastingResidualTrainingConfig(
            input_path=output_dir,
            output_dir=output_dir,
            split_manifest_path=config.manifest_path,
            window_size=config.window_size,
            stride=config.stride,
            threshold_quantile=0.95,
            aggregation='mean',
        )
    )
    return _record_from_forecasting_residual_training_result(
        key=key,
        display_name='Naive lag-one forecasting residual baseline',
        result=result,
    )


def _record_from_forecasting_residual_training_result(*, key: str, display_name: str, result: Any) -> ExperimentRecord:
    metadata_path = Path(getattr(result, 'artifact_paths')['metadata'])
    metadata = _read_json(metadata_path)
    protocol_metadata = {
        'key': key,
        'display_name': display_name,
        'status': 'completed',
        'split_protocol': SPLIT_PROTOCOL,
        'feature_mode': 'raw_lag_one_residual',
        'threshold_method': 'validation_normal_quantile',
        'calibration_split': 'validation_normal_windows',
        'threshold_calibration': 'threshold calibrated on validation lag-one residual windows with anomaly label 0 only',
        'score_orientation': 'higher_is_more_anomalous:mean_abs_scaled_lag_one_residual',
        'train_label_policy': 'fit lag-one residual scale on manifest train windows with anomaly label 0 only',
        'residual_alignment': 'target is the window final row; yhat_t is the previous row from the same window',
        'future_values_used_for_prediction': False,
        'test_metrics_policy': 'held-out test split used only for final test_ metrics after calibration',
        'test_labels_used_for_training_or_threshold': False,
        'test_scores_used_for_training_or_threshold': False,
    }
    metadata.update(protocol_metadata)
    _write_json(metadata_path, metadata)
    return _record_from_training_result(
        key=key,
        display_name=display_name,
        model_family='forecasting_residual_naive',
        feature_mode='raw_lag_one_residual',
        result=result,
        notes='Naive yhat_t=y_(t-1) residual score; residual scale fit on train-normal windows; threshold calibrated on validation-normal windows.',
        metadata=metadata,
        experiment_metadata_extra=protocol_metadata,
    )


def _run_distance_profile_experiment(config: HarnessConfig) -> ExperimentRecord:
    key = '12_distance_profile_nearest_normal'
    output_dir = config.output_dir / key
    result = train_distance_profile_from_skab(
        DistanceProfileTrainingConfig(
            input_path=output_dir,
            output_dir=output_dir,
            split_manifest_path=config.manifest_path,
            window_size=config.window_size,
            stride=config.stride,
            threshold_quantile=0.95,
            scaler='standard',
            max_reference_windows=config.distance_profile_max_reference_windows,
            max_distance_comparisons=config.distance_profile_max_distance_comparisons,
            reference_selection='evenly_spaced',
        )
    )
    return _record_from_distance_profile_training_result(
        key=key,
        display_name='Nearest-normal distance-profile baseline',
        result=result,
    )


def _record_from_distance_profile_training_result(*, key: str, display_name: str, result: Any) -> ExperimentRecord:
    metadata_path = Path(getattr(result, 'artifact_paths')['metadata'])
    metadata = _read_json(metadata_path)
    protocol_metadata = {
        'key': key,
        'display_name': display_name,
        'status': 'completed',
        'split_protocol': SPLIT_PROTOCOL,
        'feature_mode': 'raw_window_nearest_normal_distance',
        'threshold_method': 'validation_normal_quantile',
        'calibration_split': 'validation_normal_windows',
        'threshold_calibration': 'threshold calibrated on validation-normal nearest-distance scores only',
        'score_orientation': 'higher_is_more_anomalous:nearest_train_normal_window_distance',
        'train_label_policy': 'fit nearest-normal reference windows from manifest train windows with anomaly label 0 only',
        'runtime_guard': metadata.get('runtime_guard'),
        'dependency_status': metadata.get('dependency_status'),
        'test_metrics_policy': 'held-out test split used only for final test_ metrics after calibration',
        'test_labels_used_for_training_or_threshold': False,
        'test_scores_used_for_training_or_threshold': False,
    }
    metadata.update(protocol_metadata)
    _write_json(metadata_path, metadata)
    return _record_from_training_result(
        key=key,
        display_name=display_name,
        model_family='distance_profile_nearest_normal',
        feature_mode='raw_window_nearest_normal_distance',
        result=result,
        notes=(
            'Dependency-light nearest train-normal window distance; reference set is capped for runtime; '
            'threshold calibrated on validation-normal windows.'
        ),
        metadata=metadata,
        experiment_metadata_extra=protocol_metadata,
    )


def _record_from_training_result(
    *,
    key: str,
    display_name: str,
    model_family: str,
    feature_mode: str,
    result: Any,
    notes: str = '',
    metadata: dict[str, Any] | None = None,
    experiment_metadata_extra: dict[str, Any] | None = None,
) -> ExperimentRecord:
    metrics = dict(getattr(result, 'metrics'))
    params = dict(getattr(result, 'params', {}))
    output_dir = Path(getattr(result, 'output_dir'))
    artifact_paths = {name: Path(path) for name, path in getattr(result, 'artifact_paths').items()}
    _write_metrics_csv(output_dir / 'metrics.csv', metrics)
    experiment_metadata = {
        'key': key,
        'display_name': display_name,
        'status': 'completed',
        'split_protocol': SPLIT_PROTOCOL,
        'notes': notes,
    }
    if experiment_metadata_extra is not None:
        experiment_metadata.update(experiment_metadata_extra)
        experiment_metadata['notes'] = notes
    _write_experiment_metadata(
        output_dir,
        experiment_metadata,
    )
    artifact_paths['metrics_csv'] = output_dir / 'metrics.csv'
    artifact_paths['experiment_metadata'] = output_dir / 'experiment_metadata.json'
    return ExperimentRecord(
        key=key,
        display_name=display_name,
        model_family=model_family,
        feature_mode=feature_mode,
        split_protocol=SPLIT_PROTOCOL,
        status='completed',
        output_dir=output_dir,
        metrics=metrics,
        artifact_paths=artifact_paths,
        notes=notes,
        metadata=metadata,
        params=params,
    )


def _run_pca_ensemble(config: HarnessConfig, manifest: Any) -> ExperimentRecord:
    key = '03_pca_feature_bagging_ensemble'
    output_dir = config.output_dir / key
    output_dir.mkdir(parents=True, exist_ok=True)
    pca_config = PcaTrainingConfig(
        input_path=output_dir,
        output_dir=output_dir,
        split_manifest_path=config.manifest_path,
        window_size=config.window_size,
        stride=config.stride,
        threshold_quantile=0.95,
        scaler='robust',
        feature_mode='raw',
        log_mlflow=False,
    )
    train_windows = pca_training._load_windows_multi(manifest.train, pca_config)
    validation_windows = pca_training._load_windows_multi(manifest.validation, pca_config)
    test_windows = pca_training._load_windows_multi(manifest.test, pca_config) if manifest.test else None

    train_normal = train_windows.features[train_windows.labels == 0]
    validation_normal = validation_windows.features[validation_windows.labels == 0]
    if len(train_normal) == 0:
        raise ValueError('training input must contain at least one normal window')
    if len(validation_normal) == 0:
        raise ValueError('validation input must contain at least one normal window')

    members = _fit_pca_ensemble_members(train_normal, validation_normal, config)
    validation_scores, validation_predictions, validation_votes = _score_pca_ensemble(members, validation_windows.features)
    validation_labels = validation_windows.labels.astype(int, copy=False)
    metrics = evaluate_split(
        validation_labels,
        validation_predictions,
        validation_scores,
        transient_mask=validation_windows.changepoints,
    ) | {
        **_count_metrics(validation_labels, validation_windows.changepoints),
        'training_sample_count': int(len(train_windows.labels)),
        'training_normal_count': int(len(train_normal)),
        'train_count': int(len(train_windows.labels)),
        'validation_count': int(len(validation_windows.labels)),
        'ensemble_size': int(len(members)),
        'feature_subset_ratio': float(config.pca_ensemble_subset_ratio),
        'vote_fraction_threshold': 0.5,
    }

    has_test = test_windows is not None and len(test_windows.features) > 0
    if has_test and test_windows is not None:
        test_scores, test_predictions, test_votes = _score_pca_ensemble(members, test_windows.features)
        test_labels = test_windows.labels.astype(int, copy=False)
        test_metrics = evaluate_split(
            test_labels,
            test_predictions,
            test_scores,
            transient_mask=test_windows.changepoints,
        )
        metrics.update({f'test_{name}': value for name, value in test_metrics.items()})
        metrics.update(_count_metrics(test_labels, test_windows.changepoints, prefix='test_'))
        metrics['test_count'] = int(len(test_windows.labels))
        _ensemble_scores_frame(test_windows, test_labels, test_predictions, test_scores, test_votes).to_csv(
            output_dir / 'test_scores.csv', index=False
        )

    _ensemble_scores_frame(
        validation_windows,
        validation_labels,
        validation_predictions,
        validation_scores,
        validation_votes,
    ).to_csv(output_dir / 'scores.csv', index=False)

    _dump_joblib(
        [
            {
                'feature_indices': feature_indices,
                'detector': detector,
            }
            for feature_indices, detector in members
        ],
        output_dir / 'pca_ensemble.joblib',
    )
    artifact_paths = {
        'detectors': output_dir / 'pca_ensemble.joblib',
        'metrics': output_dir / 'metrics.json',
        'metrics_csv': output_dir / 'metrics.csv',
        'metadata': output_dir / 'metadata.json',
        'scores': output_dir / 'scores.csv',
        'experiment_metadata': output_dir / 'experiment_metadata.json',
    }
    if has_test:
        artifact_paths['test_scores'] = output_dir / 'test_scores.csv'
    member_metadata = [
        {
            'member_index': index,
            'feature_count': int(len(feature_indices)),
            'feature_indices': feature_indices.astype(int).tolist(),
            't2_threshold': float(detector.t2_threshold_),
            'q_threshold': float(detector.q_threshold_),
        }
        for index, (feature_indices, detector) in enumerate(members)
    ]
    metadata = {
        'model_family': 'pca_feature_bagging_ensemble',
        'status': 'completed',
        'split_protocol': SPLIT_PROTOCOL,
        'params': {
            'window_size': config.window_size,
            'stride': config.stride,
            'seed': config.seed,
            'ensemble_size': config.pca_ensemble_size,
            'subset_ratio': config.pca_ensemble_subset_ratio,
            'aggregation': 'mean_score_and_majority_vote_fraction_gte_0.5',
            'threshold_calibration': 'each member calibrated on validation windows with anomaly label 0 only',
        },
        'metrics': metrics,
        'members': member_metadata,
        'artifact_paths': {name: str(path) for name, path in artifact_paths.items()},
        'test_split_held_out': has_test,
        'split': {
            'train_files': manifest.to_payload()['train'],
            'validation_files': manifest.to_payload()['validation'],
            'test_files': manifest.to_payload()['test'],
            'train_count': int(len(train_windows.labels)),
            'validation_count': int(len(validation_windows.labels)),
            'test_count': int(len(test_windows.labels)) if test_windows is not None else 0,
        },
    }
    _write_json(output_dir / 'metrics.json', metrics)
    _write_metrics_csv(output_dir / 'metrics.csv', metrics)
    _write_json(output_dir / 'metadata.json', metadata)
    _write_experiment_metadata(output_dir, {'key': key, 'display_name': 'PCA feature-bagging ensemble', 'status': 'completed'})
    return ExperimentRecord(
        key=key,
        display_name='PCA feature-bagging ensemble',
        model_family='pca_feature_bagging_ensemble',
        feature_mode='raw_feature_subsets',
        split_protocol=SPLIT_PROTOCOL,
        status='completed',
        output_dir=output_dir,
        metrics=metrics,
        artifact_paths=artifact_paths,
        metadata=metadata,
        params=dict(metadata['params']),
    )


def _fit_pca_ensemble_members(
    train_normal_features: np.ndarray,
    validation_normal_features: np.ndarray,
    config: HarnessConfig,
) -> list[tuple[np.ndarray, PcaT2QDetector]]:
    rng = np.random.default_rng(config.seed)
    feature_count = int(train_normal_features.shape[1])
    subset_size = max(1, int(math.ceil(feature_count * config.pca_ensemble_subset_ratio)))
    subset_size = min(feature_count, subset_size)
    members: list[tuple[np.ndarray, PcaT2QDetector]] = []
    for _ in range(config.pca_ensemble_size):
        feature_indices = np.sort(rng.choice(feature_count, size=subset_size, replace=False))
        detector = PcaT2QDetector(n_components=0.9, threshold_quantile=0.95, scaler='robust')
        detector.fit(train_normal_features[:, feature_indices])
        detector.calibrate_thresholds(validation_normal_features[:, feature_indices])
        members.append((feature_indices, detector))
    return members


def _score_pca_ensemble(
    members: Sequence[tuple[np.ndarray, PcaT2QDetector]],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not members:
        raise ValueError('members must not be empty')
    member_scores = []
    member_predictions = []
    for feature_indices, detector in members:
        subset = features[:, feature_indices]
        member_scores.append(detector.score_samples(subset))
        member_predictions.append(detector.predict(subset))
    stacked_scores = np.vstack(member_scores)
    stacked_predictions = np.vstack(member_predictions)
    scores = np.mean(stacked_scores, axis=0)
    vote_fraction = np.mean(stacked_predictions, axis=0)
    predictions = (vote_fraction >= 0.5).astype(int)
    return scores, predictions, vote_fraction


def _run_lstm_dbscan_experiment(config: HarnessConfig, baseline_record: ExperimentRecord) -> ExperimentRecord:
    key = '05_lstm_ae_dbscan_threshold'
    output_dir = config.output_dir / key
    if baseline_record.status != 'completed':
        return _skipped_record(
            config,
            key=key,
            display_name='LSTM-AE DBSCAN threshold candidate',
            model_family='lstm_ae_dbscan_threshold',
            feature_mode='raw',
            reason=f'baseline LSTM-AE unavailable: {baseline_record.notes or baseline_record.status}',
        )

    baseline_scores_path = baseline_record.artifact_paths.get('scores')
    baseline_test_scores_path = baseline_record.artifact_paths.get('test_scores')
    if baseline_scores_path is None or not baseline_scores_path.exists():
        return _skipped_record(
            config,
            key=key,
            display_name='LSTM-AE DBSCAN threshold candidate',
            model_family='lstm_ae_dbscan_threshold',
            feature_mode='raw',
            reason='baseline LSTM-AE scores.csv is missing',
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    validation_frame = pd.read_csv(baseline_scores_path)
    fallback_threshold = float(baseline_record.metrics.get('threshold', np.quantile(validation_frame['score'].to_numpy(float), 0.99)))
    rule = _derive_dbscan_threshold(validation_frame['score'].to_numpy(dtype=float), fallback_threshold)
    validation_frame = _apply_score_threshold(validation_frame, rule.threshold)
    validation_labels = validation_frame['label'].to_numpy(dtype=int)
    validation_predictions = validation_frame['prediction'].to_numpy(dtype=int)
    validation_scores = validation_frame['score'].to_numpy(dtype=float)
    validation_changepoints = validation_frame['changepoint'].to_numpy(dtype=int)
    metrics = evaluate_split(
        validation_labels,
        validation_predictions,
        validation_scores,
        transient_mask=validation_changepoints,
    ) | {
        **_count_metrics(validation_labels, validation_changepoints),
        'threshold': rule.threshold,
        't2_threshold': rule.threshold,
        'q_threshold': rule.threshold,
        'dbscan_fallback_used': int(rule.fallback_used),
        'train_count': baseline_record.metrics.get('train_count'),
        'validation_count': int(len(validation_frame)),
    }
    validation_frame.to_csv(output_dir / 'scores.csv', index=False)

    has_test = baseline_test_scores_path is not None and baseline_test_scores_path.exists()
    if has_test and baseline_test_scores_path is not None:
        test_frame = _apply_score_threshold(pd.read_csv(baseline_test_scores_path), rule.threshold)
        test_labels = test_frame['label'].to_numpy(dtype=int)
        test_predictions = test_frame['prediction'].to_numpy(dtype=int)
        test_scores = test_frame['score'].to_numpy(dtype=float)
        test_changepoints = test_frame['changepoint'].to_numpy(dtype=int)
        test_metrics = evaluate_split(test_labels, test_predictions, test_scores, transient_mask=test_changepoints)
        metrics.update({f'test_{name}': value for name, value in test_metrics.items()})
        metrics.update(_count_metrics(test_labels, test_changepoints, prefix='test_'))
        metrics['test_count'] = int(len(test_frame))
        test_frame.to_csv(output_dir / 'test_scores.csv', index=False)

    metadata = {
        'model_family': 'lstm_ae_dbscan_threshold',
        'status': 'completed',
        'split_protocol': SPLIT_PROTOCOL,
        'score_source_experiment': baseline_record.key,
        'threshold_rule': rule.to_metadata(),
        'threshold_calibration': 'DBSCAN fit on validation scores only; test labels and scores are not used for selection',
        'metrics': metrics,
        'test_split_held_out': has_test,
    }
    artifact_paths = {
        'metrics': output_dir / 'metrics.json',
        'metrics_csv': output_dir / 'metrics.csv',
        'metadata': output_dir / 'metadata.json',
        'scores': output_dir / 'scores.csv',
        'experiment_metadata': output_dir / 'experiment_metadata.json',
    }
    if has_test:
        artifact_paths['test_scores'] = output_dir / 'test_scores.csv'
    _write_json(output_dir / 'metrics.json', metrics)
    _write_metrics_csv(output_dir / 'metrics.csv', metrics)
    _write_json(output_dir / 'metadata.json', metadata)
    _write_experiment_metadata(output_dir, {'key': key, 'display_name': 'LSTM-AE DBSCAN threshold candidate', 'status': 'completed'})
    return ExperimentRecord(
        key=key,
        display_name='LSTM-AE DBSCAN threshold candidate',
        model_family='lstm_ae_dbscan_threshold',
        feature_mode='raw',
        split_protocol=SPLIT_PROTOCOL,
        status='completed',
        output_dir=output_dir,
        metrics=metrics,
        artifact_paths=artifact_paths,
        notes='DBSCAN threshold calibrated from validation scores only; fallback is marked in metadata.',
        metadata=metadata,
        params={
            'threshold_method': 'dbscan',
            'source_record_key': baseline_record.key,
            'fallback_threshold': fallback_threshold,
            'fallback_used': rule.fallback_used,
        },
    )


def _run_conformal_threshold_experiment(
    config: HarnessConfig,
    source_record: ExperimentRecord,
    *,
    key: str,
    display_name: str,
    model_family: str,
    feature_mode: str,
    alpha: float,
) -> ExperimentRecord:
    output_dir = config.output_dir / key
    if source_record.status != 'completed':
        return _skipped_record(
            config,
            key=key,
            display_name=display_name,
            model_family=model_family,
            feature_mode=feature_mode,
            reason=f'source score experiment unavailable: {source_record.key} is {source_record.status}',
        )

    source_scores_path = source_record.artifact_paths.get('scores')
    source_test_scores_path = source_record.artifact_paths.get('test_scores')
    if source_scores_path is None or not source_scores_path.exists():
        return _skipped_record(
            config,
            key=key,
            display_name=display_name,
            model_family=model_family,
            feature_mode=feature_mode,
            reason=f'source score experiment {source_record.key} scores.csv is missing',
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    validation_frame = pd.read_csv(source_scores_path)
    calibration_scores = _validation_normal_scores(validation_frame)
    rule = _derive_conformal_threshold(calibration_scores, alpha=alpha)
    validation_frame = _apply_score_threshold(validation_frame, rule.threshold, inclusive=False)
    validation_labels = validation_frame['label'].to_numpy(dtype=int)
    validation_predictions = validation_frame['prediction'].to_numpy(dtype=int)
    validation_scores = validation_frame['score'].to_numpy(dtype=float)
    validation_changepoints = validation_frame['changepoint'].to_numpy(dtype=int)
    metrics = evaluate_split(
        validation_labels,
        validation_predictions,
        validation_scores,
        transient_mask=validation_changepoints,
    ) | {
        **_count_metrics(validation_labels, validation_changepoints),
        'threshold': rule.threshold,
        't2_threshold': rule.threshold,
        'q_threshold': rule.threshold,
        'conformal_alpha': rule.alpha,
        'conformal_calibration_count': rule.calibration_count,
        'conformal_corrected_quantile_rank': rule.corrected_quantile_rank,
        'train_count': source_record.metrics.get('train_count'),
        'validation_count': int(len(validation_frame)),
    }
    validation_frame.to_csv(output_dir / 'scores.csv', index=False)

    has_test = source_test_scores_path is not None and source_test_scores_path.exists()
    if has_test and source_test_scores_path is not None:
        test_frame = _apply_score_threshold(pd.read_csv(source_test_scores_path), rule.threshold, inclusive=False)
        test_labels = test_frame['label'].to_numpy(dtype=int)
        test_predictions = test_frame['prediction'].to_numpy(dtype=int)
        test_scores = test_frame['score'].to_numpy(dtype=float)
        test_changepoints = test_frame['changepoint'].to_numpy(dtype=int)
        test_metrics = evaluate_split(test_labels, test_predictions, test_scores, transient_mask=test_changepoints)
        metrics.update({f'test_{name}': value for name, value in test_metrics.items()})
        metrics.update(_count_metrics(test_labels, test_changepoints, prefix='test_'))
        metrics['test_count'] = int(len(test_frame))
        test_frame.to_csv(output_dir / 'test_scores.csv', index=False)

    protocol_metadata = {
        'key': key,
        'display_name': display_name,
        'status': 'completed',
        'split_protocol': SPLIT_PROTOCOL,
        'feature_mode': feature_mode,
        'threshold_method': 'conformal',
        'alpha': rule.alpha,
        'calibration_split': 'validation_normal_rows',
        'source_record_key': source_record.key,
        'score_source_experiment': source_record.key,
        'threshold_calibration': (
            'conformal threshold derived only from validation-normal score rows; '
            'held-out test labels were not used for thresholding; held-out test scores were not used for thresholding'
        ),
        'held_out_test_labels_used_for_thresholding': False,
        'held_out_test_scores_used_for_thresholding': False,
        'test_labels_used_for_training_or_threshold': False,
        'prediction_rule': rule.prediction_rule,
        'threshold_rule': rule.to_metadata(),
    }
    metadata = {
        'model_family': model_family,
        **protocol_metadata,
        'metrics': metrics,
        'test_split_held_out': has_test,
    }
    artifact_paths = {
        'metrics': output_dir / 'metrics.json',
        'metrics_csv': output_dir / 'metrics.csv',
        'metadata': output_dir / 'metadata.json',
        'scores': output_dir / 'scores.csv',
        'experiment_metadata': output_dir / 'experiment_metadata.json',
    }
    if has_test:
        artifact_paths['test_scores'] = output_dir / 'test_scores.csv'
    _write_json(output_dir / 'metrics.json', metrics)
    _write_metrics_csv(output_dir / 'metrics.csv', metrics)
    _write_json(output_dir / 'metadata.json', metadata)
    _write_experiment_metadata(
        output_dir,
        {
            **protocol_metadata,
            'notes': 'Conformal score-threshold wrapper only; source model training is unchanged.',
        },
    )
    return ExperimentRecord(
        key=key,
        display_name=display_name,
        model_family=model_family,
        feature_mode=feature_mode,
        split_protocol=SPLIT_PROTOCOL,
        status='completed',
        output_dir=output_dir,
        metrics=metrics,
        artifact_paths=artifact_paths,
        notes='Conformal score-threshold wrapper only; source model training is unchanged.',
        metadata=metadata,
        params={'threshold_method': 'conformal', 'alpha': rule.alpha, 'source_record_key': source_record.key},
    )


def _derive_dbscan_threshold(validation_scores: np.ndarray, fallback_threshold: float) -> DbscanThresholdRule:
    scores = np.asarray(validation_scores, dtype=np.float64).reshape(-1)
    scores = scores[np.isfinite(scores)]
    fallback = float(fallback_threshold)
    if scores.size < 4:
        return DbscanThresholdRule(fallback, fallback, True, 'not enough validation scores for DBSCAN', None, None, 0, 0, None)

    scaled = _robust_scaled_log_scores(scores).reshape(-1, 1)
    nearest_distances = _nearest_neighbor_distances(scaled.reshape(-1))
    finite_distances = nearest_distances[np.isfinite(nearest_distances) & (nearest_distances > 0.0)]
    if finite_distances.size == 0:
        return DbscanThresholdRule(fallback, fallback, True, 'validation scores are too degenerate for DBSCAN', None, None, 0, 0, None)

    eps = float(np.clip(np.quantile(finite_distances, 0.9) * 1.5, 0.05, 0.75))
    min_samples = max(2, min(10, int(round(math.sqrt(float(scores.size))))))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
    cluster_labels = sorted(label for label in set(labels.tolist()) if label != -1)
    noise_count = int(np.count_nonzero(labels == -1))
    if not cluster_labels:
        return DbscanThresholdRule(fallback, fallback, True, 'DBSCAN formed no non-noise clusters', eps, min_samples, 0, noise_count, None)

    median_score = float(np.median(scores))
    high_error_candidates: list[tuple[float, int, np.ndarray]] = []
    for label in cluster_labels:
        cluster_scores = scores[labels == label]
        if cluster_scores.size < min_samples:
            continue
        cluster_min = float(np.min(cluster_scores))
        cluster_mean = float(np.mean(cluster_scores))
        if cluster_mean > fallback and cluster_min > median_score:
            high_error_candidates.append((cluster_mean, int(label), cluster_scores))

    if not high_error_candidates:
        return DbscanThresholdRule(
            fallback,
            fallback,
            True,
            'DBSCAN did not form a useful high-error cluster above fallback threshold',
            eps,
            min_samples,
            len(cluster_labels),
            noise_count,
            None,
        )

    _cluster_mean, cluster_label, cluster_scores = max(high_error_candidates, key=lambda item: item[0])
    threshold = float(np.min(cluster_scores))
    return DbscanThresholdRule(
        threshold,
        fallback,
        False,
        'threshold is the minimum score in the highest-mean validation DBSCAN high-error cluster',
        eps,
        min_samples,
        len(cluster_labels),
        noise_count,
        cluster_label,
    )


def _derive_conformal_threshold(calibration_scores: np.ndarray, *, alpha: float) -> ConformalThresholdRule:
    parsed_alpha = float(alpha)
    if not 0.0 < parsed_alpha < 1.0:
        raise ValueError('alpha must be between 0 and 1, exclusive')

    scores = np.asarray(calibration_scores, dtype=np.float64).reshape(-1)
    scores = scores[np.isfinite(scores)]
    n = int(scores.size)
    if n == 0:
        raise ValueError('conformal calibration requires at least one finite validation-normal score')

    corrected_rank = int(math.ceil((n + 1) * (1.0 - parsed_alpha)))
    corrected_level = corrected_rank / n
    sorted_scores = np.sort(scores)
    threshold_is_infinite = corrected_rank > n
    threshold = float('inf') if threshold_is_infinite else float(sorted_scores[corrected_rank - 1])
    return ConformalThresholdRule(
        threshold=threshold,
        alpha=parsed_alpha,
        calibration_count=n,
        corrected_quantile_rank=corrected_rank,
        corrected_quantile_level=float(corrected_level),
        threshold_is_infinite=threshold_is_infinite,
        corrected_quantile_formula='ceil((n + 1) * (1 - alpha))',
        prediction_rule='prediction = score > threshold',
        reason='finite-sample corrected upper quantile of validation-normal calibration scores',
    )


def _validation_normal_scores(frame: pd.DataFrame) -> np.ndarray:
    missing_columns = {'label', 'score'} - set(frame.columns)
    if missing_columns:
        raise ValueError(f'score frame is missing required columns: {sorted(missing_columns)}')
    normal_mask = frame['label'].to_numpy(dtype=int) == 0
    scores = frame.loc[normal_mask, 'score'].to_numpy(dtype=float)
    if scores.size == 0:
        raise ValueError('score frame must contain at least one validation-normal row')
    return scores


def _robust_scaled_log_scores(scores: np.ndarray) -> np.ndarray:
    values = np.log1p(np.maximum(scores, 0.0))
    median = float(np.median(values))
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    scale = iqr if iqr > 0.0 else float(np.std(values))
    if scale <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return (values - median) / scale


def _nearest_neighbor_distances(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(values.reshape(-1))
    if sorted_values.size < 2:
        return np.empty((0,), dtype=np.float64)
    distances = np.empty_like(sorted_values, dtype=np.float64)
    distances[0] = abs(sorted_values[1] - sorted_values[0])
    distances[-1] = abs(sorted_values[-1] - sorted_values[-2])
    if sorted_values.size > 2:
        left = np.abs(sorted_values[1:-1] - sorted_values[:-2])
        right = np.abs(sorted_values[2:] - sorted_values[1:-1])
        distances[1:-1] = np.minimum(left, right)
    return distances


def _apply_score_threshold(frame: pd.DataFrame, threshold: float, *, inclusive: bool = True) -> pd.DataFrame:
    updated = frame.copy()
    scores = updated['score'].to_numpy(dtype=float)
    if inclusive:
        predictions = scores >= float(threshold)
    else:
        predictions = scores > float(threshold)
    updated['prediction'] = predictions.astype(int)
    return updated


def _skipped_record(
    config: HarnessConfig,
    *,
    key: str,
    display_name: str,
    model_family: str,
    feature_mode: str,
    reason: str,
) -> ExperimentRecord:
    output_dir = config.output_dir / key
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {'status': 'skipped', 'reason': reason}
    artifact_paths = {
        'metrics': output_dir / 'metrics.json',
        'metrics_csv': output_dir / 'metrics.csv',
        'experiment_metadata': output_dir / 'experiment_metadata.json',
    }
    _write_json(artifact_paths['metrics'], metrics)
    _write_metrics_csv(artifact_paths['metrics_csv'], metrics)
    _write_experiment_metadata(
        output_dir,
        {
            'key': key,
            'display_name': display_name,
            'status': 'skipped',
            'reason': reason,
            'split_protocol': SPLIT_PROTOCOL,
        },
    )
    return ExperimentRecord(
        key=key,
        display_name=display_name,
        model_family=model_family,
        feature_mode=feature_mode,
        split_protocol=SPLIT_PROTOCOL,
        status='skipped',
        output_dir=output_dir,
        metrics=metrics,
        artifact_paths=artifact_paths,
        notes=reason,
    )


def _ensemble_scores_frame(
    dataset: Any,
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    vote_fraction: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'timestamp': dataset.timestamps.astype(str),
            'label': labels.astype(int),
            'changepoint': dataset.changepoints.astype(int),
            'prediction': predictions.astype(int),
            'score': scores.astype(float),
            'vote_fraction': vote_fraction.astype(float),
        }
    )


def _count_metrics(labels: np.ndarray, changepoints: np.ndarray, prefix: str = '') -> dict[str, int]:
    return {
        f'{prefix}sample_count': int(len(labels)),
        f'{prefix}anomaly_count': int(np.count_nonzero(labels == 1)),
        f'{prefix}normal_count': int(np.count_nonzero(labels == 0)),
        f'{prefix}changepoint_count': int(np.count_nonzero(changepoints == 1)),
    }


def _write_root_artifacts(config: HarnessConfig, records: Sequence[ExperimentRecord]) -> None:
    summary_rows = _build_ranked_summary_rows(records)
    _write_summary_csv(config.output_dir / 'summary.csv', summary_rows)
    _write_summary_md(config.output_dir / 'summary.md', config, summary_rows)
    metadata = {
        'schema_version': 1,
        'generated_at_utc': datetime.now(UTC).isoformat(),
        'script': 'scripts/run_skab_model_experiments.py',
        'config': _jsonable(asdict(config)),
        'split_protocol': SPLIT_PROTOCOL,
        'warnings': REPORTING_CAVEATS,
        'mlflow_logging': _mlflow_reporting_status(config),
        'deployment_preference': _deployment_preference_note(summary_rows),
        'experiment_family_notes': REPORTING_EXPERIMENT_FAMILY_NOTES,
        'summary_rank': 'test_f1 descending; skipped/failed rows rank last',
        'experiments': [_record_payload(record) for record in records],
    }
    _write_json(config.output_dir / 'metadata.json', metadata)


def _mlflow_reporting_status(config: HarnessConfig) -> dict[str, Any]:
    return {
        'enabled': config.log_mlflow,
        'opt_in_flag': '--log-mlflow',
        'requires_mlflow_by_default': False,
        'experiment_name': config.mlflow_experiment_name or DEFAULT_MLFLOW_EXPERIMENT_NAME,
        'tracking_uri': config.mlflow_tracking_uri,
        'local_default_tracking_dir': str(config.output_dir / 'mlruns'),
    }


def _mlflow_summary_line(config: HarnessConfig) -> str:
    status = _mlflow_reporting_status(config)
    if status['enabled']:
        tracking_target = status['tracking_uri'] or f"local file store `{status['local_default_tracking_dir']}`"
        return f"- MLflow logging: enabled, experiment `{status['experiment_name']}`, tracking target {tracking_target}."
    return (
        '- MLflow logging: disabled by default; pass `--log-mlflow` to opt in. Local summaries do not require '
        'MLflow.'
    )


def _deployment_preference_note(rows: Sequence[dict[str, Any]]) -> str:
    pca_row = next((row for row in rows if row.get('experiment_key') == '02_pca_spectral'), None)
    pca_f1 = _metric_or_none(pca_row.get('test_f1') if pca_row else None)
    if pca_row is None or pca_row.get('status') != 'completed' or pca_f1 is None:
        return (
            '`02_pca_spectral` is the deployment-preferred baseline from prior honest runs, but this summary '
            'does not contain a completed PCA spectral held-out `test_f1`; do not make replacement claims from it.'
        )

    candidates = [
        row
        for row in rows
        if row.get('experiment_key') in DEPLOYMENT_COMPARISON_KEYS
        and row.get('experiment_key') != '02_pca_spectral'
        and row.get('status') == 'completed'
        and _metric_or_none(row.get('test_f1')) is not None
    ]
    best = max(candidates, key=_summary_test_f1, default=None)
    best_f1 = _metric_or_none(best.get('test_f1') if best else None)
    if best is not None and best_f1 is not None and best_f1 > pca_f1:
        return (
            '`02_pca_spectral` is the established deployment-preferred baseline, but '
            f"`{best['experiment_key']}` reports higher held-out `test_f1` in this rerun "
            f'({_format_metric(best_f1)} > {_format_metric(pca_f1)}); review false-alarm rate and protocol '
            'metadata '
            'before changing deployment preference.'
        )
    return (
        '`02_pca_spectral` remains the deployment-preferred baseline in this summary; no completed normal-only '
        f'comparison row reports higher held-out `test_f1` than {_format_metric(pca_f1)}.'
    )


def _format_metric(value: Any) -> str:
    metric = _metric_or_none(value)
    return 'unavailable' if metric is None else f'{metric:.6g}'


def _summary_test_f1(row: Mapping[str, Any]) -> float:
    metric = _metric_or_none(row.get('test_f1'))
    return float('-inf') if metric is None else metric


def _experiment_family_summary_lines() -> list[str]:
    lines = ['- Expanded row coverage through `12_distance_profile_nearest_normal`:']
    lines.extend(
        f"  - `{item['key']}`: {item['role']}; {item['note']}" for item in REPORTING_EXPERIMENT_FAMILY_NOTES
    )
    return lines


def _log_records_to_mlflow(config: HarnessConfig, records: Sequence[ExperimentRecord]) -> list[str]:
    if not config.log_mlflow:
        return []

    mlflow = import_module('mlflow')
    mlflow.set_tracking_uri(_mlflow_tracking_uri(config))
    mlflow.set_experiment(config.mlflow_experiment_name or DEFAULT_MLFLOW_EXPERIMENT_NAME)

    nested = _active_mlflow_run(mlflow) is not None
    seen_run_names: set[str] = set()
    logged_run_names: list[str] = []
    for record in records:
        if record.status != 'completed':
            continue
        run_name = _unique_mlflow_run_name(record.key, seen_run_names)
        with mlflow.start_run(run_name=run_name, nested=nested):
            tags = _mlflow_record_tags(record)
            if tags:
                mlflow.set_tags(tags)

            params = _mlflow_record_params(record, config)
            if params:
                mlflow.log_params(params)

            metrics = _mlflow_record_metrics(record)
            if metrics:
                mlflow.log_metrics(metrics)

            if record.output_dir.exists():
                mlflow.log_artifacts(str(record.output_dir), artifact_path=record.key)
        logged_run_names.append(run_name)
    return logged_run_names


def _mlflow_tracking_uri(config: HarnessConfig) -> str:
    if config.mlflow_tracking_uri:
        return config.mlflow_tracking_uri
    mlruns_dir = (config.output_dir / 'mlruns').resolve()
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    return mlruns_dir.as_uri()


def _active_mlflow_run(mlflow: Any) -> Any | None:
    active_run = getattr(mlflow, 'active_run', None)
    return active_run() if callable(active_run) else None


def _unique_mlflow_run_name(base_name: str, seen_run_names: set[str]) -> str:
    run_name = base_name
    if run_name not in seen_run_names:
        seen_run_names.add(run_name)
        return run_name
    suffix = 2
    while f'{base_name}_{suffix}' in seen_run_names:
        suffix += 1
    run_name = f'{base_name}_{suffix}'
    seen_run_names.add(run_name)
    return run_name


def _mlflow_record_tags(record: ExperimentRecord) -> dict[str, str]:
    tags = {
        'dataset': 'skab',
        'protocol': record.split_protocol,
        'experiment_key': record.key,
        'model_family': record.model_family,
        'feature_mode': record.feature_mode,
        'split_protocol': record.split_protocol,
        'split_strategy': record.split_protocol,
        'status': record.status,
    }
    metadata = record.metadata or {}
    for key in ('threshold_method', 'calibration_split', 'score_orientation'):
        value = _simple_tag_value(metadata.get(key))
        if value is not None:
            tags[key] = value
    return tags


def _mlflow_record_params(record: ExperimentRecord, config: HarnessConfig) -> dict[str, str | int | float | bool]:
    raw_params: dict[str, Any] = {
        name: value
        for name, value in asdict(config).items()
        if name not in {'output_dir', 'log_mlflow', 'mlflow_tracking_uri', 'mlflow_experiment_name'}
    }
    if record.params:
        raw_params.update(record.params)

    metadata_params = (record.metadata or {}).get('params')
    if isinstance(metadata_params, Mapping):
        for name, value in metadata_params.items():
            raw_params.setdefault(str(name), value)

    for metric_key in (
        'threshold',
        't2_threshold',
        'q_threshold',
        'conformal_alpha',
        'dbscan_fallback_used',
    ):
        if metric_key in record.metrics:
            raw_params.setdefault(metric_key, record.metrics[metric_key])

    params: dict[str, str | int | float | bool] = {}
    for name, value in raw_params.items():
        normalized = _simple_param_value(value)
        if normalized is not None:
            params[str(name)] = normalized
    return params


def _mlflow_record_metrics(record: ExperimentRecord) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, value in record.metrics.items():
        metric = _finite_metric_value(value)
        if metric is not None:
            metrics[str(name)] = metric
    return metrics


def _simple_tag_value(value: Any) -> str | None:
    normalized = _simple_param_value(value)
    if normalized is None:
        return None
    return str(normalized)


def _simple_param_value(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return None


def _finite_metric_value(value: Any) -> float | None:
    if isinstance(value, np.integer | np.floating):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    metric = float(value)
    return metric if math.isfinite(metric) else None


def _build_ranked_summary_rows(records: Sequence[ExperimentRecord]) -> list[dict[str, Any]]:
    rows = [_summary_row(record) for record in records]
    rows.sort(key=lambda row: (_sort_metric(row.get('test_f1')), row['experiment_key']), reverse=True)
    for index, row in enumerate(rows, start=1):
        row['rank'] = index if row['status'] == 'completed' and row.get('test_f1') is not None else ''
    return rows


def _summary_row(record: ExperimentRecord) -> dict[str, Any]:
    metrics = record.metrics
    metadata = record.metadata or {}
    return {
        'rank': '',
        'experiment_key': record.key,
        'display_name': record.display_name,
        'model_family': record.model_family,
        'feature_mode': record.feature_mode,
        'split_protocol': record.split_protocol,
        'status': record.status,
        'threshold_method': metadata.get('threshold_method'),
        'calibration_split': metadata.get('calibration_split'),
        'train_label_policy': metadata.get('train_label_policy'),
        'dependency_status': metadata.get('dependency_status'),
        'validation_f1': _metric_or_none(metrics.get('f1')),
        'test_f1': _metric_or_none(metrics.get('test_f1')),
        'test_precision': _metric_or_none(metrics.get('test_precision')),
        'test_recall': _metric_or_none(metrics.get('test_recall')),
        'test_false_alarm_rate': _metric_or_none(metrics.get('test_false_alarm_rate')),
        'test_pr_auc': _metric_or_none(metrics.get('test_pr_auc')),
        'test_roc_auc': _metric_or_none(metrics.get('test_roc_auc')),
        'output_dir': str(record.output_dir),
        'notes': record.notes,
    }


def _sort_metric(value: Any) -> float:
    metric = _metric_or_none(value)
    return float(metric) if metric is not None else float('-inf')


def _metric_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    metric = float(value)
    return metric if math.isfinite(metric) else None


def _write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        'rank',
        'experiment_key',
        'display_name',
        'model_family',
        'feature_mode',
        'split_protocol',
        'status',
        'threshold_method',
        'calibration_split',
        'train_label_policy',
        'dependency_status',
        'validation_f1',
        'test_f1',
        'test_precision',
        'test_recall',
        'test_false_alarm_rate',
        'test_pr_auc',
        'test_roc_auc',
        'output_dir',
        'notes',
    ]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def _write_summary_md(path: Path, config: HarnessConfig, rows: Sequence[dict[str, Any]]) -> None:
    columns = [
        'rank',
        'experiment_key',
        'model_family',
        'feature_mode',
        'status',
        'threshold_method',
        'calibration_split',
        'test_f1',
        'test_precision',
        'test_recall',
        'notes',
    ]
    lines = [
        '# SKAB Model Experiment Summary',
        '',
        f'- Manifest: `{config.manifest_path}`',
        f'- Split protocol: `{SPLIT_PROTOCOL}`.',
        (
            '- Honest protocol: train-normal fitting, validation-normal calibration, held-out test metrics, '
            'no point-adjustment.'
        ),
        '- Ranking: completed rows sorted by `test_f1` descending; skipped rows are unranked.',
        f'- Deployment framing: {_deployment_preference_note(rows)}',
        _mlflow_summary_line(config),
        f'- Warning: {SUPERVISED_LABEL_WARNING}',
        *[
            f'- Caution: {caveat}'
            for caveat in REPORTING_CAVEATS
            if caveat != SUPERVISED_LABEL_WARNING
        ],
        *_experiment_family_summary_lines(),
        '',
        _markdown_table(columns, rows),
        '',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')


def _markdown_table(columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> str:
    header = '| ' + ' | '.join(columns) + ' |'
    separator = '| ' + ' | '.join('---' for _ in columns) + ' |'
    body = []
    for row in rows:
        body.append('| ' + ' | '.join(_markdown_value(row.get(column)) for column in columns) + ' |')
    return '\n'.join([header, separator, *body])


def _markdown_value(value: Any) -> str:
    text = _csv_value(value)
    return str(text).replace('|', '\\|')


def _write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(metrics))
        writer.writeheader()
        writer.writerow({name: _csv_value(value) for name, value in metrics.items()})


def _csv_value(value: Any) -> str | int | float:
    jsonable = _jsonable(value)
    if jsonable is None:
        return ''
    if isinstance(jsonable, bool | int | float | str):
        return jsonable
    return json.dumps(jsonable, sort_keys=True)


def _write_experiment_metadata(output_dir: Path, payload: dict[str, Any]) -> None:
    _write_json(output_dir / 'experiment_metadata.json', payload)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'expected JSON object in {path}')
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _dump_joblib(value: Any, path: Path) -> None:
    from joblib import dump

    dump(value, path)


def _record_payload(record: ExperimentRecord) -> dict[str, Any]:
    return {
        'key': record.key,
        'display_name': record.display_name,
        'model_family': record.model_family,
        'feature_mode': record.feature_mode,
        'split_protocol': record.split_protocol,
        'status': record.status,
        'output_dir': str(record.output_dir),
        'metrics': record.metrics,
        'artifact_paths': {name: str(path) for name, path in record.artifact_paths.items()},
        'notes': record.notes,
        'metadata': record.metadata,
        'params': record.params,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _jsonable(float(value))
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


if __name__ == '__main__':
    main()
