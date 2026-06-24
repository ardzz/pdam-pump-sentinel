from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import import_module
from numbers import Real
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from routemq.job import Job  # type: ignore[reportMissingImports]
from routemq.redis_manager import redis_manager  # type: ignore[reportMissingImports]

from app.observability.annotations import post_annotation
from app.observability.metrics import ACTIVE_MODEL_AGE, RETRAIN_DURATION, RETRAINING_JOBS, set_model_info
from app.services.inference import MODEL_DIR_ENV, set_inference_service
from ml.datasets.live_windows import LiveWindowExtractionConfig, load_live_window_datasets
from ml.inference.loader import load_inference_service_from_artifacts
from ml.inference.pca_inference import PcaAnomalyInferenceService as PcaAnomalyInferenceService
from ml.monitoring.champion_challenger import should_promote
from ml.training.train_lstm_ae import (
    LstmAeTrainingConfig,
    train_lstm_ae_from_live_windows,
    train_lstm_ae_from_skab,
)
from ml.training.train_pca import PcaTrainingConfig, train_pca_from_live_windows, train_pca_from_skab
from ml.training.train_supervised import (
    SupervisedTrainingConfig,
    supervised_skip_result,
    train_supervised_from_live_windows,
)

logger = logging.getLogger(__name__)

ACTIVE_MODEL_KEY = 'pumpad:active:model'
RETRAIN_RESULT_KEY = 'pumpad:retrain:result'
DEFAULT_REGISTERED_MODEL_NAME = 'PumpAD'
SUPERVISED_MODEL_FAMILIES = {'xgboost', 'lightgbm'}


@Job.register
class RetrainingJob(Job):
    queue = 'mlops'

    async def handle(self) -> None:
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        data_source = _data_source()
        model_family = _model_family()
        try:
            config = _training_config(model_family)
            result = _train_challenger(config, data_source, model_family)
            champion_metrics = _read_champion_metrics()
            challenger_metrics = _numeric_metrics(result.metrics)
            deployment_eligible = _deployment_eligible(result, model_family)
            promoted, reason = _promotion_decision(champion_metrics, challenger_metrics, deployment_eligible, result)
            output_dir = Path(result.output_dir)
            metrics_payload = _jsonable_mapping(result.metrics)
            mlflow_version = None

            if promoted:
                hot_swapped = _hot_swap(output_dir)
                mlflow_version = _promote_mlflow_champion_alias(config.registered_model_name)
                active_model = _active_model_payload(
                    registered_model_name=config.registered_model_name,
                    alias='champion',
                    model_dir=output_dir,
                    metrics=metrics_payload,
                    reason=reason,
                    hot_swapped=hot_swapped,
                    mlflow_version=mlflow_version,
                    run_id=getattr(result, 'run_id', None),
                )
                await _write_redis_json(ACTIVE_MODEL_KEY, active_model)
                set_model_info(active_model)
                ACTIVE_MODEL_AGE.labels(
                    name=active_model['name'],
                    version=active_model['version'],
                    alias=active_model['alias'],
                ).set(0.0)

            result_label = 'skipped' if bool(getattr(result, 'skipped', False)) else 'promoted' if promoted else 'rejected'
            version = str(mlflow_version or 'challenger')
            finished_at = datetime.now(timezone.utc)
            duration_seconds = time.perf_counter() - started
            RETRAINING_JOBS.labels(result=result_label).inc()
            RETRAIN_DURATION.labels(result=result_label).observe(duration_seconds)
            post_annotation(text=f'Retraining v{version} {result_label}', tags=['retraining', 'model'])
            if promoted and mlflow_version is not None:
                post_annotation(text=f'Champion → v{version}', tags=['model-promotion', 'champion'])

            logger.info('pumpad retraining completed: promoted=%s reason=%s', promoted, reason)
            await _write_redis_json(
                RETRAIN_RESULT_KEY,
                {
                    'started_at': started_at.isoformat(),
                    'finished_at': finished_at.isoformat(),
                    'duration_seconds': duration_seconds,
                    'success': True,
                    'skipped': bool(getattr(result, 'skipped', False)),
                    'deployment_eligible': deployment_eligible,
                    'promoted': promoted,
                    'reason': reason,
                    'version': version,
                    'run_id': None if getattr(result, 'run_id', None) is None else str(getattr(result, 'run_id')),
                    'metrics': metrics_payload,
                    'data_source': data_source,
                    'model_family': model_family,
                    'provenance': _result_provenance(result),
                },
            )
        except Exception as exc:
            finished_at = datetime.now(timezone.utc)
            duration_seconds = time.perf_counter() - started
            RETRAINING_JOBS.labels(result='error').inc()
            RETRAIN_DURATION.labels(result='error').observe(duration_seconds)
            post_annotation(text='Retraining vunknown error', tags=['retraining', 'model'])
            await _write_redis_json(
                RETRAIN_RESULT_KEY,
                {
                    'started_at': started_at.isoformat(),
                    'finished_at': finished_at.isoformat(),
                    'duration_seconds': duration_seconds,
                    'success': False,
                    'promoted': False,
                    'reason': 'error',
                    'version': 'unknown',
                    'run_id': None,
                    'error': str(exc),
                    'metrics': {},
                    'data_source': data_source,
                    'model_family': model_family,
                    'provenance': None,
                },
            )
            raise


def _data_source() -> str:
    return os.getenv('PUMPAD_RETRAIN_DATA_SOURCE', 'skab').strip().lower() or 'skab'


def _model_family() -> str:
    return os.getenv('PUMPAD_RETRAIN_MODEL_FAMILY', 'pca').strip().lower() or 'pca'


def _train_challenger(
    config: PcaTrainingConfig | LstmAeTrainingConfig | SupervisedTrainingConfig,
    data_source: str,
    model_family: str,
) -> Any:
    if data_source == 'skab':
        return _train_from_skab(config, model_family)
    if data_source == 'live_clickhouse':
        live_bundle = load_live_window_datasets(_clickhouse_training_client(), _live_extraction_config(config))
        return _train_from_live_windows(config, live_bundle, model_family)
    raise ValueError('PUMPAD_RETRAIN_DATA_SOURCE must be one of: skab, live_clickhouse')


def _train_from_skab(config: Any, model_family: str) -> Any:
    if model_family in SUPERVISED_MODEL_FAMILIES:
        return supervised_skip_result(
            config,
            'supervised_skab_scheduled_retraining_not_supported',
            data_source='skab',
            provenance=None,
            label_evidence=None,
        )
    if model_family == 'lstm_ae':
        return train_lstm_ae_from_skab(config)
    return train_pca_from_skab(config)


def _train_from_live_windows(config: Any, live_bundle: Any, model_family: str) -> Any:
    if model_family in SUPERVISED_MODEL_FAMILIES:
        return train_supervised_from_live_windows(config, live_bundle)
    if model_family == 'lstm_ae':
        return train_lstm_ae_from_live_windows(config, live_bundle)
    return train_pca_from_live_windows(config, live_bundle)


def _live_extraction_config(
    training_config: PcaTrainingConfig | LstmAeTrainingConfig | SupervisedTrainingConfig,
) -> LiveWindowExtractionConfig:
    start = os.getenv('PUMPAD_LIVE_START')
    end = os.getenv('PUMPAD_LIVE_END')
    if not start or not end:
        raise ValueError('PUMPAD_LIVE_START and PUMPAD_LIVE_END are required for live_clickhouse retraining')

    station = os.getenv('PUMPAD_LIVE_STATION') or None
    device_ids = _env_csv('PUMPAD_LIVE_DEVICE_IDS')
    if not device_ids and station:
        device_ids = (station,)

    return LiveWindowExtractionConfig(
        start=start,
        end=end,
        station=station,
        device_ids=device_ids,
        window_size=training_config.window_size,
        stride=training_config.stride,
        validation_fraction=_env_float('PUMPAD_LIVE_VALIDATION_FRACTION', 0.2),
        test_fraction=_env_float('PUMPAD_LIVE_TEST_FRACTION', 0.0),
        min_train_windows=_env_int('PUMPAD_LIVE_MIN_TRAIN_WINDOWS', 1),
        include_labels=True if _is_supervised_config(training_config) else _env_bool('PUMPAD_LIVE_INCLUDE_LABELS', False),
    )


def _clickhouse_training_client() -> Any:
    import clickhouse_connect  # type: ignore[reportMissingImports]

    telemetry_url = os.getenv('TELEMETRY_URL')
    if not telemetry_url:
        raise ValueError('TELEMETRY_URL is not set')
    parsed = urlparse(telemetry_url)
    if not parsed.hostname:
        raise ValueError('TELEMETRY_URL host is not set')
    secure = parsed.scheme == 'https'
    return clickhouse_connect.get_client(
        host=parsed.hostname,
        port=parsed.port or (8443 if secure else 8123),
        username=unquote(parsed.username) if parsed.username else None,
        password=unquote(parsed.password) if parsed.password else '',
        database=parsed.path.strip('/') or '__default__',
        interface=parsed.scheme or 'http',
        secure=secure,
    )


def _training_config(model_family: str) -> PcaTrainingConfig | LstmAeTrainingConfig | SupervisedTrainingConfig:
    if model_family == 'pca':
        return _pca_training_config()
    if model_family == 'lstm_ae':
        return _lstm_ae_training_config()
    if model_family in SUPERVISED_MODEL_FAMILIES:
        return _supervised_training_config(model_family)
    raise ValueError('PUMPAD_RETRAIN_MODEL_FAMILY must be one of: pca, lstm_ae, xgboost, lightgbm')


def _retrain_dir() -> Path:
    return Path(
        os.getenv(
            'PUMPAD_RETRAIN_DIR',
            str(Path(tempfile.gettempdir()) / 'pumpad-retraining' / 'challenger'),
        )
    )


def _pca_training_config() -> PcaTrainingConfig:
    return PcaTrainingConfig(
        input_path=_env_path('PUMPAD_SKAB_INPUT_PATH', 'tests/fixtures/skab_tiny.csv'),
        output_dir=_retrain_dir(),
        validation_input_path=_env_optional_path('PUMPAD_SKAB_VALIDATION_PATH'),
        split_manifest_path=_env_optional_path('PUMPAD_SKAB_SPLIT_MANIFEST_PATH', 'PUMPAD_SPLIT_MANIFEST_PATH'),
        window_size=_env_int('PUMPAD_WINDOW_SIZE', 60),
        stride=_env_int('PUMPAD_STRIDE', 1),
        n_components=_env_n_components(),
        threshold_quantile=_env_float('PUMPAD_THRESHOLD_QUANTILE', 0.95),
        scaler=_env_scaler(),
        log_mlflow=True,
        register_model=True,
        registered_model_name=os.getenv('PUMPAD_REGISTERED_MODEL_NAME', DEFAULT_REGISTERED_MODEL_NAME),
        alias='challenger',
    )


def _lstm_ae_training_config() -> LstmAeTrainingConfig:
    return LstmAeTrainingConfig(
        input_path=_env_path('PUMPAD_SKAB_INPUT_PATH', 'tests/fixtures/skab_tiny.csv'),
        output_dir=_retrain_dir(),
        validation_input_path=_env_optional_path('PUMPAD_SKAB_VALIDATION_PATH'),
        split_manifest_path=_env_optional_path('PUMPAD_SKAB_SPLIT_MANIFEST_PATH', 'PUMPAD_SPLIT_MANIFEST_PATH'),
        window_size=_env_int('PUMPAD_WINDOW_SIZE', 60),
        stride=_env_int('PUMPAD_STRIDE', 1),
        threshold_quantile=_env_float('PUMPAD_THRESHOLD_QUANTILE', 0.99),
        log_mlflow=True,
        register_model=True,
        registered_model_name=os.getenv('PUMPAD_REGISTERED_MODEL_NAME', DEFAULT_REGISTERED_MODEL_NAME),
        alias='challenger',
        lstm_units=_env_int('PUMPAD_LSTM_UNITS', 64),
        latent_dim=_env_int('PUMPAD_LSTM_LATENT_DIM', 16),
        epochs=_env_int('PUMPAD_LSTM_EPOCHS', 100),
        batch_size=_env_int('PUMPAD_LSTM_BATCH_SIZE', 32),
        learning_rate=_env_float('PUMPAD_LSTM_LEARNING_RATE', 1e-3),
        patience=_env_int('PUMPAD_LSTM_PATIENCE', 10),
        seed=_env_int('PUMPAD_LSTM_SEED', 42),
    )


def _supervised_training_config(model_family: str) -> SupervisedTrainingConfig:
    return SupervisedTrainingConfig(
        input_path=_env_path('PUMPAD_SKAB_INPUT_PATH', 'tests/fixtures/skab_tiny.csv'),
        output_dir=_retrain_dir(),
        validation_input_path=_env_optional_path('PUMPAD_SKAB_VALIDATION_PATH'),
        split_manifest_path=_env_optional_path('PUMPAD_SKAB_SPLIT_MANIFEST_PATH', 'PUMPAD_SPLIT_MANIFEST_PATH'),
        window_size=_env_int('PUMPAD_WINDOW_SIZE', 60),
        stride=_env_int('PUMPAD_STRIDE', 1),
        threshold_quantile=None,
        scaler=_env_scaler(),
        feature_mode='enriched',
        model_type=model_family,
        n_estimators=_env_int('PUMPAD_SUPERVISED_N_ESTIMATORS', 300),
        learning_rate=_env_float('PUMPAD_SUPERVISED_LEARNING_RATE', 0.05),
        early_stopping_rounds=_env_int('PUMPAD_SUPERVISED_EARLY_STOPPING_ROUNDS', 30),
        seed=_env_int('PUMPAD_SUPERVISED_SEED', 42),
        log_mlflow=True,
        register_model=False,
        registered_model_name=os.getenv('PUMPAD_REGISTERED_MODEL_NAME', DEFAULT_REGISTERED_MODEL_NAME),
        alias=None,
    )


def _is_supervised_config(training_config: Any) -> bool:
    return isinstance(training_config, SupervisedTrainingConfig)


def _deployment_eligible(result: Any, model_family: str) -> bool:
    if model_family in SUPERVISED_MODEL_FAMILIES:
        return False
    return bool(getattr(result, 'deployment_eligible', True))


def _promotion_decision(
    champion_metrics: Mapping[str, float | None] | None,
    challenger_metrics: Mapping[str, float | None],
    deployment_eligible: bool,
    result: Any,
) -> tuple[bool, str]:
    if bool(getattr(result, 'skipped', False)):
        return False, str(getattr(result, 'skip_reason', None) or 'skipped')
    if not deployment_eligible:
        return False, 'candidate is not deployment eligible'
    return should_promote(champion_metrics or {}, challenger_metrics)


def _read_champion_metrics() -> dict[str, float | None] | None:
    model_dir = os.getenv(MODEL_DIR_ENV)
    if not model_dir:
        return None

    metadata_path = Path(model_dir) / 'metadata.json'
    if not metadata_path.exists():
        return None

    try:
        payload = json.loads(metadata_path.read_text(encoding='utf-8'))
    except Exception:
        logger.warning('could not read champion metadata from %s', metadata_path, exc_info=True)
        return None

    metrics = payload.get('metrics') if isinstance(payload, Mapping) else None
    if not isinstance(metrics, Mapping):
        return None
    return _numeric_metrics(metrics)


def _hot_swap(output_dir: Path) -> bool:
    try:
        service = load_inference_service_from_artifacts(output_dir)
        set_inference_service(service)
    except Exception:
        logger.warning('could not hot-swap inference service from %s', output_dir, exc_info=True)
        return False
    return True


def _promote_mlflow_champion_alias(model_name: str = DEFAULT_REGISTERED_MODEL_NAME) -> str | None:
    try:
        mlflow = import_module('mlflow')
        mlflow_tracking = import_module('mlflow.tracking')
    except ImportError:
        logger.info('mlflow unavailable; skipping champion alias promotion')
        return None

    try:
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        client = mlflow_tracking.MlflowClient()
        challenger = client.get_model_version_by_alias(model_name, 'challenger')
        version = getattr(challenger, 'version', None)
        if version is None:
            return None
        version_text = str(version)
        client.set_registered_model_alias(model_name, 'champion', version_text)
        return version_text
    except Exception:
        logger.warning('could not promote mlflow champion alias for %s', model_name, exc_info=True)
        return None


def _active_model_payload(
    *,
    registered_model_name: str,
    alias: str,
    model_dir: Path,
    metrics: Mapping[str, Any],
    reason: str,
    hot_swapped: bool,
    mlflow_version: str | None,
    run_id: Any = None,
) -> dict[str, Any]:
    return {
        'registered_model_name': registered_model_name,
        'alias': alias,
        'model_dir': str(model_dir),
        'metrics': metrics,
        'reason': reason,
        'hot_swapped': hot_swapped,
        'mlflow_version': mlflow_version,
        'run_id': None if run_id is None else str(run_id),
        'name': registered_model_name,
        'version': _active_model_version(mlflow_version, alias),
        'activated_at': datetime.now(timezone.utc).isoformat(),
    }


def _active_model_version(mlflow_version: str | None, alias: str) -> str:
    return str(mlflow_version) if mlflow_version is not None else f'{alias} (local)'


async def _write_redis_json(key: str, value: Mapping[str, Any]) -> None:
    try:
        if redis_manager.is_enabled():
            await redis_manager.set_json(key, _jsonable(value))
    except Exception:
        logger.warning('could not write redis key %s', key, exc_info=True)


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


def _env_optional_path(*names: str) -> Path | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return Path(value)
    return None


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == '':
        return default
    normalized = raw.strip().lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise ValueError(f'{name} must be a boolean')


def _env_csv(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, '')
    return tuple(value.strip() for value in raw.split(',') if value.strip())


def _env_n_components() -> int | float | None:
    raw = os.getenv('PUMPAD_N_COMPONENTS', '0.9')
    if raw.lower() == 'none':
        return None
    value = float(raw)
    if value.is_integer() and not any(marker in raw.lower() for marker in ('.', 'e')):
        return int(value)
    return value


def _env_scaler() -> str | None:
    raw = os.getenv('PUMPAD_SCALER', 'robust')
    if raw.lower() == 'none':
        return None
    return raw


def _numeric_metrics(metrics: Mapping[str, Any]) -> dict[str, float | None]:
    payload: dict[str, float | None] = {}
    for name, value in metrics.items():
        if value is None:
            payload[str(name)] = None
        elif not isinstance(value, bool) and isinstance(value, Real):
            payload[str(name)] = float(value)
    return payload


def _result_provenance(result: Any) -> Any:
    provenance = getattr(result, 'provenance', None)
    if provenance is not None:
        return _jsonable(provenance)
    params = getattr(result, 'params', None)
    if isinstance(params, Mapping) and 'provenance' in params:
        return _jsonable(params['provenance'])
    metrics = getattr(result, 'metrics', None)
    if isinstance(metrics, Mapping) and 'provenance' in metrics:
        return _jsonable(metrics['provenance'])
    return None


def _jsonable_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    payload = _jsonable(values)
    return payload if isinstance(payload, dict) else {}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_jsonable(item) for item in value]
    return str(value)
