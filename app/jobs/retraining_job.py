from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from importlib import import_module
from numbers import Real
from pathlib import Path
from typing import Any

from routemq.job import Job  # type: ignore[reportMissingImports]
from routemq.redis_manager import redis_manager  # type: ignore[reportMissingImports]

from app.services.inference import MODEL_DIR_ENV, set_inference_service
from ml.inference.pca_inference import PcaAnomalyInferenceService
from ml.monitoring.champion_challenger import should_promote
from ml.training.train_pca import PcaTrainingConfig, train_pca_from_skab

logger = logging.getLogger(__name__)

ACTIVE_MODEL_KEY = 'pumpad:active:model'
RETRAIN_RESULT_KEY = 'pumpad:retrain:result'
DEFAULT_REGISTERED_MODEL_NAME = 'PumpAD'


@Job.register
class RetrainingJob(Job):
    queue = 'mlops'

    async def handle(self) -> None:
        config = _training_config()
        result = train_pca_from_skab(config)
        champion_metrics = _read_champion_metrics()
        challenger_metrics = _numeric_metrics(result.metrics)
        promoted, reason = should_promote(champion_metrics or {}, challenger_metrics)
        output_dir = Path(result.output_dir)
        metrics_payload = _jsonable_mapping(result.metrics)

        if promoted:
            hot_swapped = _hot_swap(output_dir)
            mlflow_version = _promote_mlflow_champion_alias(config.registered_model_name)
            await _write_redis_json(
                ACTIVE_MODEL_KEY,
                {
                    'registered_model_name': config.registered_model_name,
                    'alias': 'champion',
                    'model_dir': str(output_dir),
                    'metrics': metrics_payload,
                    'reason': reason,
                    'hot_swapped': hot_swapped,
                    'mlflow_version': mlflow_version,
                },
            )

        logger.info('pumpad retraining completed: promoted=%s reason=%s', promoted, reason)
        await _write_redis_json(
            RETRAIN_RESULT_KEY,
            {
                'promoted': promoted,
                'reason': reason,
                'metrics': metrics_payload,
            },
        )


def _training_config() -> PcaTrainingConfig:
    output_dir = Path(
        os.getenv(
            'PUMPAD_RETRAIN_DIR',
            str(Path(tempfile.gettempdir()) / 'pumpad-retraining' / 'challenger'),
        )
    )
    return PcaTrainingConfig(
        input_path=_env_path('PUMPAD_SKAB_INPUT_PATH', 'tests/fixtures/skab_tiny.csv'),
        output_dir=output_dir,
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
        service = PcaAnomalyInferenceService.from_artifacts(output_dir)
        set_inference_service(service)
    except Exception:
        logger.warning('could not hot-swap PCA inference service from %s', output_dir, exc_info=True)
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
