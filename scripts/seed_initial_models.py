from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from routemq.redis_manager import redis_manager  # type: ignore[reportMissingImports]  # noqa: E402

from ml.training.train_pca import PcaTrainingConfig, train_pca_from_skab  # noqa: E402

ACTIVE_MODEL_KEY = 'pumpad:active:model'
DEFAULT_REGISTERED_MODEL_NAME = 'PumpAD'
DEFAULT_ALIAS = 'champion'


def main(argv: Sequence[str] | None = None) -> object:
    parser = argparse.ArgumentParser(description='Train and register the initial PumpAD champion model.')
    parser.add_argument('--input', type=Path, default=None)
    parser.add_argument('--validation-input-path', type=Path, default=None)
    parser.add_argument('--split-manifest', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--window-size', type=int, default=60)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--n-components', type=_parse_n_components, default=0.9)
    parser.add_argument('--threshold-quantile', type=float, default=0.95)
    parser.add_argument('--scaler', type=_parse_scaler, default='robust')
    args = parser.parse_args(argv)

    if args.input is None and args.split_manifest is None:
        parser.error('set --input or --split-manifest')

    config = PcaTrainingConfig(
        input_path=args.input or args.output_dir,
        output_dir=args.output_dir,
        validation_input_path=args.validation_input_path,
        split_manifest_path=args.split_manifest,
        window_size=args.window_size,
        stride=args.stride,
        n_components=args.n_components,
        threshold_quantile=args.threshold_quantile,
        scaler=args.scaler,
        log_mlflow=True,
        register_model=True,
        registered_model_name=DEFAULT_REGISTERED_MODEL_NAME,
        alias=DEFAULT_ALIAS,
    )
    result = train_pca_from_skab(config)
    _write_active_model(result, config)
    print(json.dumps(_result_payload(result), sort_keys=True))
    return result


def _write_active_model(result: object, config: PcaTrainingConfig) -> None:
    alias = config.alias or DEFAULT_ALIAS
    mlflow_version = _resolve_mlflow_alias_version(config.registered_model_name, alias)
    asyncio.run(_write_redis_json(ACTIVE_MODEL_KEY, _active_model_payload(result, config, alias, mlflow_version)))


def _active_model_payload(
    result: object,
    config: PcaTrainingConfig,
    alias: str,
    mlflow_version: str | None,
) -> dict[str, object]:
    metrics = _jsonable(getattr(result, 'metrics', {}))
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        'registered_model_name': config.registered_model_name,
        'alias': alias,
        'model_dir': str(getattr(result, 'output_dir', config.output_dir)),
        'metrics': metrics,
        'reason': 'initial champion seed',
        'hot_swapped': False,
        'mlflow_version': mlflow_version,
        'name': config.registered_model_name,
        'version': _active_model_version(mlflow_version, alias),
        'activated_at': datetime.now(timezone.utc).isoformat(),
    }


def _active_model_version(mlflow_version: str | None, alias: str) -> str:
    return str(mlflow_version) if mlflow_version is not None else f'{alias} (local)'


async def _write_redis_json(key: str, value: Mapping[str, object]) -> None:
    try:
        if redis_manager.is_enabled():
            await redis_manager.set_json(key, _jsonable(value))
    except Exception:
        return


def _resolve_mlflow_alias_version(model_name: str, alias: str) -> str | None:
    try:
        mlflow = import_module('mlflow')
        mlflow_tracking = import_module('mlflow.tracking')
    except ImportError:
        return None

    try:
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        version = mlflow_tracking.MlflowClient().get_model_version_by_alias(model_name, alias)
        value = getattr(version, 'version', None)
        return None if value is None else str(value)
    except Exception:
        return None


def _parse_n_components(value: str) -> int | float | None:
    if value.lower() == 'none':
        return None
    parsed_float = float(value)
    if parsed_float.is_integer() and not any(marker in value.lower() for marker in ('.', 'e')):
        return int(parsed_float)
    return parsed_float


def _parse_scaler(value: str) -> str | None:
    if value.lower() == 'none':
        return None
    return value


def _result_payload(result: object) -> dict[str, object]:
    return {
        name: _jsonable(getattr(result, name))
        for name in ('output_dir', 'artifact_paths', 'metrics', 'thresholds', 'sensor_columns')
        if hasattr(result, name)
    }


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if hasattr(value, '__dict__'):
        return _jsonable(vars(value))
    return str(value)


if __name__ == '__main__':
    main()
