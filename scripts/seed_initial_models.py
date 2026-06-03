from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.training.train_pca import PcaTrainingConfig, train_pca_from_skab  # noqa: E402


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
        registered_model_name='PumpAD',
        alias='champion',
    )
    result = train_pca_from_skab(config)
    print(json.dumps(_result_payload(result), sort_keys=True))
    return result


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
