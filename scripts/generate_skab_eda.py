import argparse
import importlib
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

EXPECTED_CORE_API = (
    'ml.datasets.skab_eda.generate_skab_eda_report('
    'input_path: Path | None, '
    'split_manifest_path: Path | None, '
    'output_dir: Path, '
    'include_plots: bool = True'
    ') -> Mapping[str, Path | str]'
)

def _path_to_json(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _path_to_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_path_to_json(item) for item in value]
    if isinstance(value, tuple):
        return [_path_to_json(item) for item in value]
    return value


def _artifact_payload(artifacts: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _path_to_json(value) for key, value in artifacts.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate SKAB EDA report artifacts')
    parser.add_argument(
        '--input',
        type=Path,
        default=None,
        help='Path to a SKAB CSV file or SKAB data directory',
    )
    parser.add_argument(
        '--split-manifest',
        type=Path,
        default=None,
        help='Path to split-manifest JSON for train, validation, and test files',
    )
    parser.add_argument('--output-dir', type=Path, required=True, help='Directory for EDA artifacts')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot artifact generation')
    args = parser.parse_args()

    if args.input is None and args.split_manifest is None:
        parser.error('set --input, --split-manifest, or both')

    return args


def _load_core_api() -> Callable[..., Mapping[str, object]]:
    try:
        module = importlib.import_module('ml.datasets.skab_eda')
        return getattr(module, 'generate_skab_eda_report')
    except (ImportError, AttributeError) as exc:  # pragma: no cover, depends on parallel core work
        raise SystemExit(
            'Missing SKAB EDA core API. Expected: '
            f'{EXPECTED_CORE_API}. Original import error: {exc}'
        ) from exc


def main() -> None:
    args = _parse_args()

    generate_skab_eda_report = _load_core_api()
    artifacts = generate_skab_eda_report(
        input_path=args.input,
        split_manifest_path=args.split_manifest,
        output_dir=args.output_dir,
        include_plots=not args.no_plots,
    )
    print(json.dumps(_artifact_payload(artifacts), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
