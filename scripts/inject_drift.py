from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.datasets.skab_loader import SENSOR_COLUMNS  # noqa: E402

SKAB_DELIMITER = ';'


def apply_drift(frame: pd.DataFrame, column: str, delta: float) -> pd.DataFrame:
    if column not in frame.columns:
        raise ValueError(f'column {column!r} is missing from frame')
    drifted = frame.copy()
    drifted[column] = drifted[column] + delta
    return drifted


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description='Inject a mean shift into a SKAB CSV sensor column.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--column', default='Pressure', choices=SENSOR_COLUMNS)
    parser.add_argument('--delta', type=float, required=True)
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.input, sep=SKAB_DELIMITER)
    drifted = apply_drift(frame, args.column, args.delta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    drifted.to_csv(args.output, sep=SKAB_DELIMITER, index=False)
    return args.output


if __name__ == '__main__':
    main()
