from collections.abc import Sequence
from pathlib import Path

import pandas as pd

SENSOR_COLUMNS = [
    'Accelerometer1RMS',
    'Accelerometer2RMS',
    'Current',
    'Pressure',
    'Temperature',
    'Thermocouple',
    'Voltage',
    'Volume Flow RateRMS',
]


def load_skab_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=';')
    if 'datetime' in frame.columns:
        frame['datetime'] = pd.to_datetime(frame['datetime'], utc=True)
        missing_datetimes = int(frame['datetime'].isna().sum())
        if missing_datetimes:
            raise ValueError('datetime column contains missing values after parsing')
        frame = frame.drop_duplicates(subset='datetime', keep='first')
        if not frame['datetime'].is_monotonic_increasing:
            raise ValueError(
                'datetime values must be monotonic increasing after duplicate removal'
            )
        frame = frame.reset_index(drop=True)
    for col in ('anomaly', 'changepoint'):
        if col not in frame.columns:
            frame[col] = 0
    return frame


def validate_sensor_values(
    frame: pd.DataFrame,
    sensor_columns: Sequence[str] = SENSOR_COLUMNS,
    allow_missing: bool = False,
):
    ordered_sensor_columns = tuple(sensor_columns)
    missing_columns = [col for col in ordered_sensor_columns if col not in frame.columns]
    if missing_columns:
        raise ValueError(f'missing required sensor columns: {", ".join(missing_columns)}')

    missing_counts = frame[list(ordered_sensor_columns)].isna().sum()
    row_count = len(frame)
    summary = {
        'missing_counts': {
            col: int(missing_counts[col]) for col in ordered_sensor_columns
        },
        'missing_rates': {
            col: (int(missing_counts[col]) / row_count if row_count else 0.0)
            for col in ordered_sensor_columns
        },
    }
    invalid_counts = {
        col: count for col, count in summary['missing_counts'].items() if count
    }
    if invalid_counts and not allow_missing:
        details = ', '.join(
            f'{col}={count} ({summary["missing_rates"][col]:.2%})'
            for col, count in invalid_counts.items()
        )
        raise ValueError(f'sensor columns contain missing values: {details}')
    return summary


def iter_telemetry_records(frame: pd.DataFrame, station: str):
    for _, row in frame.iterrows():
        record = {
            'station': station,
            'timestamp': _serialize_timestamp(row['datetime']),
            'sensors': {col: float(row[col]) for col in SENSOR_COLUMNS},
            'labels': {
                'anomaly': int(row['anomaly']),
                'changepoint': int(row['changepoint']),
            },
        }
        yield record


def _serialize_timestamp(value) -> str:
    timestamp = pd.Timestamp(value)
    if not isinstance(timestamp, pd.Timestamp):
        raise ValueError('datetime value cannot be missing in telemetry records')
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert('UTC')
    serialized = timestamp.isoformat()
    if serialized.endswith('+00:00'):
        return f'{serialized[:-6]}Z'
    return serialized
