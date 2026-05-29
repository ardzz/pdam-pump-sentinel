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
    for col in ('anomaly', 'changepoint'):
        if col not in frame.columns:
            frame[col] = 0
    return frame


def iter_telemetry_records(frame: pd.DataFrame, station: str):
    for _, row in frame.iterrows():
        record = {
            'station': station,
            'timestamp': str(row['datetime']),
            'sensors': {col: float(row[col]) for col in SENSOR_COLUMNS},
            'labels': {
                'anomaly': int(row['anomaly']),
                'changepoint': int(row['changepoint']),
            },
        }
        yield record
