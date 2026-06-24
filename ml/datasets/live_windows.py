from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from ml.datasets.skab_loader import SENSOR_COLUMNS
from ml.features.windowing import WindowedSensorDataset, build_sensor_windows

TELEMETRY_TABLE = 'telemetry_observations'
LABEL_TABLE = 'operator_labels'
ANOMALY_SCORE_MEASUREMENT = 'anomaly_score'


class LiveTelemetryQueryClient(Protocol):
    def query(self, query: str, parameters: Mapping[str, Any]) -> Any: ...


@dataclass(frozen=True)
class LiveWindowExtractionConfig:
    start: str | datetime
    end: str | datetime
    station: str | None = None
    device_ids: tuple[str, ...] = ()
    window_size: int = 60
    stride: int = 1
    validation_fraction: float = 0.2
    test_fraction: float = 0.0
    min_train_windows: int = 1
    include_labels: bool = False
    sensor_columns: tuple[str, ...] = tuple(SENSOR_COLUMNS)
    telemetry_table: str = TELEMETRY_TABLE
    label_table: str = LABEL_TABLE


@dataclass(frozen=True)
class LiveWindowDatasetBundle:
    train: WindowedSensorDataset
    validation: WindowedSensorDataset
    test: WindowedSensorDataset | None
    provenance: dict[str, Any]


def extract_live_sensor_frame(client: LiveTelemetryQueryClient, config: LiveWindowExtractionConfig) -> pd.DataFrame:
    normalized = _normalize_config(config)
    rows = _query_result_dicts(client.query(_telemetry_query(normalized), parameters=_query_parameters(normalized)))
    if not rows:
        raise ValueError('live telemetry query returned no rows')

    frame = _pivot_sensor_rows(rows, normalized)
    if normalized.include_labels:
        frame = _apply_labels(frame, rows)
    return frame


def build_live_window_datasets(frame: pd.DataFrame, config: LiveWindowExtractionConfig) -> LiveWindowDatasetBundle:
    normalized = _normalize_config(config)
    if frame.empty:
        raise ValueError('live telemetry frame must not be empty')

    missing_columns = [column for column in ('datetime', *normalized.sensor_columns, 'anomaly') if column not in frame.columns]
    if missing_columns:
        raise ValueError(f'missing required live frame columns: {", ".join(missing_columns)}')

    ordered = _ordered_complete_frame(frame, normalized)
    original_row_count = len(frame)
    dropped_incomplete = original_row_count - len(ordered)
    if ordered.empty:
        raise ValueError('live telemetry rows are incomplete for all required sensors')

    train_frame, validation_frame, test_frame = _chronological_split_frames(ordered, normalized)
    train = build_sensor_windows(train_frame, normalized.window_size, normalized.stride, normalized.sensor_columns)
    if len(train.labels) < normalized.min_train_windows:
        raise ValueError('live telemetry split does not produce enough training windows')

    validation = (
        build_sensor_windows(validation_frame, normalized.window_size, normalized.stride, normalized.sensor_columns)
        if len(validation_frame) > 0
        else _empty_dataset(normalized)
    )
    test = (
        build_sensor_windows(test_frame, normalized.window_size, normalized.stride, normalized.sensor_columns)
        if test_frame is not None and len(test_frame) > 0
        else None
    )

    provenance = _provenance(
        normalized,
        ordered,
        train=train,
        validation=validation,
        test=test,
        dropped_incomplete=dropped_incomplete,
        raw_row_count=original_row_count,
    )
    return LiveWindowDatasetBundle(train=train, validation=validation, test=test, provenance=provenance)


def load_live_window_datasets(
    client: LiveTelemetryQueryClient,
    config: LiveWindowExtractionConfig,
) -> LiveWindowDatasetBundle:
    frame = extract_live_sensor_frame(client, config)
    return build_live_window_datasets(frame, config)


def _normalize_config(config: LiveWindowExtractionConfig) -> LiveWindowExtractionConfig:
    _validate_positive_integer(config.window_size, 'window_size')
    _validate_positive_integer(config.stride, 'stride')
    _validate_positive_integer(config.min_train_windows, 'min_train_windows')
    _validate_fraction(config.validation_fraction, 'validation_fraction')
    _validate_fraction(config.test_fraction, 'test_fraction')
    if config.validation_fraction + config.test_fraction >= 1.0:
        raise ValueError('validation_fraction + test_fraction must be less than 1')
    if not config.sensor_columns:
        raise ValueError('sensor_columns must not be empty')
    start = _normalize_timestamp(config.start, 'start')
    end = _normalize_timestamp(config.end, 'end')
    if pd.Timestamp(start).value >= pd.Timestamp(end).value:
        raise ValueError('start must be before end')
    device_ids = tuple(device_id.strip() for device_id in config.device_ids if device_id.strip())
    return LiveWindowExtractionConfig(
        start=start,
        end=end,
        station=config.station.strip() if isinstance(config.station, str) and config.station.strip() else None,
        device_ids=device_ids,
        window_size=config.window_size,
        stride=config.stride,
        validation_fraction=float(config.validation_fraction),
        test_fraction=float(config.test_fraction),
        min_train_windows=config.min_train_windows,
        include_labels=bool(config.include_labels),
        sensor_columns=tuple(config.sensor_columns),
        telemetry_table=config.telemetry_table,
        label_table=config.label_table,
    )


def _telemetry_query(config: LiveWindowExtractionConfig) -> str:
    station_filter = "AND tags['station'] = %(station)s" if config.station else ''
    device_filter = 'AND device_id IN %(device_ids)s' if config.device_ids else ''
    return f'''
SELECT observed_at, device_id, measurement, value_float, value_int, tags
FROM {config.telemetry_table}
WHERE observed_at >= %(start)s
  AND observed_at < %(end)s
  AND (measurement IN %(sensor_columns)s OR measurement = %(anomaly_measurement)s)
  {station_filter}
  {device_filter}
ORDER BY observed_at ASC, device_id ASC, measurement ASC
'''.strip()


def _query_parameters(config: LiveWindowExtractionConfig) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        'start': config.start,
        'end': config.end,
        'sensor_columns': tuple(config.sensor_columns),
        'anomaly_measurement': ANOMALY_SCORE_MEASUREMENT,
    }
    if config.station:
        parameters['station'] = config.station
    if config.device_ids:
        parameters['device_ids'] = tuple(config.device_ids)
    return parameters


def _query_result_dicts(result: Any) -> list[dict[str, Any]]:
    named_results = getattr(result, 'named_results', None)
    if callable(named_results):
        return [dict(row) for row in cast(Iterable[Any], named_results())]
    rows = getattr(result, 'result_rows', None)
    if rows is not None:
        column_names = tuple(getattr(result, 'column_names', ())) or (
            'observed_at',
            'device_id',
            'measurement',
            'value_float',
            'value_int',
            'tags',
        )
        return [dict(zip(column_names, row)) for row in cast(Iterable[Iterable[Any]], rows)]
    if isinstance(result, Iterable) and not isinstance(result, str | bytes | Mapping):
        materialized = list(result)
        if all(isinstance(row, Mapping) for row in materialized):
            return [dict(cast(Mapping[str, Any], row)) for row in materialized]
        return [_tuple_row_dict(row) for row in materialized]
    return []


def _tuple_row_dict(row: Any) -> dict[str, Any]:
    values = tuple(row)
    column_names = ('observed_at', 'device_id', 'measurement', 'value_float', 'value_int', 'tags')
    return dict(zip(column_names, values))


def _pivot_sensor_rows(rows: Sequence[Mapping[str, Any]], config: LiveWindowExtractionConfig) -> pd.DataFrame:
    records = [_sensor_record(row, config.sensor_columns) for row in rows if row.get('measurement') in config.sensor_columns]
    if not records:
        raise ValueError('live telemetry query returned no sensor measurement rows')

    long_frame = pd.DataFrame(records)
    pivot = long_frame.pivot_table(
        index=['datetime', 'device_id', 'station'],
        columns='measurement',
        values='value',
        aggfunc='last',
        sort=False,
    )
    pivot = pivot.reindex(columns=list(config.sensor_columns)).reset_index()
    pivot['anomaly'] = 0
    pivot['changepoint'] = 0
    ordered_columns = ['datetime', 'device_id', 'station', *config.sensor_columns, 'anomaly', 'changepoint']
    ordered = cast(pd.DataFrame, pivot[ordered_columns])
    return _sort_by_datetime(ordered)


def _sensor_record(row: Mapping[str, Any], sensor_columns: Sequence[str]) -> dict[str, Any]:
    measurement = str(row.get('measurement'))
    if measurement not in sensor_columns:
        raise ValueError(f'unexpected sensor measurement: {measurement}')
    return {
        'datetime': _timestamp_key(row.get('observed_at')),
        'device_id': '' if row.get('device_id') is None else str(row.get('device_id')),
        'station': _station(row),
        'measurement': measurement,
        'value': _numeric_value(row),
    }


def _sort_by_datetime(frame: pd.DataFrame) -> pd.DataFrame:
    order = np.argsort(pd.to_datetime(frame['datetime'], utc=True).to_numpy())
    return frame.iloc[order].reset_index(drop=True)


def _numeric_value(row: Mapping[str, Any]) -> float | None:
    value = row.get('value_float')
    if value is None or pd.isna(value):
        value = row.get('value_int')
    if value is None or pd.isna(value):
        return None
    return float(value)


def _station(row: Mapping[str, Any]) -> str:
    tags = row.get('tags')
    if isinstance(tags, Mapping) and tags.get('station') is not None:
        return str(tags['station'])
    return ''


def _apply_labels(frame: pd.DataFrame, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    labeled = frame.copy()
    labels = _anomaly_score_labels(rows)
    if labels:
        labeled['anomaly'] = [labels.get(_timestamp_key(value), int(current)) for value, current in zip(labeled['datetime'], labeled['anomaly'], strict=True)]
    return labeled


def _anomaly_score_labels(rows: Sequence[Mapping[str, Any]]) -> dict[pd.Timestamp, int]:
    labels: dict[pd.Timestamp, int] = {}
    for row in rows:
        if row.get('measurement') != ANOMALY_SCORE_MEASUREMENT:
            continue
        value = row.get('value_int')
        observed_at = row.get('observed_at')
        if observed_at is None or value is None or pd.isna(value):
            continue
        labels[_timestamp_key(observed_at)] = int(value)
    return labels


def _ordered_complete_frame(frame: pd.DataFrame, config: LiveWindowExtractionConfig) -> pd.DataFrame:
    ordered = frame.copy()
    ordered['datetime'] = pd.to_datetime(ordered['datetime'], utc=True)
    ordered = _sort_by_datetime(ordered)
    complete_mask = ordered[list(config.sensor_columns)].notna().all(axis=1)
    ordered = ordered.loc[complete_mask].reset_index(drop=True)
    if 'changepoint' not in ordered.columns:
        ordered['changepoint'] = 0
    ordered['anomaly'] = ordered['anomaly'].fillna(0).astype(int)
    ordered['changepoint'] = ordered['changepoint'].fillna(0).astype(int)
    return ordered


def _chronological_split_frames(
    frame: pd.DataFrame,
    config: LiveWindowExtractionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    row_count = len(frame)
    test_rows = int(row_count * config.test_fraction)
    validation_rows = int(row_count * config.validation_fraction)
    train_rows = row_count - validation_rows - test_rows
    _validate_split_rows(train_rows, validation_rows, test_rows, config)

    train_frame = frame.iloc[:train_rows].reset_index(drop=True)
    validation_stop = train_rows + validation_rows
    validation_frame = frame.iloc[train_rows:validation_stop].reset_index(drop=True)
    test_frame = frame.iloc[validation_stop:].reset_index(drop=True) if test_rows else None
    return train_frame, validation_frame, test_frame


def _validate_split_rows(
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    config: LiveWindowExtractionConfig,
) -> None:
    if _window_count(train_rows, config.window_size, config.stride) < config.min_train_windows:
        raise ValueError('live telemetry split does not produce enough training windows')
    if config.validation_fraction > 0 and _window_count(validation_rows, config.window_size, config.stride) == 0:
        raise ValueError('live telemetry validation split does not produce any windows')
    if config.test_fraction > 0 and _window_count(test_rows, config.window_size, config.stride) == 0:
        raise ValueError('live telemetry test split does not produce any windows')


def _window_count(row_count: int, window_size: int, stride: int) -> int:
    if row_count < window_size:
        return 0
    return ((row_count - window_size) // stride) + 1


def _empty_dataset(config: LiveWindowExtractionConfig) -> WindowedSensorDataset:
    return WindowedSensorDataset(
        features=np.empty((0, config.window_size * len(config.sensor_columns)), dtype=float),
        labels=np.empty((0,), dtype=int),
        changepoints=np.empty((0,), dtype=int),
        timestamps=np.empty((0,), dtype=object),
        sensor_columns=tuple(config.sensor_columns),
        window_size=config.window_size,
        stride=config.stride,
    )


def _provenance(
    config: LiveWindowExtractionConfig,
    frame: pd.DataFrame,
    *,
    train: WindowedSensorDataset,
    validation: WindowedSensorDataset,
    test: WindowedSensorDataset | None,
    dropped_incomplete: int,
    raw_row_count: int,
) -> dict[str, Any]:
    return {
        'data_source': 'live_clickhouse',
        'source_table': config.telemetry_table,
        'label_table': config.label_table,
        'station': config.station,
        'device_ids': list(config.device_ids),
        'start': config.start,
        'end': config.end,
        'observed_start': _serialize_timestamp(frame['datetime'].iloc[0]),
        'observed_end': _serialize_timestamp(frame['datetime'].iloc[-1]),
        'sensor_columns': list(config.sensor_columns),
        'window_size': config.window_size,
        'stride': config.stride,
        'validation_fraction': config.validation_fraction,
        'test_fraction': config.test_fraction,
        'split_counts': {
            'train': int(len(train.labels)),
            'validation': int(len(validation.labels)),
            'test': int(len(test.labels)) if test is not None else 0,
        },
        'row_counts': {
            'raw': int(raw_row_count),
            'complete': int(len(frame)),
            'dropped_incomplete': int(dropped_incomplete),
        },
        'dropped_incomplete_row_count': int(dropped_incomplete),
        'include_labels': bool(config.include_labels),
    }


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def _validate_fraction(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= float(value) < 1:
        raise ValueError(f'{name} must be in [0, 1)')


def _normalize_timestamp(value: str | datetime, name: str) -> str:
    try:
        return _serialize_timestamp(_timestamp_key(value))
    except ValueError as exc:
        raise ValueError(f'{name} must be a valid timestamp') from exc


def _timestamp_key(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if str(parsed) == 'NaT':
        raise ValueError('timestamp value must be valid')
    timestamp = cast(pd.Timestamp, parsed)
    if timestamp.tzinfo is None:
        return cast(pd.Timestamp, timestamp.tz_localize(timezone.utc))
    return cast(pd.Timestamp, timestamp.tz_convert('UTC'))


def _serialize_timestamp(value: Any) -> str:
    return _timestamp_key(value).isoformat().replace('+00:00', 'Z')
