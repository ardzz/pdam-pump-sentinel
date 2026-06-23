from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest


class FakeQueryClient:
    def __init__(self, result: Any):
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def query(self, query: str, parameters: dict[str, Any]):
        self.calls.append({'query': query, 'parameters': parameters})
        return self.result


class FakeNamedResult:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def named_results(self):
        return iter(self.rows)


def _live_windows():
    return import_module('ml.datasets.live_windows')


def _sensor_columns():
    return import_module('ml.datasets.skab_loader').SENSOR_COLUMNS


def _row(index: int, measurement: str, value: float, *, device_id: str = 'ipa_01', station: str = 'ipa_01') -> dict[str, Any]:
    return {
        'observed_at': f'2026-06-01T00:00:{index:02d}Z',
        'device_id': device_id,
        'measurement': measurement,
        'value_float': value,
        'value_int': None,
        'tags': {'station': station},
    }


def _long_rows(row_count: int, *, omit: tuple[int, str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for row_index in range(row_count):
        for column_index, sensor in enumerate(_sensor_columns()):
            if omit == (row_index, sensor):
                continue
            rows.append(_row(row_index, sensor, row_index * 100.0 + column_index))
    return rows


def test_extract_live_sensor_frame_pivots_long_rows_chronologically_in_sensor_order():
    live_windows = _live_windows()
    rows = list(reversed(_long_rows(3)))
    fake = FakeQueryClient(FakeNamedResult(rows))
    config = live_windows.LiveWindowExtractionConfig(
        start='2026-06-01T00:00:00Z',
        end='2026-06-01T00:01:00Z',
        station='ipa_01',
        device_ids=('ipa_01',),
        window_size=2,
        stride=1,
    )

    frame = live_windows.extract_live_sensor_frame(fake, config)

    assert 'telemetry_observations' in fake.calls[0]['query']
    assert fake.calls[0]['parameters']['station'] == 'ipa_01'
    assert fake.calls[0]['parameters']['device_ids'] == ('ipa_01',)
    assert list(frame.columns) == ['datetime', 'device_id', 'station', *_sensor_columns(), 'anomaly', 'changepoint']
    assert frame['datetime'].astype(str).tolist() == [
        '2026-06-01 00:00:00+00:00',
        '2026-06-01 00:00:01+00:00',
        '2026-06-01 00:00:02+00:00',
    ]
    assert frame[_sensor_columns()].iloc[1].tolist() == [100.0 + index for index in range(len(_sensor_columns()))]


def test_extract_live_sensor_frame_filters_non_sensor_measurements_and_ignores_labels_by_default():
    live_windows = _live_windows()
    rows = _long_rows(2)
    rows.extend([
        _row(0, 'not_a_sensor', 999.0),
        {**_row(0, 'anomaly_score', 0.99), 'value_int': 1},
    ])
    fake = FakeQueryClient(FakeNamedResult(rows))
    config = live_windows.LiveWindowExtractionConfig(
        start='2026-06-01T00:00:00Z',
        end='2026-06-01T00:01:00Z',
        window_size=2,
        stride=1,
        include_labels=False,
    )

    frame = live_windows.extract_live_sensor_frame(fake, config)

    assert 'not_a_sensor' not in frame.columns
    assert 'anomaly_score' not in frame.columns
    assert frame['anomaly'].tolist() == [0, 0]


def test_extract_live_sensor_frame_can_map_optional_anomaly_score_labels():
    live_windows = _live_windows()
    rows = _long_rows(2)
    rows.append({**_row(1, 'anomaly_score', 0.99), 'value_int': 1})
    fake = FakeQueryClient(FakeNamedResult(rows))
    config = live_windows.LiveWindowExtractionConfig(
        start='2026-06-01T00:00:00Z',
        end='2026-06-01T00:01:00Z',
        window_size=2,
        stride=1,
        include_labels=True,
    )

    frame = live_windows.extract_live_sensor_frame(fake, config)

    assert frame['anomaly'].tolist() == [0, 1]


def test_build_live_window_datasets_drops_incomplete_rows_and_records_provenance():
    live_windows = _live_windows()
    rows = _long_rows(17, omit=(2, _sensor_columns()[0]))
    frame = live_windows.extract_live_sensor_frame(
        FakeQueryClient(FakeNamedResult(rows)),
        live_windows.LiveWindowExtractionConfig(
            start='2026-06-01T00:00:00Z',
            end='2026-06-01T00:01:00Z',
            window_size=2,
            stride=2,
        ),
    )

    bundle = live_windows.build_live_window_datasets(
        frame,
        live_windows.LiveWindowExtractionConfig(
            start='2026-06-01T00:00:00Z',
            end='2026-06-01T00:01:00Z',
            station='ipa_01',
            window_size=2,
            stride=2,
            validation_fraction=0.25,
            test_fraction=0.25,
        ),
    )

    assert bundle.provenance['dropped_incomplete_row_count'] == 1
    assert bundle.provenance['row_counts'] == {'raw': 17, 'complete': 16, 'dropped_incomplete': 1}
    assert bundle.provenance['split_counts'] == {'train': 4, 'validation': 2, 'test': 2}


def test_build_live_window_datasets_rejects_all_incomplete_sensor_rows():
    live_windows = _live_windows()
    rows = [{name: None for name in _sensor_columns()} | {'datetime': '2026-06-01T00:00:00Z', 'anomaly': 0}]
    frame = import_module('pandas').DataFrame(rows)

    with pytest.raises(ValueError, match='incomplete'):
        live_windows.build_live_window_datasets(
            frame,
            live_windows.LiveWindowExtractionConfig(
                start='2026-06-01T00:00:00Z',
                end='2026-06-01T00:01:00Z',
                window_size=2,
                stride=1,
            ),
        )


def test_build_live_window_datasets_uses_chronological_splits_without_leakage():
    live_windows = _live_windows()
    frame = live_windows.extract_live_sensor_frame(
        FakeQueryClient(FakeNamedResult(_long_rows(16))),
        live_windows.LiveWindowExtractionConfig(
            start='2026-06-01T00:00:00Z',
            end='2026-06-01T00:01:00Z',
            window_size=2,
            stride=2,
        ),
    )

    bundle = live_windows.build_live_window_datasets(
        frame,
        live_windows.LiveWindowExtractionConfig(
            start='2026-06-01T00:00:00Z',
            end='2026-06-01T00:01:00Z',
            device_ids=('ipa_01',),
            window_size=2,
            stride=2,
            validation_fraction=0.25,
            test_fraction=0.25,
            min_train_windows=1,
        ),
    )

    assert bundle.train.timestamps.tolist() == [
        '2026-06-01T00:00:01Z',
        '2026-06-01T00:00:03Z',
        '2026-06-01T00:00:05Z',
        '2026-06-01T00:00:07Z',
    ]
    assert bundle.validation.timestamps.tolist() == ['2026-06-01T00:00:09Z', '2026-06-01T00:00:11Z']
    assert bundle.test is not None
    assert bundle.test.timestamps.tolist() == ['2026-06-01T00:00:13Z', '2026-06-01T00:00:15Z']
    assert max(bundle.train.timestamps) < min(bundle.validation.timestamps)
    assert max(bundle.validation.timestamps) < min(bundle.test.timestamps)
    assert bundle.provenance['data_source'] == 'live_clickhouse'
    assert bundle.provenance['source_table'] == 'telemetry_observations'
    assert bundle.provenance['label_table'] == 'operator_labels'
    assert bundle.provenance['device_ids'] == ['ipa_01']
    assert bundle.provenance['sensor_columns'] == _sensor_columns()
    assert bundle.provenance['include_labels'] is False
