from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.models.operator_label import OperatorLabelPayload, parse_operator_label_payload


def test_operator_label_payload_accepts_bool_and_defaults_source():
    payload = parse_operator_label_payload({'source_timestamp': '2024-01-01T00:00:00Z', 'label': False})

    assert payload == OperatorLabelPayload(
        source_timestamp='2024-01-01T00:00:00Z',
        label=0,
        label_source='operator',
        operator_id=None,
        reason=None,
    )


def test_operator_label_payload_accepts_integer_labels_and_timestamp_objects():
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)

    payload = parse_operator_label_payload(
        {
            'source_timestamp': timestamp,
            'label': 1,
            'label_source': 'review_queue',
            'operator_id': 'operator-a',
            'reason': 'confirmed',
        }
    )

    assert payload.source_timestamp == '2024-01-01T00:00:00+00:00'
    assert payload.label == 1
    assert payload.label_source == 'review_queue'
    assert payload.operator_id == 'operator-a'
    assert payload.reason == 'confirmed'


@pytest.mark.parametrize(
    'payload',
    [
        None,
        [],
        {'label': 1},
        {'source_timestamp': '', 'label': 1},
        {'source_timestamp': 't0'},
        {'source_timestamp': 't0', 'label': 2},
        {'source_timestamp': 't0', 'label': 0.0},
        {'source_timestamp': 't0', 'label': '0'},
        {'source_timestamp': 't0', 'label': 1, 'label_source': 'x' * 65},
        {'source_timestamp': 't0', 'label': 1, 'reason': 5},
    ],
)
def test_operator_label_payload_rejects_invalid_shapes(payload: object):
    with pytest.raises(ValueError):
        _ = parse_operator_label_payload(payload)


def test_clickhouse_schema_defines_operator_labels_table():
    schema = Path('infra/clickhouse/init.sql').read_text(encoding='utf-8')

    assert 'CREATE TABLE IF NOT EXISTS operator_labels' in schema
    assert 'station LowCardinality(String)' in schema
    assert 'source_timestamp String' in schema
    assert 'label UInt8' in schema
    assert 'label_source LowCardinality(String)' in schema
    assert 'operator_id Nullable(String)' in schema
    assert 'reason Nullable(String)' in schema
    assert "created_at DateTime64(3, 'UTC')" in schema
    assert "updated_at DateTime64(3, 'UTC')" in schema
    assert 'ENGINE = ReplacingMergeTree(updated_at)' in schema
    assert 'ORDER BY (station, source_timestamp, label_source)' in schema
