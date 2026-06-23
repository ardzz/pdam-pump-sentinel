from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import cast

OPERATOR_LABEL_COLUMNS = (
    'station',
    'source_timestamp',
    'label',
    'label_source',
    'operator_id',
    'reason',
    'created_at',
    'updated_at',
)
DEFAULT_LABEL_SOURCE = 'operator'
MAX_LABEL_SOURCE_LENGTH = 64


@dataclass(frozen=True)
class OperatorLabelPayload:
    source_timestamp: str
    label: int
    label_source: str = DEFAULT_LABEL_SOURCE
    operator_id: str | None = None
    reason: str | None = None


def parse_operator_label_payload(payload: object) -> OperatorLabelPayload:
    if not isinstance(payload, Mapping):
        raise ValueError('operator label payload must be a mapping')
    values = cast(Mapping[str, object], payload)

    if 'source_timestamp' not in values:
        raise ValueError('operator label payload requires source_timestamp')
    source_timestamp = _source_timestamp(values.get('source_timestamp'))

    if 'label' not in values:
        raise ValueError('operator label payload requires label')
    label = _label(values.get('label'))

    return OperatorLabelPayload(
        source_timestamp=source_timestamp,
        label=label,
        label_source=_label_source(values.get('label_source', DEFAULT_LABEL_SOURCE)),
        operator_id=_optional_text(values.get('operator_id'), 'operator_id'),
        reason=_optional_text(values.get('reason'), 'reason'),
    )


def _source_timestamp(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError('operator label source_timestamp must be a non-empty string or timestamp')


def _label(value: object) -> int:
    if type(value) is bool:
        return int(value)
    if type(value) is int and value in (0, 1):
        return value
    raise ValueError('operator label must be a bool or integer 0/1')


def _label_source(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError('operator label label_source must be a string')
    text = value.strip()
    if not text:
        raise ValueError('operator label label_source must not be empty')
    if len(text) > MAX_LABEL_SOURCE_LENGTH:
        raise ValueError('operator label label_source is too long')
    return text


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f'operator label {field_name} must be a string when provided')
    return value
