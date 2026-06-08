from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any, cast
from urllib.parse import unquote, urlparse

import redis  # type: ignore[reportMissingImports]
import streamlit as st

LATEST_READING_KEY = 'pumpad:latest:reading:{station}'
LATEST_ANOMALY_KEY = 'pumpad:latest:anomaly:{station}'
DRIFT_RESULT_KEY = 'pumpad:drift:result'
RETRAIN_RESULT_KEY = 'pumpad:retrain:result'
ACTIVE_MODEL_KEY = 'pumpad:active:model'

_REDIS_CLIENT: Any | None = None
_last_error: str | None = None


def _redis_client() -> Any:
    global _REDIS_CLIENT, _last_error

    if _REDIS_CLIENT is None:
        client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            decode_responses=True,
        )
        try:
            client.ping()
        except (redis.RedisError, ConnectionError) as exc:
            _last_error = str(exc)
            raise
        _REDIS_CLIENT = client
    _last_error = None
    return _REDIS_CLIENT


def _clickhouse_client() -> Any:
    import clickhouse_connect  # type: ignore[reportMissingImports]

    telemetry_url = os.getenv('TELEMETRY_URL')
    if not telemetry_url:
        raise ValueError('TELEMETRY_URL is not set')
    parsed = urlparse(telemetry_url)
    if not parsed.hostname:
        raise ValueError('TELEMETRY_URL host is not set')
    secure = parsed.scheme == 'https'
    return clickhouse_connect.get_client(
        host=parsed.hostname,
        port=parsed.port or (8443 if secure else 8123),
        username=unquote(parsed.username) if parsed.username else None,
        password=unquote(parsed.password) if parsed.password else '',
        database=parsed.path.strip('/') or '__default__',
        interface=parsed.scheme or 'http',
        secure=secure,
    )


def _mlflow_client() -> Any:
    import mlflow  # type: ignore[reportMissingImports]
    from mlflow.tracking import MlflowClient  # type: ignore[reportMissingImports]

    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


@st.cache_data(ttl=5)
def get_latest_reading(station: str) -> dict[str, Any] | None:
    return _redis_json(LATEST_READING_KEY.format(station=station))


@st.cache_data(ttl=5)
def get_latest_anomaly(station: str) -> dict[str, Any] | None:
    return _redis_json(LATEST_ANOMALY_KEY.format(station=station))


@st.cache_data(ttl=5)
def get_drift_result() -> dict[str, Any] | None:
    return _redis_json(DRIFT_RESULT_KEY)


@st.cache_data(ttl=5)
def get_retrain_result() -> dict[str, Any] | None:
    return _redis_json(RETRAIN_RESULT_KEY)


@st.cache_data(ttl=5)
def get_active_model() -> dict[str, Any] | None:
    return _redis_json(ACTIVE_MODEL_KEY)


@st.cache_data(ttl=5)
def get_anomaly_history(station: str, limit: int = 200) -> list[dict[str, Any]]:
    try:
        client = _clickhouse_client()
        result = client.query(
            'SELECT observed_at, measurement, value_float, value_int '
            'FROM telemetry_observations '
            'WHERE device_id = %(d)s '
            'ORDER BY observed_at DESC '
            'LIMIT %(n)s',
            parameters={'d': station, 'n': max(0, int(limit))},
        )
        return _query_result_dicts(result)
    except Exception:
        return []


@st.cache_data(ttl=5)
def get_model_versions(model_name: str = 'PumpAD') -> list[dict[str, Any]]:
    try:
        client = _mlflow_client()
        versions = client.search_model_versions(f"name='{model_name}'")
        return [_model_version_dict(version) for version in versions]
    except Exception:
        return []


@st.cache_data(ttl=5)
def list_stations() -> list[str]:
    stations = [station.strip() for station in os.getenv('STATIONS', '').split(',') if station.strip()]
    return stations or ['ipa_01']


def get_last_error() -> str | None:
    return _last_error


def timestamp_age_seconds(value: Any, now: datetime | None = None) -> float | None:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return None
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    return max(0.0, (reference.astimezone(timezone.utc) - parsed).total_seconds())


def get_observability_snapshot(station: str, now: datetime | None = None) -> dict[str, Any]:
    reading = get_latest_reading(station)
    anomaly = get_latest_anomaly(station)
    drift = get_drift_result()
    retrain = get_retrain_result()
    active_model = get_active_model()

    telemetry_ts = _first_value(reading, 'timestamp', 'observed_at') or _first_value(anomaly, 'source_timestamp', 'timestamp')
    drift_ts = _first_value(drift, 'timestamp', 'finished_at', 'created_at')
    active_ts = _first_value(active_model, 'activated_at', 'activated_ts', 'updated_at')

    telemetry_age = timestamp_age_seconds(telemetry_ts, now=now)
    drift_age = timestamp_age_seconds(drift_ts, now=now)
    active_model_age = timestamp_age_seconds(active_ts, now=now)
    drift_detected = _drift_detected(drift)

    component_states = {
        'telemetry': _age_state(telemetry_age, green_seconds=60, red_seconds=300),
        'drift_report': 'RED' if drift_detected else _age_state(drift_age, green_seconds=3600, red_seconds=86400),
        'active_model': _age_state(active_model_age, green_seconds=24 * 3600, red_seconds=7 * 24 * 3600),
    }
    return {
        'state': _worst_state(component_states.values()),
        'components': component_states,
        'telemetry_age_seconds': telemetry_age,
        'drift_report_age_seconds': drift_age,
        'active_model_age_seconds': active_model_age,
        'drift_detected': drift_detected,
        'retrain_result': _retrain_result(retrain),
        'latest_reading_timestamp': telemetry_ts,
        'latest_drift_timestamp': drift_ts,
        'active_model_timestamp': active_ts,
    }


def record_operator_action(kind: str, station: str, payload: dict[str, Any], ttl_seconds: int | None) -> bool:
    global _last_error

    try:
        client = _redis_client()
        key = _operator_action_key(kind, station, payload)
        value = json.dumps({k: v for k, v in payload.items() if not k.startswith('_')})
        if ttl_seconds is None:
            client.set(key, value)
        else:
            client.set(key, value, ex=ttl_seconds)
        _last_error = None
        return True
    except (redis.RedisError, ConnectionError, TypeError, ValueError) as exc:
        _last_error = str(exc)
        return False


def _redis_json(key: str) -> dict[str, Any] | None:
    global _last_error

    try:
        client = _redis_client()
        value = client.get(key)
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        payload = json.loads(value)
        if not isinstance(payload, dict):
            return None
        _last_error = None
        return payload
    except json.JSONDecodeError:
        return None
    except (redis.RedisError, ConnectionError) as exc:
        _last_error = str(exc)
        return None


def _operator_action_key(kind: str, station: str, payload: dict[str, Any]) -> str:
    normalized_kind = kind.strip().lower()
    if normalized_kind == 'mute':
        return f'pumpad:anomaly:mute:{station}'
    raw_ts = (
        payload.get('_ts')
        or payload.get('iso_ts')
        or payload.get('source_timestamp')
        or payload.get('timestamp')
        or payload.get('observed_at')
        or 'unknown'
    )
    return f'pumpad:anomaly:{normalized_kind}:{station}:{raw_ts}'


def _first_value(payload: dict[str, Any] | None, *keys: str) -> Any:
    if not payload:
        return None
    for key in keys:
        value = payload.get(key)
        if value not in (None, ''):
            return value
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_state(age_seconds: float | None, *, green_seconds: int, red_seconds: int) -> str:
    if age_seconds is None:
        return 'RED'
    if age_seconds <= green_seconds:
        return 'GREEN'
    if age_seconds <= red_seconds:
        return 'DEGRADED'
    return 'RED'


def _worst_state(states: Iterable[str]) -> str:
    values = tuple(states)
    if 'RED' in values:
        return 'RED'
    if 'DEGRADED' in values:
        return 'DEGRADED'
    return 'GREEN'


def _drift_detected(drift: dict[str, Any] | None) -> bool:
    if not drift:
        return False
    metrics = drift.get('metrics', {}) if isinstance(drift.get('metrics'), dict) else {}
    return bool(drift.get('dataset_drift', metrics.get('drift_detected', False)))


def _retrain_result(retrain: dict[str, Any] | None) -> str:
    if not retrain:
        return 'N/A'
    if retrain.get('promoted') is True:
        return 'PROMOTED'
    if retrain.get('success') is False or retrain.get('error'):
        return 'FAILED'
    if retrain.get('success') is True:
        return 'SUCCESS'
    if retrain.get('promoted') is False:
        return 'REJECTED'
    return 'N/A'


def _query_result_dicts(result: Any) -> list[dict[str, Any]]:
    named_results = getattr(result, 'named_results', None)
    if callable(named_results):
        return [dict(row) for row in cast(Iterable[Any], named_results())]

    rows = getattr(result, 'result_rows', None)
    if rows is None:
        return []
    column_names = tuple(getattr(result, 'column_names', ())) or ('observed_at', 'measurement', 'value_float', 'value_int')
    return [dict(zip(column_names, row)) for row in cast(Iterable[Iterable[Any]], rows)]


def _model_version_dict(model_version: Any) -> dict[str, Any]:
    run_id = getattr(model_version, 'run_id', None)
    return {
        'version': str(getattr(model_version, 'version', None)),
        'aliases': _aliases(getattr(model_version, 'aliases', None)),
        'run_id': None if run_id is None else str(run_id),
    }


def _aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, bytes):
        return [str(alias) for alias in value]
    return []


__all__ = [
    'get_active_model',
    'get_anomaly_history',
    'get_drift_result',
    'get_last_error',
    'get_latest_anomaly',
    'get_latest_reading',
    'get_model_versions',
    'get_observability_snapshot',
    'get_retrain_result',
    'list_stations',
    'record_operator_action',
    'timestamp_age_seconds',
]
