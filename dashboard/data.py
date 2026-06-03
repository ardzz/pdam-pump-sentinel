from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any, cast
from urllib.parse import unquote, urlparse

LATEST_READING_KEY = 'pumpad:latest:reading:{station}'
LATEST_ANOMALY_KEY = 'pumpad:latest:anomaly:{station}'
DRIFT_RESULT_KEY = 'pumpad:drift:result'
RETRAIN_RESULT_KEY = 'pumpad:retrain:result'
ACTIVE_MODEL_KEY = 'pumpad:active:model'


def _redis_client() -> Any:
    import redis  # type: ignore[reportMissingImports]

    return redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', '6379')),
        db=int(os.getenv('REDIS_DB', '0')),
        password=os.getenv('REDIS_PASSWORD') or None,
        username=os.getenv('REDIS_USERNAME') or None,
        socket_timeout=float(os.getenv('REDIS_SOCKET_TIMEOUT', '5.0')),
        socket_connect_timeout=float(os.getenv('REDIS_SOCKET_CONNECT_TIMEOUT', '5.0')),
        decode_responses=True,
        health_check_interval=30,
    )


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


def get_latest_reading(station: str) -> dict[str, Any] | None:
    return _redis_json(LATEST_READING_KEY.format(station=station))


def get_latest_anomaly(station: str) -> dict[str, Any] | None:
    return _redis_json(LATEST_ANOMALY_KEY.format(station=station))


def get_drift_result() -> dict[str, Any] | None:
    return _redis_json(DRIFT_RESULT_KEY)


def get_retrain_result() -> dict[str, Any] | None:
    return _redis_json(RETRAIN_RESULT_KEY)


def get_active_model() -> dict[str, Any] | None:
    return _redis_json(ACTIVE_MODEL_KEY)


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


def get_model_versions(model_name: str = 'PumpAD') -> list[dict[str, Any]]:
    try:
        client = _mlflow_client()
        versions = client.search_model_versions(f"name='{model_name}'")
        return [_model_version_dict(version) for version in versions]
    except Exception:
        return []


def list_stations() -> list[str]:
    stations = [station.strip() for station in os.getenv('STATIONS', '').split(',') if station.strip()]
    return stations or ['ipa_01']


def _redis_json(key: str) -> dict[str, Any] | None:
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
        return payload
    except Exception:
        return None


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
    'get_latest_anomaly',
    'get_latest_reading',
    'get_model_versions',
    'get_retrain_result',
    'list_stations',
]
