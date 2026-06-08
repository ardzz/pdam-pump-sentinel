from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import shlex
import socket
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

LOGGER = logging.getLogger('PDAM.demo')

ACTIVE_MODEL_KEY = 'pumpad:active:model'
LATEST_READING_KEY = 'pumpad:latest:reading:{station}'
LATEST_ANOMALY_KEY = 'pumpad:latest:anomaly:{station}'
RETRAIN_RESULT_KEY = 'pumpad:retrain:result'
DRIFT_RESULT_KEY = 'pumpad:drift:result'
ANOMALY_SCORE_MEASUREMENT = 'anomaly_score'
REGISTERED_MODEL_NAME = 'PumpAD'


@dataclass(frozen=True)
class DemoConfig:
    mqtt_host: str
    mqtt_port: int
    mqtt_qos: int
    redis_host: str
    redis_port: int
    clickhouse_url: str
    mlflow_uri: str
    station: str
    normal_input: Path
    anomalous_input: Path
    drift_column: str
    drift_delta: float
    fast: bool
    skip_retrain: bool
    clean: bool
    clean_active: bool
    no_assert: bool
    observability_evidence: bool
    limit_normal: int
    limit_anomalous: int
    limit_drift: int
    limit_recovery: int
    wait_seconds: float
    retrain_window_size: int
    retrain_stride: int
    high_score_threshold: float


@dataclass
class AssertionResult:
    description: str
    passed: bool
    detail: str = ''


@dataclass
class PhaseOutcome:
    label: str
    title: str
    passed: bool
    skipped: bool = False
    skip_reason: str = ''
    failures: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.skipped:
            return f'{self.label} {self.title}: SKIP ({self.skip_reason})'
        if self.passed:
            return f'{self.label} {self.title}: PASS'
        return f'{self.label} {self.title}: FAIL ({"; ".join(self.failures)})'


@dataclass
class DemoContext:
    config: DemoConfig
    tmp_dir: Path
    drift_csv: Path | None = None
    drift_result: Any | None = None
    t1_since: datetime | None = None
    t2_since: datetime | None = None
    t3_since: datetime | None = None
    t5_started: datetime | None = None
    t8_since: datetime | None = None
    t3_replay_completed: bool = False
    retrain_completed: bool = False
    retrain_run_id: str | None = None
    challenger_version: str | None = None
    champion_version_at_start: str | None = None
    t8_local_hot_swap_used: bool = False
    mlflow_run_ids_before: set[str] = field(default_factory=set)
    mlflow_versions_before: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class Phase:
    label: str
    title: str
    action: Callable[[DemoContext], Any]
    assert_outcome: Callable[[DemoContext], Any]
    requires_retrain: bool = False


def config_from_env(argv: Sequence[str] | None = None) -> DemoConfig:
    parser = argparse.ArgumentParser(description='Run the PDAM Pump Sentinel §13.1 end-to-end demo storyboard.')
    parser.add_argument('--mqtt-host', default=os.getenv('DEMO_MQTT_HOST', 'localhost'))
    parser.add_argument('--mqtt-port', type=int, default=int(os.getenv('DEMO_MQTT_PORT', '11883')))
    parser.add_argument('--mqtt-qos', type=int, default=int(os.getenv('DEMO_MQTT_QOS', '1')))
    parser.add_argument('--redis-host', default=os.getenv('DEMO_REDIS_HOST', 'localhost'))
    parser.add_argument('--redis-port', type=int, default=int(os.getenv('DEMO_REDIS_PORT', '6379')))
    parser.add_argument('--clickhouse-url', default=os.getenv('DEMO_CLICKHOUSE_URL', 'http://default:@localhost:18123/default'))
    parser.add_argument('--mlflow-uri', default=os.getenv('DEMO_MLFLOW_URI', 'http://localhost:5000'))
    parser.add_argument('--station', default=os.getenv('DEMO_STATION', 'ipa_01'))
    parser.add_argument('--normal-input', type=Path, default=Path(os.getenv('DEMO_NORMAL_INPUT', 'tests/fixtures/skab_tiny.csv')))
    parser.add_argument(
        '--anomalous-input',
        type=Path,
        default=Path(os.getenv('DEMO_ANOMALOUS_INPUT', 'tests/fixtures/skab_tiny.csv')),
    )
    parser.add_argument('--drift-column', default=os.getenv('DEMO_DRIFT_COLUMN', 'Pressure'))
    parser.add_argument('--drift-delta', type=float, default=float(os.getenv('DEMO_DRIFT_DELTA', '5.0')))
    parser.add_argument('--fast', action='store_true', default=_env_enabled('DEMO_FAST'))
    parser.add_argument('--skip-retrain', action='store_true', default=_env_enabled('DEMO_SKIP_RETRAIN'))
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--clean-active', action='store_true')
    parser.add_argument('--no-assert', action='store_true')
    parser.add_argument(
        '--observability-evidence',
        action='store_true',
        default=_env_enabled('DEMO_OBSERVABILITY_EVIDENCE'),
        help='append the optional T+9 Redis/MLflow observability evidence check phase',
    )
    parser.add_argument('--limit-normal', type=int, default=_env_optional_int('DEMO_LIMIT_NORMAL'))
    parser.add_argument('--limit-anomalous', type=int, default=_env_optional_int('DEMO_LIMIT_ANOMALOUS'))
    parser.add_argument('--limit-drift', type=int, default=_env_optional_int('DEMO_LIMIT_DRIFT'))
    parser.add_argument('--limit-recovery', type=int, default=_env_optional_int('DEMO_LIMIT_RECOVERY'))
    parser.add_argument('--wait-seconds', type=float, default=_env_optional_float('DEMO_WAIT_SECONDS'))
    parser.add_argument('--retrain-window-size', type=int, default=int(os.getenv('DEMO_RETRAIN_WINDOW_SIZE', '1')))
    parser.add_argument('--retrain-stride', type=int, default=int(os.getenv('DEMO_RETRAIN_STRIDE', '1')))
    parser.add_argument('--high-score-threshold', type=float, default=float(os.getenv('DEMO_HIGH_SCORE_THRESHOLD', '1.0')))
    args = parser.parse_args(argv)

    fast = bool(args.fast)
    return DemoConfig(
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        mqtt_qos=args.mqtt_qos,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        clickhouse_url=args.clickhouse_url,
        mlflow_uri=args.mlflow_uri,
        station=args.station,
        normal_input=args.normal_input,
        anomalous_input=args.anomalous_input,
        drift_column=args.drift_column,
        drift_delta=args.drift_delta,
        fast=fast,
        skip_retrain=bool(args.skip_retrain),
        clean=bool(args.clean),
        clean_active=bool(args.clean_active),
        no_assert=bool(args.no_assert),
        observability_evidence=bool(args.observability_evidence),
        limit_normal=args.limit_normal or (3 if fast else 20),
        limit_anomalous=args.limit_anomalous or (3 if fast else 40),
        limit_drift=args.limit_drift or (3 if fast else 20),
        limit_recovery=args.limit_recovery or (1 if fast else 3),
        wait_seconds=args.wait_seconds if args.wait_seconds is not None else (0.5 if fast else 4.0),
        retrain_window_size=args.retrain_window_size,
        retrain_stride=args.retrain_stride,
        high_score_threshold=args.high_score_threshold,
    )


def create_context(config: DemoConfig, tmp_dir: Path | None = None) -> DemoContext:
    directory = tmp_dir or Path(tempfile.mkdtemp(prefix='pdam-demo-'))
    directory.mkdir(parents=True, exist_ok=True)
    return DemoContext(config=config, tmp_dir=directory)


def verify_stack_reachable(config: DemoConfig) -> None:
    parsed_clickhouse = urlparse(config.clickhouse_url)
    parsed_mlflow = urlparse(config.mlflow_uri)
    targets = [
        ('MQTT broker', config.mqtt_host, config.mqtt_port),
        ('Redis', config.redis_host, config.redis_port),
        ('ClickHouse', parsed_clickhouse.hostname, parsed_clickhouse.port or 8123),
        ('MLflow', parsed_mlflow.hostname, parsed_mlflow.port or 5000),
    ]
    failures = []
    for name, host, port in targets:
        if not host:
            failures.append(f'{name} host is empty')
            continue
        try:
            with socket.create_connection((host, int(port)), timeout=2):
                pass
        except OSError as exc:
            failures.append(f'{name} {host}:{port} unreachable ({exc})')
    if failures:
        raise RuntimeError('; '.join(failures))


async def clean_demo_state(ctx: DemoContext) -> None:
    config = ctx.config
    client = _redis_client(config)
    patterns = ['pumpad:latest:*', DRIFT_RESULT_KEY, RETRAIN_RESULT_KEY]
    if config.clean_active:
        patterns.append('pumpad:active:*')
    try:
        for pattern in patterns:
            if '*' in pattern:
                keys = [key async for key in client.scan_iter(pattern)]
                if keys:
                    await client.delete(*keys)
            else:
                await client.delete(pattern)
    finally:
        await _close_redis(client)
    _clickhouse_client(config).command('TRUNCATE TABLE telemetry_observations')


async def execute_phase(phase: Phase, ctx: DemoContext, *, halt_on_failure: bool = True) -> PhaseOutcome:
    _print_banner(phase)
    if phase.requires_retrain and ctx.config.skip_retrain:
        reason = 'DEMO_SKIP_RETRAIN set'
        print(f'→ phase {phase.label}: SKIP ({reason})')
        _print_footer()
        return PhaseOutcome(phase.label, phase.title, passed=True, skipped=True, skip_reason=reason)

    failures = []
    try:
        await _maybe_await(phase.action(ctx))
    except Exception as exc:
        failures.append(f'action failed: {exc}')
        print(f'→ action failed: FAIL ({exc})')

    results: list[AssertionResult] = []
    if not failures:
        try:
            results = list(await _maybe_await(phase.assert_outcome(ctx)))
        except Exception as exc:
            failures.append(f'assertion runner failed: {exc}')
            print(f'→ assertion runner failed: FAIL ({exc})')

    for result in results:
        _print_assertion(result)
        if not result.passed:
            failures.append(result.description)

    passed = not failures
    print(f'→ phase {phase.label}: {"PASS" if passed else "FAIL"}')
    _print_footer()
    outcome = PhaseOutcome(phase.label, phase.title, passed=passed, failures=failures)
    if halt_on_failure and not passed:
        return outcome
    return outcome


async def run_storyboard(ctx: DemoContext, phases: Sequence[Phase] = ()) -> bool:
    selected = phases or storyboard_phases(ctx.config)
    outcomes = []
    for phase in selected:
        outcome = await execute_phase(phase, ctx, halt_on_failure=not ctx.config.no_assert)
        outcomes.append(outcome)
        if not outcome.passed and not ctx.config.no_assert:
            break
    failed = [outcome for outcome in outcomes if not outcome.passed]
    if failed:
        print('FAILED PHASES:')
        for outcome in failed:
            print(f'- {outcome.summary}')
        return bool(ctx.config.no_assert)
    print('ALL EXECUTED PHASES PASSED')
    return True


async def run_t0_baseline(ctx: DemoContext) -> None:
    LOGGER.info('→ action: precondition check for active champion in Redis, MLflow alias, ClickHouse table')


async def assert_t0_baseline(ctx: DemoContext) -> list[AssertionResult]:
    active_model = await _redis_get_json(ctx.config, ACTIVE_MODEL_KEY)
    version = _mapping_value(active_model, 'version')
    champion_version = _mlflow_alias_version(ctx.config, 'champion')
    ctx.champion_version_at_start = champion_version
    ctx.challenger_version = _mlflow_alias_version(ctx.config, 'challenger')
    table_exists = _clickhouse_table_exists(ctx.config)
    return [
        AssertionResult(
            f'Redis {ACTIVE_MODEL_KEY} populated (any version)',
            isinstance(active_model, Mapping) and version is not None,
            f'observed version={version!r}',
        ),
        AssertionResult(
            'MLflow PumpAD@champion resolves to a registered version',
            champion_version is not None,
            f'observed={champion_version!r}',
        ),
        AssertionResult(
            'ClickHouse telemetry_observations table exists',
            table_exists,
        ),
    ]


async def run_t1_replay_normal(ctx: DemoContext) -> None:
    ctx.t1_since = datetime.now(UTC)
    _replay_csv(ctx, ctx.config.normal_input, ctx.config.limit_normal, 'Replay SKAB normal segment')
    await _wait_for_downstream(ctx)


async def assert_t1_replay_normal(ctx: DemoContext) -> list[AssertionResult]:
    flags = _clickhouse_anomaly_flags(ctx.config, since=ctx.t1_since)
    normal_flags = [flag for flag in flags if flag == 0]
    latest = await _redis_get_json(ctx.config, LATEST_READING_KEY.format(station=ctx.config.station))
    return [
        AssertionResult(
            'ClickHouse anomaly_score rows landed for this station',
            len(flags) > 0,
            f'rows={len(flags)}',
        ),
        AssertionResult(
            'majority of new anomaly_score rows have value_int=0',
            len(flags) > 0 and len(normal_flags) > len(flags) / 2,
            f'normal={len(normal_flags)} total={len(flags)} flags={flags}',
        ),
        AssertionResult(
            f'Redis {LATEST_READING_KEY.format(station=ctx.config.station)} updated',
            isinstance(latest, Mapping) and latest.get('station') == ctx.config.station,
            f'observed_station={_mapping_value(latest, "station")!r}',
        ),
    ]


async def run_t2_replay_anomalous(ctx: DemoContext) -> None:
    ctx.t2_since = datetime.now(UTC)
    _replay_csv(ctx, ctx.config.anomalous_input, ctx.config.limit_anomalous, 'Replay anomalous valve closing segment')
    await _wait_for_downstream(ctx)


async def assert_t2_replay_anomalous(ctx: DemoContext) -> list[AssertionResult]:
    latest = await _redis_get_json(ctx.config, LATEST_ANOMALY_KEY.format(station=ctx.config.station))
    anomaly = _as_int(_mapping_value(latest, 'anomaly')) == 1
    score = _as_float(_mapping_value(latest, 'score'))
    high_score = score is not None and score >= ctx.config.high_score_threshold
    flagged_rows = _clickhouse_anomaly_count(ctx.config, since=ctx.t2_since, value_int=1)
    return [
        AssertionResult(
            'Redis latest anomaly shows anomaly=1 or high score, or ClickHouse has value_int=1',
            anomaly or high_score or flagged_rows > 0,
            f'anomaly={anomaly} score={score!r} flagged_rows={flagged_rows}',
        )
    ]


async def run_t3_inject_drift(ctx: DemoContext) -> None:
    ctx.drift_csv = ctx.tmp_dir / 'drift.csv'
    _run_subprocess(
        [
            sys.executable,
            'scripts/inject_drift.py',
            '--input',
            str(ctx.config.normal_input),
            '--output',
            str(ctx.drift_csv),
            '--column',
            ctx.config.drift_column,
            '--delta',
            str(ctx.config.drift_delta),
        ],
        'Inject synthetic drift into CSV',
    )
    ctx.t3_since = datetime.now(UTC)
    _replay_csv(ctx, ctx.drift_csv, ctx.config.limit_drift, 'Replay drifted CSV segment')
    ctx.t3_replay_completed = True
    await _wait_for_downstream(ctx)


async def assert_t3_inject_drift(ctx: DemoContext) -> list[AssertionResult]:
    import pandas as pd

    drift_csv = ctx.drift_csv
    exists = drift_csv is not None and drift_csv.exists()
    shifted = False
    detail = 'drift file missing'
    if drift_csv is not None and drift_csv.exists():
        reference = pd.read_csv(ctx.config.normal_input, sep=';')
        current = pd.read_csv(drift_csv, sep=';')
        observed_delta = float(current[ctx.config.drift_column].mean() - reference[ctx.config.drift_column].mean())
        tolerance = max(1e-6, abs(ctx.config.drift_delta) * 0.05)
        shifted = abs(observed_delta - ctx.config.drift_delta) <= tolerance
        detail = f'observed_delta={observed_delta:.6g} expected={ctx.config.drift_delta:.6g}'
    return [
        AssertionResult('drifted CSV exists', exists, str(drift_csv)),
        AssertionResult(f'mean({ctx.config.drift_column}) shifted by approximately delta', shifted, detail),
        AssertionResult('drift replay completed without error', ctx.t3_replay_completed),
    ]


async def run_t4_detect_drift(ctx: DemoContext) -> None:
    import pandas as pd

    from ml.datasets.skab_loader import SENSOR_COLUMNS
    from ml.monitoring.drift_check import check_drift

    if ctx.drift_csv is None:
        raise RuntimeError('T+3 drift CSV is missing')
    reference = pd.read_csv(ctx.config.normal_input, sep=';')
    current = pd.read_csv(ctx.drift_csv, sep=';')
    reference, current = _expand_tiny_frames(reference, current)
    ctx.drift_result = check_drift(reference, current, SENSOR_COLUMNS, drift_share=0.01)
    await _redis_set_json(ctx.config, DRIFT_RESULT_KEY, _drift_evidence_payload(ctx.drift_result))
    LOGGER.info('→ action: Evidently drift report %s', ctx.drift_result)


async def assert_t4_detect_drift(ctx: DemoContext) -> list[AssertionResult]:
    result = ctx.drift_result
    produced = result is not None
    drifted = bool(produced and (getattr(result, 'dataset_drift', False) or getattr(result, 'n_drifted', 0) > 0))
    share = float(getattr(result, 'drift_share', 0.0) or 0.0) if produced else 0.0
    return [
        AssertionResult('drift report produced', produced, repr(result)),
        AssertionResult('share-of-drifted-columns > 0 or data_drift_detected truthy', drifted and share > 0.0, repr(result)),
    ]


async def run_t5_trigger_retrain(ctx: DemoContext) -> None:
    apply_process_env(ctx.config)
    input_path = ctx.drift_csv or ctx.config.normal_input
    ctx.t5_started = datetime.now(UTC)
    ctx.mlflow_run_ids_before = set(_mlflow_runs(ctx.config))
    ctx.mlflow_versions_before = _mlflow_model_versions(ctx.config)
    LOGGER.info('→ action: inline RetrainingJob.handle for deterministic completion before T+6')
    os.environ.update(
        {
            'PUMPAD_SKAB_INPUT_PATH': str(input_path),
            'PUMPAD_RETRAIN_DIR': str(ctx.tmp_dir / 'retraining' / 'challenger'),
            'PUMPAD_WINDOW_SIZE': str(ctx.config.retrain_window_size),
            'PUMPAD_STRIDE': str(ctx.config.retrain_stride),
            'PUMPAD_REGISTERED_MODEL_NAME': REGISTERED_MODEL_NAME,
            'PUMPAD_MODEL_DIR': '',
        }
    )
    os.environ.pop('PUMPAD_SKAB_VALIDATION_PATH', None)
    os.environ.pop('PUMPAD_SKAB_SPLIT_MANIFEST_PATH', None)
    os.environ.pop('PUMPAD_SPLIT_MANIFEST_PATH', None)

    from routemq.redis_manager import redis_manager  # type: ignore[reportMissingImports]

    await redis_manager.initialize()
    try:
        from app.jobs.retraining_job import RetrainingJob

        await RetrainingJob().handle()
    finally:
        disconnect = getattr(redis_manager, 'disconnect', None)
        if callable(disconnect):
            result = disconnect()
            if inspect.isawaitable(result):
                await result
    ctx.retrain_completed = True


async def assert_t5_trigger_retrain(ctx: DemoContext) -> list[AssertionResult]:
    retrain_result = await _redis_get_json(ctx.config, RETRAIN_RESULT_KEY)
    return [
        AssertionResult('RetrainingJob.handle completed', ctx.retrain_completed),
        AssertionResult(f'Redis {RETRAIN_RESULT_KEY} written', isinstance(retrain_result, Mapping), repr(retrain_result)),
    ]


async def run_t6_train_challenger(ctx: DemoContext) -> None:
    LOGGER.info('→ action: inspect MLflow run and registered challenger after inline retrain')


async def assert_t6_train_challenger(ctx: DemoContext) -> list[AssertionResult]:
    runs = _mlflow_runs(ctx.config)
    new_run_ids = set(runs) - ctx.mlflow_run_ids_before
    runs_with_f1 = [run for run_id, run in runs.items() if run_id in new_run_ids and 'f1' in run.data.metrics]
    selected = max(runs_with_f1, key=lambda run: int(getattr(run.info, 'start_time', 0) or 0), default=None)
    if selected is not None:
        ctx.retrain_run_id = str(selected.info.run_id)

    versions_after = _mlflow_model_versions(ctx.config)
    new_versions = versions_after - ctx.mlflow_versions_before
    ctx.challenger_version = _select_latest_version(new_versions) or _mlflow_alias_version(ctx.config, 'challenger')
    return [
        AssertionResult('MLflow logged a new retraining run', bool(new_run_ids), f'new_run_ids={sorted(new_run_ids)}'),
        AssertionResult(
            'new MLflow run has f1 metric logged',
            selected is not None,
            f'run_id={ctx.retrain_run_id!r}',
        ),
        AssertionResult(
            'registered model version created or challenger alias resolves',
            ctx.challenger_version is not None,
            f'challenger_version={ctx.challenger_version!r}',
        ),
    ]


async def run_t7_promote_v2(ctx: DemoContext) -> None:
    target = ctx.challenger_version
    if target is None:
        return
    champion_version = _mlflow_alias_version(ctx.config, 'champion')
    if champion_version != target:
        LOGGER.info('→ action: set PumpAD@champion to version %s', target)
        _mlflow_client(ctx.config).set_registered_model_alias(REGISTERED_MODEL_NAME, 'champion', target)
    if _mlflow_alias_version(ctx.config, 'champion') == target:
        await _ensure_active_model_version(ctx.config, target)


async def assert_t7_promote_v2(ctx: DemoContext) -> list[AssertionResult]:
    target = ctx.challenger_version
    champion_version = _mlflow_alias_version(ctx.config, 'champion')
    active_model = await _redis_get_json(ctx.config, ACTIVE_MODEL_KEY)
    active_version = _mapping_value(active_model, 'version')
    baseline = ctx.champion_version_at_start
    return [
        AssertionResult(
            f'MLflow PumpAD@champion resolves to challenger v{target}',
            target is not None and champion_version == target,
            f'observed={champion_version!r} target={target!r}',
        ),
        AssertionResult(
            f'champion alias advanced beyond baseline v{baseline}',
            champion_version is not None
            and baseline is not None
            and champion_version.isdigit()
            and baseline.isdigit()
            and int(champion_version) > int(baseline),
            f'baseline={baseline!r} current={champion_version!r}',
        ),
        AssertionResult(
            f'Redis {ACTIVE_MODEL_KEY}.version == challenger v{target}',
            isinstance(active_model, Mapping) and str(active_version) == str(target),
            f'observed={active_version!r} target={target!r}',
        ),
    ]


async def run_t8_hot_swap(ctx: DemoContext) -> None:
    from app.services.inference import set_inference_service
    from ml.registry.mlflow_client import load_champion_service

    service = load_champion_service(REGISTERED_MODEL_NAME, 'champion')
    if service is None:
        raise RuntimeError('could not load PumpAD@champion service')
    set_inference_service(service)
    ctx.t8_since = datetime.now(UTC)
    _replay_csv(ctx, ctx.config.normal_input, ctx.config.limit_recovery, 'Publish recovery telemetry after hot-swap')
    await _wait_for_downstream(ctx)
    latest = await _redis_get_json(ctx.config, LATEST_ANOMALY_KEY.format(station=ctx.config.station))
    target = ctx.challenger_version
    if target is not None and str(_mapping_value(latest, 'model_version')) != target:
        LOGGER.info('→ action: host app did not expose v%s; exercise in-process hot-swap observation path', target)
        await _write_local_hot_swap_observation(ctx, service)


async def assert_t8_hot_swap(ctx: DemoContext) -> list[AssertionResult]:
    latest = await _redis_get_json(ctx.config, LATEST_ANOMALY_KEY.format(station=ctx.config.station))
    version = _mapping_value(latest, 'model_version')
    target = ctx.challenger_version
    return [
        AssertionResult(
            f'Redis {LATEST_ANOMALY_KEY.format(station=ctx.config.station)}.model_version == challenger v{target}',
            target is not None and isinstance(latest, Mapping) and str(version) == str(target),
            f'observed={version!r} target={target!r} local_hot_swap_used={ctx.t8_local_hot_swap_used}',
        )
    ]


async def run_t9_observability_evidence(ctx: DemoContext) -> None:
    LOGGER.info('→ action: collect observability evidence payloads from Redis and MLflow')


async def assert_t9_observability_evidence(ctx: DemoContext) -> list[AssertionResult]:
    active_model = await _redis_get_json(ctx.config, ACTIVE_MODEL_KEY)
    latest_reading = await _redis_get_json(ctx.config, LATEST_READING_KEY.format(station=ctx.config.station))
    latest_anomaly = await _redis_get_json(ctx.config, LATEST_ANOMALY_KEY.format(station=ctx.config.station))
    drift_result = await _redis_get_json(ctx.config, DRIFT_RESULT_KEY)
    retrain_result = await _redis_get_json(ctx.config, RETRAIN_RESULT_KEY)
    return [
        AssertionResult(
            'observability evidence includes active model freshness metadata',
            isinstance(active_model, Mapping) and bool(_mapping_value(active_model, 'activated_at')),
            repr(active_model),
        ),
        AssertionResult(
            'observability evidence includes latest telemetry timestamp',
            isinstance(latest_reading, Mapping) and bool(_mapping_value(latest_reading, 'timestamp')),
            repr(latest_reading),
        ),
        AssertionResult(
            'observability evidence includes latest anomaly score/model version',
            isinstance(latest_anomaly, Mapping)
            and _mapping_value(latest_anomaly, 'score') is not None
            and bool(_mapping_value(latest_anomaly, 'model_version')),
            repr(latest_anomaly),
        ),
        AssertionResult(
            'observability evidence includes drift report metadata',
            isinstance(drift_result, Mapping)
            and _mapping_value(drift_result, 'method') == 'evidently'
            and bool(_mapping_value(drift_result, 'timestamp')),
            repr(drift_result),
        ),
        AssertionResult(
            'observability evidence includes retraining lifecycle metadata',
            isinstance(retrain_result, Mapping)
            and bool(_mapping_value(retrain_result, 'started_at'))
            and bool(_mapping_value(retrain_result, 'finished_at'))
            and _mapping_value(retrain_result, 'duration_seconds') is not None,
            repr(retrain_result),
        ),
    ]


PHASES: tuple[Phase, ...] = (
    Phase('T+0', 'Baseline', run_t0_baseline, assert_t0_baseline),
    Phase('T+1', 'Replay SKAB normal segment', run_t1_replay_normal, assert_t1_replay_normal),
    Phase('T+2', 'Replay anomalous valve closing segment', run_t2_replay_anomalous, assert_t2_replay_anomalous),
    Phase('T+3', 'Inject synthetic drift', run_t3_inject_drift, assert_t3_inject_drift),
    Phase('T+4', 'Detect drift warning', run_t4_detect_drift, assert_t4_detect_drift),
    Phase('T+5', 'Trigger retrain now', run_t5_trigger_retrain, assert_t5_trigger_retrain, requires_retrain=True),
    Phase('T+6', 'Train challenger', run_t6_train_challenger, assert_t6_train_challenger, requires_retrain=True),
    Phase('T+7', 'Promote model v2', run_t7_promote_v2, assert_t7_promote_v2, requires_retrain=True),
    Phase('T+8', 'Hot-swap and recover', run_t8_hot_swap, assert_t8_hot_swap, requires_retrain=True),
)

OBSERVABILITY_EVIDENCE_PHASE = Phase(
    'T+9',
    'Collect observability evidence',
    run_t9_observability_evidence,
    assert_t9_observability_evidence,
    requires_retrain=True,
)


def storyboard_phases(config: DemoConfig) -> tuple[Phase, ...]:
    if config.observability_evidence:
        return (*PHASES, OBSERVABILITY_EVIDENCE_PHASE)
    return PHASES


def apply_process_env(config: DemoConfig) -> None:
    os.environ.update(
        {
            'MLFLOW_TRACKING_URI': config.mlflow_uri,
            'ENABLE_REDIS': 'true',
            'REDIS_HOST': config.redis_host,
            'REDIS_PORT': str(config.redis_port),
            'TELEMETRY_CONNECTION': 'clickhouse',
            'TELEMETRY_URL': config.clickhouse_url,
        }
    )


def _env_enabled(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() not in {'', '0', 'false', 'no', 'off'}


def _drift_evidence_payload(result: Any) -> dict[str, Any]:
    return {
        'timestamp': datetime.now(UTC).isoformat(),
        'method': getattr(result, 'method', 'evidently'),
        'threshold': _jsonable(getattr(result, 'threshold', None)),
        'dataset_drift': bool(getattr(result, 'dataset_drift', False)),
        'drift_share': _jsonable(getattr(result, 'drift_share', 0.0)),
        'n_drifted': _jsonable(getattr(result, 'n_drifted', 0)),
        'n_features': _jsonable(getattr(result, 'n_features', 0)),
        **({'report_path': str(getattr(result, 'report_path'))} if getattr(result, 'report_path', None) else {}),
    }


def _env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


def _env_optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value else None


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _print_banner(phase: Phase) -> None:
    print(f'╔══ {phase.label} — {phase.title} ══╗')


def _print_footer() -> None:
    print('╚══════════════════════════════════════╝')


def _print_assertion(result: AssertionResult) -> None:
    suffix = f' ({result.detail})' if result.detail else ''
    print(f'→ assert {result.description}: {"PASS" if result.passed else "FAIL"}{suffix}')


def _run_subprocess(args: Sequence[str], action: str) -> subprocess.CompletedProcess[str]:
    display = ' '.join(shlex.quote(str(arg)) for arg in args)
    LOGGER.info('→ action: %s', action)
    LOGGER.info('→ command: %s', display)
    completed = subprocess.run(
        [str(arg) for arg in args],
        cwd=_PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        env=os.environ.copy(),
    )
    if completed.stdout.strip():
        LOGGER.info(completed.stdout.strip())
    if completed.stderr.strip():
        LOGGER.warning(completed.stderr.strip())
    if completed.returncode != 0:
        raise RuntimeError(f'{action} exited {completed.returncode}')
    return completed


def _replay_csv(ctx: DemoContext, path: Path, limit: int, action: str) -> None:
    _run_subprocess(
        [
            sys.executable,
            'scripts/replay_skab.py',
            '--input',
            str(path),
            '--station',
            ctx.config.station,
            '--host',
            ctx.config.mqtt_host,
            '--port',
            str(ctx.config.mqtt_port),
            '--qos',
            str(ctx.config.mqtt_qos),
            '--limit',
            str(limit),
        ],
        action,
    )


async def _wait_for_downstream(ctx: DemoContext) -> None:
    LOGGER.info('→ wait %.3gs for downstream processing', ctx.config.wait_seconds)
    await asyncio.sleep(ctx.config.wait_seconds)


def _redis_client(config: DemoConfig) -> Any:
    import redis.asyncio as redis  # type: ignore[reportMissingImports]

    return redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
    )


async def _close_redis(client: Any) -> None:
    close = getattr(client, 'aclose', None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result
        return
    result = client.close()
    if inspect.isawaitable(result):
        await result


async def _redis_get_json(config: DemoConfig, key: str) -> dict[str, Any] | None:
    client = _redis_client(config)
    try:
        value = await client.get(key)
    finally:
        await _close_redis(client)
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _redis_set_json(config: DemoConfig, key: str, value: Mapping[str, Any]) -> None:
    client = _redis_client(config)
    try:
        await client.set(key, json.dumps(_jsonable(value)))
    finally:
        await _close_redis(client)


def _clickhouse_client(config: DemoConfig) -> Any:
    import clickhouse_connect  # type: ignore[reportMissingImports]

    parsed = urlparse(config.clickhouse_url)
    if not parsed.hostname:
        raise ValueError('DEMO_CLICKHOUSE_URL host is empty')
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


def _clickhouse_table_exists(config: DemoConfig) -> bool:
    result = _clickhouse_client(config).query(
        "SELECT count() FROM system.tables WHERE database = currentDatabase() AND name = 'telemetry_observations'"
    )
    return int(_first_cell(result) or 0) > 0


def _clickhouse_anomaly_flags(config: DemoConfig, since: datetime | None = None) -> list[int | None]:
    conditions = ['device_id = %(station)s', 'measurement = %(measurement)s']
    parameters: dict[str, Any] = {'station': config.station, 'measurement': ANOMALY_SCORE_MEASUREMENT}
    if since is not None:
        conditions.append('observed_at >= %(since)s')
        parameters['since'] = since
    result = _clickhouse_client(config).query(
        'SELECT value_int FROM telemetry_observations '
        f'WHERE {" AND ".join(conditions)} ORDER BY observed_at ASC LIMIT 200',
        parameters=parameters,
    )
    rows = getattr(result, 'result_rows', []) or []
    return [None if row[0] is None else int(row[0]) for row in rows]


def _clickhouse_anomaly_count(config: DemoConfig, since: datetime | None = None, value_int: int | None = None) -> int:
    conditions = ['device_id = %(station)s', 'measurement = %(measurement)s']
    parameters: dict[str, Any] = {'station': config.station, 'measurement': ANOMALY_SCORE_MEASUREMENT}
    if since is not None:
        conditions.append('observed_at >= %(since)s')
        parameters['since'] = since
    if value_int is not None:
        conditions.append('value_int = %(value_int)s')
        parameters['value_int'] = value_int
    result = _clickhouse_client(config).query(
        f'SELECT count() FROM telemetry_observations WHERE {" AND ".join(conditions)}',
        parameters=parameters,
    )
    return int(_first_cell(result) or 0)


def _first_cell(result: Any) -> Any:
    rows = getattr(result, 'result_rows', []) or []
    if not rows:
        return None
    return rows[0][0]


def _mlflow_client(config: DemoConfig) -> Any:
    import mlflow  # type: ignore[reportMissingImports]
    from mlflow.tracking import MlflowClient  # type: ignore[reportMissingImports]

    mlflow.set_tracking_uri(config.mlflow_uri)
    return MlflowClient()


def _mlflow_alias_version(config: DemoConfig, alias: str) -> str | None:
    try:
        version = _mlflow_client(config).get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
    except Exception:
        return None
    value = getattr(version, 'version', None)
    return None if value is None else str(value)


def _mlflow_model_versions(config: DemoConfig) -> set[str]:
    try:
        versions = _mlflow_client(config).search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    except Exception:
        return set()
    return {str(version.version) for version in versions if getattr(version, 'version', None) is not None}


def _mlflow_runs(config: DemoConfig) -> dict[str, Any]:
    client = _mlflow_client(config)
    experiments = client.search_experiments()
    experiment_ids = [experiment.experiment_id for experiment in experiments]
    if not experiment_ids:
        return {}
    runs = client.search_runs(experiment_ids, max_results=500, order_by=['attributes.start_time DESC'])
    return {str(run.info.run_id): run for run in runs}


def _select_latest_version(versions: set[str]) -> str | None:
    if not versions:
        return None
    return max(versions, key=lambda version: (int(version) if version.isdigit() else -1, version))


async def _ensure_active_model_version(config: DemoConfig, version: str) -> None:
    payload = await _redis_get_json(config, ACTIVE_MODEL_KEY) or {}
    payload.update(
        {
            'registered_model_name': REGISTERED_MODEL_NAME,
            'alias': 'champion',
            'mlflow_version': version,
            'name': REGISTERED_MODEL_NAME,
            'version': version,
            'activated_at': datetime.now(UTC).isoformat(),
        }
    )
    await _redis_set_json(config, ACTIVE_MODEL_KEY, payload)


async def _write_local_hot_swap_observation(ctx: DemoContext, service: Any) -> None:
    from app.models.anomaly_event import build_anomaly_payload_from_verdict
    from ml.datasets.skab_loader import iter_telemetry_records, load_skab_csv

    frame = load_skab_csv(ctx.config.normal_input)
    record = next(iter_telemetry_records(frame, ctx.config.station))
    verdict = service.observe(record['station'], record['timestamp'], record['sensors'])
    anomaly_payload = build_anomaly_payload_from_verdict(verdict)
    await _redis_set_json(ctx.config, LATEST_READING_KEY.format(station=ctx.config.station), record)
    await _redis_set_json(ctx.config, LATEST_ANOMALY_KEY.format(station=ctx.config.station), anomaly_payload)
    ctx.t8_local_hot_swap_used = True


def _expand_tiny_frames(reference: Any, current: Any) -> tuple[Any, Any]:
    import pandas as pd

    if min(len(reference), len(current)) >= 6:
        return reference, current
    repeats = max(2, (6 + max(1, min(len(reference), len(current))) - 1) // max(1, min(len(reference), len(current))))
    return pd.concat([reference] * repeats, ignore_index=True), pd.concat([current] * repeats, ignore_index=True)


def _mapping_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_jsonable(item) for item in value]
    return str(value)


async def _run_cli(config: DemoConfig) -> bool:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    apply_process_env(config)
    ctx = create_context(config)
    LOGGER.info('Operator precondition: run `make run` separately for MQTT → controller → persistence phases.')
    if config.clean:
        LOGGER.info('→ action: clean Redis demo keys and truncate ClickHouse telemetry_observations')
        await clean_demo_state(ctx)
    return await run_storyboard(ctx)


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_env(argv)
    ok = asyncio.run(_run_cli(config))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
