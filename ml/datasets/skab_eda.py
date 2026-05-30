from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from ml.datasets.skab_loader import SENSOR_COLUMNS, load_skab_csv
from ml.datasets.skab_manifest import load_skab_split_manifest


@dataclass
class SkabEdaConfig:
    plots_enabled: bool = False


@dataclass
class SkabEdaResult:
    row_count: int = 0
    file_count: int = 0
    sensor_columns: list[str] = field(default_factory=list)
    anomaly_count: int = 0
    changepoint_count: int = 0
    missing_counts: dict[str, int] = field(default_factory=dict)
    time_range: dict[str, str] = field(default_factory=dict)
    sensor_statistics: dict[str, dict[str, float]] = field(default_factory=dict)
    plots_available: bool = False
    source: str = ''

    def to_payload(self) -> dict:
        return asdict(self)


def _compute_sensor_statistics(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats = {}
    for col in SENSOR_COLUMNS:
        if col in frame.columns:
            series = frame[col]
            stats[col] = {
                'mean': float(series.mean()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
            }
    return stats


def _compute_missing_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {col: int(frame[col].isna().sum()) for col in frame.columns}


def _compute_time_range(frame: pd.DataFrame) -> dict[str, str]:
    if 'datetime' not in frame.columns or frame.empty:
        return {}
    return {
        'start': str(frame['datetime'].iloc[0]),
        'end': str(frame['datetime'].iloc[-1]),
    }


def summarize_skab_csv(path: Path) -> SkabEdaResult:
    frame = load_skab_csv(path)
    return SkabEdaResult(
        row_count=len(frame),
        file_count=1,
        sensor_columns=list(SENSOR_COLUMNS),
        anomaly_count=int(frame['anomaly'].sum()),
        changepoint_count=int(frame['changepoint'].sum()),
        missing_counts=_compute_missing_counts(frame),
        time_range=_compute_time_range(frame),
        sensor_statistics=_compute_sensor_statistics(frame),
        plots_available=False,
        source=str(path),
    )


def _aggregate_results(results: list[SkabEdaResult]) -> SkabEdaResult:
    if not results:
        return SkabEdaResult(
            row_count=0,
            file_count=0,
            sensor_columns=list(SENSOR_COLUMNS),
            anomaly_count=0,
            changepoint_count=0,
            missing_counts={col: 0 for col in ['datetime', *SENSOR_COLUMNS, 'anomaly', 'changepoint']},
            time_range={},
            sensor_statistics={},
            plots_available=False,
            source='',
        )

    total_rows = sum(r.row_count for r in results)
    total_files = sum(r.file_count for r in results)
    total_anomalies = sum(r.anomaly_count for r in results)
    total_changepoints = sum(r.changepoint_count for r in results)

    missing_counts: dict[str, int] = {}
    for r in results:
        for col, count in r.missing_counts.items():
            missing_counts[col] = missing_counts.get(col, 0) + count

    time_starts = [r.time_range['start'] for r in results if r.time_range.get('start')]
    time_ends = [r.time_range['end'] for r in results if r.time_range.get('end')]
    time_range = {}
    if time_starts:
        time_range['start'] = min(time_starts)
    if time_ends:
        time_range['end'] = max(time_ends)

    frames = []
    for r in results:
        if r.source:
            frames.append(load_skab_csv(Path(r.source)))
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        sensor_statistics = _compute_sensor_statistics(combined)
    else:
        sensor_statistics = {}

    return SkabEdaResult(
        row_count=total_rows,
        file_count=total_files,
        sensor_columns=list(SENSOR_COLUMNS),
        anomaly_count=total_anomalies,
        changepoint_count=total_changepoints,
        missing_counts=missing_counts,
        time_range=time_range,
        sensor_statistics=sensor_statistics,
        plots_available=False,
        source='',
    )


def summarize_skab_manifest(manifest_path: Path) -> dict[str, SkabEdaResult]:
    manifest = load_skab_split_manifest(manifest_path)
    results = {}
    for split_name, paths in [
        ('train', manifest.train),
        ('validation', manifest.validation),
        ('test', manifest.test),
    ]:
        split_results = [summarize_skab_csv(p) for p in paths]
        aggregated = _aggregate_results(split_results)
        aggregated.source = f"manifest:{manifest_path.name} split={split_name}"
        results[split_name] = aggregated
    return results


def write_skab_eda_report(result: SkabEdaResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / 'summary.json'
    payload = result.to_payload()
    summary_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2),
        encoding='utf-8',
    )

    report_path = output_dir / 'report.md'
    lines = [
        '# SKAB EDA Report',
        '',
        f"**Source:** {result.source}",
        '',
        f"**Row count:** {result.row_count}",
        f"**File count:** {result.file_count}",
        f"**Anomalies:** {result.anomaly_count}",
        f"**Changepoints:** {result.changepoint_count}",
        '',
    ]

    if result.time_range:
        lines.append(
            f"**Time range:** {result.time_range.get('start', 'N/A')} to {result.time_range.get('end', 'N/A')}"
        )
        lines.append('')

    lines.append('## Sensor Statistics')
    lines.append('')
    lines.append('| Sensor | Mean | Std | Min | Max |')
    lines.append('| --- | --- | --- | --- | --- |')
    for col in result.sensor_columns:
        stats = result.sensor_statistics.get(col, {})
        mean = stats.get('mean', 0.0)
        std = stats.get('std', 0.0)
        min_val = stats.get('min', 0.0)
        max_val = stats.get('max', 0.0)
        lines.append(f"| {col} | {mean:.4f} | {std:.4f} | {min_val:.4f} | {max_val:.4f} |")
    lines.append('')

    lines.append(f"**Plots available:** {result.plots_available}")
    lines.append('')

    report_path.write_text('\n'.join(lines), encoding='utf-8')


def generate_skab_eda_report(
    input_path: Path | None,
    split_manifest_path: Path | None,
    output_dir: Path,
    include_plots: bool = True,
) -> Mapping[str, Path | str]:
    """Generate JSON and Markdown EDA artifacts for a CSV input and/or split manifest."""

    artifacts: dict[str, Path | str] = {}

    if input_path is not None:
        result = summarize_skab_csv(input_path)
        write_skab_eda_report(result, output_dir)
        artifacts['input_summary_json'] = output_dir / 'summary.json'
        artifacts['input_report_md'] = output_dir / 'report.md'

    if split_manifest_path is not None:
        split_results = summarize_skab_manifest(split_manifest_path)
        for split_name, result in split_results.items():
            split_dir = output_dir / split_name
            write_skab_eda_report(result, split_dir)
            artifacts[f'{split_name}_summary_json'] = split_dir / 'summary.json'
            artifacts[f'{split_name}_report_md'] = split_dir / 'report.md'

    if include_plots:
        artifacts['plots_note'] = 'plots unavailable'

    return artifacts
