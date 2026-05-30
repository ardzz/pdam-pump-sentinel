from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

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
    missing_rates: dict[str, float] = field(default_factory=dict)
    time_range: dict[str, str] = field(default_factory=dict)
    sensor_statistics: dict[str, dict[str, Any]] = field(default_factory=dict)
    correlation_matrix: dict[str, dict[str, float | None]] = field(default_factory=dict)
    sensor_distributions: dict[str, dict[str, Any]] = field(default_factory=dict)
    rolling_statistics: dict[str, dict[str, Any]] = field(default_factory=dict)
    timestamp_quality: dict[str, Any] = field(default_factory=dict)
    constant_column_flags: dict[str, bool] = field(default_factory=dict)
    label_ranges: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    label_overlay: list[dict[str, Any]] = field(default_factory=list)
    changepoint_timestamps: list[str] = field(default_factory=list)
    split_aggregation: dict[str, Any] = field(default_factory=dict)
    plots_available: bool = False
    source: str = ''

    def to_payload(self) -> dict:
        return asdict(self)


_BASE_COLUMNS = ['datetime', *SENSOR_COLUMNS, 'anomaly', 'changepoint']
_QUANTILE_KEYS = ('p01', 'p05', 'p25', 'p50', 'p75', 'p95', 'p99')
_ROLLING_WINDOW = 3

_ARTIFACT_FILENAMES = {
    'summary_json': 'summary.json',
    'report_md': 'report.md',
    'sensor_statistics_csv': 'sensor_statistics.csv',
    'missingness_csv': 'missingness.csv',
    'timestamp_quality_json': 'timestamp_quality.json',
    'label_ranges_csv': 'label_ranges.csv',
    'correlation_matrix_csv': 'correlation_matrix.csv',
    'sensor_distributions_csv': 'sensor_distributions.csv',
    'rolling_statistics_csv': 'rolling_statistics.csv',
    'label_overlay_csv': 'label_overlay.csv',
}

_PLOT_ARTIFACT_FILENAMES = {
    'label_overlay_plot_json': 'label_overlay_plot.json',
    'label_overlay_plot_html': 'label_overlay_plot.html',
}

_SENSOR_STAT_COLUMNS = [
    'sensor',
    'count',
    'missing_count',
    'missing_rate',
    'mean',
    'std',
    'min',
    'p01',
    'p05',
    'p25',
    'p50',
    'p75',
    'p95',
    'p99',
    'max',
    'is_constant',
]

_SENSOR_DISTRIBUTION_COLUMNS = [
    'sensor',
    'count',
    'missing_count',
    'missing_rate',
    'min',
    *_QUANTILE_KEYS,
    'max',
    'iqr',
]

_ROLLING_STAT_COLUMNS = [
    'sensor',
    'window',
    'valid_points',
    'rolling_mean_min',
    'rolling_mean_max',
    'rolling_mean_last',
    'rolling_std_min',
    'rolling_std_max',
    'rolling_std_mean',
    'rolling_std_last',
]


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _json_float(value: Any) -> float | None:
    if _is_missing_scalar(value):
        return None
    return float(value)


def _format_timestamp(value: Any) -> str | None:
    if _is_missing_scalar(value):
        return None
    timestamp: Any = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.isoformat()
    return timestamp.isoformat().replace('+00:00', 'Z')


def _timestamp_text(value: Any) -> str:
    try:
        formatted = _format_timestamp(value)
    except (TypeError, ValueError):
        return str(value)
    return str(value) if formatted is None else formatted


def _report_number(value: Any) -> str:
    if _is_missing_scalar(value):
        return 'N/A'
    return f'{float(value):.4f}'


def _numeric_sensor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = {}
    for col in SENSOR_COLUMNS:
        if col in frame.columns:
            data[col] = cast(pd.Series, pd.to_numeric(frame[col], errors='coerce'))
    return pd.DataFrame(data, index=frame.index)


def _last_valid(series: pd.Series) -> float | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return _json_float(valid.iloc[-1])


def _compute_sensor_statistics(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    stats = {}
    for col in SENSOR_COLUMNS:
        if col in frame.columns:
            series = cast(pd.Series, pd.to_numeric(frame[col], errors='coerce'))
            quantiles = series.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
            stats[col] = {
                'count': int(series.count()),
                'missing_count': int(series.isna().sum()),
                'missing_rate': _missing_rate(int(series.isna().sum()), len(frame)),
                'mean': _json_float(series.mean()),
                'std': _json_float(series.std()),
                'min': _json_float(series.min()),
                'p01': _json_float(quantiles.loc[0.01]),
                'p05': _json_float(quantiles.loc[0.05]),
                'p25': _json_float(quantiles.loc[0.25]),
                'p50': _json_float(quantiles.loc[0.50]),
                'p75': _json_float(quantiles.loc[0.75]),
                'p95': _json_float(quantiles.loc[0.95]),
                'p99': _json_float(quantiles.loc[0.99]),
                'max': _json_float(series.max()),
            }
    return stats


def _compute_correlation_matrix(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    numeric = _numeric_sensor_frame(frame)
    correlation = numeric.corr() if not numeric.empty else pd.DataFrame()
    matrix: dict[str, dict[str, float | None]] = {}
    for row_sensor in SENSOR_COLUMNS:
        matrix[row_sensor] = {}
        for col_sensor in SENSOR_COLUMNS:
            if row_sensor in correlation.index and col_sensor in correlation.columns:
                matrix[row_sensor][col_sensor] = _json_float(correlation.loc[row_sensor, col_sensor])
            else:
                matrix[row_sensor][col_sensor] = None
    return matrix


def _compute_sensor_distributions(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    stats = _compute_sensor_statistics(frame)
    distributions: dict[str, dict[str, Any]] = {}
    for sensor, sensor_stats in stats.items():
        quantiles = {key: sensor_stats.get(key) for key in _QUANTILE_KEYS}
        p25 = quantiles.get('p25')
        p75 = quantiles.get('p75')
        distributions[sensor] = {
            'count': sensor_stats.get('count', 0),
            'missing_count': sensor_stats.get('missing_count', 0),
            'missing_rate': sensor_stats.get('missing_rate', 0.0),
            'min': sensor_stats.get('min'),
            'max': sensor_stats.get('max'),
            'quantiles': quantiles,
            'iqr': None if p25 is None or p75 is None else float(p75 - p25),
        }
    return distributions


def _compute_rolling_statistics(frame: pd.DataFrame, window: int = _ROLLING_WINDOW) -> dict[str, dict[str, Any]]:
    numeric = _numeric_sensor_frame(frame)
    rolling: dict[str, dict[str, Any]] = {}
    for sensor in SENSOR_COLUMNS:
        if sensor not in numeric.columns:
            continue
        series = numeric[sensor]
        rolling_mean = cast(pd.Series, series.rolling(window=window, min_periods=1).mean())
        rolling_std = cast(pd.Series, series.rolling(window=window, min_periods=2).std())
        rolling[sensor] = {
            'window': window,
            'valid_points': int(series.count()),
            'rolling_mean_min': _json_float(rolling_mean.min()),
            'rolling_mean_max': _json_float(rolling_mean.max()),
            'rolling_mean_last': _last_valid(rolling_mean),
            'rolling_std_min': _json_float(rolling_std.min()),
            'rolling_std_max': _json_float(rolling_std.max()),
            'rolling_std_mean': _json_float(rolling_std.mean()),
            'rolling_std_last': _last_valid(rolling_std),
        }
    return rolling


def _compute_missing_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {col: int(frame[col].isna().sum()) for col in frame.columns}


def _missing_rate(missing_count: int, row_count: int) -> float:
    if row_count == 0:
        return 0.0
    return float(missing_count / row_count)


def _compute_missing_rates(frame: pd.DataFrame) -> dict[str, float]:
    row_count = len(frame)
    return {col: _missing_rate(int(frame[col].isna().sum()), row_count) for col in frame.columns}


def _compute_time_range(frame: pd.DataFrame) -> dict[str, str]:
    if 'datetime' not in frame.columns or frame.empty:
        return {}
    return {
        'start': _timestamp_text(frame['datetime'].iloc[0]),
        'end': _timestamp_text(frame['datetime'].iloc[-1]),
    }


def _compute_timestamp_quality(frame: pd.DataFrame) -> dict[str, Any]:
    base: dict[str, Any] = {
        'column': 'datetime',
        'present': 'datetime' in frame.columns,
        'row_count': len(frame),
        'parsed_datetime_count': 0,
        'invalid_timestamp_count': 0,
        'duplicate_timestamp_count': 0,
        'is_monotonic_increasing': True,
        'is_monotonic_decreasing': True,
        'is_sorted_ascending': True,
        'first_valid_timestamp': None,
        'last_valid_timestamp': None,
        'cadence_mode_seconds': None,
        'max_gap_seconds': None,
    }
    if 'datetime' not in frame.columns:
        return base

    parsed = pd.to_datetime(frame['datetime'], errors='coerce', utc=True)
    valid = parsed.dropna()
    base['parsed_datetime_count'] = int(valid.shape[0])
    base['invalid_timestamp_count'] = int(parsed.isna().sum())
    base['duplicate_timestamp_count'] = int(valid.duplicated().sum())
    base['is_monotonic_increasing'] = bool(valid.is_monotonic_increasing)
    base['is_monotonic_decreasing'] = bool(valid.is_monotonic_decreasing)
    sorted_valid = valid.sort_values(kind='mergesort').reset_index(drop=True)
    base['is_sorted_ascending'] = bool(valid.reset_index(drop=True).equals(sorted_valid))

    if not valid.empty:
        base['first_valid_timestamp'] = _format_timestamp(valid.min())
        base['last_valid_timestamp'] = _format_timestamp(valid.max())

    unique_sorted = valid.drop_duplicates().sort_values(kind='mergesort')
    if unique_sorted.shape[0] >= 2:
        gaps = unique_sorted.diff().dropna().dt.total_seconds()
        positive_gaps = gaps[gaps > 0]
        if not positive_gaps.empty:
            modes = positive_gaps.mode()
            base['cadence_mode_seconds'] = float(modes.iloc[0])
            base['max_gap_seconds'] = float(positive_gaps.max())
    return base


def _compute_constant_column_flags(frame: pd.DataFrame) -> dict[str, bool]:
    flags = {}
    for col in frame.columns:
        series = frame[col].dropna()
        flags[col] = bool(len(series) > 0 and series.nunique(dropna=True) <= 1)
    return flags


def _compute_label_ranges(frame: pd.DataFrame, label_col: str) -> list[dict[str, Any]]:
    if label_col not in frame.columns:
        return []

    label_values = cast(pd.Series, pd.to_numeric(frame[label_col], errors='coerce'))
    values = label_values.fillna(0).astype(int).tolist()
    datetimes = [_timestamp_text(value) for value in frame['datetime'].tolist()] if 'datetime' in frame.columns else [''] * len(frame)
    ranges: list[dict[str, Any]] = []
    start: int | None = None

    for idx, value in enumerate(values):
        if value == 1 and start is None:
            start = idx
        if start is not None and (value != 1 or idx == len(values) - 1):
            end = idx if value == 1 and idx == len(values) - 1 else idx - 1
            ranges.append(
                {
                    'start_index': int(start),
                    'end_index': int(end),
                    'length': int(end - start + 1),
                    'start_timestamp': datetimes[start],
                    'end_timestamp': datetimes[end],
                }
            )
            start = None

    return ranges


def _compute_all_label_ranges(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    return {
        'anomaly': _compute_label_ranges(frame, 'anomaly'),
        'changepoint': _compute_label_ranges(frame, 'changepoint'),
    }


def _compute_changepoint_timestamps(frame: pd.DataFrame) -> list[str]:
    if 'changepoint' not in frame.columns or 'datetime' not in frame.columns:
        return []
    changepoints = cast(pd.Series, pd.to_numeric(frame['changepoint'], errors='coerce'))
    mask = changepoints.fillna(0).astype(int) == 1
    return [_timestamp_text(value) for value in frame.loc[mask, 'datetime'].tolist()]


def _compute_label_overlay(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    datetimes = [_timestamp_text(value) for value in frame['datetime'].tolist()] if 'datetime' in frame.columns else [''] * len(frame)
    anomalies = (
        cast(pd.Series, pd.to_numeric(frame['anomaly'], errors='coerce')).fillna(0).astype(int)
        if 'anomaly' in frame.columns
        else pd.Series([0] * len(frame), index=frame.index)
    )
    changepoints = (
        cast(pd.Series, pd.to_numeric(frame['changepoint'], errors='coerce')).fillna(0).astype(int)
        if 'changepoint' in frame.columns
        else pd.Series([0] * len(frame), index=frame.index)
    )
    numeric = _numeric_sensor_frame(frame)

    overlay = []
    for idx in range(len(frame)):
        anomaly = int(anomalies.iloc[idx])
        changepoint = int(changepoints.iloc[idx])
        if anomaly and changepoint:
            label = 'anomaly_changepoint'
        elif anomaly:
            label = 'anomaly'
        elif changepoint:
            label = 'changepoint'
        else:
            label = 'normal'
        row: dict[str, Any] = {
            'row_index': idx,
            'timestamp': datetimes[idx],
            'anomaly': anomaly,
            'changepoint': changepoint,
            'label': label,
        }
        for sensor in SENSOR_COLUMNS:
            row[sensor] = _json_float(numeric[sensor].iloc[idx]) if sensor in numeric.columns else None
        overlay.append(row)
    return overlay


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=cast(Any, _BASE_COLUMNS))


def _load_skab_csv_for_eda(path: Path) -> pd.DataFrame:
    try:
        return load_skab_csv(path)
    except (TypeError, ValueError):
        frame = pd.read_csv(path, sep=';')
        for col in ('anomaly', 'changepoint'):
            if col not in frame.columns:
                frame[col] = 0
        return frame


def _artifact_paths(output_dir: Path, include_plots: bool = False) -> dict[str, Path]:
    filenames = dict(_ARTIFACT_FILENAMES)
    if include_plots:
        filenames.update(_PLOT_ARTIFACT_FILENAMES)
    return {key: output_dir / filename for key, filename in filenames.items()}


def _add_artifact_paths(
    artifacts: dict[str, Path | str], prefix: str, output_dir: Path, include_plots: bool = False
) -> None:
    for key, path in _artifact_paths(output_dir, include_plots=include_plots).items():
        artifacts[f'{prefix}_{key}'] = path


def _compute_split_aggregation(
    split_name: str,
    paths: list[Path],
    split_results: list[SkabEdaResult],
    aggregated: SkabEdaResult,
) -> dict[str, Any]:
    return {
        'split': split_name,
        'source_files': [str(path) for path in paths],
        'file_count': aggregated.file_count,
        'row_count': aggregated.row_count,
        'anomaly_count': aggregated.anomaly_count,
        'changepoint_count': aggregated.changepoint_count,
        'rows_by_file': {result.source: result.row_count for result in split_results},
        'anomalies_by_file': {result.source: result.anomaly_count for result in split_results},
        'changepoints_by_file': {result.source: result.changepoint_count for result in split_results},
    }


def summarize_skab_csv(path: Path) -> SkabEdaResult:
    frame = _load_skab_csv_for_eda(path)
    return SkabEdaResult(
        row_count=len(frame),
        file_count=1,
        sensor_columns=list(SENSOR_COLUMNS),
        anomaly_count=int(frame['anomaly'].sum()),
        changepoint_count=int(frame['changepoint'].sum()),
        missing_counts=_compute_missing_counts(frame),
        missing_rates=_compute_missing_rates(frame),
        time_range=_compute_time_range(frame),
        sensor_statistics=_compute_sensor_statistics(frame),
        correlation_matrix=_compute_correlation_matrix(frame),
        sensor_distributions=_compute_sensor_distributions(frame),
        rolling_statistics=_compute_rolling_statistics(frame),
        timestamp_quality=_compute_timestamp_quality(frame),
        constant_column_flags=_compute_constant_column_flags(frame),
        label_ranges=_compute_all_label_ranges(frame),
        label_overlay=_compute_label_overlay(frame),
        changepoint_timestamps=_compute_changepoint_timestamps(frame),
        plots_available=False,
        source=str(path),
    )


def _aggregate_results(results: list[SkabEdaResult]) -> SkabEdaResult:
    if not results:
        frame = _empty_frame()
        return SkabEdaResult(
            row_count=0,
            file_count=0,
            sensor_columns=list(SENSOR_COLUMNS),
            anomaly_count=0,
            changepoint_count=0,
            missing_counts={col: 0 for col in ['datetime', *SENSOR_COLUMNS, 'anomaly', 'changepoint']},
            missing_rates={col: 0.0 for col in ['datetime', *SENSOR_COLUMNS, 'anomaly', 'changepoint']},
            time_range={},
            sensor_statistics={},
            correlation_matrix=_compute_correlation_matrix(frame),
            sensor_distributions=_compute_sensor_distributions(frame),
            rolling_statistics=_compute_rolling_statistics(frame),
            timestamp_quality=_compute_timestamp_quality(frame),
            constant_column_flags=_compute_constant_column_flags(frame),
            label_ranges=_compute_all_label_ranges(frame),
            label_overlay=[],
            changepoint_timestamps=[],
            split_aggregation={},
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
    missing_rates = {col: _missing_rate(count, total_rows) for col, count in missing_counts.items()}

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
            frames.append(_load_skab_csv_for_eda(Path(r.source)))
    combined = pd.concat(frames, ignore_index=True) if frames else _empty_frame()
    sensor_statistics = _compute_sensor_statistics(combined) if frames else {}

    return SkabEdaResult(
        row_count=total_rows,
        file_count=total_files,
        sensor_columns=list(SENSOR_COLUMNS),
        anomaly_count=total_anomalies,
        changepoint_count=total_changepoints,
        missing_counts=missing_counts,
        missing_rates=missing_rates,
        time_range=time_range,
        sensor_statistics=sensor_statistics,
        correlation_matrix=_compute_correlation_matrix(combined),
        sensor_distributions=_compute_sensor_distributions(combined),
        rolling_statistics=_compute_rolling_statistics(combined),
        timestamp_quality=_compute_timestamp_quality(combined),
        constant_column_flags=_compute_constant_column_flags(combined),
        label_ranges=_compute_all_label_ranges(combined),
        label_overlay=_compute_label_overlay(combined),
        changepoint_timestamps=_compute_changepoint_timestamps(combined),
        split_aggregation={},
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
        aggregated.split_aggregation = _compute_split_aggregation(split_name, paths, split_results, aggregated)
        results[split_name] = aggregated
    return results


def _write_sensor_statistics_csv(result: SkabEdaResult, path: Path) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=_SENSOR_STAT_COLUMNS)
        writer.writeheader()
        for sensor in result.sensor_columns:
            stats = result.sensor_statistics.get(sensor, {})
            writer.writerow(
                {
                    'sensor': sensor,
                    'count': stats.get('count', ''),
                    'missing_count': stats.get('missing_count', ''),
                    'missing_rate': stats.get('missing_rate', ''),
                    'mean': stats.get('mean', ''),
                    'std': stats.get('std', ''),
                    'min': stats.get('min', ''),
                    'p01': stats.get('p01', ''),
                    'p05': stats.get('p05', ''),
                    'p25': stats.get('p25', ''),
                    'p50': stats.get('p50', ''),
                    'p75': stats.get('p75', ''),
                    'p95': stats.get('p95', ''),
                    'p99': stats.get('p99', ''),
                    'max': stats.get('max', ''),
                    'is_constant': result.constant_column_flags.get(sensor, False),
                }
            )


def _write_missingness_csv(result: SkabEdaResult, path: Path) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['column', 'missing_count', 'missing_rate'])
        writer.writeheader()
        for column in result.missing_counts:
            writer.writerow(
                {
                    'column': column,
                    'missing_count': result.missing_counts[column],
                    'missing_rate': result.missing_rates.get(column, 0.0),
                }
            )


def _write_label_ranges_csv(result: SkabEdaResult, path: Path) -> None:
    fieldnames = ['label', 'start_index', 'end_index', 'length', 'start_timestamp', 'end_timestamp']
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label in ('anomaly', 'changepoint'):
            for label_range in result.label_ranges.get(label, []):
                writer.writerow({'label': label, **label_range})


def _write_correlation_matrix_csv(result: SkabEdaResult, path: Path) -> None:
    fieldnames = ['sensor', *result.sensor_columns]
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_sensor in result.sensor_columns:
            correlations = result.correlation_matrix.get(row_sensor, {})
            writer.writerow({'sensor': row_sensor, **{col: correlations.get(col) for col in result.sensor_columns}})


def _write_sensor_distributions_csv(result: SkabEdaResult, path: Path) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=_SENSOR_DISTRIBUTION_COLUMNS)
        writer.writeheader()
        for sensor in result.sensor_columns:
            distribution = result.sensor_distributions.get(sensor, {})
            quantiles = distribution.get('quantiles', {})
            writer.writerow(
                {
                    'sensor': sensor,
                    'count': distribution.get('count', ''),
                    'missing_count': distribution.get('missing_count', ''),
                    'missing_rate': distribution.get('missing_rate', ''),
                    'min': distribution.get('min', ''),
                    **{key: quantiles.get(key, '') for key in _QUANTILE_KEYS},
                    'max': distribution.get('max', ''),
                    'iqr': distribution.get('iqr', ''),
                }
            )


def _write_rolling_statistics_csv(result: SkabEdaResult, path: Path) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=_ROLLING_STAT_COLUMNS)
        writer.writeheader()
        for sensor in result.sensor_columns:
            stats = result.rolling_statistics.get(sensor, {})
            writer.writerow({'sensor': sensor, **{col: stats.get(col, '') for col in _ROLLING_STAT_COLUMNS if col != 'sensor'}})


def _write_label_overlay_csv(result: SkabEdaResult, path: Path) -> None:
    fieldnames = ['row_index', 'timestamp', 'anomaly', 'changepoint', 'label', *result.sensor_columns]
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.label_overlay:
            writer.writerow({field: row.get(field, '') for field in fieldnames})


def _label_overlay_plot_payload(result: SkabEdaResult) -> dict[str, Any]:
    primary_sensor = result.sensor_columns[0] if result.sensor_columns else ''
    timestamps = [row.get('timestamp', '') for row in result.label_overlay]
    sensor_values = [row.get(primary_sensor) for row in result.label_overlay]
    anomaly_rows = [row for row in result.label_overlay if int(row.get('anomaly', 0)) == 1]
    changepoint_rows = [row for row in result.label_overlay if int(row.get('changepoint', 0)) == 1]
    return {
        'data': [
            {
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': primary_sensor,
                'x': timestamps,
                'y': sensor_values,
            },
            {
                'type': 'scatter',
                'mode': 'markers',
                'name': 'anomaly',
                'x': [row.get('timestamp', '') for row in anomaly_rows],
                'y': [row.get(primary_sensor) for row in anomaly_rows],
                'marker': {'color': '#d62728', 'size': 10, 'symbol': 'x'},
            },
            {
                'type': 'scatter',
                'mode': 'markers',
                'name': 'changepoint',
                'x': [row.get('timestamp', '') for row in changepoint_rows],
                'y': [row.get(primary_sensor) for row in changepoint_rows],
                'marker': {'color': '#ff7f0e', 'size': 10, 'symbol': 'diamond'},
            },
        ],
        'layout': {
            'title': {'text': 'SKAB anomaly/changepoint overlay'},
            'xaxis': {'title': {'text': 'datetime'}},
            'yaxis': {'title': {'text': primary_sensor}},
            'template': 'plotly_white',
        },
    }


def _write_label_overlay_plot_artifacts(result: SkabEdaResult, json_path: Path, html_path: Path) -> None:
    payload = _label_overlay_plot_payload(result)
    payload_text = json.dumps(payload, sort_keys=True, indent=2)
    json_path.write_text(payload_text, encoding='utf-8')
    html = (
        '<!doctype html>\n'
        '<html lang="en">\n'
        '<head><meta charset="utf-8"><title>SKAB anomaly/changepoint overlay</title>'
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>\n'
        '<body><div id="skab-label-overlay"></div>\n'
        f'<script>const figure = {json.dumps(payload, sort_keys=True)};'
        "Plotly.newPlot('skab-label-overlay', figure.data, figure.layout, {responsive: true});</script>\n"
        '</body></html>\n'
    )
    html_path.write_text(html, encoding='utf-8')


def write_skab_eda_report(result: SkabEdaResult, output_dir: Path, include_plots: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    result.plots_available = include_plots
    paths = _artifact_paths(output_dir, include_plots=include_plots)
    summary_path = paths['summary_json']
    payload = result.to_payload()
    summary_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2),
        encoding='utf-8',
    )

    paths['timestamp_quality_json'].write_text(
        json.dumps(result.timestamp_quality, sort_keys=True, indent=2),
        encoding='utf-8',
    )
    _write_sensor_statistics_csv(result, paths['sensor_statistics_csv'])
    _write_missingness_csv(result, paths['missingness_csv'])
    _write_label_ranges_csv(result, paths['label_ranges_csv'])
    _write_correlation_matrix_csv(result, paths['correlation_matrix_csv'])
    _write_sensor_distributions_csv(result, paths['sensor_distributions_csv'])
    _write_rolling_statistics_csv(result, paths['rolling_statistics_csv'])
    _write_label_overlay_csv(result, paths['label_overlay_csv'])
    if include_plots:
        _write_label_overlay_plot_artifacts(
            result,
            paths['label_overlay_plot_json'],
            paths['label_overlay_plot_html'],
        )

    report_path = paths['report_md']
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

    if result.timestamp_quality:
        lines.append('## Timestamp Quality')
        lines.append('')
        lines.append(f"**Parsed datetimes:** {result.timestamp_quality.get('parsed_datetime_count', 0)}")
        lines.append(f"**Invalid timestamps:** {result.timestamp_quality.get('invalid_timestamp_count', 0)}")
        lines.append(f"**Duplicate timestamps:** {result.timestamp_quality.get('duplicate_timestamp_count', 0)}")
        lines.append(f"**Cadence mode seconds:** {result.timestamp_quality.get('cadence_mode_seconds', 'N/A')}")
        lines.append(f"**Max gap seconds:** {result.timestamp_quality.get('max_gap_seconds', 'N/A')}")
        lines.append('')

    lines.append('## Sensor Statistics')
    lines.append('')
    lines.append('| Sensor | Mean | Std | Min | Max |')
    lines.append('| --- | --- | --- | --- | --- |')
    for col in result.sensor_columns:
        stats = result.sensor_statistics.get(col, {})
        mean = _report_number(stats.get('mean', 0.0))
        std = _report_number(stats.get('std', 0.0))
        min_val = _report_number(stats.get('min', 0.0))
        max_val = _report_number(stats.get('max', 0.0))
        lines.append(f'| {col} | {mean} | {std} | {min_val} | {max_val} |')
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
        write_skab_eda_report(result, output_dir, include_plots=include_plots)
        _add_artifact_paths(artifacts, 'input', output_dir, include_plots=include_plots)

    if split_manifest_path is not None:
        split_results = summarize_skab_manifest(split_manifest_path)
        for split_name, result in split_results.items():
            split_dir = output_dir / split_name
            write_skab_eda_report(result, split_dir, include_plots=include_plots)
            _add_artifact_paths(artifacts, split_name, split_dir, include_plots=include_plots)

    if include_plots:
        artifacts['plots_note'] = 'plots available'

    return artifacts
