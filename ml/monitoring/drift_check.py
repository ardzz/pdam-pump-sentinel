from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from evidently import DataDefinition, Dataset
from evidently.core.report import Report
from evidently.presets import DataDriftPreset


@dataclass(frozen=True)
class DriftResult:
    dataset_drift: bool
    drift_share: float
    n_drifted: int
    n_features: int
    method: str = 'evidently'
    threshold: float = 0.5
    report_path: str | None = None


def check_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Sequence[str],
    drift_share: float = 0.5,
) -> DriftResult:
    selected_columns = list(columns)
    data_definition = DataDefinition(numerical_columns=selected_columns)
    reference_ds = Dataset.from_pandas(reference, data_definition=data_definition)
    current_ds = Dataset.from_pandas(current, data_definition=data_definition)
    snapshot = Report([DataDriftPreset(drift_share=drift_share)]).run(current_ds, reference_ds)
    value = _drifted_columns_count(snapshot)
    n_drifted = int(float(value.get('count', 0.0)))
    observed_share = float(value.get('share', 0.0))
    return DriftResult(
        dataset_drift=n_drifted > 0 and observed_share >= float(drift_share),
        drift_share=observed_share,
        n_drifted=n_drifted,
        n_features=len(selected_columns),
        method='evidently',
        threshold=float(drift_share),
    )


def _drifted_columns_count(snapshot: Any) -> Mapping[str, Any]:
    payload = snapshot.dict() if callable(getattr(snapshot, 'dict', None)) else snapshot.model_dump()
    metrics = payload.get('metrics', []) if isinstance(payload, Mapping) else []
    for metric in metrics:
        if not isinstance(metric, Mapping):
            continue
        config = metric.get('config')
        metric_name = str(metric.get('metric_name', ''))
        if isinstance(config, Mapping) and config.get('type') == 'evidently:metric_v2:DriftedColumnsCount':
            value = metric.get('value', {})
            if isinstance(value, Mapping):
                return value
        if metric_name.startswith('DriftedColumnsCount'):
            value = metric.get('value', {})
            if isinstance(value, Mapping):
                return value
    raise ValueError('Evidently DriftedColumnsCount metric not found')


__all__ = ['DriftResult', 'check_drift']
