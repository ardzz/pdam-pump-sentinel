from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PcaArtifactVisualizationConfig:
    artifact_dir: Path
    output_dir: Path | None = None
    include_test: bool = True


def generate_pca_artifact_visualizations(
    config: PcaArtifactVisualizationConfig,
) -> dict[str, Path]:
    artifact_dir = Path(config.artifact_dir)
    output_dir = Path(config.output_dir) if config.output_dir is not None else artifact_dir / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = artifact_dir / 'scores.csv'
    metrics_path = artifact_dir / 'metrics.json'
    metadata_path = artifact_dir / 'metadata.json'

    pd = import_module('pandas')
    scores = pd.read_csv(scores_path)
    _validate_score_columns(scores)

    metrics = json.loads(metrics_path.read_text())
    metadata = json.loads(metadata_path.read_text())

    thresholds = _resolve_thresholds(metrics, metadata)

    artifacts: dict[str, Path] = {}
    artifacts.update(_generate_score_artifacts(scores, thresholds, output_dir, prefix='scores'))

    test_scores_path = artifact_dir / 'test_scores.csv'
    if config.include_test and test_scores_path.exists():
        test_scores = pd.read_csv(test_scores_path)
        _validate_score_columns(test_scores)
        artifacts.update(_generate_score_artifacts(test_scores, thresholds, output_dir, prefix='test_scores'))

    summary = _build_summary(scores, thresholds, metadata, artifacts, test_scores_path.exists() and config.include_test)
    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    artifacts['summary'] = summary_path

    return artifacts


def _validate_score_columns(scores: Any) -> None:
    required = {'timestamp', 'label', 'prediction', 't2', 'q', 'score'}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f'scores CSV missing required columns: {sorted(missing)}')


def _resolve_thresholds(metrics: dict[str, Any], metadata: dict[str, Any]) -> dict[str, float]:
    t2_threshold = metrics.get('t2_threshold')
    q_threshold = metrics.get('q_threshold')

    if t2_threshold is None:
        t2_threshold = metadata.get('thresholds', {}).get('t2_threshold')
    if q_threshold is None:
        q_threshold = metadata.get('thresholds', {}).get('q_threshold')

    if t2_threshold is None or q_threshold is None:
        raise ValueError('t2_threshold and q_threshold must be present in metrics.json or metadata.json thresholds')

    return {'t2_threshold': float(t2_threshold), 'q_threshold': float(q_threshold)}


def _generate_score_artifacts(
    scores: Any,
    thresholds: dict[str, float],
    output_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    scatter_fig = _build_t2q_scatter(scores, thresholds, prefix)
    timeline_fig = _build_timeline(scores, thresholds, prefix)

    scatter_json = output_dir / f'{prefix}_scatter_plotly.json'
    scatter_html = output_dir / f'{prefix}_scatter_plotly.html'
    timeline_json = output_dir / f'{prefix}_timeline_plotly.json'
    timeline_html = output_dir / f'{prefix}_timeline_plotly.html'

    _write_plotly_artifacts(scatter_fig, scatter_json, scatter_html, div_id=f'{prefix}-t2q-scatter')
    _write_plotly_artifacts(timeline_fig, timeline_json, timeline_html, div_id=f'{prefix}-timeline')

    return {
        f'{prefix}_scatter_plotly_json': scatter_json,
        f'{prefix}_scatter_plotly_html': scatter_html,
        f'{prefix}_timeline_plotly_json': timeline_json,
        f'{prefix}_timeline_plotly_html': timeline_html,
    }


def _write_plotly_artifacts(fig: Any, json_path: Path, html_path: Path, div_id: str) -> None:
    plotly_io = import_module('plotly.io')
    plotly_io.write_json(fig, json_path, pretty=True, remove_uids=True, engine='json')
    plotly_io.write_html(fig, html_path, include_plotlyjs='cdn', full_html=True, div_id=div_id)


def _build_t2q_scatter(scores: Any, thresholds: dict[str, float], prefix: str) -> Any:
    go = import_module('plotly.graph_objects')

    normal_mask = scores['label'] == 0
    anomaly_mask = scores['label'] == 1

    traces = [
        go.Scatter(
            x=scores.loc[normal_mask, 't2'].tolist(),
            y=scores.loc[normal_mask, 'q'].tolist(),
            mode='markers',
            name='Normal',
            marker={'color': '#2ecc71', 'size': 8},
        ),
        go.Scatter(
            x=scores.loc[anomaly_mask, 't2'].tolist(),
            y=scores.loc[anomaly_mask, 'q'].tolist(),
            mode='markers',
            name='Anomaly',
            marker={'color': '#e74c3c', 'size': 8},
        ),
        go.Scatter(
            x=[thresholds['t2_threshold'], thresholds['t2_threshold']],
            y=[scores['q'].min(), scores['q'].max()],
            mode='lines',
            name='T² threshold',
            line={'color': '#3498db', 'dash': 'dash', 'width': 2},
        ),
        go.Scatter(
            x=[scores['t2'].min(), scores['t2'].max()],
            y=[thresholds['q_threshold'], thresholds['q_threshold']],
            mode='lines',
            name='Q threshold',
            line={'color': '#9b59b6', 'dash': 'dash', 'width': 2},
        ),
    ]

    layout = go.Layout(
        title=f'{prefix.replace("_", " ").title()} T²/Q Scatter',
        xaxis={'title': 'T² statistic'},
        yaxis={'title': 'Q statistic'},
        hovermode='closest',
    )

    return go.Figure(data=traces, layout=layout)


def _build_timeline(scores: Any, thresholds: dict[str, float], prefix: str) -> Any:
    go = import_module('plotly.graph_objects')

    timestamps = scores['timestamp'].astype(str).tolist()

    traces = [
        go.Scatter(
            x=timestamps,
            y=scores['t2'].tolist(),
            mode='lines',
            name='T²',
            line={'color': '#3498db', 'width': 2},
        ),
        go.Scatter(
            x=timestamps,
            y=scores['q'].tolist(),
            mode='lines',
            name='Q',
            line={'color': '#9b59b6', 'width': 2},
        ),
        go.Scatter(
            x=timestamps,
            y=scores['score'].tolist(),
            mode='lines',
            name='Score',
            line={'color': '#f39c12', 'width': 2},
        ),
        go.Scatter(
            x=[timestamps[0], timestamps[-1]],
            y=[thresholds['t2_threshold'], thresholds['t2_threshold']],
            mode='lines',
            name='T² threshold',
            line={'color': '#3498db', 'dash': 'dash', 'width': 2},
        ),
        go.Scatter(
            x=[timestamps[0], timestamps[-1]],
            y=[thresholds['q_threshold'], thresholds['q_threshold']],
            mode='lines',
            name='Q threshold',
            line={'color': '#9b59b6', 'dash': 'dash', 'width': 2},
        ),
    ]

    layout = go.Layout(
        title=f'{prefix.replace("_", " ").title()} Timeline',
        xaxis={'title': 'Timestamp'},
        yaxis={'title': 'Value'},
        hovermode='x unified',
    )

    return go.Figure(data=traces, layout=layout)


def _build_summary(
    scores: Any,
    thresholds: dict[str, float],
    metadata: dict[str, Any],
    artifacts: dict[str, Path],
    has_test: bool,
) -> dict[str, Any]:
    return {
        'artifact_count': len(artifacts),
        'artifact_paths': {name: str(path) for name, path in artifacts.items()},
        'has_test_scores': has_test,
        'metrics': {
            'sample_count': int(len(scores)),
            'anomaly_count': int(scores['label'].sum()),
            'normal_count': int(len(scores) - scores['label'].sum()),
        },
        'thresholds': {
            't2_threshold': thresholds['t2_threshold'],
            'q_threshold': thresholds['q_threshold'],
        },
        'metadata_keys': sorted(metadata.keys()),
    }
