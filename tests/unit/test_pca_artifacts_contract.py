import json
from importlib import import_module
from pathlib import Path

import pytest


def _pca_artifacts():
    return import_module('ml.visualization.pca_artifacts')


def _pandas():
    return import_module('pandas')


def _write_scores_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    pd = _pandas()
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def _write_metrics_json(path: Path, t2_threshold: float = 5.0, q_threshold: float = 3.0) -> Path:
    payload = {
        't2_threshold': t2_threshold,
        'q_threshold': q_threshold,
        'precision': 0.85,
        'recall': 0.90,
        'f1': 0.875,
        'sample_count': 4,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return path


def _write_metadata_json(path: Path) -> Path:
    payload = {
        'params': {'window_size': 60},
        'thresholds': {'t2_threshold': 5.0, 'q_threshold': 3.0},
        'sensor_columns': ['sensor1', 'sensor2'],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return path


def _synthetic_score_rows(count: int = 6) -> list[dict[str, object]]:
    rows = []
    for i in range(count):
        rows.append(
            {
                'timestamp': f'2024-01-01T00:00:{i:02d}Z',
                'label': 1 if i >= count - 2 else 0,
                'prediction': 1 if i >= count - 1 else 0,
                't2': float(i * 1.5 + 1.0),
                'q': float(i * 0.8 + 0.5),
                'score': float(i * 0.3 + 0.1),
            }
        )
    return rows


def test_generate_pca_artifact_visualizations_creates_expected_artifacts(tmp_path):
    pca_artifacts = _pca_artifacts()
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()

    _write_scores_csv(artifact_dir / 'scores.csv', _synthetic_score_rows())
    _write_metrics_json(artifact_dir / 'metrics.json')
    _write_metadata_json(artifact_dir / 'metadata.json')

    config = pca_artifacts.PcaArtifactVisualizationConfig(artifact_dir=artifact_dir)
    result = pca_artifacts.generate_pca_artifact_visualizations(config)

    expected_keys = {
        'scores_scatter_plotly_json',
        'scores_scatter_plotly_html',
        'scores_timeline_plotly_json',
        'scores_timeline_plotly_html',
        'summary',
    }
    assert set(result.keys()) == expected_keys
    for path in result.values():
        assert path.exists()
    assert result['summary'].parent == artifact_dir / 'visualizations'


def test_generate_pca_artifact_visualizations_skips_test_when_not_present(tmp_path):
    pca_artifacts = _pca_artifacts()
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()

    _write_scores_csv(artifact_dir / 'scores.csv', _synthetic_score_rows())
    _write_metrics_json(artifact_dir / 'metrics.json')
    _write_metadata_json(artifact_dir / 'metadata.json')

    config = pca_artifacts.PcaArtifactVisualizationConfig(artifact_dir=artifact_dir, include_test=True)
    result = pca_artifacts.generate_pca_artifact_visualizations(config)

    assert not any('test_scores' in name for name in result.keys())
    summary = json.loads(result['summary'].read_text())
    assert summary['has_test_scores'] is False


def test_generate_pca_artifact_visualizations_includes_test_when_present(tmp_path):
    pca_artifacts = _pca_artifacts()
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()

    _write_scores_csv(artifact_dir / 'scores.csv', _synthetic_score_rows())
    _write_scores_csv(artifact_dir / 'test_scores.csv', _synthetic_score_rows(count=4))
    _write_metrics_json(artifact_dir / 'metrics.json')
    _write_metadata_json(artifact_dir / 'metadata.json')

    config = pca_artifacts.PcaArtifactVisualizationConfig(artifact_dir=artifact_dir, include_test=True)
    result = pca_artifacts.generate_pca_artifact_visualizations(config)

    assert 'test_scores_scatter_plotly_json' in result
    assert 'test_scores_scatter_plotly_html' in result
    assert 'test_scores_timeline_plotly_json' in result
    assert 'test_scores_timeline_plotly_html' in result
    summary = json.loads(result['summary'].read_text())
    assert summary['has_test_scores'] is True


def test_generate_pca_artifact_visualizations_respects_no_test_flag(tmp_path):
    pca_artifacts = _pca_artifacts()
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()

    _write_scores_csv(artifact_dir / 'scores.csv', _synthetic_score_rows())
    _write_scores_csv(artifact_dir / 'test_scores.csv', _synthetic_score_rows(count=4))
    _write_metrics_json(artifact_dir / 'metrics.json')
    _write_metadata_json(artifact_dir / 'metadata.json')

    config = pca_artifacts.PcaArtifactVisualizationConfig(artifact_dir=artifact_dir, include_test=False)
    result = pca_artifacts.generate_pca_artifact_visualizations(config)

    assert not any('test_scores' in name for name in result.keys())
    summary = json.loads(result['summary'].read_text())
    assert summary['has_test_scores'] is False


def test_generate_pca_artifact_visualizations_raises_on_missing_columns(tmp_path):
    pca_artifacts = _pca_artifacts()
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()

    pd = _pandas()
    bad_scores = pd.DataFrame({'timestamp': ['2024-01-01T00:00:00Z'], 'label': [0]})
    bad_scores.to_csv(artifact_dir / 'scores.csv', index=False)
    _write_metrics_json(artifact_dir / 'metrics.json')
    _write_metadata_json(artifact_dir / 'metadata.json')

    config = pca_artifacts.PcaArtifactVisualizationConfig(artifact_dir=artifact_dir)
    with pytest.raises(ValueError, match='missing required columns'):
        pca_artifacts.generate_pca_artifact_visualizations(config)


def test_generate_pca_artifact_visualizations_summary_is_deterministic(tmp_path):
    pca_artifacts = _pca_artifacts()
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()

    _write_scores_csv(artifact_dir / 'scores.csv', _synthetic_score_rows())
    _write_metrics_json(artifact_dir / 'metrics.json')
    _write_metadata_json(artifact_dir / 'metadata.json')

    config = pca_artifacts.PcaArtifactVisualizationConfig(artifact_dir=artifact_dir)
    result1 = pca_artifacts.generate_pca_artifact_visualizations(config)
    result2 = pca_artifacts.generate_pca_artifact_visualizations(config)

    summary1 = json.loads(result1['summary'].read_text())
    summary2 = json.loads(result2['summary'].read_text())
    assert summary1 == summary2


def test_generate_pca_artifact_visualizations_plotly_json_is_deterministic(tmp_path):
    pca_artifacts = _pca_artifacts()
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()

    _write_scores_csv(artifact_dir / 'scores.csv', _synthetic_score_rows())
    _write_metrics_json(artifact_dir / 'metrics.json')
    _write_metadata_json(artifact_dir / 'metadata.json')

    first_output = tmp_path / 'first'
    second_output = tmp_path / 'second'
    first = pca_artifacts.generate_pca_artifact_visualizations(
        pca_artifacts.PcaArtifactVisualizationConfig(artifact_dir=artifact_dir, output_dir=first_output)
    )
    second = pca_artifacts.generate_pca_artifact_visualizations(
        pca_artifacts.PcaArtifactVisualizationConfig(artifact_dir=artifact_dir, output_dir=second_output)
    )

    assert first['scores_scatter_plotly_json'].read_text() == second['scores_scatter_plotly_json'].read_text()
    assert first['scores_timeline_plotly_json'].read_text() == second['scores_timeline_plotly_json'].read_text()


def test_generate_pca_artifact_visualizations_html_uses_stable_div_ids(tmp_path):
    pca_artifacts = _pca_artifacts()
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()

    _write_scores_csv(artifact_dir / 'scores.csv', _synthetic_score_rows())
    _write_metrics_json(artifact_dir / 'metrics.json')
    _write_metadata_json(artifact_dir / 'metadata.json')

    result = pca_artifacts.generate_pca_artifact_visualizations(
        pca_artifacts.PcaArtifactVisualizationConfig(artifact_dir=artifact_dir)
    )

    assert 'id="scores-t2q-scatter"' in result['scores_scatter_plotly_html'].read_text()
    assert 'id="scores-timeline"' in result['scores_timeline_plotly_html'].read_text()


def test_generate_pca_artifact_visualizations_uses_custom_output_dir(tmp_path):
    pca_artifacts = _pca_artifacts()
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()
    custom_output = tmp_path / 'custom_viz'

    _write_scores_csv(artifact_dir / 'scores.csv', _synthetic_score_rows())
    _write_metrics_json(artifact_dir / 'metrics.json')
    _write_metadata_json(artifact_dir / 'metadata.json')

    config = pca_artifacts.PcaArtifactVisualizationConfig(artifact_dir=artifact_dir, output_dir=custom_output)
    result = pca_artifacts.generate_pca_artifact_visualizations(config)

    assert result['summary'].parent == custom_output


def test_generate_pca_artifact_visualizations_reads_thresholds_from_metadata_when_missing_in_metrics(tmp_path):
    pca_artifacts = _pca_artifacts()
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()

    _write_scores_csv(artifact_dir / 'scores.csv', _synthetic_score_rows())
    metrics = {'precision': 0.5}
    (artifact_dir / 'metrics.json').write_text(json.dumps(metrics) + '\n')
    _write_metadata_json(artifact_dir / 'metadata.json')

    config = pca_artifacts.PcaArtifactVisualizationConfig(artifact_dir=artifact_dir)
    result = pca_artifacts.generate_pca_artifact_visualizations(config)

    summary = json.loads(result['summary'].read_text())
    assert summary['thresholds']['t2_threshold'] == 5.0
    assert summary['thresholds']['q_threshold'] == 3.0


def test_cli_prints_sorted_json_paths(tmp_path, capsys):
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()

    _write_scores_csv(artifact_dir / 'scores.csv', _synthetic_score_rows())
    _write_metrics_json(artifact_dir / 'metrics.json')
    _write_metadata_json(artifact_dir / 'metadata.json')

    cli = import_module('scripts.generate_pca_visualizations')
    result = cli.main(
        [
            '--artifact-dir',
            str(artifact_dir),
        ]
    )

    captured = capsys.readouterr().out
    output = json.loads(captured)
    assert sorted(output.keys()) == sorted(result.keys())
    for name, path_str in output.items():
        assert Path(path_str).exists()


def test_cli_respects_no_test_flag(tmp_path, capsys):
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()

    _write_scores_csv(artifact_dir / 'scores.csv', _synthetic_score_rows())
    _write_scores_csv(artifact_dir / 'test_scores.csv', _synthetic_score_rows(count=4))
    _write_metrics_json(artifact_dir / 'metrics.json')
    _write_metadata_json(artifact_dir / 'metadata.json')

    cli = import_module('scripts.generate_pca_visualizations')
    cli.main(
        [
            '--artifact-dir',
            str(artifact_dir),
            '--no-test',
        ]
    )

    captured = capsys.readouterr().out
    output = json.loads(captured)
    assert not any('test_scores' in name for name in output.keys())


def test_cli_uses_custom_output_dir(tmp_path, capsys):
    artifact_dir = tmp_path / 'artifacts'
    artifact_dir.mkdir()
    custom_output = tmp_path / 'custom_out'

    _write_scores_csv(artifact_dir / 'scores.csv', _synthetic_score_rows())
    _write_metrics_json(artifact_dir / 'metrics.json')
    _write_metadata_json(artifact_dir / 'metadata.json')

    cli = import_module('scripts.generate_pca_visualizations')
    cli.main(
        [
            '--artifact-dir',
            str(artifact_dir),
            '--output-dir',
            str(custom_output),
        ]
    )

    captured = capsys.readouterr().out
    output = json.loads(captured)
    assert all(Path(path_str).parent == custom_output for path_str in output.values())
