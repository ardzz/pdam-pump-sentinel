import json
from importlib import import_module


def _skab_loader():
    return import_module('ml.datasets.skab_loader')


def _joblib():
    return import_module('joblib')


def _pandas():
    return import_module('pandas')


def _train_pca():
    return import_module('ml.training.train_pca')


def _write_skab_csv(path, row_count=24, anomaly_start=None, offset=0.0):
    sensor_columns = _skab_loader().SENSOR_COLUMNS
    rows = [['datetime', *sensor_columns, 'anomaly', 'changepoint']]
    for row_index in range(row_count):
        anomaly = int(anomaly_start is not None and row_index >= anomaly_start)
        sensor_values = []
        for column_index, _ in enumerate(sensor_columns):
            baseline = 10.0 + offset + row_index * 0.05 + column_index * 0.1
            sensor_values.append(f'{baseline + anomaly * 8.0:.4f}')
        rows.append([
            f'2024-01-01T00:00:{row_index:02d}Z',
            *sensor_values,
            str(anomaly),
            '0',
        ])
    path.write_text('\n'.join(';'.join(row) for row in rows) + '\n')
    return path


def test_train_pca_from_skab_writes_deterministic_artifact_contract(tmp_path):
    train_pca = _train_pca()
    input_path = _write_skab_csv(tmp_path / 'train.csv', anomaly_start=20)
    output_dir = tmp_path / 'artifacts'
    config = train_pca.PcaTrainingConfig(
        input_path=input_path,
        output_dir=output_dir,
        window_size=4,
        stride=4,
        threshold_quantile=0.9,
    )

    result = train_pca.train_pca_from_skab(config)

    assert set(result.artifact_paths) == {'detector', 'metadata', 'metrics', 'scores'}
    for path in result.artifact_paths.values():
        assert path.exists()
        assert path.parent == output_dir
    assert result.sensor_columns == tuple(_skab_loader().SENSOR_COLUMNS)
    assert result.thresholds['t2_threshold'] > 0
    assert result.thresholds['q_threshold'] > 0

    detector = _joblib().load(result.artifact_paths['detector'])
    assert detector.predict([[0.0] * (4 * len(result.sensor_columns))]).shape == (1,)

    metrics = json.loads(result.artifact_paths['metrics'].read_text())
    assert metrics['sample_count'] == 6
    assert metrics['anomaly_count'] == 1
    assert metrics['normal_count'] == 5
    assert metrics['training_sample_count'] == 6
    assert metrics['training_normal_count'] == 5
    for metric_name in ('precision', 'recall', 'f1', 'false_alarm_rate'):
        assert 0.0 <= metrics[metric_name] <= 1.0

    metadata = json.loads(result.artifact_paths['metadata'].read_text())
    assert metadata['sensor_columns'] == _skab_loader().SENSOR_COLUMNS
    assert metadata['artifact_paths']['detector'] == str(output_dir / 'pca_detector.joblib')
    assert metadata['params']['window_size'] == 4

    scores = _pandas().read_csv(result.artifact_paths['scores'])
    assert list(scores.columns) == ['timestamp', 'label', 'prediction', 't2', 'q', 'score']
    assert scores['label'].tolist() == [0, 0, 0, 0, 0, 1]


def test_train_pca_uses_validation_input_for_scoring_when_provided(tmp_path):
    train_pca = _train_pca()
    input_path = _write_skab_csv(tmp_path / 'train.csv', row_count=24)
    validation_path = _write_skab_csv(tmp_path / 'validation.csv', row_count=12, anomaly_start=8, offset=1.0)
    config = train_pca.PcaTrainingConfig(
        input_path=input_path,
        validation_input_path=validation_path,
        output_dir=tmp_path / 'validation-artifacts',
        window_size=4,
        stride=4,
        threshold_quantile=0.9,
    )

    result = train_pca.train_pca_from_skab(config)

    assert result.metrics['sample_count'] == 3
    assert result.metrics['anomaly_count'] == 1
    assert result.metrics['training_sample_count'] == 6
    assert result.metrics['training_normal_count'] == 6
    scores = _pandas().read_csv(result.artifact_paths['scores'])
    assert scores['timestamp'].tolist() == [
        '2024-01-01T00:00:03Z',
        '2024-01-01T00:00:07Z',
        '2024-01-01T00:00:11Z',
    ]
    assert scores['label'].tolist() == [0, 0, 1]


def test_main_trains_without_starting_mlflow(tmp_path, capsys):
    train_pca = _train_pca()
    input_path = _write_skab_csv(tmp_path / 'cli-train.csv', anomaly_start=20)
    output_dir = tmp_path / 'cli-artifacts'

    result = train_pca.main([
        str(input_path),
        str(output_dir),
        '--window-size',
        '4',
        '--stride',
        '4',
        '--n-components',
        '0.9',
        '--threshold-quantile',
        '0.9',
    ])

    output = json.loads(capsys.readouterr().out)
    assert result.artifact_paths['detector'] == output_dir / 'pca_detector.joblib'
    assert result.output_dir == output_dir
    assert output['output_dir'] == str(output_dir)
    assert output['artifact_paths']['scores'] == str(output_dir / 'scores.csv')
    assert output['metrics']['sample_count'] == 6


def test_train_pca_calls_mlflow_helper_only_when_enabled(tmp_path, monkeypatch):
    train_pca = _train_pca()
    input_path = _write_skab_csv(tmp_path / 'mlflow-train.csv', anomaly_start=20)
    calls = []

    def log_pca_training_run(result, detector, config):
        calls.append((result, detector, config))
        return 'run-123'

    mlflow_client = import_module('ml.registry.mlflow_client')
    monkeypatch.setattr(mlflow_client, 'log_pca_training_run', log_pca_training_run)

    config = train_pca.PcaTrainingConfig(
        input_path=input_path,
        output_dir=tmp_path / 'mlflow-artifacts',
        window_size=4,
        stride=4,
        threshold_quantile=0.9,
        log_mlflow=True,
        register_model=True,
        alias='champion',
    )

    result = train_pca.train_pca_from_skab(config)

    assert len(calls) == 1
    logged_result, detector, logged_config = calls[0]
    assert logged_result == result
    assert detector.predict([[0.0] * (4 * len(result.sensor_columns))]).shape == (1,)
    assert logged_config.register_model is True
    assert logged_config.alias == 'champion'
