import json
from importlib import import_module

import pytest


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


def _write_split_manifest(path, train, validation, test):
    payload = {
        'train': [str(p.relative_to(path.parent)) for p in train],
        'validation': [str(p.relative_to(path.parent)) for p in validation],
        'test': [str(p.relative_to(path.parent)) for p in test],
    }
    path.write_text(json.dumps(payload, indent=2) + '\n')
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


def test_split_manifest_config_exists(tmp_path):
    train_pca = _train_pca()
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps({'train': [], 'validation': [], 'test': []}) + '\n')
    config = train_pca.PcaTrainingConfig(
        input_path=tmp_path / 'dummy.csv',
        output_dir=tmp_path / 'out',
        split_manifest_path=manifest_path,
    )
    assert config.split_manifest_path == manifest_path


def test_split_manifest_train_files_loaded_without_cross_file_mixing(tmp_path):
    train_pca = _train_pca()
    train1 = _write_skab_csv(tmp_path / 'train1.csv', row_count=24, offset=0.0)
    train2 = _write_skab_csv(tmp_path / 'train2.csv', row_count=24, offset=100.0)
    val = _write_skab_csv(tmp_path / 'val.csv', row_count=12, anomaly_start=8, offset=200.0)
    test = _write_skab_csv(tmp_path / 'test.csv', row_count=12, anomaly_start=8, offset=300.0)
    manifest = _write_split_manifest(tmp_path / 'manifest.json', [train1, train2], [val], [test])

    config = train_pca.PcaTrainingConfig(
        input_path=tmp_path / 'dummy.csv',
        output_dir=tmp_path / 'out',
        split_manifest_path=manifest,
        window_size=4,
        stride=4,
        threshold_quantile=0.9,
    )

    result = train_pca.train_pca_from_skab(config)

    assert result.metrics['training_sample_count'] == 12
    assert result.metrics['training_normal_count'] == 12


def test_split_manifest_scaler_pca_fit_uses_train_normal_only(tmp_path, monkeypatch):
    train_pca = _train_pca()
    train1 = _write_skab_csv(tmp_path / 'train1.csv', row_count=24, anomaly_start=20, offset=0.0)
    train2 = _write_skab_csv(tmp_path / 'train2.csv', row_count=24, anomaly_start=20, offset=100.0)
    val = _write_skab_csv(tmp_path / 'val.csv', row_count=12, anomaly_start=8, offset=200.0)
    test = _write_skab_csv(tmp_path / 'test.csv', row_count=12, anomaly_start=8, offset=300.0)
    manifest = _write_split_manifest(tmp_path / 'manifest.json', [train1, train2], [val], [test])

    fit_calls = []
    original_fit = train_pca.PcaT2QDetector.fit

    def patched_fit(self, X, y=None):
        fit_calls.append(X)
        return original_fit(self, X, y)

    monkeypatch.setattr(train_pca.PcaT2QDetector, 'fit', patched_fit)

    config = train_pca.PcaTrainingConfig(
        input_path=tmp_path / 'dummy.csv',
        output_dir=tmp_path / 'out',
        split_manifest_path=manifest,
        window_size=4,
        stride=4,
        threshold_quantile=0.9,
    )

    train_pca.train_pca_from_skab(config)

    assert len(fit_calls) == 1
    fitted_X = fit_calls[0]
    assert fitted_X.shape[0] == 10


def test_split_manifest_thresholds_calibrated_from_validation_normal(tmp_path, monkeypatch):
    train_pca = _train_pca()
    train1 = _write_skab_csv(tmp_path / 'train1.csv', row_count=24, offset=0.0)
    train2 = _write_skab_csv(tmp_path / 'train2.csv', row_count=24, offset=100.0)
    val = _write_skab_csv(tmp_path / 'val.csv', row_count=12, anomaly_start=8, offset=200.0)
    test = _write_skab_csv(tmp_path / 'test.csv', row_count=12, anomaly_start=8, offset=300.0)
    manifest = _write_split_manifest(tmp_path / 'manifest.json', [train1, train2], [val], [test])

    calibrate_calls = []
    original_calibrate = train_pca.PcaT2QDetector.calibrate_thresholds

    def patched_calibrate(self, X):
        calibrate_calls.append(X)
        return original_calibrate(self, X)

    monkeypatch.setattr(train_pca.PcaT2QDetector, 'calibrate_thresholds', patched_calibrate)

    config = train_pca.PcaTrainingConfig(
        input_path=tmp_path / 'dummy.csv',
        output_dir=tmp_path / 'out',
        split_manifest_path=manifest,
        window_size=4,
        stride=4,
        threshold_quantile=0.9,
    )

    train_pca.train_pca_from_skab(config)

    assert len(calibrate_calls) == 1
    calibrated_X = calibrate_calls[0]
    assert calibrated_X.shape[0] == 2


def test_split_manifest_requires_validation_normal_windows_for_calibration(tmp_path):
    train_pca = _train_pca()
    train1 = _write_skab_csv(tmp_path / 'train1.csv', row_count=24, offset=0.0)
    val = _write_skab_csv(tmp_path / 'val.csv', row_count=12, anomaly_start=0, offset=200.0)
    manifest = _write_split_manifest(tmp_path / 'manifest.json', [train1], [val], [])

    config = train_pca.PcaTrainingConfig(
        input_path=tmp_path / 'dummy.csv',
        output_dir=tmp_path / 'out',
        split_manifest_path=manifest,
        window_size=4,
        stride=4,
        threshold_quantile=0.9,
    )

    with pytest.raises(ValueError, match='validation input must contain at least one normal window'):
        train_pca.train_pca_from_skab(config)


def test_split_manifest_writes_split_manifest_artifact(tmp_path):
    train_pca = _train_pca()
    train1 = _write_skab_csv(tmp_path / 'train1.csv', row_count=24, offset=0.0)
    train2 = _write_skab_csv(tmp_path / 'train2.csv', row_count=24, offset=100.0)
    val = _write_skab_csv(tmp_path / 'val.csv', row_count=12, anomaly_start=8, offset=200.0)
    test = _write_skab_csv(tmp_path / 'test.csv', row_count=12, anomaly_start=8, offset=300.0)
    manifest = _write_split_manifest(tmp_path / 'manifest.json', [train1, train2], [val], [test])

    config = train_pca.PcaTrainingConfig(
        input_path=tmp_path / 'dummy.csv',
        output_dir=tmp_path / 'out',
        split_manifest_path=manifest,
        window_size=4,
        stride=4,
        threshold_quantile=0.9,
    )

    result = train_pca.train_pca_from_skab(config)

    assert 'split_manifest' in result.artifact_paths
    split_manifest_path = result.artifact_paths['split_manifest']
    assert split_manifest_path.exists()
    payload = json.loads(split_manifest_path.read_text())
    assert set(payload.keys()) == {'train', 'validation', 'test'}
    assert len(payload['train']) == 2
    assert len(payload['validation']) == 1
    assert len(payload['test']) == 1


def test_split_manifest_metadata_and_metrics_include_split_info(tmp_path):
    train_pca = _train_pca()
    train1 = _write_skab_csv(tmp_path / 'train1.csv', row_count=24, offset=0.0)
    train2 = _write_skab_csv(tmp_path / 'train2.csv', row_count=24, offset=100.0)
    val = _write_skab_csv(tmp_path / 'val.csv', row_count=12, anomaly_start=8, offset=200.0)
    test = _write_skab_csv(tmp_path / 'test.csv', row_count=12, anomaly_start=8, offset=300.0)
    manifest = _write_split_manifest(tmp_path / 'manifest.json', [train1, train2], [val], [test])

    config = train_pca.PcaTrainingConfig(
        input_path=tmp_path / 'dummy.csv',
        output_dir=tmp_path / 'out',
        split_manifest_path=manifest,
        window_size=4,
        stride=4,
        threshold_quantile=0.9,
    )

    result = train_pca.train_pca_from_skab(config)

    metadata = json.loads(result.artifact_paths['metadata'].read_text())
    assert 'split' in metadata
    assert metadata['split']['train_files'] == [str(train1.name), str(train2.name)]
    assert metadata['split']['validation_files'] == [str(val.name)]
    assert metadata['split']['test_files'] == [str(test.name)]
    assert metadata['split']['train_count'] == 12
    assert metadata['split']['validation_count'] == 3
    assert metadata['split']['test_count'] == 3

    metrics = json.loads(result.artifact_paths['metrics'].read_text())
    assert metrics['train_count'] == 12
    assert metrics['validation_count'] == 3
    assert metrics['test_count'] == 3


def test_split_manifest_scores_csv_represents_validation_scoring(tmp_path):
    train_pca = _train_pca()
    train1 = _write_skab_csv(tmp_path / 'train1.csv', row_count=24, offset=0.0)
    val = _write_skab_csv(tmp_path / 'val.csv', row_count=12, anomaly_start=8, offset=200.0)
    test = _write_skab_csv(tmp_path / 'test.csv', row_count=12, anomaly_start=8, offset=300.0)
    manifest = _write_split_manifest(tmp_path / 'manifest.json', [train1], [val], [test])

    config = train_pca.PcaTrainingConfig(
        input_path=tmp_path / 'dummy.csv',
        output_dir=tmp_path / 'out',
        split_manifest_path=manifest,
        window_size=4,
        stride=4,
        threshold_quantile=0.9,
    )

    result = train_pca.train_pca_from_skab(config)

    scores = _pandas().read_csv(result.artifact_paths['scores'])
    assert len(scores) == 3
    assert scores['label'].tolist() == [0, 0, 1]


def test_split_manifest_test_scores_csv_exists_when_test_has_files(tmp_path):
    train_pca = _train_pca()
    train1 = _write_skab_csv(tmp_path / 'train1.csv', row_count=24, offset=0.0)
    val = _write_skab_csv(tmp_path / 'val.csv', row_count=12, anomaly_start=8, offset=200.0)
    test = _write_skab_csv(tmp_path / 'test.csv', row_count=12, anomaly_start=8, offset=300.0)
    manifest = _write_split_manifest(tmp_path / 'manifest.json', [train1], [val], [test])

    config = train_pca.PcaTrainingConfig(
        input_path=tmp_path / 'dummy.csv',
        output_dir=tmp_path / 'out',
        split_manifest_path=manifest,
        window_size=4,
        stride=4,
        threshold_quantile=0.9,
    )

    result = train_pca.train_pca_from_skab(config)

    assert 'test_scores' in result.artifact_paths
    test_scores_path = result.artifact_paths['test_scores']
    assert test_scores_path.exists()
    test_scores = _pandas().read_csv(test_scores_path)
    assert len(test_scores) == 3
    assert list(test_scores.columns) == ['timestamp', 'label', 'prediction', 't2', 'q', 'score']


def test_split_manifest_duplicate_files_fail_before_fitting(tmp_path):
    train_pca = _train_pca()
    train1 = _write_skab_csv(tmp_path / 'train1.csv', row_count=24, offset=0.0)
    val = _write_skab_csv(tmp_path / 'val.csv', row_count=12, anomaly_start=8, offset=200.0)
    manifest = _write_split_manifest(tmp_path / 'manifest.json', [train1], [val, train1], [])

    config = train_pca.PcaTrainingConfig(
        input_path=tmp_path / 'dummy.csv',
        output_dir=tmp_path / 'out',
        split_manifest_path=manifest,
        window_size=4,
        stride=4,
        threshold_quantile=0.9,
    )

    with pytest.raises(ValueError, match='duplicate file'):
        train_pca.train_pca_from_skab(config)


def test_split_manifest_cli_supports_split_manifest_flag(tmp_path, capsys):
    train_pca = _train_pca()
    train1 = _write_skab_csv(tmp_path / 'train1.csv', row_count=24, offset=0.0)
    val = _write_skab_csv(tmp_path / 'val.csv', row_count=12, anomaly_start=8, offset=200.0)
    test = _write_skab_csv(tmp_path / 'test.csv', row_count=12, anomaly_start=8, offset=300.0)
    manifest = _write_split_manifest(tmp_path / 'manifest.json', [train1], [val], [test])
    output_dir = tmp_path / 'cli-out'

    result = train_pca.main([
        str(tmp_path / 'dummy.csv'),
        str(output_dir),
        '--split-manifest',
        str(manifest),
        '--window-size',
        '4',
        '--stride',
        '4',
        '--threshold-quantile',
        '0.9',
    ])

    output = json.loads(capsys.readouterr().out)
    assert result.output_dir == output_dir
    assert output['output_dir'] == str(output_dir)
    assert 'split_manifest' in result.artifact_paths
    assert result.artifact_paths['split_manifest'].exists()


def test_split_manifest_cli_allows_output_dir_without_dummy_input(tmp_path, capsys):
    train_pca = _train_pca()
    train1 = _write_skab_csv(tmp_path / 'train1.csv', row_count=24, offset=0.0)
    val = _write_skab_csv(tmp_path / 'val.csv', row_count=12, anomaly_start=8, offset=200.0)
    manifest = _write_split_manifest(tmp_path / 'manifest.json', [train1], [val], [])
    output_dir = tmp_path / 'cli-out'

    result = train_pca.main([
        str(output_dir),
        '--split-manifest',
        str(manifest),
        '--window-size',
        '4',
        '--stride',
        '4',
        '--threshold-quantile',
        '0.9',
    ])

    output = json.loads(capsys.readouterr().out)
    assert result.output_dir == output_dir
    assert output['artifact_paths']['split_manifest'] == str(output_dir / 'split_manifest.json')


def test_default_scaler_is_robust(tmp_path):
    train_pca = _train_pca()
    config = train_pca.PcaTrainingConfig(
        input_path=tmp_path / 'dummy.csv',
        output_dir=tmp_path / 'out',
    )
    assert config.scaler == 'robust'
