import json
import math
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _harness():
    return import_module('scripts.run_skab_model_experiments')


class _FakeDetector:
    def __init__(self, *, score_offset=0.0, t2_threshold=1.0, q_threshold=1.0):
        self._score_offset = score_offset
        self.t2_threshold_ = t2_threshold
        self.q_threshold_ = q_threshold

    def score_samples(self, features):
        return features[:, 0].astype(float) + self._score_offset

    def predict(self, features):
        return (self.score_samples(features) >= self.t2_threshold_).astype(int)


def _harness_config(harness, tmp_path, output_dir_name='artifacts'):
    return harness.HarnessConfig(
        manifest_path=Path(tmp_path / 'manifest.json'),
        output_dir=Path(tmp_path / output_dir_name),
        window_size=4,
        stride=1,
        seed=7,
        lstm_epochs=1,
        lstm_batch_size=2,
        lstm_patience=1,
        supervised_estimators=4,
        isoforest_estimators=5,
        oneclass_svm_nu=0.05,
        oneclass_svm_max_iter=200,
        distance_profile_max_reference_windows=8,
        distance_profile_max_distance_comparisons=1000,
        pca_ensemble_size=2,
        pca_ensemble_subset_ratio=0.5,
        pca_variant_feature_mode='spectral',
        skip_lstm=True,
        smoke=True,
    )


def _score_csv(path, scores, labels, changepoints=None):
    changepoints = list(changepoints or [0] * len(scores))
    rows = ['timestamp,label,changepoint,prediction,score']
    for index, (score, label, changepoint) in enumerate(zip(scores, labels, changepoints, strict=True)):
        rows.append(f'2024-01-01T00:00:{index:02d}Z,{label},{changepoint},0,{score}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    return path


def _baseline_lstm_record(harness, tmp_path, validation_scores_path, test_scores_path):
    return harness.ExperimentRecord(
        key='04_lstm_ae',
        display_name='LSTM-AE threshold baseline',
        model_family='lstm_ae',
        feature_mode='raw',
        split_protocol=harness.SPLIT_PROTOCOL,
        status='completed',
        output_dir=Path(tmp_path / '04_lstm_ae'),
        metrics={'threshold': 2.5, 'train_count': 8},
        artifact_paths={'scores': validation_scores_path, 'test_scores': test_scores_path},
    )


def _baseline_score_record(harness, tmp_path, validation_scores_path, test_scores_path):
    return harness.ExperimentRecord(
        key='08_isolation_forest_spectral',
        display_name='Isolation Forest spectral baseline',
        model_family='isolation_forest',
        feature_mode='spectral',
        split_protocol=harness.SPLIT_PROTOCOL,
        status='completed',
        output_dir=Path(tmp_path / '08_isolation_forest_spectral'),
        metrics={'threshold': 2.5, 'train_count': 8},
        artifact_paths={'scores': validation_scores_path, 'test_scores': test_scores_path},
    )


def _stub_record(harness, tmp_path, key, *, status='completed'):
    return harness.ExperimentRecord(
        key=key,
        display_name=key,
        model_family=key,
        feature_mode='stub',
        split_protocol=harness.SPLIT_PROTOCOL,
        status=status,
        output_dir=Path(tmp_path / key),
        metrics={} if status == 'completed' else {'status': status},
        artifact_paths={},
    )


def test_dbscan_threshold_uses_validation_score_cluster_without_labels():
    harness = _harness()
    validation_scores = np.asarray([0.10, 0.11, 0.12, 0.13, 3.00, 3.05, 3.10, 3.15], dtype=float)

    rule = harness._derive_dbscan_threshold(validation_scores, fallback_threshold=2.50)

    assert rule.fallback_used is False
    assert rule.high_error_cluster is not None
    assert 2.99 <= rule.threshold <= 3.01


def test_dbscan_threshold_falls_back_when_high_error_cluster_is_not_useful():
    harness = _harness()
    validation_scores = np.asarray([0.10, 0.11, 0.12, 0.13, 0.14, 0.15], dtype=float)

    rule = harness._derive_dbscan_threshold(validation_scores, fallback_threshold=0.30)

    assert rule.fallback_used is True
    assert rule.threshold == 0.30
    assert 'high-error cluster' in rule.reason or 'degenerate' in rule.reason


def test_conformal_threshold_uses_finite_sample_corrected_upper_quantile():
    harness = _harness()

    rule = harness._derive_conformal_threshold(np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]), alpha=0.5)

    assert rule.corrected_quantile_formula == 'ceil((n + 1) * (1 - alpha))'
    assert rule.calibration_count == 5
    assert rule.corrected_quantile_rank == math.ceil((5 + 1) * (1 - 0.5))
    assert rule.corrected_quantile_rank == 3
    assert rule.corrected_quantile_level == 3 / 5
    assert rule.threshold == 3.0
    assert rule.threshold_is_infinite is False

    conservative_rule = harness._derive_conformal_threshold(np.asarray([1.0, 2.0, 3.0]), alpha=0.1)
    assert conservative_rule.corrected_quantile_rank == math.ceil((3 + 1) * (1 - 0.1))
    assert conservative_rule.corrected_quantile_rank == 4
    assert conservative_rule.corrected_quantile_level == 4 / 3
    assert math.isinf(conservative_rule.threshold)
    assert conservative_rule.threshold_is_infinite is True


def test_conformal_experiment_metadata_uses_validation_normal_rows_only(tmp_path):
    harness = _harness()
    validation_scores_path = _score_csv(
        tmp_path / 'source' / 'scores.csv',
        scores=[0.10, 0.20, 0.30, 0.40, 0.50, 99.0],
        labels=[0, 0, 0, 0, 0, 1],
    )
    test_scores_path = _score_csv(
        tmp_path / 'source' / 'test_scores.csv',
        scores=[0.20, 0.60],
        labels=[0, 1],
    )

    record = harness._run_conformal_threshold_experiment(
        _harness_config(harness, tmp_path),
        _baseline_score_record(harness, tmp_path, validation_scores_path, test_scores_path),
        key='10_isolation_forest_conformal',
        display_name='Isolation Forest conformal threshold wrapper',
        model_family='isolation_forest_conformal_threshold',
        feature_mode='spectral',
        alpha=0.4,
    )

    assert record.key == '10_isolation_forest_conformal'
    assert record.model_family == 'isolation_forest_conformal_threshold'
    assert record.feature_mode == 'spectral'
    assert record.metrics['threshold'] == 0.40
    assert record.metrics['conformal_corrected_quantile_rank'] == math.ceil((5 + 1) * (1 - 0.4))
    assert record.metrics['conformal_calibration_count'] == 5
    assert record.metadata['threshold_method'] == 'conformal'
    assert record.metadata['alpha'] == 0.4
    assert record.metadata['calibration_split'] == 'validation_normal_rows'
    assert record.metadata['source_record_key'] == '08_isolation_forest_spectral'
    assert record.metadata['score_source_experiment'] == '08_isolation_forest_spectral'
    assert record.metadata['held_out_test_labels_used_for_thresholding'] is False
    assert record.metadata['held_out_test_scores_used_for_thresholding'] is False
    assert 'held-out test labels were not used for thresholding' in record.metadata['threshold_calibration']
    assert record.metadata['threshold_rule']['corrected_quantile_formula'] == 'ceil((n + 1) * (1 - alpha))'
    assert record.metadata['threshold_rule']['prediction_rule'] == 'prediction = score > threshold'
    assert record.artifact_paths['scores'].exists()
    assert record.artifact_paths['test_scores'].exists()


def test_summary_rows_rank_by_held_out_test_f1_and_leave_skipped_unranked(tmp_path):
    harness = _harness()
    records = [
        harness.ExperimentRecord(
            key='06_xgboost',
            display_name='XGBoost supervised upper bound',
            model_family='xgboost',
            feature_mode='enriched',
            split_protocol=harness.SPLIT_PROTOCOL,
            status='completed',
            output_dir=Path(tmp_path / 'xgb'),
            metrics={'f1': 0.4, 'test_f1': 0.7, 'test_precision': 0.8},
            artifact_paths={},
            notes=harness.SUPERVISED_LABEL_WARNING,
        ),
        harness.ExperimentRecord(
            key='01_pca_raw',
            display_name='PCA T²/Q baseline',
            model_family='pca',
            feature_mode='raw',
            split_protocol=harness.SPLIT_PROTOCOL,
            status='completed',
            output_dir=Path(tmp_path / 'pca'),
            metrics={'f1': 0.3, 'test_f1': 0.2, 'test_precision': 0.5},
            artifact_paths={},
        ),
        harness.ExperimentRecord(
            key='04_lstm_ae',
            display_name='LSTM-AE threshold baseline',
            model_family='lstm_ae',
            feature_mode='raw',
            split_protocol=harness.SPLIT_PROTOCOL,
            status='completed',
            output_dir=Path(tmp_path / 'lstm'),
            metrics={'f1': 0.99, 'test_f1': 0.1, 'test_precision': 0.5},
            artifact_paths={},
        ),
        harness.ExperimentRecord(
            key='07_lightgbm',
            display_name='LightGBM supervised upper bound',
            model_family='lightgbm',
            feature_mode='enriched',
            split_protocol=harness.SPLIT_PROTOCOL,
            status='skipped',
            output_dir=Path(tmp_path / 'lgbm'),
            metrics={'status': 'skipped', 'reason': 'training labels unavailable'},
            artifact_paths={},
            notes='training labels unavailable',
        ),
    ]

    rows = harness._build_ranked_summary_rows(records)

    assert [row['experiment_key'] for row in rows] == ['06_xgboost', '01_pca_raw', '04_lstm_ae', '07_lightgbm']
    assert [row['rank'] for row in rows] == [1, 2, 3, '']
    assert rows[2]['validation_f1'] == 0.99
    assert rows[0]['notes'] == harness.SUPERVISED_LABEL_WARNING
    assert all(row['split_protocol'] == harness.SPLIT_PROTOCOL for row in rows)


def test_summary_artifacts_document_wave5_rows_caveats_and_mlflow_opt_in(tmp_path):
    harness = _harness()
    config = _harness_config(harness, tmp_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    def record(key, *, test_f1, model_family, feature_mode, notes=''):
        return harness.ExperimentRecord(
            key=key,
            display_name=key,
            model_family=model_family,
            feature_mode=feature_mode,
            split_protocol=harness.SPLIT_PROTOCOL,
            status='completed',
            output_dir=Path(tmp_path / key),
            metrics={
                'f1': test_f1 / 2,
                'test_f1': test_f1,
                'test_precision': 0.5,
                'test_recall': 0.6,
            },
            artifact_paths={},
            notes=notes,
        )

    records = [
        record('02_pca_spectral', test_f1=0.56, model_family='pca', feature_mode='spectral'),
        record(
            '06_xgboost',
            test_f1=0.90,
            model_family='xgboost',
            feature_mode='enriched',
            notes=harness.SUPERVISED_LABEL_WARNING,
        ),
        record(
            '08_isolation_forest_spectral',
            test_f1=0.40,
            model_family='isolation_forest',
            feature_mode='spectral',
        ),
        record('09_oneclass_svm_spectral', test_f1=0.30, model_family='oneclass_svm', feature_mode='spectral'),
        record(
            '10_isolation_forest_conformal',
            test_f1=0.42,
            model_family='isolation_forest_conformal_threshold',
            feature_mode='spectral',
        ),
        record(
            '11_forecasting_residual_naive',
            test_f1=0.35,
            model_family='forecasting_residual_naive',
            feature_mode='raw_lag_one_residual',
        ),
        record(
            '12_distance_profile_nearest_normal',
            test_f1=0.33,
            model_family='distance_profile_nearest_normal',
            feature_mode='raw_window_nearest_normal_distance',
        ),
    ]

    harness._write_root_artifacts(config, records)

    summary = (config.output_dir / 'summary.md').read_text(encoding='utf-8')
    metadata = json.loads((config.output_dir / 'metadata.json').read_text(encoding='utf-8'))
    assert '`02_pca_spectral` remains the deployment-preferred baseline' in summary
    assert 'MLflow logging: disabled by default; pass `--log-mlflow` to opt in' in summary
    assert 'train-normal fitting, validation-normal calibration, held-out test metrics' in summary
    assert 'Expanded row coverage through `12_distance_profile_nearest_normal`' in summary
    assert '`08_isolation_forest_spectral`' in summary
    assert '`09_oneclass_svm_spectral`' in summary
    assert '`10_isolation_forest_conformal`: threshold calibration variant' in summary
    assert '`11_forecasting_residual_naive`' in summary
    assert '`12_distance_profile_nearest_normal`: bounded optional comparative baseline' in summary
    assert 'not a separately trained model family' in summary
    assert 'supervised upper-bound experiments only' in summary
    assert metadata['mlflow_logging']['enabled'] is False
    assert metadata['mlflow_logging']['requires_mlflow_by_default'] is False
    assert metadata['experiment_family_notes'][-1]['key'] == '12_distance_profile_nearest_normal'


def test_deployment_preference_note_requires_actual_higher_normal_only_metrics(tmp_path):
    harness = _harness()
    rows_with_supervised_leader = harness._build_ranked_summary_rows(
        [
            harness.ExperimentRecord(
                key='02_pca_spectral',
                display_name='PCA spectral',
                model_family='pca',
                feature_mode='spectral',
                split_protocol=harness.SPLIT_PROTOCOL,
                status='completed',
                output_dir=Path(tmp_path / 'pca'),
                metrics={'test_f1': 0.50},
                artifact_paths={},
            ),
            harness.ExperimentRecord(
                key='06_xgboost',
                display_name='XGBoost supervised upper bound',
                model_family='xgboost',
                feature_mode='enriched',
                split_protocol=harness.SPLIT_PROTOCOL,
                status='completed',
                output_dir=Path(tmp_path / 'xgb'),
                metrics={'test_f1': 0.95},
                artifact_paths={},
            ),
        ]
    )
    assert '`02_pca_spectral` remains the deployment-preferred baseline' in harness._deployment_preference_note(
        rows_with_supervised_leader
    )

    rows_with_normal_only_leader = harness._build_ranked_summary_rows(
        [
            harness.ExperimentRecord(
                key='02_pca_spectral',
                display_name='PCA spectral',
                model_family='pca',
                feature_mode='spectral',
                split_protocol=harness.SPLIT_PROTOCOL,
                status='completed',
                output_dir=Path(tmp_path / 'pca'),
                metrics={'test_f1': 0.50},
                artifact_paths={},
            ),
            harness.ExperimentRecord(
                key='08_isolation_forest_spectral',
                display_name='Isolation Forest spectral baseline',
                model_family='isolation_forest',
                feature_mode='spectral',
                split_protocol=harness.SPLIT_PROTOCOL,
                status='completed',
                output_dir=Path(tmp_path / 'if'),
                metrics={'test_f1': 0.55},
                artifact_paths={},
            ),
        ]
    )
    preference_note = harness._deployment_preference_note(rows_with_normal_only_leader)
    assert '`08_isolation_forest_spectral` reports higher held-out `test_f1` in this rerun' in preference_note
    assert 'before changing deployment preference' in preference_note


def test_pca_ensemble_record_exposes_threshold_metadata_and_validation_normal_calibration(monkeypatch, tmp_path):
    harness = _harness()
    manifest = SimpleNamespace(
        train=('train.csv',),
        validation=('validation.csv',),
        test=('test.csv',),
        to_payload=lambda: {
            'train': ['train.csv'],
            'validation': ['validation.csv'],
            'test': ['test.csv'],
        },
    )
    train_windows = SimpleNamespace(
        features=np.asarray([[0.0, 0.1, 0.2, 0.3], [0.2, 0.3, 0.4, 0.5], [0.4, 0.5, 0.6, 0.7]], dtype=float),
        labels=np.asarray([0, 0, 0], dtype=int),
        changepoints=np.asarray([0, 0, 0], dtype=int),
        timestamps=np.asarray(['2024-01-01T00:00:00Z', '2024-01-01T00:00:01Z', '2024-01-01T00:00:02Z']),
    )
    validation_windows = SimpleNamespace(
        features=np.asarray([[0.0, 0.1, 0.2, 0.3], [0.2, 0.3, 0.4, 0.5], [3.0, 3.1, 3.2, 3.3]], dtype=float),
        labels=np.asarray([0, 0, 1], dtype=int),
        changepoints=np.asarray([0, 0, 1], dtype=int),
        timestamps=np.asarray(['2024-01-01T00:01:00Z', '2024-01-01T00:01:01Z', '2024-01-01T00:01:02Z']),
    )
    test_windows = SimpleNamespace(
        features=np.asarray([[0.1, 0.2, 0.3, 0.4], [3.1, 3.2, 3.3, 3.4], [0.2, 0.3, 0.4, 0.5]], dtype=float),
        labels=np.asarray([0, 1, 0], dtype=int),
        changepoints=np.asarray([0, 0, 0], dtype=int),
        timestamps=np.asarray(['2024-01-01T00:02:00Z', '2024-01-01T00:02:01Z', '2024-01-01T00:02:02Z']),
    )

    def load_windows(files, _config):
        if files == manifest.train:
            return train_windows
        if files == manifest.validation:
            return validation_windows
        if files == manifest.test:
            return test_windows
        raise AssertionError(f'unexpected split files: {files!r}')

    captured = {}

    def fit_members(train_normal_features, validation_normal_features, _config):
        captured['train_normal_count'] = len(train_normal_features)
        captured['validation_normal_count'] = len(validation_normal_features)
        return [
            (np.asarray([0, 1], dtype=int), _FakeDetector(t2_threshold=1.0, q_threshold=1.5)),
            (np.asarray([2, 3], dtype=int), _FakeDetector(score_offset=0.2, t2_threshold=1.2, q_threshold=1.7)),
        ]

    monkeypatch.setattr(harness.pca_training, '_load_windows_multi', load_windows)
    monkeypatch.setattr(harness, '_fit_pca_ensemble_members', fit_members)
    monkeypatch.setattr(harness, '_dump_joblib', lambda _value, path: path.write_text('stub\n', encoding='utf-8'))

    record = harness._run_pca_ensemble(_harness_config(harness, tmp_path), manifest)

    assert captured == {'train_normal_count': 3, 'validation_normal_count': 2}
    assert record.metadata['params']['threshold_calibration'] == (
        'each member calibrated on validation windows with anomaly label 0 only'
    )
    assert record.metadata['test_split_held_out'] is True
    assert record.metadata['split']['test_count'] == 3
    assert record.metrics['test_sample_count'] == 3
    assert 'test_f1' in record.metrics
    assert record.artifact_paths['test_scores'].exists()
    assert [member['feature_indices'] for member in record.metadata['members']] == [[0, 1], [2, 3]]
    assert all('t2_threshold' in member and 'q_threshold' in member for member in record.metadata['members'])


def test_run_experiments_registers_unsupervised_spectral_rows_after_lightgbm_without_full_training(monkeypatch, tmp_path):
    harness = _harness()
    manifest = SimpleNamespace(train=(), validation=(), test=(), to_payload=lambda: {'train': [], 'validation': [], 'test': []})
    captured = {}

    monkeypatch.setattr(harness, 'load_skab_split_manifest', lambda _path: manifest)
    monkeypatch.setattr(
        harness,
        '_run_pca_experiment',
        lambda _config, key, _display_name, _feature_mode: _stub_record(harness, tmp_path, key),
    )
    monkeypatch.setattr(harness, '_run_pca_ensemble', lambda _config, _manifest: _stub_record(harness, tmp_path, '03_pca_feature_bagging_ensemble'))
    monkeypatch.setattr(harness, '_run_lstm_dbscan_experiment', lambda _config, _baseline: _stub_record(harness, tmp_path, '05_lstm_ae_dbscan_threshold'))
    monkeypatch.setattr(
        harness,
        '_run_supervised_experiment',
        lambda _config, key, _display_name, _model_type: _stub_record(harness, tmp_path, key),
    )
    monkeypatch.setattr(harness, '_run_isoforest_experiment', lambda _config: _stub_record(harness, tmp_path, '08_isolation_forest_spectral'))
    monkeypatch.setattr(harness, '_run_oneclass_svm_experiment', lambda _config: _stub_record(harness, tmp_path, '09_oneclass_svm_spectral'))
    monkeypatch.setattr(
        harness,
        '_run_forecasting_residual_experiment',
        lambda _config: _stub_record(harness, tmp_path, '11_forecasting_residual_naive'),
    )
    monkeypatch.setattr(
        harness,
        '_run_distance_profile_experiment',
        lambda _config: _stub_record(harness, tmp_path, '12_distance_profile_nearest_normal'),
    )

    def fake_conformal(_config, source_record, **kwargs):
        captured['conformal_source'] = source_record.key
        captured['conformal_kwargs'] = kwargs
        return _stub_record(harness, tmp_path, kwargs['key'])

    monkeypatch.setattr(harness, '_run_conformal_threshold_experiment', fake_conformal)
    monkeypatch.setattr(harness, '_write_root_artifacts', lambda _config, records: captured.setdefault('records', list(records)))

    records = harness.run_experiments(_harness_config(harness, tmp_path))

    keys = [record.key for record in records]
    assert keys == [
        '01_pca_raw',
        '02_pca_spectral',
        '03_pca_feature_bagging_ensemble',
        '04_lstm_ae',
        '05_lstm_ae_dbscan_threshold',
        '06_xgboost',
        '07_lightgbm',
        '08_isolation_forest_spectral',
        '09_oneclass_svm_spectral',
        '10_isolation_forest_conformal',
        '11_forecasting_residual_naive',
        '12_distance_profile_nearest_normal',
    ]
    assert captured['conformal_source'] == '08_isolation_forest_spectral'
    assert captured['conformal_kwargs']['alpha'] == 0.1
    assert captured['records'] == records


def test_isolation_forest_harness_uses_existing_spectral_trainer_and_writes_protocol_metadata(monkeypatch, tmp_path):
    harness = _harness()
    captured = {}

    def fake_train(config):
        captured['config'] = config
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            'f1': 0.5,
            'test_f1': 0.25,
            'threshold': 1.2,
            'train_count': 3,
            'validation_count': 2,
            'test_count': 2,
        }
        artifact_paths = {
            'model': output_dir / 'isoforest.joblib',
            'metadata': output_dir / 'metadata.json',
            'metrics': output_dir / 'metrics.json',
            'scores': output_dir / 'scores.csv',
            'split_manifest': output_dir / 'split_manifest.json',
            'test_scores': output_dir / 'test_scores.csv',
        }
        artifact_paths['metadata'].write_text(
            json.dumps(
                {
                    'model_family': 'isolation_forest',
                    'params': {
                        'feature_mode': config.feature_mode,
                        'n_estimators': config.n_estimators,
                        'threshold_quantile': config.threshold_quantile,
                    },
                    'split': {'train_count': 3, 'validation_count': 2, 'test_count': 2},
                    'test_split_held_out': True,
                }
            )
            + '\n',
            encoding='utf-8',
        )
        return SimpleNamespace(output_dir=output_dir, artifact_paths=artifact_paths, metrics=metrics)

    monkeypatch.setattr(harness, 'train_isoforest_from_skab', fake_train)

    config = _harness_config(harness, tmp_path)
    record = harness._run_isoforest_experiment(config)

    trainer_config = captured['config']
    assert isinstance(trainer_config, harness.IsoForestTrainingConfig)
    assert trainer_config.input_path == config.output_dir / '08_isolation_forest_spectral'
    assert trainer_config.output_dir == config.output_dir / '08_isolation_forest_spectral'
    assert trainer_config.split_manifest_path == config.manifest_path
    assert trainer_config.feature_mode == 'spectral'
    assert trainer_config.scaler == 'standard'
    assert trainer_config.n_estimators == config.isoforest_estimators
    assert trainer_config.seed == config.seed
    assert trainer_config.log_mlflow is False

    assert record.key == '08_isolation_forest_spectral'
    assert record.display_name == 'Isolation Forest spectral baseline'
    assert record.model_family == 'isolation_forest'
    assert record.feature_mode == 'spectral'
    assert record.status == 'completed'
    assert record.metrics['test_f1'] == 0.25
    assert record.artifact_paths['metrics_csv'].exists()
    assert record.artifact_paths['experiment_metadata'].exists()

    metadata = json.loads(record.artifact_paths['metadata'].read_text(encoding='utf-8'))
    experiment_metadata = json.loads(record.artifact_paths['experiment_metadata'].read_text(encoding='utf-8'))
    for payload in (metadata, experiment_metadata, record.metadata):
        assert payload['key'] == '08_isolation_forest_spectral'
        assert payload['split_protocol'] == harness.SPLIT_PROTOCOL
        assert payload['threshold_method'] == 'validation_normal_quantile'
        assert payload['calibration_split'] == 'validation_normal_windows'
        assert payload['test_labels_used_for_training_or_threshold'] is False
        assert 'anomaly label 0 only' in payload['train_label_policy']
        assert 'held-out test split' in payload['test_metrics_policy']
    assert metadata['test_split_held_out'] is True
    assert metadata['params']['feature_mode'] == 'spectral'
    assert experiment_metadata['notes'] == record.notes


def test_oneclass_svm_harness_uses_spectral_trainer_and_writes_protocol_metadata(monkeypatch, tmp_path):
    harness = _harness()
    captured = {}

    def fake_train(config):
        captured['config'] = config
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            'f1': 0.4,
            'test_f1': 0.2,
            'threshold': 0.8,
            'train_count': 3,
            'validation_count': 2,
            'test_count': 2,
        }
        artifact_paths = {
            'model': output_dir / 'oneclass_svm.joblib',
            'metadata': output_dir / 'metadata.json',
            'metrics': output_dir / 'metrics.json',
            'scores': output_dir / 'scores.csv',
            'split_manifest': output_dir / 'split_manifest.json',
            'test_scores': output_dir / 'test_scores.csv',
        }
        artifact_paths['metadata'].write_text(
            json.dumps(
                {
                    'model_family': 'oneclass_svm',
                    'params': {
                        'feature_mode': config.feature_mode,
                        'kernel': config.kernel,
                        'nu': config.nu,
                        'gamma': config.gamma,
                        'max_iter': config.max_iter,
                        'score_orientation': 'higher_is_more_anomalous:-decision_function',
                    },
                    'split': {'train_count': 3, 'validation_count': 2, 'test_count': 2},
                    'test_split_held_out': True,
                }
            )
            + '\n',
            encoding='utf-8',
        )
        return SimpleNamespace(output_dir=output_dir, artifact_paths=artifact_paths, metrics=metrics)

    monkeypatch.setattr(harness, 'train_oneclass_svm_from_skab', fake_train)

    config = _harness_config(harness, tmp_path)
    record = harness._run_oneclass_svm_experiment(config)

    trainer_config = captured['config']
    assert isinstance(trainer_config, harness.OneClassSvmTrainingConfig)
    assert trainer_config.input_path == config.output_dir / '09_oneclass_svm_spectral'
    assert trainer_config.output_dir == config.output_dir / '09_oneclass_svm_spectral'
    assert trainer_config.split_manifest_path == config.manifest_path
    assert trainer_config.feature_mode == 'spectral'
    assert trainer_config.scaler == 'standard'
    assert trainer_config.kernel == 'rbf'
    assert trainer_config.nu == config.oneclass_svm_nu
    assert trainer_config.gamma == 'scale'
    assert trainer_config.max_iter == config.oneclass_svm_max_iter
    assert trainer_config.log_mlflow is False

    assert record.key == '09_oneclass_svm_spectral'
    assert record.display_name == 'One-Class SVM spectral baseline'
    assert record.model_family == 'oneclass_svm'
    assert record.feature_mode == 'spectral'
    assert record.status == 'completed'
    assert record.metrics['test_f1'] == 0.2
    assert record.artifact_paths['metrics_csv'].exists()
    assert record.artifact_paths['experiment_metadata'].exists()

    metadata = json.loads(record.artifact_paths['metadata'].read_text(encoding='utf-8'))
    experiment_metadata = json.loads(record.artifact_paths['experiment_metadata'].read_text(encoding='utf-8'))
    for payload in (metadata, experiment_metadata, record.metadata):
        assert payload['key'] == '09_oneclass_svm_spectral'
        assert payload['split_protocol'] == harness.SPLIT_PROTOCOL
        assert payload['threshold_method'] == 'validation_normal_quantile'
        assert payload['calibration_split'] == 'validation_normal_windows'
        assert payload['score_orientation'] == 'higher_is_more_anomalous:-decision_function'
        assert payload['test_labels_used_for_training_or_threshold'] is False
        assert 'anomaly label 0 only' in payload['train_label_policy']
        assert 'held-out test split' in payload['test_metrics_policy']
    assert metadata['test_split_held_out'] is True
    assert metadata['params']['feature_mode'] == 'spectral'
    assert metadata['params']['kernel'] == 'rbf'
    assert experiment_metadata['notes'] == record.notes


def test_forecasting_residual_harness_uses_naive_trainer_and_writes_protocol_metadata(monkeypatch, tmp_path):
    harness = _harness()
    captured = {}

    def fake_train(config):
        captured['config'] = config
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            'f1': 0.6,
            'test_f1': 0.3,
            'threshold': 1.1,
            'train_count': 3,
            'validation_count': 2,
            'test_count': 2,
        }
        artifact_paths = {
            'model': output_dir / 'forecasting_residual_naive.joblib',
            'metadata': output_dir / 'metadata.json',
            'metrics': output_dir / 'metrics.json',
            'scores': output_dir / 'scores.csv',
            'split_manifest': output_dir / 'split_manifest.json',
            'test_scores': output_dir / 'test_scores.csv',
        }
        artifact_paths['metadata'].write_text(
            json.dumps(
                {
                    'model_family': 'forecasting_residual_naive',
                    'params': {
                        'feature_mode': 'raw_lag_one_residual',
                        'aggregation': config.aggregation,
                        'threshold_quantile': config.threshold_quantile,
                        'score_orientation': 'higher_is_more_anomalous:mean_abs_scaled_lag_one_residual',
                    },
                    'split': {'train_count': 3, 'validation_count': 2, 'test_count': 2},
                    'test_split_held_out': True,
                }
            )
            + '\n',
            encoding='utf-8',
        )
        return SimpleNamespace(output_dir=output_dir, artifact_paths=artifact_paths, metrics=metrics)

    monkeypatch.setattr(harness, 'train_forecasting_residual_from_skab', fake_train)

    config = _harness_config(harness, tmp_path)
    record = harness._run_forecasting_residual_experiment(config)

    trainer_config = captured['config']
    assert isinstance(trainer_config, harness.ForecastingResidualTrainingConfig)
    assert trainer_config.input_path == config.output_dir / '11_forecasting_residual_naive'
    assert trainer_config.output_dir == config.output_dir / '11_forecasting_residual_naive'
    assert trainer_config.split_manifest_path == config.manifest_path
    assert trainer_config.window_size == config.window_size
    assert trainer_config.stride == config.stride
    assert trainer_config.threshold_quantile == 0.95
    assert trainer_config.aggregation == 'mean'

    assert record.key == '11_forecasting_residual_naive'
    assert record.display_name == 'Naive lag-one forecasting residual baseline'
    assert record.model_family == 'forecasting_residual_naive'
    assert record.feature_mode == 'raw_lag_one_residual'
    assert record.status == 'completed'
    assert record.metrics['test_f1'] == 0.3
    assert record.artifact_paths['metrics_csv'].exists()
    assert record.artifact_paths['experiment_metadata'].exists()

    metadata = json.loads(record.artifact_paths['metadata'].read_text(encoding='utf-8'))
    experiment_metadata = json.loads(record.artifact_paths['experiment_metadata'].read_text(encoding='utf-8'))
    for payload in (metadata, experiment_metadata, record.metadata):
        assert payload['key'] == '11_forecasting_residual_naive'
        assert payload['split_protocol'] == harness.SPLIT_PROTOCOL
        assert payload['threshold_method'] == 'validation_normal_quantile'
        assert payload['calibration_split'] == 'validation_normal_windows'
        assert payload['score_orientation'] == 'higher_is_more_anomalous:mean_abs_scaled_lag_one_residual'
        assert payload['future_values_used_for_prediction'] is False
        assert payload['test_labels_used_for_training_or_threshold'] is False
        assert payload['test_scores_used_for_training_or_threshold'] is False
        assert 'anomaly label 0 only' in payload['train_label_policy']
        assert 'previous row' in payload['residual_alignment']
        assert 'held-out test split' in payload['test_metrics_policy']
    assert metadata['test_split_held_out'] is True
    assert metadata['params']['feature_mode'] == 'raw_lag_one_residual'
    assert metadata['params']['aggregation'] == 'mean'
    assert experiment_metadata['notes'] == record.notes


def test_distance_profile_harness_uses_bounded_trainer_and_writes_protocol_metadata(monkeypatch, tmp_path):
    harness = _harness()
    captured = {}

    def fake_train(config):
        captured['config'] = config
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            'f1': 0.55,
            'test_f1': 0.35,
            'threshold': 1.4,
            'train_count': 3,
            'validation_count': 2,
            'test_count': 2,
        }
        artifact_paths = {
            'model': output_dir / 'distance_profile_nearest_normal.joblib',
            'scaler': output_dir / 'scaler.joblib',
            'metadata': output_dir / 'metadata.json',
            'metrics': output_dir / 'metrics.json',
            'scores': output_dir / 'scores.csv',
            'split_manifest': output_dir / 'split_manifest.json',
            'test_scores': output_dir / 'test_scores.csv',
        }
        artifact_paths['metadata'].write_text(
            json.dumps(
                {
                    'model_family': 'distance_profile_nearest_normal',
                    'params': {
                        'feature_mode': 'raw_window_nearest_normal_distance',
                        'threshold_quantile': config.threshold_quantile,
                        'score_orientation': 'higher_is_more_anomalous:nearest_train_normal_window_distance',
                        'max_reference_windows': config.max_reference_windows,
                        'max_distance_comparisons_per_batch': config.max_distance_comparisons,
                    },
                    'runtime_guard': {
                        'reference_selection': config.reference_selection,
                        'max_reference_windows': config.max_reference_windows,
                        'reference_window_count': 3,
                        'max_distance_comparisons_per_batch': config.max_distance_comparisons,
                    },
                    'dependency_status': 'implemented_with_existing_numpy_sklearn_only_no_stumpy_dependency',
                    'split': {'train_count': 3, 'validation_count': 2, 'test_count': 2},
                    'test_split_held_out': True,
                }
            )
            + '\n',
            encoding='utf-8',
        )
        return SimpleNamespace(output_dir=output_dir, artifact_paths=artifact_paths, metrics=metrics)

    monkeypatch.setattr(harness, 'train_distance_profile_from_skab', fake_train)

    config = _harness_config(harness, tmp_path)
    record = harness._run_distance_profile_experiment(config)

    trainer_config = captured['config']
    assert isinstance(trainer_config, harness.DistanceProfileTrainingConfig)
    assert trainer_config.input_path == config.output_dir / '12_distance_profile_nearest_normal'
    assert trainer_config.output_dir == config.output_dir / '12_distance_profile_nearest_normal'
    assert trainer_config.split_manifest_path == config.manifest_path
    assert trainer_config.window_size == config.window_size
    assert trainer_config.stride == config.stride
    assert trainer_config.threshold_quantile == 0.95
    assert trainer_config.scaler == 'standard'
    assert trainer_config.max_reference_windows == config.distance_profile_max_reference_windows
    assert trainer_config.max_distance_comparisons == config.distance_profile_max_distance_comparisons
    assert trainer_config.reference_selection == 'evenly_spaced'

    assert record.key == '12_distance_profile_nearest_normal'
    assert record.display_name == 'Nearest-normal distance-profile baseline'
    assert record.model_family == 'distance_profile_nearest_normal'
    assert record.feature_mode == 'raw_window_nearest_normal_distance'
    assert record.status == 'completed'
    assert record.metrics['test_f1'] == 0.35
    assert record.artifact_paths['metrics_csv'].exists()
    assert record.artifact_paths['experiment_metadata'].exists()

    metadata = json.loads(record.artifact_paths['metadata'].read_text(encoding='utf-8'))
    experiment_metadata = json.loads(record.artifact_paths['experiment_metadata'].read_text(encoding='utf-8'))
    for payload in (metadata, experiment_metadata, record.metadata):
        assert payload['key'] == '12_distance_profile_nearest_normal'
        assert payload['split_protocol'] == harness.SPLIT_PROTOCOL
        assert payload['threshold_method'] == 'validation_normal_quantile'
        assert payload['calibration_split'] == 'validation_normal_windows'
        assert payload['score_orientation'] == 'higher_is_more_anomalous:nearest_train_normal_window_distance'
        assert payload['test_labels_used_for_training_or_threshold'] is False
        assert payload['test_scores_used_for_training_or_threshold'] is False
        assert payload['dependency_status'] == 'implemented_with_existing_numpy_sklearn_only_no_stumpy_dependency'
        assert payload['runtime_guard']['max_reference_windows'] == config.distance_profile_max_reference_windows
        assert 'anomaly label 0 only' in payload['train_label_policy']
        assert 'held-out test split' in payload['test_metrics_policy']
    assert metadata['test_split_held_out'] is True
    assert metadata['params']['feature_mode'] == 'raw_window_nearest_normal_distance'
    assert experiment_metadata['notes'] == record.notes


def test_lstm_dbscan_calibration_metadata_is_independent_of_test_labels_and_scores(tmp_path):
    harness = _harness()
    validation_scores_path = _score_csv(
        tmp_path / 'baseline' / 'scores.csv',
        scores=[0.10, 0.11, 0.12, 0.13, 3.00, 3.05, 3.10, 3.15],
        labels=[0, 0, 0, 0, 1, 1, 1, 1],
    )
    honest_test_scores_path = _score_csv(
        tmp_path / 'baseline' / 'test_scores_honest.csv',
        scores=[0.2, 3.2, 3.3, 0.1],
        labels=[0, 1, 1, 0],
    )
    adversarial_test_scores_path = _score_csv(
        tmp_path / 'baseline' / 'test_scores_adversarial.csv',
        scores=[9.0, 9.1, 9.2, 9.3],
        labels=[0, 0, 0, 0],
    )

    honest_record = harness._run_lstm_dbscan_experiment(
        _harness_config(harness, tmp_path, 'honest_artifacts'),
        _baseline_lstm_record(harness, tmp_path, validation_scores_path, honest_test_scores_path),
    )
    adversarial_record = harness._run_lstm_dbscan_experiment(
        _harness_config(harness, tmp_path, 'adversarial_artifacts'),
        _baseline_lstm_record(harness, tmp_path, validation_scores_path, adversarial_test_scores_path),
    )

    assert honest_record.metadata['threshold_rule'] == adversarial_record.metadata['threshold_rule']
    assert honest_record.metrics['threshold'] == adversarial_record.metrics['threshold']
    assert honest_record.metrics['test_f1'] == 1.0
    assert adversarial_record.metrics['test_f1'] == 0.0
    assert honest_record.metadata['threshold_calibration'] == (
        'DBSCAN fit on validation scores only; test labels and scores are not used for selection'
    )
    assert honest_record.metadata['test_split_held_out'] is True
    assert honest_record.artifact_paths['test_scores'].exists()


def test_conformal_calibration_is_independent_of_held_out_test_labels_and_scores(tmp_path):
    harness = _harness()
    validation_scores_path = _score_csv(
        tmp_path / 'source' / 'scores.csv',
        scores=[0.10, 0.20, 0.30, 0.40, 0.50, 99.0],
        labels=[0, 0, 0, 0, 0, 1],
    )
    honest_test_scores_path = _score_csv(
        tmp_path / 'source' / 'test_scores_honest.csv',
        scores=[0.20, 0.70, 0.80, 0.10],
        labels=[0, 1, 1, 0],
    )
    adversarial_test_scores_path = _score_csv(
        tmp_path / 'source' / 'test_scores_adversarial.csv',
        scores=[9.0, 9.1, 9.2, 9.3],
        labels=[0, 0, 0, 0],
    )

    honest_record = harness._run_conformal_threshold_experiment(
        _harness_config(harness, tmp_path, 'honest_artifacts'),
        _baseline_score_record(harness, tmp_path, validation_scores_path, honest_test_scores_path),
        key='10_isolation_forest_conformal',
        display_name='Isolation Forest conformal threshold wrapper',
        model_family='isolation_forest_conformal_threshold',
        feature_mode='spectral',
        alpha=0.4,
    )
    adversarial_record = harness._run_conformal_threshold_experiment(
        _harness_config(harness, tmp_path, 'adversarial_artifacts'),
        _baseline_score_record(harness, tmp_path, validation_scores_path, adversarial_test_scores_path),
        key='10_isolation_forest_conformal',
        display_name='Isolation Forest conformal threshold wrapper',
        model_family='isolation_forest_conformal_threshold',
        feature_mode='spectral',
        alpha=0.4,
    )

    assert honest_record.metadata['threshold_rule'] == adversarial_record.metadata['threshold_rule']
    assert honest_record.metrics['threshold'] == adversarial_record.metrics['threshold'] == 0.40
    assert honest_record.metrics['test_f1'] == 1.0
    assert adversarial_record.metrics['test_f1'] == 0.0
    assert honest_record.metadata['threshold_calibration'] == (
        'conformal threshold derived only from validation-normal score rows; '
        'held-out test labels were not used for thresholding; held-out test scores were not used for thresholding'
    )
    assert honest_record.metadata['test_split_held_out'] is True
    assert honest_record.artifact_paths['test_scores'].exists()


def test_supervised_rows_skip_honestly_when_manifest_train_is_normal_only(monkeypatch, tmp_path):
    harness = _harness()

    def raise_normal_only_train(_config):
        raise ValueError('training split must contain both binary classes, got [0]')

    monkeypatch.setattr(harness, 'train_supervised_from_skab', raise_normal_only_train)

    record = harness._run_supervised_experiment(
        _harness_config(harness, tmp_path),
        '06_xgboost',
        'XGBoost supervised upper bound',
        'xgboost',
    )

    assert record.status == 'skipped'
    assert record.metrics['status'] == 'skipped'
    assert 'training split must contain both binary classes' in record.metrics['reason']
    assert 'supervised models require labels in manifest train split' in record.metrics['reason']
    assert 'test_f1' not in record.metrics

    metrics = json.loads(record.artifact_paths['metrics'].read_text(encoding='utf-8'))
    experiment_metadata = json.loads(record.artifact_paths['experiment_metadata'].read_text(encoding='utf-8'))
    assert metrics == record.metrics
    assert experiment_metadata['status'] == 'skipped'
    assert experiment_metadata['split_protocol'] == harness.SPLIT_PROTOCOL
    assert experiment_metadata['reason'] == record.notes
