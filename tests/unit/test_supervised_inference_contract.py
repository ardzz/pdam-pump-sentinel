import json
from importlib import import_module

from app.models.anomaly_event import build_anomaly_payload_from_verdict
from ml.datasets.skab_loader import SENSOR_COLUMNS, iter_telemetry_records, load_skab_csv


def _write_skab_csv(path, row_count=24, anomaly_start=None, offset=0.0):
    rows = [['datetime', *SENSOR_COLUMNS, 'anomaly', 'changepoint']]
    for row_index in range(row_count):
        anomaly = int(anomaly_start is not None and row_index >= anomaly_start)
        sensor_values = []
        for column_index, _column in enumerate(SENSOR_COLUMNS):
            baseline = 5.0 + offset + row_index * 0.04 + column_index * 0.12
            sensor_values.append(f'{baseline + anomaly * (9.0 + column_index):.6f}')
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
        'train': [str(item.relative_to(path.parent)) for item in train],
        'validation': [str(item.relative_to(path.parent)) for item in validation],
        'test': [str(item.relative_to(path.parent)) for item in test],
    }
    path.write_text(json.dumps(payload, indent=2) + '\n')
    return path


def _train_artifacts(tmp_path):
    train_supervised = import_module('ml.training.train_supervised')
    train = _write_skab_csv(tmp_path / 'train.csv', anomaly_start=12, offset=0.0)
    validation = _write_skab_csv(tmp_path / 'validation.csv', anomaly_start=12, offset=80.0)
    test = _write_skab_csv(tmp_path / 'test.csv', anomaly_start=12, offset=120.0)
    manifest = _write_split_manifest(tmp_path / 'manifest.json', [train], [validation], [test])
    artifacts = tmp_path / 'artifacts'
    train_supervised.train_supervised_from_skab(
        train_supervised.SupervisedTrainingConfig(
            input_path=tmp_path / 'unused.csv',
            output_dir=artifacts,
            split_manifest_path=manifest,
            window_size=4,
            stride=4,
            scaler='standard',
            model_type='lightgbm',
            n_estimators=12,
            learning_rate=0.2,
            early_stopping_rounds=3,
            seed=7,
        )
    )
    return train, artifacts


def test_supervised_service_observe_warms_up_then_scores_window(tmp_path):
    supervised_inference = import_module('ml.inference.supervised_inference')
    input_path, artifacts = _train_artifacts(tmp_path)

    service = supervised_inference.SupervisedAnomalyInferenceService.from_artifacts(artifacts)
    assert service.window_size == 4
    assert service.sensor_columns == tuple(SENSOR_COLUMNS)
    assert service.model_family == 'lightgbm'
    assert service.model_version == 'lightgbm-local'

    records = list(iter_telemetry_records(load_skab_csv(input_path), station='ipa_01'))[:4]
    warmup = service.observe('ipa_01', records[0]['timestamp'], records[0]['sensors'])
    assert isinstance(warmup, supervised_inference.SupervisedAnomalyVerdict)
    assert warmup.window_filled is False
    assert warmup.anomaly is None and warmup.t2 is None and warmup.score is None
    assert warmup.t2_threshold == service.threshold

    for record in records[1:3]:
        service.observe('ipa_01', record['timestamp'], record['sensors'])
    scored = service.observe('ipa_01', records[3]['timestamp'], records[3]['sensors'])

    assert scored.window_filled is True
    assert scored.t2 is None and scored.q is None
    assert isinstance(scored.score, float)
    assert 0.0 <= scored.score <= 1.0
    assert scored.anomaly in (0, 1)
    assert scored.t2_threshold == service.threshold
    assert scored.q_threshold == service.threshold
    assert scored.top_contributing_sensor in (*SENSOR_COLUMNS, None)

    payload = build_anomaly_payload_from_verdict(scored)
    assert payload['model_version'] == 'lightgbm-local'
    assert payload['score'] == scored.score
    assert payload['anomaly'] in (0, 1)


def test_supervised_service_from_artifacts_accepts_explicit_model_version(tmp_path):
    supervised_inference = import_module('ml.inference.supervised_inference')
    input_path, artifacts = _train_artifacts(tmp_path)

    service = supervised_inference.SupervisedAnomalyInferenceService.from_artifacts(artifacts, model_version='gbm-v1')
    assert service.model_version == 'gbm-v1'

    first = next(iter_telemetry_records(load_skab_csv(input_path), station='ipa_01'))
    verdict = service.observe('ipa_01', first['timestamp'], first['sensors'])
    assert verdict.model_version == 'gbm-v1'


def test_supervised_inference_exported_from_package():
    inference = import_module('ml.inference')
    supervised_inference = import_module('ml.inference.supervised_inference')
    assert inference.SupervisedAnomalyInferenceService is supervised_inference.SupervisedAnomalyInferenceService
    assert inference.SupervisedAnomalyVerdict is supervised_inference.SupervisedAnomalyVerdict
