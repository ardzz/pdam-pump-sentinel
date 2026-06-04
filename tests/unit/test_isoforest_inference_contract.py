from importlib import import_module

from app.models.anomaly_event import build_anomaly_payload_from_verdict
from ml.datasets.skab_loader import SENSOR_COLUMNS, iter_telemetry_records, load_skab_csv


def _write_skab_csv(path, row_count=16, anomaly_start=None, offset=0.0):
    rows = [['datetime', *SENSOR_COLUMNS, 'anomaly', 'changepoint']]
    for row_index in range(row_count):
        anomaly = int(anomaly_start is not None and row_index >= anomaly_start)
        sensor_values = []
        for column_index, _column in enumerate(SENSOR_COLUMNS):
            baseline = 5.0 + offset + row_index * 0.04 + column_index * 0.12
            sensor_values.append(f'{baseline + anomaly * 10.0:.6f}')
        rows.append([
            f'2024-01-01T00:00:{row_index:02d}Z',
            *sensor_values,
            str(anomaly),
            '0',
        ])
    path.write_text('\n'.join(';'.join(row) for row in rows) + '\n')
    return path


def test_isoforest_service_observe_warms_up_then_scores_window(tmp_path):
    isoforest_inference = import_module('ml.inference.isoforest_inference')
    train_isoforest = import_module('ml.training.train_isoforest')
    input_path = _write_skab_csv(tmp_path / 'train.csv')
    train_isoforest.train_isoforest_from_skab(
        train_isoforest.IsoForestTrainingConfig(
            input_path=input_path,
            output_dir=tmp_path / 'artifacts',
            window_size=4,
            stride=4,
            threshold_quantile=0.9,
            scaler='standard',
            feature_mode='spectral',
            n_estimators=8,
            seed=7,
        )
    )

    service = isoforest_inference.IsoForestAnomalyInferenceService.from_artifacts(tmp_path / 'artifacts')
    assert service.window_size == 4
    assert service.sensor_columns == tuple(SENSOR_COLUMNS)
    assert service.feature_mode == 'spectral'
    assert service.model_version == 'isoforest-local'

    records = list(iter_telemetry_records(load_skab_csv(input_path), station='ipa_01'))[:4]
    warmup = service.observe('ipa_01', records[0]['timestamp'], records[0]['sensors'])
    assert isinstance(warmup, isoforest_inference.IsoForestAnomalyVerdict)
    assert warmup.window_filled is False
    assert warmup.anomaly is None and warmup.t2 is None and warmup.score is None
    assert warmup.t2_threshold == service.threshold

    for record in records[1:3]:
        service.observe('ipa_01', record['timestamp'], record['sensors'])
    scored = service.observe('ipa_01', records[3]['timestamp'], records[3]['sensors'])

    assert scored.window_filled is True
    assert scored.t2 is None and scored.q is None
    assert isinstance(scored.score, float)
    assert scored.anomaly in (0, 1)
    assert scored.t2_threshold == service.threshold
    assert scored.q_threshold == service.threshold
    assert scored.top_contributing_sensor in SENSOR_COLUMNS

    payload = build_anomaly_payload_from_verdict(scored)
    assert payload['model_version'] == 'isoforest-local'
    assert payload['score'] == scored.score
    assert payload['anomaly'] in (0, 1)


def test_isoforest_service_from_artifacts_accepts_explicit_model_version(tmp_path):
    isoforest_inference = import_module('ml.inference.isoforest_inference')
    train_isoforest = import_module('ml.training.train_isoforest')
    input_path = _write_skab_csv(tmp_path / 'train.csv')
    artifacts = tmp_path / 'artifacts'
    train_isoforest.train_isoforest_from_skab(
        train_isoforest.IsoForestTrainingConfig(
            input_path=input_path,
            output_dir=artifacts,
            window_size=2,
            stride=2,
            threshold_quantile=0.9,
            scaler='none',
            feature_mode='raw',
            n_estimators=4,
        )
    )

    service = isoforest_inference.IsoForestAnomalyInferenceService.from_artifacts(artifacts, model_version='if-v1')
    assert service.model_version == 'if-v1'
    assert service.feature_mode == 'raw'

    first = next(iter_telemetry_records(load_skab_csv(input_path), station='ipa_01'))
    verdict = service.observe('ipa_01', first['timestamp'], first['sensors'])
    assert verdict.model_version == 'if-v1'


def test_isoforest_inference_exported_from_package():
    inference = import_module('ml.inference')
    isoforest_inference = import_module('ml.inference.isoforest_inference')
    assert inference.IsoForestAnomalyInferenceService is isoforest_inference.IsoForestAnomalyInferenceService
    assert inference.IsoForestAnomalyVerdict is isoforest_inference.IsoForestAnomalyVerdict
