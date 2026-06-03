import json
from importlib import import_module
from typing import Any

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from app.models.anomaly_event import build_anomaly_payload_from_verdict
from ml.inference.lstm_ae_inference import LstmAeAnomalyInferenceService, LstmAeAnomalyVerdict


def _write_lstm_artifacts(path):
    keras = import_module('keras')
    joblib = import_module('joblib')
    path.mkdir()

    inputs = keras.Input(shape=(2, 2))
    outputs = keras.layers.TimeDistributed(
        keras.layers.Dense(2, kernel_initializer='zeros', bias_initializer='zeros')
    )(inputs)
    model = keras.Model(inputs, outputs)
    model.save(str(path / 'lstm_ae.keras'))

    scaler = StandardScaler().fit(np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float))
    joblib.dump(scaler, path / 'scaler.joblib')
    (path / 'metadata.json').write_text(
        json.dumps(
            {
                'model_family': 'lstm_ae',
                'sensor_columns': ['a', 'b'],
                'params': {'window_size': 2, 'threshold': 0.5, 'score_type': 'mae'},
                'thresholds': {'threshold': 0.5, 't2_threshold': 0.5, 'q_threshold': 0.5},
            }
        )
        + '\n'
    )
    return path


def test_lstm_ae_service_loads_warms_up_scores_and_maps_payload(tmp_path):
    artifact_dir = _write_lstm_artifacts(tmp_path / 'artifacts')
    service = LstmAeAnomalyInferenceService.from_artifacts(artifact_dir, model_version='v7')

    assert service.window_size == 2
    assert service.sensor_columns == ('a', 'b')
    assert service.model_version == 'v7'
    assert service.threshold == 0.5

    warmup = service.observe('s1', 't0', {'a': 10.0, 'b': 0.0})
    assert isinstance(warmup, LstmAeAnomalyVerdict)
    assert warmup.window_filled is False
    assert warmup.score is None and warmup.anomaly is None and warmup.t2 is None and warmup.q is None

    verdict = service.observe('s1', 't1', {'a': 10.0, 'b': 0.0})
    assert verdict.window_filled is True
    assert verdict.t2 is None and verdict.q is None
    assert verdict.t2_threshold == 0.5 and verdict.q_threshold == 0.5
    assert isinstance(verdict.score, float)
    assert verdict.anomaly == 1
    assert verdict.top_contributing_sensor == 'a'
    assert verdict.as_dict()['model_version'] == 'v7'

    payload = build_anomaly_payload_from_verdict(verdict)
    assert payload['station'] == 's1'
    assert payload['source_timestamp'] == 't1'
    assert payload['status'] == 'ok'
    assert payload['score'] == verdict.score
    assert payload['anomaly'] == 1


def test_lstm_ae_service_keeps_buffers_independent_and_resets(tmp_path):
    service = LstmAeAnomalyInferenceService.from_artifacts(_write_lstm_artifacts(tmp_path / 'artifacts'))

    service.observe('s1', 't0', {'a': 1.0, 'b': 1.0})
    other = service.observe('s2', 't0', {'a': 1.0, 'b': 1.0})
    assert other.window_filled is False

    service.observe('s1', 't1', {'a': 1.0, 'b': 1.0})
    service.reset('s1')
    after_reset = service.observe('s1', 't2', {'a': 1.0, 'b': 1.0})
    assert after_reset.window_filled is False


def test_lstm_ae_service_rejects_bad_inputs(tmp_path):
    service = LstmAeAnomalyInferenceService.from_artifacts(_write_lstm_artifacts(tmp_path / 'artifacts'))

    with pytest.raises(KeyError):
        service.observe('s1', 't0', {'a': 1.0})
    not_a_mapping: Any = [1.0, 2.0]
    with pytest.raises(TypeError):
        service.observe('s1', 't0', not_a_mapping)
