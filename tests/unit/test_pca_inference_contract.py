from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.models.anomaly_event import build_anomaly_payload_from_verdict
from ml.datasets.skab_loader import SENSOR_COLUMNS, iter_telemetry_records, load_skab_csv
from ml.inference.pca_inference import AnomalyVerdict, PcaAnomalyInferenceService
from ml.training.pca_detector import PcaT2QDetector
from ml.training.train_pca import PcaTrainingConfig, train_pca_from_skab

_FIXTURE = Path(__file__).parent.parent / 'fixtures' / 'skab_tiny.csv'


def _fit_detector(window_size: int, sensor_count: int) -> PcaT2QDetector:
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 1.0, size=(200, window_size * sensor_count))
    return PcaT2QDetector(n_components=2, threshold_quantile=0.95, scaler='standard').fit(normal)


def test_service_warms_up_then_scores_window():
    detector = _fit_detector(window_size=2, sensor_count=2)
    service = PcaAnomalyInferenceService(detector, ['a', 'b'], window_size=2, model_version='test')

    warmup = service.observe('s1', 't0', {'a': 0.0, 'b': 0.0})
    assert isinstance(warmup, AnomalyVerdict)
    assert warmup.window_filled is False
    assert warmup.anomaly is None and warmup.t2 is None and warmup.score is None
    assert warmup.t2_threshold == float(detector.t2_threshold_)

    scored = service.observe('s1', 't1', {'a': 0.0, 'b': 0.0})
    assert scored.window_filled is True
    assert isinstance(scored.t2, float) and isinstance(scored.q, float)
    assert scored.anomaly in (0, 1)
    assert scored.top_contributing_sensor in ('a', 'b')


def test_service_flags_extreme_window_as_anomaly():
    detector = _fit_detector(window_size=2, sensor_count=2)
    service = PcaAnomalyInferenceService(detector, ['a', 'b'], window_size=2)

    service.observe('s1', 't0', {'a': 100.0, 'b': 100.0})
    verdict = service.observe('s1', 't1', {'a': 100.0, 'b': 100.0})
    assert verdict.anomaly == 1


def test_service_keeps_station_buffers_independent():
    detector = _fit_detector(window_size=2, sensor_count=2)
    service = PcaAnomalyInferenceService(detector, ['a', 'b'], window_size=2)

    service.observe('s1', 't0', {'a': 0.0, 'b': 0.0})
    other = service.observe('s2', 't0', {'a': 0.0, 'b': 0.0})
    assert other.window_filled is False


def test_service_reset_restarts_warmup():
    detector = _fit_detector(window_size=2, sensor_count=2)
    service = PcaAnomalyInferenceService(detector, ['a', 'b'], window_size=2)

    service.observe('s1', 't0', {'a': 0.0, 'b': 0.0})
    service.observe('s1', 't1', {'a': 0.0, 'b': 0.0})
    service.reset('s1')
    after_reset = service.observe('s1', 't2', {'a': 0.0, 'b': 0.0})
    assert after_reset.window_filled is False


def test_service_rejects_bad_inputs():
    detector = _fit_detector(window_size=2, sensor_count=2)
    service = PcaAnomalyInferenceService(detector, ['a', 'b'], window_size=2)

    with pytest.raises(KeyError):
        service.observe('s1', 't0', {'a': 1.0})
    not_a_mapping: Any = [1.0, 2.0]
    with pytest.raises(TypeError):
        service.observe('s1', 't0', not_a_mapping)


def test_service_rejects_feature_dimension_mismatch():
    detector = _fit_detector(window_size=2, sensor_count=2)
    with pytest.raises(ValueError):
        PcaAnomalyInferenceService(detector, ['a', 'b'], window_size=3)


def test_service_loads_from_train_pca_artifacts(tmp_path):
    train_pca_from_skab(
        PcaTrainingConfig(
            input_path=_FIXTURE,
            output_dir=tmp_path,
            window_size=1,
            stride=1,
            n_components=0.9,
            threshold_quantile=0.95,
            scaler='robust',
        )
    )

    service = PcaAnomalyInferenceService.from_artifacts(tmp_path)
    assert service.window_size == 1
    assert service.sensor_columns == tuple(SENSOR_COLUMNS)
    assert service.model_version == 'PumpAD'

    record = next(iter_telemetry_records(load_skab_csv(_FIXTURE), station='ipa_01'))
    verdict = service.observe('ipa_01', record['timestamp'], record['sensors'])
    assert verdict.window_filled is True
    assert isinstance(verdict.t2, float) and isinstance(verdict.q, float)
    assert verdict.anomaly in (0, 1)


def test_build_anomaly_payload_from_verdict_maps_fields():
    scored = AnomalyVerdict(
        station='ipa_01',
        timestamp='2024-01-01T00:00:00Z',
        window_filled=True,
        window_size=60,
        model_version='PumpAD',
        t2_threshold=1.0,
        q_threshold=2.0,
        t2=3.0,
        q=0.5,
        score=3.0,
        anomaly=1,
        top_contributing_sensor='Pressure',
    )
    payload = build_anomaly_payload_from_verdict(scored)
    assert payload['station'] == 'ipa_01'
    assert payload['source_timestamp'] == '2024-01-01T00:00:00Z'
    assert payload['status'] == 'ok'
    assert payload['anomaly'] == 1
    assert payload['t2'] == 3.0 and payload['q_threshold'] == 2.0
    assert payload['top_contributing_sensor'] == 'Pressure'

    warming = AnomalyVerdict(
        station='ipa_01',
        timestamp='2024-01-01T00:00:01Z',
        window_filled=False,
        window_size=60,
        model_version='PumpAD',
        t2_threshold=1.0,
        q_threshold=2.0,
    )
    warming_payload = build_anomaly_payload_from_verdict(warming)
    assert warming_payload['status'] == 'warming_up'
    assert warming_payload['anomaly'] is None
