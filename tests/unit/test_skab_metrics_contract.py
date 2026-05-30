from importlib import import_module

import pytest


def _metrics():
    return import_module('ml.evaluation.metrics')


def test_point_classification_metrics_preserve_existing_far_semantics():
    metrics = _metrics()

    result = metrics.point_classification_metrics(
        labels=[0, 0, 1, 1, 0, 1],
        predictions=[0, 1, 1, 0, 0, 1],
    )

    assert result == {
        'precision': pytest.approx(2 / 3),
        'recall': pytest.approx(2 / 3),
        'f1': pytest.approx(2 / 3),
        'false_alarm_rate': pytest.approx(1 / 3),
    }
    assert metrics.point_classification_metrics([0, 0], [0, 0]) == {
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'false_alarm_rate': 0.0,
    }


def test_score_threshold_free_metrics_return_auc_or_none_for_single_class():
    metrics = _metrics()

    result = metrics.score_threshold_free_metrics(labels=[0, 0, 1, 1], scores=[0.1, 0.4, 0.35, 0.8])

    assert result['pr_auc'] == pytest.approx(5 / 6)
    assert result['roc_auc'] == pytest.approx(0.75)
    assert metrics.score_threshold_free_metrics(labels=[0, 0, 0], scores=[0.1, 0.2, 0.3]) == {
        'pr_auc': None,
        'roc_auc': None,
    }


def test_contiguous_ranges_use_half_open_positive_runs():
    metrics = _metrics()

    assert metrics.contiguous_ranges([0, 1, 1, 0, 1, 0, 1, 1, 1]) == [(1, 3), (4, 5), (6, 9)]
    assert metrics.contiguous_ranges([0, 0, 0]) == []


def test_event_metrics_detect_events_misses_false_alarms_and_delay():
    metrics = _metrics()

    result = metrics.event_metrics(
        labels=[0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
        predictions=[1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1],
    )

    assert result == {
        'event_count': 2,
        'event_recall': pytest.approx(0.5),
        'missed_events': 1,
        'false_alarm_events': 3,
        'mean_detection_delay_windows': pytest.approx(2.0),
    }
    assert metrics.event_metrics([0, 0, 0], [0, 1, 1]) == {
        'event_count': 0,
        'event_recall': None,
        'missed_events': 0,
        'false_alarm_events': 1,
        'mean_detection_delay_windows': None,
    }


def test_evaluate_split_emits_transient_exclusion_suffix_metrics():
    metrics = _metrics()

    result = metrics.evaluate_split(
        labels=[0, 1, 1, 0, 1],
        predictions=[0, 1, 0, 1, 1],
        scores=[0.1, 0.8, 0.2, 0.7, 0.9],
        transient_mask=[False, False, True, True, False],
    )

    assert result['precision'] == pytest.approx(2 / 3)
    assert result['recall'] == pytest.approx(2 / 3)
    assert result['pr_auc'] is not None
    assert result['precision_excluding_transient'] == pytest.approx(1.0)
    assert result['recall_excluding_transient'] == pytest.approx(1.0)
    assert result['false_alarm_rate_excluding_transient'] == pytest.approx(0.0)
    assert result['roc_auc_excluding_transient'] == pytest.approx(1.0)
    assert result['event_count_excluding_transient'] == 1
    assert result['mean_detection_delay_windows_excluding_transient'] == pytest.approx(0.0)
