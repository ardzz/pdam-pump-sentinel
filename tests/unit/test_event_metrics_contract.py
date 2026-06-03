from importlib import import_module


def _metrics():
    return import_module('ml.evaluation.metrics')


def test_event_metrics_partial_segment_detection_uses_point_precision_and_event_recall():
    metrics = _metrics()
    labels = [0, 1, 1, 0, 1, 0]
    predictions = [0, 1, 0, 0, 0, 0]

    assert metrics.composite_f_score(labels, predictions) == {
        'point_precision': 1.0,
        'event_recall': 0.5,
        'composite_f1': 2 / 3,
    }
    assert metrics.event_precision_recall(labels, predictions) == {
        'event_precision': 1.0,
        'event_recall': 0.5,
        'event_f1': 2 / 3,
    }
    assert metrics.evaluate_events(labels, predictions) == {
        'point_precision': 1.0,
        'event_recall': 0.5,
        'composite_f1': 2 / 3,
        'event_precision': 1.0,
        'event_f1': 2 / 3,
    }


def test_event_metrics_no_ground_truth_anomalies_keep_recall_undefined():
    metrics = _metrics()
    labels = [0, 0, 0, 0]
    predictions = [0, 1, 1, 0]

    assert metrics.composite_f_score(labels, predictions) == {
        'point_precision': 0.0,
        'event_recall': None,
        'composite_f1': None,
    }
    assert metrics.event_precision_recall(labels, predictions) == {
        'event_precision': 0.0,
        'event_recall': None,
        'event_f1': None,
    }


def test_event_metrics_all_anomalies_count_any_overlapping_prediction_segments():
    metrics = _metrics()
    labels = [1, 1, 1, 1]
    predictions = [1, 0, 1, 0]

    assert metrics.composite_f_score(labels, predictions) == {
        'point_precision': 1.0,
        'event_recall': 1.0,
        'composite_f1': 1.0,
    }
    assert metrics.event_precision_recall(labels, predictions) == {
        'event_precision': 1.0,
        'event_recall': 1.0,
        'event_f1': 1.0,
    }


def test_event_metrics_no_predictions_keep_event_precision_undefined():
    metrics = _metrics()
    labels = [0, 1, 1, 0]
    predictions = [0, 0, 0, 0]

    assert metrics.composite_f_score(labels, predictions) == {
        'point_precision': 0.0,
        'event_recall': 0.0,
        'composite_f1': 0.0,
    }
    assert metrics.event_precision_recall(labels, predictions) == {
        'event_precision': None,
        'event_recall': 0.0,
        'event_f1': None,
    }


def test_event_metrics_perfect_prediction_scores_one():
    metrics = _metrics()
    labels = [0, 1, 1, 0, 1]
    predictions = [0, 1, 1, 0, 1]

    assert metrics.composite_f_score(labels, predictions) == {
        'point_precision': 1.0,
        'event_recall': 1.0,
        'composite_f1': 1.0,
    }
    assert metrics.event_precision_recall(labels, predictions) == {
        'event_precision': 1.0,
        'event_recall': 1.0,
        'event_f1': 1.0,
    }
