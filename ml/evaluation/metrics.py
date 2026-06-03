from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

MetricValue = float | int | None


def point_classification_metrics(labels: Any, predictions: Any) -> dict[str, float]:
    """Return point-wise precision/recall/F1/FAR with the existing PCA semantics.

    Undefined denominators intentionally resolve to 0.0 to preserve the current
    training helper contract in ml.training.train_pca.
    """

    labels_array, predictions_array = _binary_pair(labels, predictions)

    positive = labels_array == 1
    negative = labels_array == 0
    predicted_positive = predictions_array == 1
    predicted_negative = predictions_array == 0

    true_positive = int(np.count_nonzero(predicted_positive & positive))
    false_positive = int(np.count_nonzero(predicted_positive & negative))
    false_negative = int(np.count_nonzero(predicted_negative & positive))
    true_negative = int(np.count_nonzero(predicted_negative & negative))

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    return {
        'precision': precision,
        'recall': recall,
        'f1': _safe_divide(2.0 * precision * recall, precision + recall),
        'false_alarm_rate': _safe_divide(false_positive, false_positive + true_negative),
    }


def score_threshold_free_metrics(labels: Any, scores: Any) -> dict[str, float | None]:
    """Return PR-AUC and ROC-AUC, or None when undefined/unavailable."""

    labels_array = _as_binary_array(labels, 'labels')
    scores_array = _as_float_array(scores, 'scores')
    _require_same_length(labels_array, scores_array, 'labels', 'scores')

    if labels_array.size == 0 or np.unique(labels_array).size < 2:
        return {'pr_auc': None, 'roc_auc': None}

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception:
        return {'pr_auc': None, 'roc_auc': None}

    return {
        'pr_auc': _safe_metric_call(average_precision_score, labels_array, scores_array),
        'roc_auc': _safe_metric_call(roc_auc_score, labels_array, scores_array),
    }


def contiguous_ranges(labels: Any) -> list[tuple[int, int]]:
    """Return half-open [start, end) ranges for contiguous positive labels."""

    labels_array = _as_binary_array(labels, 'labels')
    ranges: list[tuple[int, int]] = []
    start: int | None = None

    for index, value in enumerate(labels_array):
        if value == 1 and start is None:
            start = index
        elif value == 0 and start is not None:
            ranges.append((start, index))
            start = None

    if start is not None:
        ranges.append((start, int(labels_array.size)))

    return ranges


def event_metrics(labels: Any, predictions: Any) -> dict[str, MetricValue]:
    """Evaluate predictions against contiguous ground-truth anomaly events."""

    labels_array, predictions_array = _binary_pair(labels, predictions)
    label_ranges = contiguous_ranges(labels_array)
    prediction_ranges = contiguous_ranges(predictions_array)

    detection_delays: list[int] = []
    for start, end in label_ranges:
        predicted_offsets = np.flatnonzero(predictions_array[start:end] == 1)
        if predicted_offsets.size:
            detection_delays.append(int(predicted_offsets[0]))

    event_count = len(label_ranges)
    detected_events = len(detection_delays)
    false_alarm_events = sum(
        1 for prediction_range in prediction_ranges if not any(_ranges_overlap(prediction_range, r) for r in label_ranges)
    )

    return {
        'event_count': event_count,
        'event_recall': None if event_count == 0 else float(detected_events / event_count),
        'missed_events': event_count - detected_events,
        'false_alarm_events': int(false_alarm_events),
        'mean_detection_delay_windows': None
        if not detection_delays
        else float(sum(detection_delays) / len(detection_delays)),
    }


def composite_f_score(labels: Any, predictions: Any) -> dict[str, MetricValue]:
    labels_array, predictions_array = _binary_pair(labels, predictions)
    positive = labels_array == 1
    negative = labels_array == 0
    predicted_positive = predictions_array == 1
    true_positive = int(np.count_nonzero(predicted_positive & positive))
    false_positive = int(np.count_nonzero(predicted_positive & negative))
    point_precision = _safe_divide(true_positive, true_positive + false_positive)

    label_ranges = contiguous_ranges(labels_array)
    prediction_ranges = contiguous_ranges(predictions_array)
    event_recall = _event_recall(label_ranges, prediction_ranges)
    composite_f1 = None if event_recall is None else _safe_divide(2.0 * point_precision * event_recall, point_precision + event_recall)
    return {
        'point_precision': point_precision,
        'event_recall': event_recall,
        'composite_f1': composite_f1,
    }


def event_precision_recall(labels: Any, predictions: Any) -> dict[str, MetricValue]:
    labels_array, predictions_array = _binary_pair(labels, predictions)
    label_ranges = contiguous_ranges(labels_array)
    prediction_ranges = contiguous_ranges(predictions_array)

    event_precision = None
    if prediction_ranges:
        event_precision = float(_matched_range_count(prediction_ranges, label_ranges) / len(prediction_ranges))
    event_recall = _event_recall(label_ranges, prediction_ranges)
    event_f1 = _harmonic_mean_or_none(event_precision, event_recall)
    return {
        'event_precision': event_precision,
        'event_recall': event_recall,
        'event_f1': event_f1,
    }


def evaluate_events(labels: Any, predictions: Any) -> dict[str, MetricValue]:
    return composite_f_score(labels, predictions) | event_precision_recall(labels, predictions)


def evaluate_split(labels: Any, predictions: Any, scores: Any, transient_mask: Any | None = None) -> dict[str, MetricValue]:
    """Return point, threshold-free, and event metrics for one evaluation split.

    When transient_mask is supplied, all metrics are recomputed after dropping
    mask=True windows and emitted with the ``_excluding_transient`` suffix.
    """

    labels_array, predictions_array = _binary_pair(labels, predictions)
    scores_array = _as_float_array(scores, 'scores')
    _require_same_length(labels_array, scores_array, 'labels', 'scores')

    metrics = _evaluate_arrays(labels_array, predictions_array, scores_array)
    if transient_mask is None:
        return metrics

    transient_array = _as_bool_array(transient_mask, 'transient_mask')
    _require_same_length(labels_array, transient_array, 'labels', 'transient_mask')
    keep = ~transient_array
    non_transient_metrics = _evaluate_arrays(labels_array[keep], predictions_array[keep], scores_array[keep])
    metrics.update({f'{name}_excluding_transient': value for name, value in non_transient_metrics.items()})
    return metrics


def _evaluate_arrays(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict[str, MetricValue]:
    return {
        **point_classification_metrics(labels, predictions),
        **score_threshold_free_metrics(labels, scores),
        **event_metrics(labels, predictions),
    }


def _binary_pair(labels: Any, predictions: Any) -> tuple[np.ndarray, np.ndarray]:
    labels_array = _as_binary_array(labels, 'labels')
    predictions_array = _as_binary_array(predictions, 'predictions')
    _require_same_length(labels_array, predictions_array, 'labels', 'predictions')
    return labels_array, predictions_array


def _as_binary_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        return array.astype(int)

    numeric = array.astype(int)
    if not np.array_equal(array, numeric) or not np.isin(numeric, [0, 1]).all():
        raise ValueError(f'{name} must contain only binary 0/1 values')
    return numeric


def _as_float_array(values: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    return array


def _as_bool_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        return array.astype(bool)
    if not np.isin(array, [False, True, 0, 1]).all():
        raise ValueError(f'{name} must contain only boolean values')
    return array.astype(bool)


def _require_same_length(left: np.ndarray, right: np.ndarray, left_name: str, right_name: str) -> None:
    if left.size != right.size:
        raise ValueError(f'{left_name} and {right_name} must have the same length')


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _safe_metric_call(metric_func: Any, labels: np.ndarray, scores: np.ndarray) -> float | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            value = float(metric_func(labels, scores))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _event_recall(label_ranges: list[tuple[int, int]], prediction_ranges: list[tuple[int, int]]) -> float | None:
    return None if not label_ranges else float(_matched_range_count(label_ranges, prediction_ranges) / len(label_ranges))


def _matched_range_count(left_ranges: list[tuple[int, int]], right_ranges: list[tuple[int, int]]) -> int:
    return sum(1 for left_range in left_ranges if any(_ranges_overlap(left_range, right_range) for right_range in right_ranges))


def _harmonic_mean_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return _safe_divide(2.0 * left * right, left + right)


def _ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]
