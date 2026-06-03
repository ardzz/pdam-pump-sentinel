from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real


def should_promote(
    champion_metrics: Mapping[str, float | None],
    challenger_metrics: Mapping[str, float | None],
    f1_margin: float = 0.02,
    far_guard: float = 1.05,
) -> tuple[bool, str]:
    challenger_f1 = _metric(challenger_metrics, 'f1')
    challenger_far = _metric(challenger_metrics, 'false_alarm_rate')
    if challenger_f1 is None or challenger_far is None:
        return False, 'challenger metrics missing f1 or false_alarm_rate'

    champion_f1 = _metric(champion_metrics, 'f1')
    champion_far = _metric(champion_metrics, 'false_alarm_rate')
    if champion_f1 is None or champion_far is None:
        return True, 'no incumbent champion; challenger metrics are valid'

    required_f1 = champion_f1 + float(f1_margin)
    if challenger_f1 <= required_f1:
        return False, f'challenger f1 {challenger_f1:.6g} must exceed {required_f1:.6g}'

    max_far = champion_far * float(far_guard)
    if challenger_far > max_far:
        return False, f'challenger false_alarm_rate {challenger_far:.6g} exceeds guard {max_far:.6g}'

    return True, 'challenger beats f1 margin and satisfies false_alarm_rate guard'


def _metric(metrics: Mapping[str, float | None], name: str) -> float | None:
    value = metrics.get(name)
    if isinstance(value, bool) or value is None or not isinstance(value, Real):
        return None
    metric = float(value)
    return metric if math.isfinite(metric) else None


__all__ = ['should_promote']
