from __future__ import annotations

from typing import Any


class _Value:
    def __init__(self) -> None:
        self._value = 0.0

    def get(self) -> float:
        return self._value

    def set(self, value: float) -> None:
        self._value = float(value)

    def inc(self, amount: float = 1.0) -> None:
        self._value += float(amount)


class _MetricChild:
    def __init__(self) -> None:
        self._value = _Value()

    def set(self, value: float) -> None:
        self._value.set(value)

    def inc(self, amount: float = 1.0) -> None:
        self._value.inc(amount)

    def observe(self, value: float) -> None:
        self._value.inc(value)


class _Metric:
    _type = 'untyped'

    def __init__(self, name: str, documentation: str, labelnames: list[str] | tuple[str, ...] = (), **kwargs: Any) -> None:
        self._name = name
        self._documentation = documentation
        self._labelnames = tuple(labelnames)
        self._metrics: dict[tuple[str, ...], _MetricChild] = {}
        self._value = _Value()
        self._kwargs = kwargs

    def labels(self, *labelvalues: Any, **labelkwargs: Any) -> _MetricChild:
        if labelkwargs:
            key = tuple(str(labelkwargs[name]) for name in self._labelnames)
        else:
            key = tuple(str(value) for value in labelvalues)
        if len(key) != len(self._labelnames):
            raise ValueError('incorrect label count')
        child = self._metrics.get(key)
        if child is None:
            child = _MetricChild()
            self._metrics[key] = child
        return child

    def clear(self) -> None:
        self._metrics.clear()

    def set(self, value: float) -> None:
        self._value.set(value)

    def inc(self, amount: float = 1.0) -> None:
        self._value.inc(amount)

    def observe(self, value: float) -> None:
        self._value.inc(value)


class Gauge(_Metric):
    _type = 'gauge'


class Histogram(_Metric):
    _type = 'histogram'


class Counter(_Metric):
    _type = 'counter'

    def __init__(self, name: str, documentation: str, labelnames: list[str] | tuple[str, ...] = (), **kwargs: Any) -> None:
        super().__init__(name.removesuffix('_total'), documentation, labelnames, **kwargs)
