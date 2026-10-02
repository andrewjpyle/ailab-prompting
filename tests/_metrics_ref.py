"""Frozen reference copy of the classification metric core.

The production copy lives in ``ailab_prompting.metrics``. ``tests/test_metric_parity.py``
asserts the two agree on a shared fixture, so the production maths cannot silently drift from
this reference (the metric-parity test approach ported from ailab-rag @ 3cd8f82).
"""

from __future__ import annotations

from collections.abc import Sequence


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)


def _per_class_f1(y_true: Sequence[str], y_pred: Sequence[str], label: str) -> float:
    pairs = list(zip(y_true, y_pred, strict=True))
    tp = sum(t == label and p == label for t, p in pairs)
    fp = sum(t != label and p == label for t, p in pairs)
    fn = sum(t == label and p != label for t, p in pairs)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return _safe_div(2 * precision * recall, precision + recall)


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    return sum(_per_class_f1(y_true, y_pred, label) for label in labels) / len(labels)
