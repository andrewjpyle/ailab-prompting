"""Extraction metrics: field accuracy, record exact, schema macro-F1, and the CIs."""

from __future__ import annotations

from ailab_prompting.metrics import (
    accuracy,
    bootstrap_diff_ci,
    confusion_matrix,
    field_accuracy,
    macro_f1,
    per_class,
    record_exact,
    record_exact_flags,
    schema_macro_f1,
    wilson_ci,
)

GOLD = [
    {"intent": "bug", "product": "sync", "severity": "high"},
    {"intent": "billing", "product": "web", "severity": "low"},
]


def test_field_accuracy_counts_every_field() -> None:
    perfect = field_accuracy(GOLD, GOLD)
    assert perfect == 1.0
    # One wrong field out of six -> 5/6.
    pred = [dict(GOLD[0]), {**GOLD[1], "severity": "high"}]
    assert round(field_accuracy(GOLD, pred), 4) == round(5 / 6, 4)


def test_missing_field_scores_zero() -> None:
    pred: list[dict[str, str]] = [{}, {}]
    assert field_accuracy(GOLD, pred) == 0.0
    assert record_exact(GOLD, pred) == 0.0


def test_record_exact_and_flags() -> None:
    pred = [dict(GOLD[0]), {"intent": "x", "product": "y", "severity": "z"}]
    assert record_exact(GOLD, pred) == 0.5
    assert record_exact_flags(GOLD, pred) == [1, 0]


def test_schema_macro_f1_is_perfect_on_perfect_pred() -> None:
    assert schema_macro_f1(GOLD, GOLD) == 1.0


def test_classification_core() -> None:
    assert accuracy(["a", "b"], ["a", "a"]) == 0.5
    assert 0.0 <= macro_f1(["a", "b"], ["a", "a"]) <= 1.0
    assert per_class(["a"], ["a"])["a"].f1 == 1.0
    assert confusion_matrix(["a", "b"], ["a", "a"]) == {
        "a": {"a": 1, "b": 0},
        "b": {"a": 1, "b": 0},
    }


def test_wilson_ci_brackets_the_point() -> None:
    lo, hi = wilson_ci(8, 10)
    assert lo < 0.8 < hi
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_bootstrap_diff_ci_is_deterministic_and_signed() -> None:
    xs = [1.0] * 10 + [0.0] * 0
    ys = [0.0] * 10
    lo, hi, point = bootstrap_diff_ci(xs, ys)
    assert point == 1.0
    assert lo > 0.0  # a clear positive difference excludes zero
    # Deterministic: same seed, same interval.
    assert bootstrap_diff_ci(xs, ys) == (lo, hi, point)


def test_bootstrap_diff_ci_overlaps_zero_for_equal_samples() -> None:
    xs = [1.0, 0.0, 1.0, 0.0]
    lo, hi, point = bootstrap_diff_ci(xs, xs)
    assert point == 0.0
    assert lo <= 0.0 <= hi
