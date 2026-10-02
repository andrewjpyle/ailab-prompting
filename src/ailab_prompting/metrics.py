"""Metrics for the prompting lab.

Two layers:

* The classification core (``accuracy``, ``per_class``, ``macro_f1``, ``confusion_matrix``)
  is the scikit-learn-style maths shared with the template and ailab-rag. It is kept
  verbatim and guarded by ``tests/test_metric_parity.py`` against a frozen reference so it
  cannot drift.
* The extraction layer scores the JSON-record task: per-field exact match, whole-record
  exact match, schema macro-F1, the ``format_valid`` rate, the vendor token cost, and the
  confidence intervals (Wilson for a proportion, a seeded bootstrap for a delta) that every
  headline number carries.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ailab_prompting.schema import FIELDS

# --- classification core (parity-checked; do not re-derive) --------------------------


@dataclass(frozen=True, slots=True)
class ClassScores:
    precision: float
    recall: float
    f1: float
    support: int


def _check(y_true: Sequence[str], y_pred: Sequence[str]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} gold vs {len(y_pred)} predicted")
    if not y_true:
        raise ValueError("cannot score an empty prediction set")


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    _check(y_true, y_pred)
    return sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def per_class(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, ClassScores]:
    """Precision/recall/F1 per label.

    Labels are the union of gold and predicted labels, so a model that invents a label is
    penalised (its precision for that label is 0) instead of ignored. Undefined ratios
    (0/0) are scored 0.0, matching scikit-learn's ``zero_division=0`` convention.
    """
    _check(y_true, y_pred)
    scores: dict[str, ClassScores] = {}
    for label in sorted(set(y_true) | set(y_pred)):
        pairs = list(zip(y_true, y_pred, strict=True))
        tp = sum(t == label and p == label for t, p in pairs)
        fp = sum(t != label and p == label for t, p in pairs)
        fn = sum(t == label and p != label for t, p in pairs)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        scores[label] = ClassScores(precision, recall, f1, support=tp + fn)
    return scores


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Unweighted mean of per-label F1: every class counts equally, however rare."""
    scores = per_class(y_true, y_pred)
    return sum(s.f1 for s in scores.values()) / len(scores)


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, dict[str, int]]:
    """``matrix[gold][predicted] -> count`` over the union of labels."""
    _check(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    matrix = {gold: dict.fromkeys(labels, 0) for gold in labels}
    for t, p in zip(y_true, y_pred, strict=True):
        matrix[t][p] += 1
    return matrix


# --- extraction layer ----------------------------------------------------------------

Record = Mapping[str, str]


def field_accuracy(gold: Sequence[Record], pred: Sequence[Record]) -> float:
    """Mean exact-match over every (example, field). A missing or wrong value scores 0."""
    _check([g.get("intent", "") for g in gold], [p.get("intent", "") for p in pred])
    hits = sum(
        g.get(field) == p.get(field, "")
        for g, p in zip(gold, pred, strict=True)
        for field in FIELDS
    )
    return hits / (len(gold) * len(FIELDS))


def record_exact(gold: Sequence[Record], pred: Sequence[Record]) -> float:
    """Fraction of examples where every schema field matches the gold record."""
    if not gold:
        return 0.0
    exact = sum(
        all(g.get(field) == p.get(field, "") for field in FIELDS)
        for g, p in zip(gold, pred, strict=True)
    )
    return exact / len(gold)


def schema_macro_f1(gold: Sequence[Record], pred: Sequence[Record]) -> float:
    """Mean of the per-field macro-F1 over the three schema fields."""
    if not gold:
        return 0.0
    scores = []
    for field in FIELDS:
        y_true = [g.get(field, "") for g in gold]
        y_pred = [p.get(field, "") or "<none>" for p in pred]
        scores.append(macro_f1(y_true, y_pred))
    return sum(scores) / len(scores)


def record_exact_flags(gold: Sequence[Record], pred: Sequence[Record]) -> list[int]:
    """Per-example 1/0 whole-record-correct flags (for a bootstrap CI)."""
    return [
        int(all(g.get(field) == p.get(field, "") for field in FIELDS))
        for g, p in zip(gold, pred, strict=True)
    ]


def mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or ``None`` for an empty sequence."""
    return sum(values) / len(values) if values else None


def wilson_ci(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default).

    A proper small-sample interval, unlike the normal approximation, so a rate from a few
    dozen examples carries an honest band rather than a false point estimate.
    """
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_diff_ci(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    n_boot: int = 4000,
    seed: int = 12345,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for ``mean(xs) - mean(ys)``.

    Equal-length samples are resampled paired (same index into both), which is right for two
    runs over the SAME examples; unequal lengths are resampled independently. Seeded, so the
    interval is deterministic and replayable. Returns ``(low, high, point)``.
    """
    if not xs or not ys:
        return (0.0, 0.0, 0.0)
    point = sum(xs) / len(xs) - sum(ys) / len(ys)
    rng = random.Random(seed)
    diffs: list[float] = []
    paired = len(xs) == len(ys)
    for _ in range(n_boot):
        if paired:
            idx = [rng.randrange(len(xs)) for _ in range(len(xs))]
            dx = sum(xs[i] for i in idx) / len(idx)
            dy = sum(ys[i] for i in idx) / len(idx)
        else:
            dx = sum(rng.choice(xs) for _ in xs) / len(xs)
            dy = sum(rng.choice(ys) for _ in ys) / len(ys)
        diffs.append(dx - dy)
    diffs.sort()
    lo = diffs[int((alpha / 2) * n_boot)]
    hi = diffs[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi, point)
