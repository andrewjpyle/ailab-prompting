"""Parity test: the production classification metrics must agree with the frozen reference.

The metric-parity approach ported from ailab-rag @ 3cd8f82: a change to the production
``accuracy``/``macro_f1`` that disagrees with the reference fails here."""

from __future__ import annotations

import random

from ailab_prompting import metrics as prod
from tests import _metrics_ref as frozen


def _fixture() -> list[tuple[list[str], list[str]]]:
    rng = random.Random(2024)
    labels = ["a", "b", "c", "d", "<none>"]
    cases: list[tuple[list[str], list[str]]] = []
    for _ in range(50):
        n = rng.randint(1, 20)
        y_true = [rng.choice(labels) for _ in range(n)]
        y_pred = [rng.choice(labels) for _ in range(n)]
        cases.append((y_true, y_pred))
    return cases


def test_accuracy_and_macro_f1_match_reference() -> None:
    for y_true, y_pred in _fixture():
        assert prod.accuracy(y_true, y_pred) == frozen.accuracy(y_true, y_pred)
        assert prod.macro_f1(y_true, y_pred) == frozen.macro_f1(y_true, y_pred)
