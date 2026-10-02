"""Two baselines the prompting strategies must beat, or they are not worth their cost.

* :class:`EmptyBaseline` emits nothing. It is the floor for ``format_valid`` (0.0) and for
  field accuracy (0.0): a strategy that cannot beat emitting nothing has failed.
* :class:`MajorityBaseline` always emits the most common value for each field, learned from
  the dataset. It is format valid but blind to the message, so it is the floor a real reader
  of the message must clear on field accuracy.

Neither baseline calls a model; both produce records directly, so they stay deterministic
and free.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from ailab_prompting.data import Example
from ailab_prompting.schema import FIELDS


class EmptyBaseline:
    """Predicts an empty record: the floor for format-valid and field accuracy."""

    name = "baseline"
    model = "empty-baseline-v1"

    def predict(self, message: str) -> dict[str, str]:
        return {}


class MajorityBaseline:
    """Predicts the per-field majority value learned from the dataset."""

    name = "baseline"
    model = "majority-baseline-v1"

    def __init__(self, record: dict[str, str]) -> None:
        self.record = dict(record)

    @classmethod
    def fit(cls, examples: Sequence[Example]) -> MajorityBaseline:
        if not examples:
            raise ValueError("cannot fit on an empty dataset")
        record: dict[str, str] = {}
        for field in FIELDS:
            counts = Counter(getattr(ex, field) for ex in examples)
            # Sort first for a deterministic tie-break.
            record[field] = max(sorted(counts), key=lambda value: counts[value])
        return cls(record)

    def predict(self, message: str) -> dict[str, str]:
        return dict(self.record)
