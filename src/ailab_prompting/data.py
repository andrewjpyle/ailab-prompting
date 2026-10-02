"""Dataset loading for the prompting lab.

Two JSONL datasets are read here, both validated loudly so a bad fixture fails with a
``file:line`` message instead of silently shrinking the eval set.

* An extraction row is
  ``{"id", "message", "intent", "product", "severity"}``; the three label fields are the
  gold record and each must be in its schema enum (see :mod:`ailab_prompting.schema`).
* An injection row adds ``"family"`` (the injection family) and its ``message`` hides an
  indirect prompt injection. The three label fields are the TRUE record a defended model
  should still return.

Each dataset file also carries a sha256 of its raw bytes, recorded in the results so a
number can be traced back to the exact file that produced it (the frozen dataset).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ailab_prompting.schema import ENUMS, FIELDS

INJECTION_FAMILIES = ("fake_system", "instruction_override", "role_play")


@dataclass(frozen=True, slots=True)
class Example:
    """One extraction example with its gold record."""

    id: str
    message: str
    intent: str
    product: str
    severity: str

    @property
    def record(self) -> dict[str, str]:
        return {"intent": self.intent, "product": self.product, "severity": self.severity}


@dataclass(frozen=True, slots=True)
class InjectionExample:
    """One injection example: the TRUE record plus the family of the hidden attack."""

    id: str
    message: str
    intent: str
    product: str
    severity: str
    family: str

    @property
    def record(self) -> dict[str, str]:
        return {"intent": self.intent, "product": self.product, "severity": self.severity}


class DatasetError(ValueError):
    """Raised when a dataset file is malformed."""


def sha256_file(path: str | Path) -> str:
    """Return the sha256 hex digest of a file's raw bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(path: Path) -> list[tuple[int, dict[str, object]]]:
    """Yield ``(lineno, object)`` pairs, skipping blank lines and validating JSON."""
    rows: list[tuple[int, dict[str, object]]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"{path}:{lineno}: expected a JSON object")
            rows.append((lineno, row))
    return rows


def _str_field(path: Path, lineno: int, row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise DatasetError(f"{path}:{lineno}: need a non-empty string {key!r}")
    return value


def _label_fields(path: Path, lineno: int, row: dict[str, object]) -> dict[str, str]:
    """Read and enum-check the three gold label fields."""
    labels: dict[str, str] = {}
    for field in FIELDS:
        value = _str_field(path, lineno, row, field)
        if value not in ENUMS[field]:
            raise DatasetError(f"{path}:{lineno}: {field!r} {value!r} is not one of {ENUMS[field]}")
        labels[field] = value
    return labels


def load_extractions(path: str | Path) -> list[Example]:
    """Load the extraction JSONL. Raises :class:`DatasetError` on any malformed row."""
    path = Path(path)
    examples: list[Example] = []
    seen: set[str] = set()
    for lineno, row in _rows(path):
        ex_id = _str_field(path, lineno, row, "id")
        if ex_id in seen:
            raise DatasetError(f"{path}:{lineno}: duplicate example id {ex_id!r}")
        seen.add(ex_id)
        labels = _label_fields(path, lineno, row)
        examples.append(
            Example(id=ex_id, message=_str_field(path, lineno, row, "message"), **labels)
        )
    if not examples:
        raise DatasetError(f"{path}: dataset is empty")
    return examples


def load_injections(path: str | Path) -> list[InjectionExample]:
    """Load the injection JSONL. Raises :class:`DatasetError` on any malformed row."""
    path = Path(path)
    examples: list[InjectionExample] = []
    seen: set[str] = set()
    for lineno, row in _rows(path):
        ex_id = _str_field(path, lineno, row, "id")
        if ex_id in seen:
            raise DatasetError(f"{path}:{lineno}: duplicate injection id {ex_id!r}")
        seen.add(ex_id)
        family = _str_field(path, lineno, row, "family")
        if family not in INJECTION_FAMILIES:
            raise DatasetError(
                f"{path}:{lineno}: family {family!r} is not one of {INJECTION_FAMILIES}"
            )
        labels = _label_fields(path, lineno, row)
        examples.append(
            InjectionExample(
                id=ex_id,
                message=_str_field(path, lineno, row, "message"),
                family=family,
                **labels,
            )
        )
    if not examples:
        raise DatasetError(f"{path}: injection set is empty")
    return examples
