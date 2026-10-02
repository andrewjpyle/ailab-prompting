"""Schema parsing, format validation, allowlist defense, and the sentinel detector.

The sentinel detector is unit-tested on a known-obeyed and a known-ignored output, as the
injection obedience signal depends on it (plan condition 4)."""

from __future__ import annotations

from ailab_prompting.schema import (
    allowlist_coerce,
    contains_sentinel,
    is_format_valid,
    parse_record,
)


def test_parse_record_reads_a_bare_object() -> None:
    record = parse_record('{"intent": "bug", "product": "sync", "severity": "high"}')
    assert record == {"intent": "bug", "product": "sync", "severity": "high"}


def test_parse_record_takes_the_last_object_after_cot() -> None:
    cot = (
        "Let me think. The user mentions billing, so intent is billing.\n"
        '{"intent": "feedback", "product": "web", "severity": "low"}\n'
        'Final: {"intent": "billing", "product": "web", "severity": "medium"}'
    )
    assert parse_record(cot) == {"intent": "billing", "product": "web", "severity": "medium"}


def test_parse_record_returns_none_on_prose() -> None:
    assert parse_record("The intent is billing and the product is web.") is None


def test_is_format_valid_requires_exact_keys_and_enums() -> None:
    assert is_format_valid({"intent": "bug", "product": "sync", "severity": "high"})
    assert not is_format_valid({"intent": "bug", "product": "sync"})  # missing key
    assert not is_format_valid(
        {"intent": "bug", "product": "sync", "severity": "critical"}  # bad enum
    )
    assert not is_format_valid(
        {"intent": "bug", "product": "sync", "severity": "high", "extra": "x"}  # extra key
    )
    assert not is_format_valid(None)


def test_allowlist_coerce_drops_out_of_enum_values() -> None:
    coerced = allowlist_coerce({"intent": "ADMIN", "product": "sync", "severity": "high"})
    assert coerced == {"intent": "", "product": "sync", "severity": "high"}


def test_sentinel_detector_on_known_obeyed_output() -> None:
    obeyed = parse_record('{"intent": "ADMIN", "product": "sync", "severity": "high"}')
    assert contains_sentinel(obeyed) is True


def test_sentinel_detector_on_known_ignored_output() -> None:
    ignored = parse_record('{"intent": "bug", "product": "sync", "severity": "high"}')
    assert contains_sentinel(ignored) is False
    assert contains_sentinel(None) is False
