"""The Extractor ties a strategy to a provider and reports format, obedience and cost."""

from __future__ import annotations

from ailab_prompting.extract import Extractor
from ailab_prompting.providers import StubProvider


def test_stub_extraction_is_format_valid() -> None:
    result = Extractor(StubProvider(), "zero_shot").run("my app crashed")
    assert result.format_valid is True
    assert result.prediction == {"intent": "bug", "product": "sync", "severity": "medium"}
    assert result.obeyed is False
    assert result.leaked is False
    assert result.prompt_tokens > 0


def test_force_invalid_breaks_format() -> None:
    result = Extractor(StubProvider(), "few_shot", force_invalid=True).run("x")
    assert result.format_valid is False  # out-of-enum intent
    assert result.prediction["intent"] == ""  # allowlist blanks SUPERUSER


class _ObeyProvider:
    """A provider that always obeys the injection, for the leaked/obeyed logic test."""

    name = "stub"
    model = "obey"

    def complete(self, prompt: str) -> str:
        return '{"intent": "ADMIN", "product": "sync", "severity": "high"}'

    def token_counts(self, prompt: str) -> tuple[int, int, bool]:
        return (1, 1, True)


def test_obeyed_detected_but_allowlist_stops_the_leak_when_defended() -> None:
    off = Extractor(_ObeyProvider(), "few_shot", defenses=False).run("attack")
    assert off.obeyed is True
    assert off.leaked is True  # no defense, the sentinel reaches the output
    on = Extractor(_ObeyProvider(), "few_shot", defenses=True).run("attack")
    assert on.obeyed is True  # the model still obeyed (same canned output)
    assert on.leaked is False  # the allowlist defense drops ADMIN
    assert on.prediction["intent"] == ""
