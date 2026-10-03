"""LLM provider seam for the extractor.

The discipline-agnostic pieces (the ``LLMProvider`` protocol, the optional
``SupportsTokenCounts`` capability, ``cassette_key``, ``CassetteMissError``,
``ReplayProvider`` and ``OllamaProvider``) now live in ``ailab_core`` and are re-exported here
so existing imports keep working. Only :class:`StubProvider` is prompting-specific: it returns a
fixed, format-valid record (the stub proves the plumbing, never the prompt effect) and reports a
whitespace token estimate for the cost axis.
"""

from __future__ import annotations

import re

from ailab_core.providers import (
    CassetteMissError,
    LLMProvider,
    OllamaProvider,
    ReplayProvider,
    SupportsTokenCounts,
    cassette_key,
)

_WORD = re.compile(r"\S+")


class StubProvider:
    """Deterministic offline backend. No network, no key. FORMAT path only.

    Returns the same format-valid record for every prompt, so every strategy looks
    identical. That is the point: the stub proves the plumbing, never the prompt effect.
    Token counts are a whitespace ESTIMATE, flagged as such.
    """

    name = "stub"
    model = "stub-extractor-v1"
    FIXED_RECORD = '{"intent": "bug", "product": "sync", "severity": "medium"}'

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return self.FIXED_RECORD

    def token_counts(self, prompt: str) -> tuple[int, int, bool]:
        prompt_tokens = len(_WORD.findall(prompt))
        completion_tokens = len(_WORD.findall(self.FIXED_RECORD))
        return prompt_tokens, completion_tokens, True  # whitespace estimate


__all__ = [
    "CassetteMissError",
    "LLMProvider",
    "OllamaProvider",
    "ReplayProvider",
    "StubProvider",
    "SupportsTokenCounts",
    "cassette_key",
]
