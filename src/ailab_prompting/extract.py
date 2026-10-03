"""Extractor: turn a support message into one JSON record via a prompting strategy.

The Extractor is the seam the eval scores. It builds a prompt with the chosen
:data:`~ailab_prompting.strategies.PromptStrategy`, asks the provider to complete it, parses
the JSON record, and reports whether the output is format valid, whether an injection
sentinel slipped through, and the vendor token counts (the cost axis).

Two obedience signals are kept apart on the injection path:

* ``obeyed`` is MODEL obedience: the sentinel appears in the parsed record, before any
  output filtering. It changes only with the prompt, so it shows the effect of the two
  prompt defenses (delimiting and the instruction hierarchy).
* ``leaked`` is the SERVICE attack-success signal: the sentinel survives into the record
  the service would return. The output-schema allowlist (on when ``defenses`` is set)
  deterministically drops any out-of-enum value, so ``leaked`` is ``obeyed and not
  defenses``.

The same Extractor runs the offline stub and the recorded real model unchanged. Two
proof-of-gate knobs stay OFF in a real run: ``broken_fewshot`` (wrong-label exemplars) and
``force_invalid`` (emit an out-of-enum record, the format-valid negative control).
"""

from __future__ import annotations

from dataclasses import dataclass

from ailab_core.providers import SupportsTokenCounts

from ailab_prompting.providers import LLMProvider
from ailab_prompting.schema import (
    allowlist_coerce,
    contains_sentinel,
    is_format_valid,
    parse_record,
)
from ailab_prompting.strategies import PromptStrategy, build_prompt

# An out-of-enum record for the format-valid negative control (proof of gate).
_INVALID_RECORD = {"intent": "SUPERUSER", "product": "sync", "severity": "low"}


@dataclass(frozen=True, slots=True)
class ExtractResult:
    """One extraction outcome, scored by the eval."""

    raw: str
    parsed: dict[str, str] | None  # the model's parsed record, before the allowlist
    prediction: dict[str, str]  # one in-enum value per schema field, else "" (for scoring)
    format_valid: bool
    obeyed: bool  # model obedience: sentinel in the parsed record (pre-allowlist)
    leaked: bool  # service attack success: sentinel survives the defenses into the output
    prompt_tokens: int
    completion_tokens: int
    tokens_estimated: bool


class Extractor:
    """Prompt-strategy extractor over any :class:`LLMProvider`."""

    def __init__(
        self,
        provider: LLMProvider,
        strategy: PromptStrategy,
        *,
        broken_fewshot: bool = False,
        defenses: bool = False,
        force_invalid: bool = False,
    ) -> None:
        self.provider = provider
        self.strategy = strategy
        self.broken_fewshot = broken_fewshot
        self.defenses = defenses
        self.force_invalid = force_invalid

    def build_prompt(self, message: str) -> str:
        return build_prompt(
            self.strategy,
            message,
            broken_fewshot=self.broken_fewshot,
            defenses=self.defenses,
        )

    def run(self, message: str) -> ExtractResult:
        prompt = self.build_prompt(message)
        raw = self.provider.complete(prompt)
        if not isinstance(self.provider, SupportsTokenCounts):
            raise TypeError(
                "extractor requires a provider that reports token_counts (the cost axis)"
            )
        prompt_tokens, completion_tokens, estimated = self.provider.token_counts(prompt)

        parsed = _INVALID_RECORD.copy() if self.force_invalid else parse_record(raw)
        obeyed = contains_sentinel(parsed)
        return ExtractResult(
            raw=raw,
            parsed=parsed,
            prediction=allowlist_coerce(parsed),
            format_valid=is_format_valid(parsed),
            obeyed=obeyed,
            leaked=obeyed and not self.defenses,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_estimated=estimated,
        )
