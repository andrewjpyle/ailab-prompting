"""Tiny demo: show each strategy's prompt on one message, then run the eval.

The demo runs on the offline :class:`~ailab_prompting.providers.StubProvider`, so it needs
no network and shows the PLUMBING only. The stub returns one fixed record, so every strategy
prints the same output here; the real prompt effect is in `make eval` (the recorded pass).
"""

from __future__ import annotations

from collections.abc import Sequence

from ailab_prompting.eval import main as eval_main
from ailab_prompting.extract import Extractor
from ailab_prompting.providers import StubProvider
from ailab_prompting.strategies import STRATEGIES

SAMPLE = "The mobile app keeps crashing when I sync a folder and now I cannot see any of my files."


def main(argv: Sequence[str] | None = None) -> int:
    print("== ailab-prompting demo (stub provider, format path only) ==\n")
    print(f"Message: {SAMPLE}\n")
    for strategy in STRATEGIES:
        extractor = Extractor(StubProvider(), strategy)
        result = extractor.run(SAMPLE)
        print(f"- {strategy}: {result.raw}  (format_valid={result.format_valid})")
    print("\n== eval (replays the real recorded cassette and enforces the gate) ==\n")
    return eval_main(list(argv or []))


if __name__ == "__main__":
    raise SystemExit(main())
