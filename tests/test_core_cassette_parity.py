"""Golden parity: the committed cassette loads identically through ailab_core.

Spar condition 1 for the ailab-core extraction: adopting the shared ReplayProvider must not
silently invalidate the real-model cassette. This asserts the core provider reads the SAME keys
and values the committed file holds, with zero misses, and that the vendor token counts (the cost
axis prompting depends on) survive. The before/after eval-metrics half is enforced by the `eval`
CI job.
"""

from __future__ import annotations

import json

from ailab_core.providers import ReplayProvider, SupportsTokenCounts

from tests.conftest import CASSETTE


def test_core_replay_reads_committed_cassette_identically() -> None:
    raw = json.loads(CASSETTE.read_text(encoding="utf-8"))
    entries = raw["entries"]
    rp = ReplayProvider(CASSETTE, name="ollama", model=raw["model"])

    assert rp.name == raw["provider"]
    assert rp.entry_count == len(entries)

    misses = 0
    for key, entry in entries.items():
        stored = rp._entries.get(key)
        assert stored == entry, f"entry {key[:12]} changed under ailab_core"
        if stored is None:
            misses += 1
    assert misses == 0  # a stale cassette can never pass


def test_core_replay_preserves_vendor_token_counts() -> None:
    # prompting has a cost axis, so token_counts must survive the move to ailab_core.
    raw = json.loads(CASSETTE.read_text(encoding="utf-8"))
    _, entry = next(iter(raw["entries"].items()))
    assert "prompt_eval_count" in entry
    assert "eval_count" in entry
    rp = ReplayProvider(CASSETTE, name="ollama", model=raw["model"])
    assert isinstance(rp, SupportsTokenCounts)
