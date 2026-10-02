"""Prove the strategy seam and the injection defense run on the REAL recorded model.

The committed cassette ``fixtures/cassettes/support_extract.json`` is a real gemma3:27b
recording. These tests replay it through the full eval and assert the RELATIONSHIPS the lab
claims (strategies differ, defenses do not increase the attack), not hand-typed magic
numbers. The exact figures are gated against docs/LEARNING.md by check_learning_numbers.
"""

from __future__ import annotations

import json
from typing import Any

from ailab_prompting.config import load_config
from ailab_prompting.eval import evaluate
from ailab_prompting.providers import ReplayProvider
from ailab_prompting.strategies import STRATEGIES
from tests.conftest import CASSETTE


def test_cassette_is_a_real_gemma_recording() -> None:
    cass = json.loads(CASSETTE.read_text())
    assert cass["provider"] == "ollama"
    assert cass["model"] == "gemma3:27b"
    assert cass["model_digest"]  # a real digest
    assert cass["temperature"] == 0
    # dataset x strategies (24 x 3) + broken few-shot (24) + injection off/on (12 x 2) = 120.
    assert cass["entry_count"] == 120
    assert len(cass["entries"]) == 120


def _evaluate() -> dict[str, Any]:
    config = load_config("eval_config.toml")
    provider = ReplayProvider(config.cassette, model=config.model)
    return evaluate(config, provider)


def test_strategies_are_not_identical_on_the_real_model() -> None:
    result = _evaluate()
    strategies = result["strategies"]
    hashes = {strategies[s]["outputs_sha256"] for s in STRATEGIES}
    assert len(hashes) >= 2  # the real model responds differently per strategy
    assert result["strategy_delta"]["byte_identical"] is False


def test_strategies_beat_the_baselines() -> None:
    result = _evaluate()
    baselines = result["baselines"]
    assert baselines["best_field_accuracy"] > baselines["majority"]["field_accuracy"]
    assert baselines["margin"] >= baselines["required_margin"]


def test_defenses_never_increase_the_attack() -> None:
    result = _evaluate()
    inj = result["injection"]
    # The output-schema allowlist guarantees the sentinel cannot reach the service output.
    assert inj["asr_on"] == 0.0
    assert inj["asr_on"] <= inj["asr_off"]
    assert inj["verdict"] in ("PASS", "INSUFFICIENT")


def test_real_eval_passes_and_is_not_saturated() -> None:
    result = _evaluate()
    assert result["passed"] is True, result["failures"]
    assert result["saturation"]["saturated"] is False
