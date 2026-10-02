"""Prove the injection ASR gate is wired, without faking a model recording.

The real model resisted every frozen injection (asr_off = 0, verdict INSUFFICIENT), so the
ASR gate cannot fire on real data. Authoring a synthetic "obey" cassette is forbidden
(vendor responses must come from the vendor). Instead these tests drive the gate FUNCTION
with constructed numbers, which is pure gate logic, not a fabricated recording.
"""

from __future__ import annotations

from typing import Any

from ailab_prompting.config import EvalConfig
from ailab_prompting.data import load_extractions, load_injections
from ailab_prompting.eval import _gate, _injection_scores
from ailab_prompting.providers import ReplayProvider
from tests.conftest import CASSETTE, EXTRACTIONS, INJECTIONS


class _FakeOllama:
    name = "ollama"
    model = "gemma3:27b"
    calls = 5


def _strategies() -> dict[str, dict[str, Any]]:
    # Distinct outputs per strategy and metrics that clear every non-injection gate.
    return {
        s: {
            "field_accuracy": 0.90,
            "format_valid": 1.0,
            "record_exact": 0.80,
            "outputs_sha256": f"hash-{s}",
        }
        for s in ("zero_shot", "few_shot", "cot")
    }


def _base_kwargs() -> dict[str, Any]:
    strategies = _strategies()
    return {
        "config": EvalConfig(),
        "strategies": strategies,
        "primary": strategies["few_shot"],
        "baselines": {},
        "margin": 0.60,
        "saturated": False,
        "byte_identical": False,
        "delta": {"value": 0.10, "metric": "record_exact"},
        "provider": _FakeOllama(),
        "max_asr": None,
    }


def test_asr_gate_fires_when_defenses_leave_a_high_asr_on() -> None:
    injection = {
        "verdict": "PASS",
        "asr_off": 0.80,
        "asr_on": 0.50,  # defenses did not stop it
        "reduction": {"value": 0.30},
    }
    failures = _gate(injection=injection, **_base_kwargs())
    assert any("asr_on" in f for f in failures)


def test_asr_gate_fires_when_reduction_is_below_margin() -> None:
    injection = {
        "verdict": "PASS",
        "asr_off": 0.12,
        "asr_on": 0.08,
        "reduction": {"value": 0.04},  # below the 0.10 margin
    }
    failures = _gate(injection=injection, **_base_kwargs())
    assert any("reduction" in f for f in failures)


def test_max_asr_override_fires_on_a_nonzero_asr() -> None:
    injection = {
        "verdict": "INSUFFICIENT",
        "asr_off": 0.25,
        "asr_on": 0.0,
        "reduction": {"value": 0.25},
    }
    kwargs = _base_kwargs()
    kwargs["max_asr"] = 0.0
    failures = _gate(injection=injection, **kwargs)
    assert any("asr" in f.lower() for f in failures)


def test_insufficient_verdict_does_not_fail_the_gate() -> None:
    injection = {
        "verdict": "INSUFFICIENT",
        "asr_off": 0.0,
        "asr_on": 0.0,
        "reduction": {"value": 0.0},
    }
    failures = _gate(injection=injection, **_base_kwargs())
    assert failures == []  # INSUFFICIENT is reported, never a hard failure


def test_real_injection_run_is_insufficient_because_the_model_resisted() -> None:
    config = EvalConfig(dataset=EXTRACTIONS, injections=INJECTIONS)
    provider = ReplayProvider(CASSETTE, model="gemma3:27b")
    load_extractions(config.dataset)  # sanity: the dataset loads
    scores = _injection_scores(provider, "few_shot", load_injections(config.injections), config)
    assert scores["asr_off"] == 0.0  # the frozen injections did not fool gemma3:27b
    assert scores["verdict"] == "INSUFFICIENT"
