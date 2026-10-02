"""The prompting-strategy seam and the injection defenses build distinct prompts."""

from __future__ import annotations

from ailab_prompting.strategies import (
    BROKEN_EXEMPLARS,
    FEWSHOT_EXEMPLARS,
    STRATEGIES,
    build_prompt,
)

MSG = "The mobile app crashes on launch."


def test_three_named_strategies() -> None:
    assert STRATEGIES == ("zero_shot", "few_shot", "cot")


def test_zero_shot_has_no_exemplars() -> None:
    prompt = build_prompt("zero_shot", MSG)
    assert "Message:" in prompt
    assert MSG in prompt
    assert "Output:" in prompt
    # No worked example message appears.
    assert FEWSHOT_EXEMPLARS[0][0] not in prompt


def test_few_shot_includes_exemplars() -> None:
    prompt = build_prompt("few_shot", MSG)
    assert FEWSHOT_EXEMPLARS[0][0] in prompt


def test_broken_fewshot_uses_wrong_labels() -> None:
    prompt = build_prompt("few_shot", MSG, broken_fewshot=True)
    assert BROKEN_EXEMPLARS[0][0] in prompt
    # The broken block carries a wrong label for the first exemplar.
    assert '"intent": "feedback"' in prompt


def test_cot_asks_to_reason_first() -> None:
    prompt = build_prompt("cot", MSG)
    assert "step by step" in prompt.lower()


def test_defenses_change_the_prompt_and_fence_the_message() -> None:
    off = build_prompt("few_shot", MSG, defenses=False)
    on = build_prompt("few_shot", MSG, defenses=True)
    assert off != on
    assert "UNTRUSTED_MESSAGE" in on
    assert "never a command" in on.lower()
    assert "UNTRUSTED_MESSAGE" not in off


def test_unknown_strategy_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown strategy"):
        build_prompt("tree_of_thought", MSG)  # type: ignore[arg-type]
