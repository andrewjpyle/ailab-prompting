"""The one-variable run comparator and its bootstrap CI."""

from __future__ import annotations

import json
from pathlib import Path

from ailab_prompting.compare import changed_variables, main, metric_deltas, render


def _run(strategy: str, fa: float, flags: list[int]) -> dict[str, object]:
    return {
        "primary_strategy": strategy,
        "provider": "ollama",
        "passed": True,
        "primary_metric": "field_accuracy",
        "config": {"primary_strategy": strategy, "provider": "ollama", "tolerance": 0.05},
        "metrics": {"field_accuracy": fa, "record_exact": sum(flags) / len(flags), "n": len(flags)},
        "primary_flags": {"record_exact": flags, "format_valid": [1] * len(flags)},
    }


def test_changed_variables_and_deltas() -> None:
    a = _run("zero_shot", 0.90, [1, 0, 1, 0])
    b = _run("few_shot", 0.93, [1, 1, 1, 0])
    assert changed_variables(a, b) == {"primary_strategy": ("zero_shot", "few_shot")}
    deltas = {row[0]: row[3] for row in metric_deltas(a, b)}
    assert round(deltas["field_accuracy"], 4) == 0.03


def test_render_reports_single_variable_and_ci() -> None:
    a = _run("zero_shot", 0.90, [1, 0, 1, 0])
    b = _run("few_shot", 0.93, [1, 1, 1, 1])
    out = render(a, b)
    assert "single changed variable: primary_strategy: zero_shot -> few_shot" in out
    assert "record_exact delta" in out
    assert "CI" in out


def test_render_refuses_more_than_one_changed_variable() -> None:
    a = _run("zero_shot", 0.90, [1, 0])
    b = _run("few_shot", 0.93, [1, 1])
    b["config"]["provider"] = "stub"  # type: ignore[index]
    out = render(a, b)
    assert "ERROR" in out


def test_main_on_committed_runs(capsys: object) -> None:
    code = main(["results/run_zero_shot.json", "results/run_few_shot.json"])
    assert code == 0


def test_main_bad_args() -> None:
    assert main(["only-one.json"]) == 2


def test_main_error_exit(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    ra = _run("zero_shot", 0.9, [1, 0])
    rb = _run("few_shot", 0.9, [1, 1])
    rb["config"]["tolerance"] = 0.1  # type: ignore[index]  # a second changed variable
    a.write_text(json.dumps(ra))
    b.write_text(json.dumps(rb))
    assert main([str(a), str(b)]) == 1
