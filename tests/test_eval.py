"""Eval gate behavior, results contract, and the proof-of-gate red cases.

The stub path proves the vacuous-stub guard (byte-identical outputs fail the gate). The red
cases run against the committed REAL cassette, so a broken-few-shot or out-of-enum run
reddens on a real metric, not a cassette miss (plan conditions 1 and 3)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ailab_prompting.eval import current_commit, main
from tests.conftest import CASSETTE, EXTRACTIONS, INJECTIONS

# Flags that pin the real cassette, used by every replay-backed test.
REPLAY = [
    "--provider", "replay", "--cassette", str(CASSETTE), "--model", "gemma3:27b",
    "--dataset", str(EXTRACTIONS), "--injections", str(INJECTIONS),
]  # fmt: skip


def test_stub_fails_the_provider_and_vacuous_guards(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "r.json"
    code = main(["--provider", "stub", "--dataset", str(EXTRACTIONS),
                 "--injections", str(INJECTIONS), "--output", str(out)])  # fmt: skip
    assert code == 1
    err = capsys.readouterr().err
    assert "provider 'stub' != 'ollama'" in err
    assert "byte-identical" in err
    record = json.loads(out.read_text())
    assert record["provider"] == "stub"
    assert record["strategy_delta"]["byte_identical"] is True


def test_results_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_SHA", "f" * 40)
    out = tmp_path / "eval_results.json"
    main([*REPLAY, "--output", str(out)])
    record = json.loads(out.read_text())
    contract: dict[str, type | tuple[type, ...]] = {
        "schema_version": int, "lab": str, "dataset": str, "provider": str, "model": str,
        "primary_metric": str, "metrics": dict, "threshold": float, "passed": bool,
        "commit": (str, type(None)), "generated_at": str,
    }  # fmt: skip
    for key, kind in contract.items():
        assert isinstance(record[key], kind), key
    assert record["schema_version"] == 1
    assert record["provider"] == "ollama"
    assert record["model"] == "gemma3:27b"
    assert record["commit"] == "f" * 40
    assert set(record["metrics"]) >= {"field_accuracy", "format_valid", "record_exact", "n"}
    assert record["metrics"]["n"] == 24
    assert datetime.fromisoformat(record["generated_at"]).utcoffset() == timedelta(0)


def test_real_replay_passes_the_gate(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    code = main([*REPLAY, "--output", str(out)])
    record = json.loads(out.read_text())
    assert code == 0, record["failures"]
    assert record["passed"] is True
    assert record["strategy_delta"]["byte_identical"] is False
    assert record["cassette"]["provider"] == "ollama"
    assert record["cassette"]["call_count"] > 0


def test_red_broken_fewshot_regresses_field_accuracy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Floor 0.97 (band 0.92): the clean few-shot (0.9306) would pass, the broken run
    # (0.9028) fails, so the break is what trips the gate, not just a high floor.
    code = main([*REPLAY, "--output", str(tmp_path / "r.json"),
                 "--break-fewshot", "--min-field-accuracy", "0.97"])  # fmt: skip
    assert code == 1
    assert "REGRESSION: field_accuracy" in capsys.readouterr().err


def test_red_force_invalid_regresses_format_valid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([*REPLAY, "--output", str(tmp_path / "r.json"),
                 "--force-invalid", "--min-format-valid", "0.95"])  # fmt: skip
    assert code == 1
    assert "REGRESSION: format_valid" in capsys.readouterr().err


def test_red_empty_dataset_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    code = main(["--provider", "replay", "--cassette", str(CASSETTE),
                 "--dataset", str(empty), "--injections", str(INJECTIONS)])  # fmt: skip
    assert code == 2
    assert "empty" in capsys.readouterr().err


def test_table_from_existing_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "r.json"
    main([*REPLAY, "--output", str(out)])
    capsys.readouterr()
    assert main(["--table-from", str(out)]) == 0
    assert "| field_acc |" in capsys.readouterr().out
    (tmp_path / "bad.json").write_text("{}")
    assert main(["--table-from", str(tmp_path / "bad.json")]) == 2


def test_writes_github_step_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    main([*REPLAY, "--output", str(tmp_path / "r.json")])
    assert "### Eval results" in summary.read_text()


def test_commit_is_null_when_unknown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))  # no git binary reachable
    assert current_commit() is None
    monkeypatch.setenv("AILAB_COMMIT", "unknown")
    assert current_commit() is None
