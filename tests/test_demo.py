"""The demo shows each strategy's stub prompt, then runs the real replay eval."""

from __future__ import annotations

from pathlib import Path

import pytest

from ailab_prompting.demo import main as demo_main
from tests.conftest import CASSETTE, EXTRACTIONS, INJECTIONS


def test_demo_runs_and_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = demo_main(
        [
            "--provider",
            "replay",
            "--cassette",
            str(CASSETTE),
            "--model",
            "gemma3:27b",
            "--dataset",
            str(EXTRACTIONS),
            "--injections",
            str(INJECTIONS),
            "--output",
            str(tmp_path / "r.json"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "zero_shot:" in out
    assert "few_shot:" in out
    assert "cot:" in out
    assert "| field_acc |" in out
