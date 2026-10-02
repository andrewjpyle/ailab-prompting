"""Dataset loading validation and config layering."""

from __future__ import annotations

from pathlib import Path

import pytest

from ailab_prompting.config import ConfigError, EvalConfig, load_config
from ailab_prompting.data import (
    DatasetError,
    load_extractions,
    load_injections,
    sha256_file,
)
from tests.conftest import EXTRACTIONS, INJECTIONS


def test_load_extractions_and_injections() -> None:
    examples = load_extractions(EXTRACTIONS)
    injections = load_injections(INJECTIONS)
    assert len(examples) == 24
    assert len(injections) == 12
    assert examples[0].record == {"intent": "account", "product": "web", "severity": "medium"}
    assert injections[0].family == "fake_system"


def test_dataset_sha256_is_stable() -> None:
    assert sha256_file(EXTRACTIONS) == sha256_file(EXTRACTIONS)


def test_bad_enum_value_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"id": "x", "message": "hi", "intent": "zzz", "product": "web", "severity": "low"}\n'
    )
    with pytest.raises(DatasetError, match="not one of"):
        load_extractions(bad)


def test_empty_dataset_raises(tmp_path: Path) -> None:
    empty = tmp_path / "e.jsonl"
    empty.write_text("")
    with pytest.raises(DatasetError, match="empty"):
        load_extractions(empty)


def test_bad_injection_family_raises(tmp_path: Path) -> None:
    bad = tmp_path / "b.jsonl"
    bad.write_text(
        '{"id": "x", "message": "hi", "intent": "bug", "product": "web", '
        '"severity": "low", "family": "nope"}\n'
    )
    with pytest.raises(DatasetError, match="family"):
        load_injections(bad)


def test_duplicate_id_raises(tmp_path: Path) -> None:
    dup = tmp_path / "d.jsonl"
    row = '{"id": "x", "message": "hi", "intent": "bug", "product": "web", "severity": "low"}\n'
    dup.write_text(row + row)
    with pytest.raises(DatasetError, match="duplicate"):
        load_extractions(dup)


def test_config_defaults_and_thresholds() -> None:
    cfg = EvalConfig()
    assert cfg.provider == "replay"
    assert cfg.primary_strategy == "few_shot"
    assert cfg.thresholds == {"field_accuracy": 0.55, "format_valid": 0.70}


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ConfigError, match="provider"):
        EvalConfig(provider="openai")
    with pytest.raises(ConfigError, match="within"):
        EvalConfig(tolerance=2.0)
    with pytest.raises(ConfigError, match="primary_strategy"):
        EvalConfig(primary_strategy="tree")


def test_config_loads_committed_toml() -> None:
    cfg = load_config("eval_config.toml")
    assert cfg.lab == "ailab-prompting"
    assert cfg.min_field_accuracy == 0.55
    assert cfg.asr_off_floor == 0.15


def test_env_overrides_apply(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # no eval_config.toml here, so defaults apply
    monkeypatch.setenv("AILAB_MIN_FIELD_ACCURACY", "0.9")
    monkeypatch.setenv("AILAB_PROVIDER", "stub")
    cfg = load_config()
    assert cfg.min_field_accuracy == 0.9
    assert cfg.provider == "stub"


def test_missing_explicit_config_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")
