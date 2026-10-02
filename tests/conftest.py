from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTIONS = REPO_ROOT / "fixtures" / "support_extractions.jsonl"
INJECTIONS = REPO_ROOT / "fixtures" / "injection_attacks.jsonl"
CASSETTE = REPO_ROOT / "fixtures" / "cassettes" / "support_extract.json"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep host env vars from leaking into config-sensitive tests."""
    for var in (
        "AILAB_CONFIG",
        "AILAB_LAB",
        "AILAB_DATASET",
        "AILAB_INJECTIONS",
        "AILAB_OUTPUT",
        "AILAB_PROVIDER",
        "AILAB_CASSETTE",
        "AILAB_MODEL",
        "AILAB_PRIMARY_STRATEGY",
        "AILAB_PRIMARY_METRIC",
        "AILAB_MIN_FIELD_ACCURACY",
        "AILAB_MIN_FORMAT_VALID",
        "AILAB_TOLERANCE",
        "AILAB_BASELINE_MARGIN",
        "AILAB_STRATEGY_DELTA_MIN",
        "AILAB_COMMIT",
        "GITHUB_SHA",
        "GITHUB_STEP_SUMMARY",
    ):
        monkeypatch.delenv(var, raising=False)
