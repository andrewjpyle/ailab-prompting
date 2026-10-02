"""Provider seam: the stub format path, the cassette key, and the hard replay miss.

Plan condition 2 is enforced here: a ReplayProvider miss is a hard CassetteMissError naming
the prompt hash, never a stub fallback. The test mutates a recorded prompt and expects the
miss."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ailab_prompting.providers import (
    CassetteMissError,
    LLMProvider,
    OllamaProvider,
    ReplayProvider,
    StubProvider,
    cassette_key,
)
from tests.conftest import CASSETTE


def test_stub_is_offline_and_returns_a_fixed_valid_record() -> None:
    stub = StubProvider()
    out = stub.complete("any prompt")
    assert out == StubProvider.FIXED_RECORD
    assert stub.calls == 1
    assert isinstance(stub, LLMProvider)
    pt, _ct, estimated = stub.token_counts("a b c")
    assert (pt, estimated) == (3, True)  # whitespace estimate


def test_cassette_key_is_stable() -> None:
    assert cassette_key("m", "p") == cassette_key("m", "p")
    assert cassette_key("m", "p") != cassette_key("m", "q")


def test_replay_miss_raises_with_prompt_hash(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"provider": "ollama", "model": "m", "entries": {}}))
    replay = ReplayProvider(path, model="m")
    with pytest.raises(CassetteMissError, match="prompt-hash"):
        replay.complete("a prompt that was never recorded")


def test_replay_mutated_prompt_misses(tmp_path: Path) -> None:
    key = cassette_key("m", "exact prompt")
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps(
            {
                "provider": "ollama",
                "model": "m",
                "entries": {key: {"model": "m", "response": "ok", "eval_count": 1}},
            }
        )
    )
    replay = ReplayProvider(path, model="m")
    assert replay.complete("exact prompt") == "ok"
    # One trailing space is a different prompt, so it must miss.
    with pytest.raises(CassetteMissError):
        replay.complete("exact prompt ")


def test_replay_reads_header_and_vendor_token_counts() -> None:
    cass = json.loads(CASSETTE.read_text())
    assert cass["provider"] == "ollama"
    replay = ReplayProvider(CASSETTE, model=cass["model"])
    assert replay.name == "ollama"
    assert replay.model_digest  # non-empty digest from the real recording
    # Pick any recorded prompt and confirm vendor counts come back, not an estimate.
    some_key = next(iter(cass["entries"]))
    entry = cass["entries"][some_key]
    assert "prompt_eval_count" in entry
    assert "eval_count" in entry
    assert isinstance(entry["eval_count"], int)


def test_ollama_provider_builds_localhost_url_with_no_committed_host() -> None:
    p = OllamaProvider("gemma3:27b", host="localhost:11434")
    assert p.url == "http://localhost:11434/api/generate"
    assert p.model == "gemma3:27b"
    assert p.name == "ollama"
    assert isinstance(p, LLMProvider)


def test_ollama_provider_defaults_to_localhost_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    p = OllamaProvider("gemma3:27b")
    assert p.base == "http://localhost:11434"  # no committed private host


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_ollama_generate_and_token_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"response": "ANSWER", "prompt_eval_count": 42, "eval_count": 7}
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: _FakeResponse(payload))
    p = OllamaProvider("gemma3:27b", host="localhost:11434")
    assert p.complete("hi") == "ANSWER"
    assert p.token_counts("hi") == (42, 7, False)
