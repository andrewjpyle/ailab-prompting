"""LLM provider seam for the extractor. Stdlib only: no SDKs, no third-party HTTP.

Ported from ``ailab-rag`` @ 3cd8f82 (``src/ailab_rag/providers.py``); the cassette key,
:class:`CassetteMissError`, :class:`ReplayProvider` and :class:`OllamaProvider` are that
module adapted for extraction (vendor token counts added to the replay path).

* :class:`StubProvider` is a deterministic offline backend. It echoes a fixed, format-valid
  record so the prompt -> completion -> parse path runs with no network. The stub CANNOT
  show how a strategy changes a model, so it is a FORMAT path only; the real claim needs
  the recorded Ollama pass (see docs/LEARNING.md).
* :class:`ReplayProvider` serves a committed JSON cassette keyed by ``sha256(model,
  prompt)``. A miss RAISES :class:`CassetteMissError` with the prompt hash, so a stale
  cassette can never silently pass and never falls back to the stub.
* :class:`OllamaProvider` POSTs to a local Ollama server with ``urllib`` from stdlib, used
  only in record mode to fill a cassette.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

_WORD = re.compile(r"\S+")


@runtime_checkable
class LLMProvider(Protocol):
    """The minimum a text-completion backend must offer.

    A real implementation wraps a vendor SDK or a local model server and reads its
    credentials from the environment. Keeping the protocol this narrow is what makes
    providers swappable and the eval reproducible.
    """

    name: str  # provider id recorded in results, e.g. "stub" or "ollama"
    model: str

    def complete(self, prompt: str) -> str: ...

    def token_counts(self, prompt: str) -> tuple[int, int, bool]:
        """Return ``(prompt_tokens, completion_tokens, is_estimate)`` for the cost axis."""
        ...


class StubProvider:
    """Deterministic offline backend. No network, no key. FORMAT path only.

    Returns the same format-valid record for every prompt, so every strategy looks
    identical. That is the point: the stub proves the plumbing, never the prompt effect.
    Token counts are a whitespace ESTIMATE, flagged as such.
    """

    name = "stub"
    model = "stub-extractor-v1"
    FIXED_RECORD = '{"intent": "bug", "product": "sync", "severity": "medium"}'

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return self.FIXED_RECORD

    def token_counts(self, prompt: str) -> tuple[int, int, bool]:
        prompt_tokens = len(_WORD.findall(prompt))
        completion_tokens = len(_WORD.findall(self.FIXED_RECORD))
        return prompt_tokens, completion_tokens, True  # whitespace estimate


def cassette_key(model: str, prompt: str) -> str:
    """Stable key for a cassette entry: ``sha256(model, prompt)``.

    Ported verbatim from ailab-rag @ 3cd8f82.
    """
    raw = json.dumps([model, prompt], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CassetteMissError(RuntimeError):
    """Raised when a ReplayProvider is asked for a prompt its cassette does not hold."""


class ReplayProvider:
    """Replay a committed JSON cassette; a miss raises so a stale cassette cannot pass.

    Ported from ailab-rag @ 3cd8f82. The cassette is a JSON object with a header (provider,
    model, model_digest, ollama_version, temperature, seed) and ``entries`` mapping
    ``sha256(model, prompt)`` to ``{"model", "response", "prompt_eval_count",
    "eval_count"}``. ``name`` mirrors the recorded ``provider`` so the results record names
    what was replayed; the eval gate asserts it is ``ollama``.
    """

    def __init__(self, path: str | Path, model: str = "replay") -> None:
        self.path = Path(path)
        self.model = model
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.name = str(raw.get("provider", "replay"))
        self.model_digest = str(raw.get("model_digest", ""))
        self._entries: dict[str, dict[str, Any]] = raw.get("entries", raw)
        self.calls = 0

    def _hit(self, prompt: str) -> dict[str, Any]:
        key = cassette_key(self.model, prompt)
        hit = self._entries.get(key)
        if hit is None:
            raise CassetteMissError(
                f"cassette miss for model {self.model!r} (prompt-hash {key}); "
                "re-record the cassette with `make record`"
            )
        return hit

    def complete(self, prompt: str) -> str:
        entry = self._hit(prompt)
        self.calls += 1
        return str(entry["response"])

    def token_counts(self, prompt: str) -> tuple[int, int, bool]:
        entry = self._hit(prompt)
        return (
            int(entry.get("prompt_eval_count", 0) or 0),
            int(entry.get("eval_count", 0) or 0),
            False,  # vendor counts, not an estimate
        )

    @property
    def call_count(self) -> int:
        return self.calls

    @property
    def entry_count(self) -> int:
        return len(self._entries)


class OllamaProvider:
    """Local models via Ollama ``/api/generate``. Used only in record mode. Stdlib HTTP.

    Ported from ailab-rag @ 3cd8f82. The server address comes from ``OLLAMA_HOST`` (default
    ``http://localhost:11434``, never a committed tailnet host) and is never stored.
    Generation is pinned to ``temperature`` 0 and a fixed ``seed`` so a re-record is
    deterministic, which keeps the replayed numbers stable.
    """

    name = "ollama"
    DEFAULT_SEED = 7

    def __init__(
        self, model: str, host: str | None = None, timeout: float = 300.0, seed: int = DEFAULT_SEED
    ) -> None:
        self.model = model
        base = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        if not base.startswith("http"):
            base = "http://" + base
        self.base = base.rstrip("/")
        self.url = self.base + "/api/generate"
        self.timeout = timeout
        self.seed = seed

    def generate(self, prompt: str) -> dict[str, Any]:
        """Return the raw Ollama response dict (text plus vendor token counts)."""
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "seed": self.seed},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data: dict[str, Any] = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError) as exc:  # pragma: no cover - network
            raise RuntimeError(f"ollama: {exc}") from exc
        return data

    def complete(self, prompt: str) -> str:
        return str(self.generate(prompt).get("response", ""))

    def token_counts(self, prompt: str) -> tuple[int, int, bool]:  # pragma: no cover - network
        raw = self.generate(prompt)
        return (
            int(raw.get("prompt_eval_count", 0) or 0),
            int(raw.get("eval_count", 0) or 0),
            False,
        )
