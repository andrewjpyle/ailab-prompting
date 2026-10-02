#!/usr/bin/env python3
"""Record a REAL extraction cassette from a live Ollama model.

Adapted from ailab-rag @ 3cd8f82 (``scripts/record_cassette.py``). It rebuilds EVERY prompt
the eval and the proof-of-gate will ask for, so a replay never misses and a red case reddens
on a real metric, not a cassette miss. The recorded set is:

* the three strategies (zero_shot, few_shot, cot) with defenses OFF, over every extraction
  example (dataset x strategies);
* the wrong-label few-shot variant over every extraction example (the broken-few-shot red
  case);
* the primary strategy over every injection example, with defenses OFF and ON (the injection
  sub-experiment and the defenses-off red case).

A record-time preflight asserts the server is up, the model is present, and one real
completion works, failing loudly otherwise. Generation is pinned to temperature 0 and a
fixed seed; the model digest and Ollama version go in the header. The cassette is written
atomically after the entry count is verified.

The model id comes from ``AILAB_RECORD_MODEL`` (default ``gemma3:27b``). The server URL comes
from ``OLLAMA_HOST`` and is NEVER written to disk (only the prompt hash and the response are
stored).

Usage::

    OLLAMA_HOST=http://<host>:11434 make record
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ailab_prompting.config import load_config  # noqa: E402
from ailab_prompting.data import load_extractions, load_injections  # noqa: E402
from ailab_prompting.providers import OllamaProvider, cassette_key  # noqa: E402
from ailab_prompting.strategies import STRATEGIES, build_prompt  # noqa: E402

CASSETTE = REPO / "fixtures" / "cassettes" / "support_extract.json"


def _get_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        out: dict[str, Any] = json.loads(resp.read())
    return out


def _planned_prompts(config: Any) -> list[str]:
    """Every prompt the eval and the proof-of-gate will build, de-duplicated in order."""
    extractions = load_extractions(config.dataset)
    injections = load_injections(config.injections)
    prompts: list[str] = []
    seen: set[str] = set()

    def add(prompt: str) -> None:
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)

    for strat in STRATEGIES:
        for ex in extractions:
            add(build_prompt(strat, ex.message, defenses=False))
    for ex in extractions:
        add(build_prompt("few_shot", ex.message, broken_fewshot=True, defenses=False))
    for x in injections:
        add(build_prompt("few_shot", x.message, defenses=False))
        add(build_prompt("few_shot", x.message, defenses=True))
    return prompts


def main() -> int:
    if not os.environ.get("OLLAMA_HOST"):
        print("record_cassette: set OLLAMA_HOST (never hardcoded here)", file=sys.stderr)
        return 2
    model = os.environ.get("AILAB_RECORD_MODEL", "gemma3:27b")
    config = load_config()
    provider = OllamaProvider(model)

    # --- preflight: server up, model present, one real completion works -----------------
    try:
        tags = _get_json(f"{provider.base}/api/tags").get("models", [])
        names = [str(m.get("name", "")) for m in tags]
        if model not in names:
            print(f"record_cassette: model {model!r} not in {names}; aborting", file=sys.stderr)
            return 2
        version = str(_get_json(f"{provider.base}/api/version").get("version", ""))
        digest = next((str(m.get("digest", "")) for m in tags if m.get("name") == model), "")
        probe = provider.generate("Reply with the single word: ok")
        if not str(probe.get("response", "")).strip():
            print("record_cassette: preflight completion was empty; aborting", file=sys.stderr)
            return 1
    except Exception as exc:  # preflight must fail loudly on any error
        print(f"record_cassette: preflight failed ({exc}); aborting", file=sys.stderr)
        return 1

    prompts = _planned_prompts(config)
    expected = len(prompts)
    print(f"record_cassette: recording {expected} prompts from {model} ({version})", flush=True)

    entries: dict[str, dict[str, Any]] = {}
    for i, prompt in enumerate(prompts, start=1):
        print(f"[{i}/{expected}] calling {model} ...", flush=True)
        raw = provider.generate(prompt)
        response = str(raw.get("response", ""))
        if not response.strip():
            print(f"record_cassette: empty response at prompt {i}; aborting", file=sys.stderr)
            return 1
        entries[cassette_key(model, prompt)] = {
            "model": model,
            "response": response,
            # Vendor's real token counts (not a whitespace estimate).
            "prompt_eval_count": int(raw.get("prompt_eval_count", 0) or 0),
            "eval_count": int(raw.get("eval_count", 0) or 0),
        }

    if len(entries) != expected:
        print(
            f"record_cassette: entry count {len(entries)} != expected {expected}; aborting",
            file=sys.stderr,
        )
        return 1

    payload = {
        "model": model,
        "provider": "ollama",
        "model_digest": digest,
        "ollama_version": version,
        "temperature": 0,
        "seed": OllamaProvider.DEFAULT_SEED,
        "recorded_on": datetime.now(UTC).date().isoformat(),
        "entry_count": len(entries),
        "note": (
            "Real recording from a local Ollama server (host from OLLAMA_HOST, never stored). "
            "Keys are sha256(model, prompt). Covers the three strategies over the extraction "
            "set, the wrong-label few-shot red variant, and the primary strategy over the "
            "injection set with defenses off and on. Pinned temperature 0 and a fixed seed."
        ),
        "entries": entries,
    }
    CASSETTE.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: temp file in the same dir, then os.replace.
    with tempfile.NamedTemporaryFile(
        "w", dir=CASSETTE.parent, prefix=".tmp-", suffix=".json", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        tmp = Path(handle.name)
    os.replace(tmp, CASSETTE)
    print(f"record_cassette: wrote {len(entries)} entries for model {model} to {CASSETTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
