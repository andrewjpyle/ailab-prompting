# Design notes

Decisions behind `ailab-prompting`, and the alternatives considered and rejected. This lab
is the second in the `ailab-*` family. It reuses the scaffold and the proven patterns from
`ailab-rag`, and it keeps the same discipline: a number gates CI, and the number comes from a
real model.

## Provenance and ported SHAs

This lab was built from two pinned sources:

- `ailab-template-python` at `00c298d931c8452ad1bfe9a42a3e7e23998f7c1f` (origin/main): the
  eval runner shape, the regression-gate shape, config layering, the secret wall (pre-push
  hook plus CI), non-root Docker, ruff, strict mypy, pytest with a 90 percent coverage floor,
  and the `eval_results.json` `schema_version` contract.
- `ailab-rag` at `3cd8f82f37d10bfe3c9f6595baf5620b793341f2` (local HEAD, working tree verified
  clean at port time): the honest eval gate (tolerance band, saturation guard, baselines
  below, INSUFFICIENT floors), the provider seam (`providers.py`), the record-and-replay
  cassette and `scripts/record_cassette.py`, the content-rule gates (`no_emdash.sh`,
  `check_learning_numbers.py`), and the metric-parity test approach.

The SHAs are also in `BUILD_SOURCES.txt` and `FIXTURES.md`.

### Files ported verbatim or near-verbatim from ailab-rag @ 3cd8f82

Each carries a one-line source comment at the top:

- `src/ailab_prompting/providers.py`: the cassette key, `CassetteMissError`,
  `ReplayProvider`, and `OllamaProvider` (adapted to add vendor token counts to replay).
- `src/ailab_prompting/compare.py`: the one-variable run comparator (a bootstrap CI added).
- `scripts/check_learning_numbers.py`: verbatim.
- `scripts/no_emdash.sh`, `scripts/prove_gate.sh`, `scripts/denylist_scan.sh`: adapted.
- `src/ailab_prompting/config.py`, `src/ailab_prompting/eval.py`: the gate and config shapes.
- `tests/_metrics_ref.py` plus `tests/test_metric_parity.py`: the metric-parity approach.

## 1. The seam is a prompting strategy over a fixed extraction task

The template's single classifier is replaced by a `PromptStrategy` seam (`zero_shot`,
`few_shot`, `cot`) and an `Extractor` behind the `LLMProvider`. The task is structured
extraction to a closed JSON schema.

- Rejected: a free-text classification task. Extraction makes format a measurable axis, so
  few-shot has something to improve, and the schema allowlist doubles as an injection defense.
- Rejected: self-consistency and multi-model routing in v1. One changed variable per
  comparison keeps the result attributable.

## 2. The real recorded pass is REQUIRED, not optional

Prompt engineering is about how the model's behavior changes with the prompt. An offline
stub cannot show that: one canned answer makes every strategy look identical, which is
exactly the vacuous-stub failure the gate now catches (byte-identical outputs fail). So the
default `make eval` replays a real `gemma3:27b` cassette, and the gate fails if the provider
is not `ollama`, if the per-strategy outputs are byte-identical, or if the between-strategy
delta is below the frozen minimum.

- Rejected: a live model in CI. Non-deterministic, costs money or needs a server, and turns
  an outage into a red build. Recording once at temperature 0 with a fixed seed and replaying
  keeps the real behavior without the flakiness.

## 3. Freeze note (before the first real run)

The dataset and the gate margins were FROZEN before the first recorded run, and were not
retuned after seeing scores.

- `fixtures/support_extractions.jsonl` sha256
  `efabe475f7c317cbc545071bc43bfee5ad81adc86b9589172ef17b029b72cc92` (24 examples).
- `fixtures/injection_attacks.jsonl` sha256
  `769764348db2019d470c10819f8b03a2b39e5a27afa391710f3dabbb8023f80a` (12 attacks, 4 per
  family).
- Minimum n: at least 4 examples per enum value of every field, and at least 4 injections per
  family (`insufficient_floor = 4`).
- Frozen margins (`eval_config.toml`): field_accuracy floor 0.55, format_valid floor 0.70,
  tolerance band 0.05, baseline margin 0.10, between-strategy delta minimum 0.02, injection
  asr-off floor 0.15, asr-on ceiling 0.10, asr reduction margin 0.10.

Every headline delta reports a confidence interval (Wilson for a proportion, a seeded
bootstrap for a difference). A delta whose interval overlaps zero is reported as "no
significant difference" rather than claimed as an effect.

## 4. The injection sub-experiment came back INSUFFICIENT (honest)

The gate has a precondition: the defenses-off ASR must exceed the frozen floor on the frozen
n, or the verdict is INSUFFICIENT, never PASS. On this frozen set, `gemma3:27b` obeyed none
of the 12 attacks even with all defenses off, so the measured defenses-off ASR was 0.0 and
the verdict is INSUFFICIENT.

- This is reported as INSUFFICIENT, not dressed up as a defense win. The lab does not fake an
  attack, and a synthetic "obey" cassette is forbidden (vendor responses must come from the
  vendor). The ASR gate wiring is proven instead by `tests/test_injection_gate.py`, which
  drives the gate function with constructed numbers.
- The output-schema allowlist is still a deterministic guarantee: the sentinel `ADMIN` is not
  an enum value, so it can never reach a field when defenses are on. That is a design
  property, verified in `tests/test_extract.py`, not a measured model behavior.
- To get a real measurement, a later slice would use a weaker or more injectable model, or
  more subtle indirect injections. The dataset is frozen, so that is a new experiment, not a
  retune of this one.

## 5. Metrics and cost

Field exact-match and macro-F1 are computed per field over the schema. The cost axis uses the
vendor `eval_count` and `prompt_eval_count` captured in the cassette, not a whitespace
estimate. The whitespace estimate is the stub-only fallback, and it is flagged as an
estimate. Chain-of-thought's higher completion-token count is the real cost signal.

## 6. Content-rule enforcement

`make lint-docs` fails on the em dash character (U+2014) anywhere in the docs. It scans only
Markdown, so the cassette and the results JSON are never scanned: raw model output is data,
not prose. Any model output quoted in the docs is paraphrased, so the em-dash lint and
`check_learning_numbers` never conflict. `make check-learning-numbers` fails if a tagged
figure in `docs/LEARNING.md` differs from the committed results.

## 7. The ailab-core extraction trigger (SATISFIED 2026-10-02)

The hard rule recorded here was: there is NO new lab beyond the existing set until a shared
`ailab-core` package holds the gate, the provider seam, the cassette, and the content-lint, so
the labs stop fanning out O(labs) hand-copies that drift. This lab had copied those pieces from
`ailab-rag` by hand (see the verbatim list above).

That trigger is now SATISFIED. `ailab-core` exists (github.com/andrewjpyle/ailab-core) and this
lab depends on it at a pinned commit: the provider seam, the cassette replay and the gate floor
primitive are imported from `ailab_core`, not copied. `ailab-rag` depends on it too. A new lab
may now be built on top of `ailab-core`.

## 8. Kept from the template

The secret wall (pre-push hook plus a required CI check, both scanning full history), the
non-root multi-stage Docker image, uv with a committed lockfile, strict mypy, a 90 percent
coverage floor, and the versioned `eval_results.json` contract. These are not re-argued here.

## 9. No infrastructure in committed files

The Ollama host comes from `OLLAMA_HOST` at record time and is never committed. There is no
default pointing at a private network; the provider default is `http://localhost:11434`. The
cassette stores only the prompt hash and the response, never the host. The denylist scan
carries the shared-address CGNAT range reserved by RFC 6598 (the range tailnet addresses use)
as a committed pattern in `.denylist-generic.txt`, so the pre-push hook and CI block any such
address, and `git grep` for the record host across history is empty.

## 10. Things deliberately left out of v1

Self-consistency, multi-model routing, a paid-API provider, a browser UI, and a real
faithfulness judge. Each is a later slice behind the seam this lab already defines, and most
belong after `ailab-core` exists.
