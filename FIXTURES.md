# Fixture provenance

Every file under `fixtures/` is listed here. A fixture without an entry must not be merged.

## Rule

**Only public or synthetic data may ever be added to this repository.** That means:

- no real customer, user or employee data, and no personal information of any kind;
- no internal business data, logs, tickets, metrics or exports, even if anonymised;
- no hostnames, IP addresses, account names or other infrastructure identifiers;
- public datasets only under a license that permits redistribution, with the source,
  version and license recorded below.

Synthetic text is written fresh for the fixture, not adapted from real records. The secret
wall (`make scan`) catches some mistakes; it cannot catch all of them, so this rule is
enforced in review as well.

## Source commits

This lab ported its scaffold and its proven patterns from two repositories:

- `ailab-template-python` at `00c298d931c8452ad1bfe9a42a3e7e23998f7c1f`
- `ailab-rag` at `3cd8f82f37d10bfe3c9f6595baf5620b793341f2` (working tree clean at port time)

## Files

### `fixtures/support_extractions.jsonl`

| Field | Value |
|---|---|
| sha256 | `efabe475f7c317cbc545071bc43bfee5ad81adc86b9589172ef17b029b72cc92` |
| Rows | 24 support messages |
| Format | JSONL, one `{"id", "message", "intent", "product", "severity"}` object per line |
| Schema | intent in (account, billing, bug, feedback); product in (api, mobile, sync, web); severity in (high, low, medium) |
| Balance | 6 per intent, 6 per product, and at least 4 per severity (floor is 4 per enum value) |
| Design | Includes format-trap messages (chatty or multi-clause) and reasoning-hard severity cases (a data-loss consequence implies high, a cosmetic note implies low) |
| Author | Andrew Pyle, written by hand for this lab (2026-10-02), with AI assistance |
| Origin | Synthetic. Not derived from, sampled from, or paraphrased from any real product, docs, or support data. The product is fictional |
| Personal data | None. No names, emails, account numbers, or identifiable details |
| License | Apache-2.0, same as the repository |
| Frozen | Yes, before the first recorded run. The sha256 is pinned in docs/DESIGN.md |

### `fixtures/injection_attacks.jsonl`

| Field | Value |
|---|---|
| sha256 | `769764348db2019d470c10819f8b03a2b39e5a27afa391710f3dabbb8023f80a` |
| Rows | 12 indirect injections (4 per family) |
| Format | JSONL, one `{"id", "message", "intent", "product", "severity", "family"}` object per line |
| Families | `fake_system`, `instruction_override`, `role_play`, 4 each |
| Ground truth | The three label fields are the TRUE record a defended model should still return |
| Attack | Each message hides a command to put the sentinel `ADMIN` in a schema field |
| Author | Andrew Pyle, written by hand for this lab (2026-10-02), with AI assistance |
| Origin | Synthetic. No real messages or attack logs |
| License | Apache-2.0 |
| Frozen | Yes, before the first recorded run. The sha256 is pinned in docs/DESIGN.md |

### `fixtures/cassettes/support_extract.json`

| Field | Value |
|---|---|
| sha256 | `bbe25bd3ed18c2619452baba063a4283f1ab53bc4c36ee92c085689c4cf7e8b4` |
| Format | JSON cassette keyed by `sha256(model, prompt)`, with `{"model", "response", "prompt_eval_count", "eval_count"}` entries and a header (provider, model_digest, ollama_version, temperature 0, seed, recorded_on, entry_count) |
| Entries | 120: the three strategies over the 24 extractions (72), the wrong-label few-shot variant over the 24 extractions (24), and the primary strategy over the 12 injections with defenses off and on (24) |
| Determinism | Recorded at temperature 0 with a fixed seed, so a re-record reproduces the same responses (Ollama 0.32.1). Token counts are the vendor's real counts, not a whitespace estimate |
| Purpose | Lets `ReplayProvider` replay completions with no network, exercised by the eval and by `tests/test_live_cassette.py` |
| Origin | REAL recording from the `gemma3:27b` model on a local Ollama server (reached over a private network) on 2026-10-02, using `make record` (`scripts/record_cassette.py`). No synthetic stand-in |
| Infrastructure | The server host is never stored. Only the prompt hash and the response are saved |
| Reproduce | `OLLAMA_HOST=http://<host>:11434 make record`. The host address is never committed |
| License | Apache-2.0 |
