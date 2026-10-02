#!/usr/bin/env bash
# Proof-of-gate (make red): the regression gate MUST fail on each sabotage, and name the
# failing metric in stderr. Any case that does not fail, or does not name its metric, fails
# this script. A gate you cannot see fail is not a gate. Adapted from ailab-rag @ 3cd8f82.
#
# Every case runs against the committed REAL cassette (--provider replay), so the broken
# few-shot and defenses-off cases redden on a real metric regression, not a cassette miss
# (the red-variant prompts were recorded at record time).
set -uo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
out="$tmp/r.json"
empty="$tmp/empty.jsonl"
: >"$empty"

status=0

# $1 human label, $2 expected exit code, $3 metric token expected in stderr, rest: eval args
check() {
  local label="$1" want_code="$2" want_metric="$3"
  shift 3
  local err
  err="$(uv run ailab-eval --output "$out" "$@" 2>&1 >/dev/null)"
  local code=$?
  if [[ "$code" -ne "$want_code" ]]; then
    echo "FAIL [$label]: exit $code, expected $want_code" >&2
    echo "$err" | sed 's/^/    /' >&2
    status=1
  elif ! grep -qiE "$want_metric" <<<"$err"; then
    echo "FAIL [$label]: stderr did not name '$want_metric'" >&2
    echo "$err" | sed 's/^/    /' >&2
    status=1
  else
    echo "PASS [$label]: exit $code, named '$want_metric'"
    grep -iE "REGRESSION|error" <<<"$err" | sed 's/^/    /'
  fi
}

echo "== make red: proving the gate fails on each sabotage =="
check "1 broken few-shot (wrong-label exemplars)" 1 "field_accuracy"  --break-fewshot --min-field-accuracy 0.97
check "2 empty dataset (0 examples)"              2 "empty"           --dataset "$empty"
check "3 out-of-enum output (format-valid)"       1 "format_valid"    --force-invalid --min-format-valid 0.95

echo ""
echo "note: the defenses-off ASR red case is NOT shown here. On the frozen injection set,"
echo "gemma3:27b never obeyed even undefended (asr_off = 0.0, verdict INSUFFICIENT). Faking an"
echo "attack, or authoring a synthetic 'obey' cassette, is forbidden; the ASR gate wiring is"
echo "proven instead by the unit test tests/test_injection_gate.py (the gate fires on asr_on >"
echo "max and on the --max-asr override)."

if [[ "$status" -eq 0 ]]; then
  echo "== all live red cases failed the gate as required =="
else
  echo "== RED PROOF FAILED: a sabotage did not trip the gate ==" >&2
fi
exit "$status"
