#!/usr/bin/env bash
# Fail if the em dash character (U+2014) appears in any Markdown doc. The content rule for
# this repo bans the em dash; use commas, periods, or parentheses instead.
#
# Scope note (ported from ailab-rag, with the content-vs-data rule made explicit): this lint
# scans docs/ plus every top-level Markdown file ONLY. It deliberately never scans the
# cassette or the results JSON: raw model output is DATA, not prose, and any em dash a model
# emits there must not fail a content lint. Model output quoted in the docs is paraphrased,
# so the lint and check_learning_numbers never conflict.
set -uo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

# Scan docs/ plus every top-level Markdown file (tracked or not). JSON (cassette, results)
# is never matched because only *.md is listed.
files=()
while IFS= read -r f; do files+=("$f"); done < <(
  { find docs -name '*.md' 2>/dev/null; find . -maxdepth 1 -name '*.md' 2>/dev/null; } | sort -u
)

hits=0
for f in "${files[@]}"; do
  [[ -f "$f" ]] || continue
  if grep -nP '\x{2014}' "$f" 2>/dev/null; then
    echo "no_emdash: HIT em dash (U+2014) in $f" >&2
    hits=$((hits + 1))
  elif LC_ALL=C grep -n $'\xe2\x80\x94' "$f" >/dev/null 2>&1; then
    # Fallback for greps without -P: match the UTF-8 bytes of U+2014.
    LC_ALL=C grep -n $'\xe2\x80\x94' "$f" >&2
    echo "no_emdash: HIT em dash (U+2014) in $f" >&2
    hits=$((hits + 1))
  fi
done

if [[ "$hits" -gt 0 ]]; then
  echo "no_emdash: FAIL: $hits file(s) contain an em dash. Replace it with a comma, period, or parentheses." >&2
  exit 1
fi
echo "no_emdash: OK: no em dash (U+2014) in ${#files[@]} Markdown file(s)."
