"""Compare two eval result files that change exactly ONE config variable.

Usage::

    python -m ailab_prompting.compare results/run_zero_shot.json results/run_few_shot.json

It prints the single changed variable and the delta on each shared top-level metric, plus a
seeded bootstrap confidence interval on the primary-metric delta (from the per-example flags
each run stores). It refuses to compare two runs that differ in more than one config
variable, because then the delta could not be attributed to one cause. Adapted from
ailab-rag @ 3cd8f82 (``src/ailab_rag/compare.py``), with the CI added.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ailab_prompting.metrics import bootstrap_diff_ci


def changed_variables(a: dict[str, Any], b: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Return the config keys whose value differs between two result records."""
    cfg_a, cfg_b = a["config"], b["config"]
    keys = sorted(set(cfg_a) | set(cfg_b))
    return {k: (cfg_a.get(k), cfg_b.get(k)) for k in keys if cfg_a.get(k) != cfg_b.get(k)}


def metric_deltas(a: dict[str, Any], b: dict[str, Any]) -> list[tuple[str, float, float, float]]:
    """Return ``(metric, a_value, b_value, delta)`` for each shared numeric metric."""
    ma, mb = a["metrics"], b["metrics"]
    rows: list[tuple[str, float, float, float]] = []
    for key in ma:
        if key == "n" or key not in mb:
            continue
        va, vb = ma[key], mb[key]
        if isinstance(va, int | float) and isinstance(vb, int | float):
            rows.append((key, float(va), float(vb), float(vb) - float(va)))
    return rows


def render(a: dict[str, Any], b: dict[str, Any]) -> str:
    changed = changed_variables(a, b)
    lines = [
        f"A: {a['primary_strategy']} provider={a['provider']} (passed={a['passed']})",
        f"B: {b['primary_strategy']} provider={b['provider']} (passed={b['passed']})",
    ]
    if len(changed) != 1:
        lines.append(f"ERROR: runs differ in {len(changed)} variables ({sorted(changed)}); need 1.")
        return "\n".join(lines)
    name, (va, vb) = next(iter(changed.items()))
    lines.append(f"\nsingle changed variable: {name}: {va} -> {vb}\n")
    lines.append(f"{'metric':<16} {'A':>10} {'B':>10} {'delta':>10}")
    for metric, av, bv, delta in metric_deltas(a, b):
        lines.append(f"{metric:<16} {av:>10.4f} {bv:>10.4f} {delta:>+10.4f}")
    metric = a["primary_metric"]
    flags = metric if metric in ("record_exact", "format_valid") else "record_exact"
    fa = a.get("primary_flags", {}).get(flags)
    fb = b.get("primary_flags", {}).get(flags)
    if fa and fb:
        lo, hi, point = bootstrap_diff_ci([float(x) for x in fb], [float(x) for x in fa])
        sig = "significant" if (lo > 0.0 or hi < 0.0) else "no significant difference"
        lines.append(
            f"\n{flags} delta (B - A): {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  ({sig})"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: python -m ailab_prompting.compare RUN_A.json RUN_B.json", file=sys.stderr)
        return 2
    a = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    b = json.loads(Path(args[1]).read_text(encoding="utf-8"))
    rendered = render(a, b)
    print(rendered)
    return 1 if "ERROR" in rendered else 0


if __name__ == "__main__":
    raise SystemExit(main())
