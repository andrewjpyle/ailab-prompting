"""Prompting eval runner and regression gate.

Usage::

    ailab-eval                                   # the real recorded pass (replay cassette)
    ailab-eval --provider stub                   # plumbing only; fails the provider gate
    ailab-eval --break-fewshot --min-field-accuracy 0.95   # proof of gate

One run scores all three strategies (zero_shot, few_shot, cot) over the frozen extraction
set with defenses OFF, the two baselines, and the injection sub-experiment (defenses OFF vs
ON) over the injection set. The gate uses a TOLERANCE BAND, not an exact threshold.

Exit codes: 0 = gate passed, 1 = a regression (a floor breach, a baseline not beaten, a
saturated task, strategies that do not differ, a provider that is not the recorded model, or
a failed injection defense), 2 = bad config or data (empty dataset, bad TOML, cassette miss).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ailab_core.gate import floor_breach

from ailab_prompting.baselines import EmptyBaseline, MajorityBaseline
from ailab_prompting.config import ConfigError, EvalConfig, load_config
from ailab_prompting.data import (
    INJECTION_FAMILIES,
    DatasetError,
    Example,
    InjectionExample,
    load_extractions,
    load_injections,
    sha256_file,
)
from ailab_prompting.extract import Extractor
from ailab_prompting.metrics import (
    bootstrap_diff_ci,
    field_accuracy,
    mean,
    record_exact,
    record_exact_flags,
    schema_macro_f1,
    wilson_ci,
)
from ailab_prompting.providers import CassetteMissError, LLMProvider, ReplayProvider, StubProvider
from ailab_prompting.schema import ENUMS, FIELDS, is_format_valid
from ailab_prompting.strategies import STRATEGIES, PromptStrategy

SCHEMA_VERSION = 1

# The injection sub-experiment runs on one fixed strategy, independent of which strategy is
# the extraction table's primary row. few_shot is the strongest extractor, so it is the
# fairest host for the attack. Keeping it fixed also means the committed cassette only needs
# injection prompts for this one strategy.
INJECTION_STRATEGY: PromptStrategy = "few_shot"

TABLE_HEADER = (
    "| Date | Commit | Provider | Strategy | field_acc | format_valid | record_exact "
    "| macro_f1 | Notes |\n"
    "|---|---|---|---|---|---|---|---|---|"
)


def current_commit() -> str | None:
    """Commit for the results record: env first (CI, Docker), then git, else None."""
    for var in ("GITHUB_SHA", "AILAB_COMMIT"):
        if sha := os.environ.get(var):
            return sha if sha != "unknown" else None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strategy_scores(
    provider: LLMProvider,
    strategy: PromptStrategy,
    examples: Sequence[Example],
    *,
    broken: bool = False,
    force_invalid: bool = False,
) -> dict[str, Any]:
    """Run one strategy (defenses off) over the extraction set and score it."""
    extractor = Extractor(
        provider, strategy, broken_fewshot=broken, defenses=False, force_invalid=force_invalid
    )
    results = [extractor.run(ex.message) for ex in examples]
    preds = [r.prediction for r in results]
    gold = [ex.record for ex in examples]
    fv_flags = [1 if r.format_valid else 0 for r in results]
    return {
        "field_accuracy": field_accuracy(gold, preds),
        "format_valid": sum(fv_flags) / len(fv_flags),
        "record_exact": record_exact(gold, preds),
        "schema_macro_f1": schema_macro_f1(gold, preds),
        "record_exact_flags": record_exact_flags(gold, preds),
        "format_valid_flags": fv_flags,
        "mean_prompt_tokens": mean([r.prompt_tokens for r in results]),
        "mean_completion_tokens": mean([r.completion_tokens for r in results]),
        "tokens_estimated": any(r.tokens_estimated for r in results),
        "outputs_sha256": _sha("␞".join(r.raw for r in results)),
    }


def _baseline_scores(examples: Sequence[Example]) -> dict[str, Any]:
    gold = [ex.record for ex in examples]
    empty = EmptyBaseline()
    majority = MajorityBaseline.fit(examples)
    empty_preds = [empty.predict(ex.message) for ex in examples]
    maj_preds = [majority.predict(ex.message) for ex in examples]
    empty_fv = sum(is_format_valid(p) for p in empty_preds) / len(empty_preds)
    maj_fv = sum(is_format_valid(p) for p in maj_preds) / len(maj_preds)
    return {
        "empty": {
            "field_accuracy": field_accuracy(gold, empty_preds),
            "format_valid": empty_fv,
        },
        "majority": {
            "field_accuracy": field_accuracy(gold, maj_preds),
            "format_valid": maj_fv,
            "record": majority.record,
        },
    }


def _injection_run(
    provider: LLMProvider,
    strategy: PromptStrategy,
    injections: Sequence[InjectionExample],
    *,
    defenses: bool,
) -> dict[str, Any]:
    extractor = Extractor(provider, strategy, defenses=defenses)
    results = [extractor.run(x.message) for x in injections]
    leaked = [1 if r.leaked else 0 for r in results]
    obeyed = [1 if r.obeyed else 0 for r in results]
    return {"leaked": leaked, "obeyed": obeyed}


def _injection_scores(
    provider: LLMProvider,
    strategy: PromptStrategy,
    injections: Sequence[InjectionExample],
    config: EvalConfig,
) -> dict[str, Any]:
    off = _injection_run(provider, strategy, injections, defenses=False)
    on = _injection_run(provider, strategy, injections, defenses=True)
    n = len(injections)
    asr_off = sum(off["leaked"]) / n
    asr_on = sum(on["leaked"]) / n
    reduction_ci = bootstrap_diff_ci(
        [float(v) for v in off["leaked"]], [float(v) for v in on["leaked"]]
    )
    by_family: dict[str, dict[str, Any]] = {}
    for fam in INJECTION_FAMILIES:
        idx = [i for i, x in enumerate(injections) if x.family == fam]
        if not idx:
            continue
        by_family[fam] = {
            "n": len(idx),
            "asr_off": _round(sum(off["leaked"][i] for i in idx) / len(idx)),
            "asr_on": _round(sum(on["leaked"][i] for i in idx) / len(idx)),
            "obeyed_off": _round(sum(off["obeyed"][i] for i in idx) / len(idx)),
        }
    precondition_met = n >= config.insufficient_floor and asr_off >= config.asr_off_floor
    verdict = "PASS" if precondition_met else "INSUFFICIENT"
    lo, hi = wilson_ci(sum(off["leaked"]), n)
    return {
        "n": n,
        "strategy": strategy,
        "asr_off": asr_off,
        "asr_on": asr_on,
        "obeyed_off": sum(off["obeyed"]) / n,
        "obeyed_on": sum(on["obeyed"]) / n,
        "asr_off_ci": [_round(lo), _round(hi)],
        "reduction": {
            "value": _round(reduction_ci[2]),
            "ci": [_round(reduction_ci[0]), _round(reduction_ci[1])],
            "significant": reduction_ci[0] > 0.0,
        },
        "by_family": by_family,
        "precondition_floor": config.asr_off_floor,
        "asr_on_max": config.asr_on_max,
        "reduction_margin": config.asr_reduction_margin,
        "verdict": verdict,
        "leaked_off_flags": off["leaked"],
        "leaked_on_flags": on["leaked"],
    }


def _class_counts(
    examples: Sequence[Example], injections: Sequence[InjectionExample], floor: int
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for fieldname in FIELDS:
        for value in ENUMS[fieldname]:
            counts[f"{fieldname}:{value}"] = sum(
                1 for ex in examples if getattr(ex, fieldname) == value
            )
    for fam in INJECTION_FAMILIES:
        counts[f"family:{fam}"] = sum(1 for x in injections if x.family == fam)
    insufficient = [name for name, c in counts.items() if c < floor]
    return {"counts": counts, "insufficient": insufficient, "floor": floor}


def _between_strategy_delta(strategies: dict[str, dict[str, Any]]) -> dict[str, Any]:
    best_metric = ""
    best_value = -1.0
    for metric in ("field_accuracy", "format_valid", "record_exact"):
        values = [strategies[s][metric] for s in STRATEGIES]
        spread = max(values) - min(values)
        if spread > best_value:
            best_value, best_metric = spread, metric
    return {"value": best_value, "metric": best_metric}


def evaluate(
    config: EvalConfig,
    provider: LLMProvider,
    *,
    max_asr: float | None = None,
    broken: bool = False,
    force_invalid: bool = False,
) -> dict[str, Any]:
    """Run one eval and return the results record written to ``eval_results.json``.

    Raises :class:`DatasetError` for an empty dataset (exit 2). ``max_asr`` adds a
    proof-of-gate ceiling on the defenses-off attack-success-rate.
    """
    examples = load_extractions(config.dataset)
    injections = load_injections(config.injections)

    strategies: dict[str, dict[str, Any]] = {}
    for strat in STRATEGIES:
        strategies[strat] = _strategy_scores(
            provider, strat, examples, broken=broken, force_invalid=force_invalid
        )

    baselines = _baseline_scores(examples)
    injection = _injection_scores(provider, INJECTION_STRATEGY, injections, config)
    classes = _class_counts(examples, injections, config.insufficient_floor)

    primary = strategies[config.primary_strategy]
    delta = _between_strategy_delta(strategies)

    best_strategy_fa = max(strategies[s]["field_accuracy"] for s in STRATEGIES)
    best_baseline_fa = max(
        baselines["empty"]["field_accuracy"], baselines["majority"]["field_accuracy"]
    )
    margin = best_strategy_fa - best_baseline_fa

    record_exact_max = max(strategies[s]["record_exact"] for s in STRATEGIES)
    saturated = record_exact_max >= 1.0

    outputs = {strategies[s]["outputs_sha256"] for s in STRATEGIES}
    byte_identical = len(outputs) == 1

    # Headline strategy comparison: the primary strategy vs zero_shot on record_exact.
    other = "zero_shot" if config.primary_strategy != "zero_shot" else "few_shot"
    head_ci = bootstrap_diff_ci(
        [float(v) for v in primary["record_exact_flags"]],
        [float(v) for v in strategies[other]["record_exact_flags"]],
    )

    failures = _gate(
        config,
        strategies,
        primary,
        baselines,
        margin,
        saturated,
        byte_identical,
        delta,
        injection,
        provider,
        max_asr,
    )

    metrics = {
        "field_accuracy": _round(primary["field_accuracy"]),
        "format_valid": _round(primary["format_valid"]),
        "record_exact": _round(primary["record_exact"]),
        "schema_macro_f1": _round(primary["schema_macro_f1"]),
        "n": len(examples),
    }
    strategy_table = {
        s: {
            "field_accuracy": _round(strategies[s]["field_accuracy"]),
            "format_valid": _round(strategies[s]["format_valid"]),
            "record_exact": _round(strategies[s]["record_exact"]),
            "schema_macro_f1": _round(strategies[s]["schema_macro_f1"]),
            "record_exact_ci": list(
                map(_round, wilson_ci(sum(strategies[s]["record_exact_flags"]), len(examples)))
            ),
            "mean_prompt_tokens": _round(strategies[s]["mean_prompt_tokens"]),
            "mean_completion_tokens": _round(strategies[s]["mean_completion_tokens"]),
            "tokens_estimated": strategies[s]["tokens_estimated"],
            "outputs_sha256": strategies[s]["outputs_sha256"],
        }
        for s in STRATEGIES
    }
    cassette_info = _cassette_info(provider)
    return {
        "schema_version": SCHEMA_VERSION,
        "lab": config.lab,
        "dataset": config.dataset.stem,
        "provider": provider.name,
        "model": provider.model,
        "primary_metric": config.primary_metric,
        "metrics": metrics,
        "threshold": config.thresholds.get(config.primary_metric, config.min_field_accuracy),
        "passed": not failures,
        "commit": current_commit(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        # Diagnostic extras below (not part of the schema_version 1 contract).
        "primary_strategy": config.primary_strategy,
        "strategies": strategy_table,
        "primary_flags": {
            "record_exact": primary["record_exact_flags"],
            "format_valid": primary["format_valid_flags"],
        },
        "baselines": {
            "empty": {k: _round(v) for k, v in baselines["empty"].items()},
            "majority": {
                "field_accuracy": _round(baselines["majority"]["field_accuracy"]),
                "format_valid": _round(baselines["majority"]["format_valid"]),
                "record": baselines["majority"]["record"],
            },
            "best_field_accuracy": _round(best_strategy_fa),
            "best_baseline_field_accuracy": _round(best_baseline_fa),
            "margin": _round(margin),
            "required_margin": config.baseline_margin,
        },
        "strategy_delta": {
            "value": _round(delta["value"]),
            "metric": delta["metric"],
            "required_min": config.strategy_delta_min,
            "byte_identical": byte_identical,
            "headline": {
                "a": config.primary_strategy,
                "b": other,
                "metric": "record_exact",
                "delta": _round(head_ci[2]),
                "ci": [_round(head_ci[0]), _round(head_ci[1])],
                "significant": head_ci[0] > 0.0 or head_ci[1] < 0.0,
            },
        },
        "saturation": {"record_exact_max": _round(record_exact_max), "saturated": saturated},
        "injection": {
            "n": injection["n"],
            "strategy": injection["strategy"],
            "asr_off": _round(injection["asr_off"]),
            "asr_on": _round(injection["asr_on"]),
            "obeyed_off": _round(injection["obeyed_off"]),
            "obeyed_on": _round(injection["obeyed_on"]),
            "asr_off_ci": injection["asr_off_ci"],
            "reduction": injection["reduction"],
            "by_family": injection["by_family"],
            "precondition_floor": injection["precondition_floor"],
            "asr_on_max": injection["asr_on_max"],
            "reduction_margin": injection["reduction_margin"],
            "verdict": injection["verdict"],
        },
        "cost": {
            s: {"mean_completion_tokens": _round(strategies[s]["mean_completion_tokens"])}
            for s in STRATEGIES
        },
        "cassette": cassette_info,
        "classes": classes,
        "config": {
            "provider": config.provider,
            "primary_strategy": config.primary_strategy,
            "primary_metric": config.primary_metric,
            "tolerance": config.tolerance,
            "baseline_margin": config.baseline_margin,
            "strategy_delta_min": config.strategy_delta_min,
        },
        "dataset_sha256": sha256_file(config.dataset),
        "injections_sha256": sha256_file(config.injections),
        "tolerance": config.tolerance,
        "thresholds": config.thresholds,
        "failures": failures,
    }


def _cassette_info(provider: LLMProvider) -> dict[str, Any]:
    if isinstance(provider, ReplayProvider):
        return {
            "provider": provider.name,
            "model": provider.model,
            "model_digest": provider.model_digest,
            "entry_count": provider.entry_count,
            "call_count": provider.call_count,
        }
    calls = getattr(provider, "calls", 0)
    return {
        "provider": provider.name,
        "model": provider.model,
        "model_digest": "",
        "entry_count": 0,
        "call_count": calls,
    }


def _gate(
    config: EvalConfig,
    strategies: dict[str, dict[str, Any]],
    primary: dict[str, Any],
    baselines: dict[str, Any],
    margin: float,
    saturated: bool,
    byte_identical: bool,
    delta: dict[str, Any],
    injection: dict[str, Any],
    provider: LLMProvider,
    max_asr: float | None,
) -> list[str]:
    """Build the list of regression messages; empty means the gate passes."""
    tol = config.tolerance
    failures: list[str] = []

    if provider.name != "ollama":
        failures.append(
            f"provider {provider.name!r} != 'ollama'; make eval requires the recorded real pass"
        )
    call_count = _cassette_info(provider)["call_count"]
    if provider.name == "ollama" and call_count <= 0:
        failures.append("cassette call_count is 0; no recorded completions were replayed")
    if byte_identical:
        failures.append(
            "per-strategy outputs are byte-identical; the provider shows no prompt effect"
        )
    if delta["value"] < config.strategy_delta_min:
        failures.append(
            f"between-strategy delta {delta['value']:.4f} < required "
            f"{config.strategy_delta_min:.4f} on {delta['metric']}; strategies do not differ"
        )
    if msg := floor_breach(
        "field_accuracy", primary["field_accuracy"], config.min_field_accuracy, tol
    ):
        failures.append(msg)
    if msg := floor_breach("format_valid", primary["format_valid"], config.min_format_valid, tol):
        failures.append(msg)
    if margin < config.baseline_margin:
        failures.append(
            f"baseline_margin {margin:.4f} < required {config.baseline_margin:.4f}; "
            "the strategies do not beat the baselines"
        )
    if saturated:
        failures.append(
            "saturation record_exact is 1.0000 for a strategy; the task cannot detect regressions"
        )
    # Injection gate: only when the precondition is met (else INSUFFICIENT, reported not failed).
    if injection["verdict"] == "PASS":
        if injection["asr_on"] > config.asr_on_max:
            failures.append(
                f"injection asr_on {injection['asr_on']:.4f} > max {config.asr_on_max:.4f}"
            )
        reduction = injection["reduction"]["value"] or 0.0
        if reduction < config.asr_reduction_margin:
            failures.append(
                f"injection asr reduction {reduction:.4f} < margin "
                f"{config.asr_reduction_margin:.4f}; defenses do not reduce the attack"
            )
    if max_asr is not None and injection["asr_off"] > max_asr:
        failures.append(
            f"injection attack_success_rate (asr) {injection['asr_off']:.4f} > max {max_asr:.4f}"
        )
    return failures


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def markdown_rows(result: dict[str, Any]) -> str:
    """README "Eval results" rows, one per strategy, derived only from a results record."""
    commit = (result["commit"] or "unknown")[:7]
    note = "PASS" if result["passed"] else "FAIL"
    rows = []
    for strat in STRATEGIES:
        s = result["strategies"][strat]
        mark = note if strat == result["primary_strategy"] else ""
        rows.append(
            f"| {result['generated_at'][:10]} | {commit} | {result['provider']} | {strat} "
            f"| {_fmt(s['field_accuracy'])} | {_fmt(s['format_valid'])} "
            f"| {_fmt(s['record_exact'])} | {_fmt(s['schema_macro_f1'])} | {mark} |"
        )
    return "\n".join(rows)


def _print_summary(result: dict[str, Any]) -> None:
    b = result["baselines"]
    print(
        f"baselines: empty_fa={_fmt(b['empty']['field_accuracy'])} "
        f"majority_fa={_fmt(b['majority']['field_accuracy'])} "
        f"best_strategy_fa={_fmt(b['best_field_accuracy'])} margin={_fmt(b['margin'])} "
        f"(required {b['required_margin']:.2f})"
    )
    d = result["strategy_delta"]
    print(
        f"between-strategy delta: {_fmt(d['value'])} on {d['metric']} "
        f"(required {d['required_min']:.2f}); byte_identical={d['byte_identical']}"
    )
    h = d["headline"]
    sig = "significant" if h["significant"] else "no significant difference"
    print(
        f"headline {h['a']} vs {h['b']} on {h['metric']}: delta {_fmt(h['delta'])} "
        f"CI [{_fmt(h['ci'][0])}, {_fmt(h['ci'][1])}] ({sig})"
    )
    inj = result["injection"]
    print(
        f"injection ({inj['strategy']}, n={inj['n']}): asr_off={_fmt(inj['asr_off'])} "
        f"asr_on={_fmt(inj['asr_on'])} obeyed_off={_fmt(inj['obeyed_off'])} "
        f"obeyed_on={_fmt(inj['obeyed_on'])} -> {inj['verdict']}"
    )
    sat = result["saturation"]
    print(
        f"saturation guard: record_exact_max={_fmt(sat['record_exact_max'])} "
        f"-> {'SATURATED' if sat['saturated'] else 'ok'}"
    )
    classes = result["classes"]
    if classes["insufficient"]:
        print(f"INSUFFICIENT classes (< {classes['floor']}): {', '.join(classes['insufficient'])}")
    cass = result["cassette"]
    print(
        f"cassette: provider={cass['provider']} model={cass['model']} "
        f"entries={cass['entry_count']} replayed_calls={cass['call_count']}"
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ailab-eval", description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="TOML config path ($AILAB_CONFIG or eval_config.toml)")
    parser.add_argument("--dataset", type=Path, help="override the extraction JSONL")
    parser.add_argument("--injections", type=Path, help="override the injection JSONL")
    parser.add_argument("--provider", choices=["stub", "replay"], help="override the provider")
    parser.add_argument("--cassette", type=Path, help="cassette path for --provider replay")
    parser.add_argument("--model", help="recorded model id to replay")
    parser.add_argument(
        "--primary-strategy",
        dest="primary_strategy",
        choices=list(STRATEGIES),
        help="which strategy's metrics are the top-level contract row",
    )
    parser.add_argument("--output", type=Path, help="override the results JSON path")
    parser.add_argument("--min-field-accuracy", type=float, dest="min_field_accuracy")
    parser.add_argument("--min-format-valid", type=float, dest="min_format_valid")
    parser.add_argument("--tolerance", type=float)
    parser.add_argument(
        "--break-fewshot",
        action="store_true",
        help="proof-of-gate: use wrong-label few-shot exemplars (accuracy regression)",
    )
    parser.add_argument(
        "--force-invalid",
        action="store_true",
        help="proof-of-gate: emit an out-of-enum record (format-valid negative control)",
    )
    parser.add_argument(
        "--max-asr",
        type=float,
        dest="max_asr",
        help="proof-of-gate: fail if the defenses-off attack-success-rate exceeds this",
    )
    parser.add_argument(
        "--table-from",
        type=Path,
        metavar="RESULTS_JSON",
        help="print README table rows from an existing results file, without running",
    )
    return parser.parse_args(argv)


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "dataset",
        "injections",
        "provider",
        "cassette",
        "model",
        "primary_strategy",
        "output",
        "min_field_accuracy",
        "min_format_valid",
        "tolerance",
    )
    return {key: getattr(args, key) for key in keys if getattr(args, key) is not None}


def _build_provider(config: EvalConfig) -> LLMProvider:
    if config.provider == "replay":
        return ReplayProvider(config.cassette, model=config.model)
    return StubProvider()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.table_from is not None:
        try:
            record = json.loads(args.table_from.read_text(encoding="utf-8"))
            print(f"{TABLE_HEADER}\n{markdown_rows(record)}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"ailab-eval: error: cannot read {args.table_from}: {exc!r}", file=sys.stderr)
            return 2
        return 0
    try:
        config = replace(load_config(args.config), **_overrides(args))
        provider = _build_provider(config)
        result = evaluate(
            config,
            provider,
            max_asr=args.max_asr,
            broken=args.break_fewshot,
            force_invalid=args.force_invalid,
        )
    except (ConfigError, DatasetError, OSError, CassetteMissError) as exc:
        print(f"ailab-eval: error: {exc}", file=sys.stderr)
        return 2

    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    table = f"{TABLE_HEADER}\n{markdown_rows(result)}"
    print(table)
    _print_summary(result)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"### Eval results\n\n{table}\n\n")

    if not result["passed"]:
        for failure in result["failures"]:
            print(f"REGRESSION: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
