"""Build the README graphics for ailab-prompting.

Run from the repo root after vendoring the kit:
    uv run --with playwright --with pillow python docs/assets/src/render.py docs/assets/src docs/assets

Four graphics:
  hero      structural wheel of the three strategies plus the injection probe
  flow      how it works: message -> strategy -> replay -> parse -> validate -> gate -> results
  anatomy   the REAL eval output, built from a committed capture (never typed)
  catalog   the make targets, read from the Makefile workflow

Data rule: the anatomy numbers come only from captures/eval_run.json via load_capture().
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import readme_kit as k  # noqa: E402

CAP_DATE = "2026-10-03"  # the capture's captured_at date, used only in the footer label


def parse_eval(output: str) -> dict:
    """Pull the strategy rows and the summary numbers out of the captured eval stdout."""
    rows: list[dict] = []
    summary: dict[str, str] = {}
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("| 20"):  # a strategy data row (starts with the date)
            cells = [c.strip() for c in line.strip("|").split("|")]
            # date, commit, provider, strategy, field_acc, format_valid, record_exact, macro_f1, notes
            rows.append(
                {
                    "strategy": cells[3],
                    "field_acc": cells[4],
                    "format_valid": cells[5],
                    "record_exact": cells[6],
                    "macro_f1": cells[7],
                    "notes": cells[8],
                }
            )
        elif line.startswith("baselines:"):
            summary["baselines"] = line
        elif line.startswith("between-strategy delta:"):
            summary["delta"] = line
        elif line.startswith("injection"):
            summary["injection"] = line
        elif line.startswith("saturation guard:"):
            summary["saturation"] = line
        elif line.startswith("cassette:"):
            summary["cassette"] = line

    def field(line: str, key: str) -> str:
        for tok in line.replace("(", " ").replace(")", " ").split():
            if tok.startswith(key + "="):
                return tok.split("=", 1)[1]
        return ""

    return {
        "rows": rows,
        "majority_fa": field(summary["baselines"], "majority_fa"),
        "best_fa": field(summary["baselines"], "best_strategy_fa"),
        "margin": field(summary["baselines"], "margin"),
        "delta_val": summary["delta"].split(":", 1)[1].split("on")[0].strip(),
        "asr_off": field(summary["injection"], "asr_off"),
        "injection_n": summary["injection"].split("n=")[1].split(")")[0],
        "injection_verdict": summary["injection"].split("->")[-1].strip(),
        "sat_max": field(summary["saturation"], "record_exact_max"),
        "cassette_model": field(summary["cassette"], "model"),
        "cassette_entries": field(summary["cassette"], "entries"),
        "cassette_calls": field(summary["cassette"], "replayed_calls"),
    }


def build_hero() -> str:
    right = k.wheel(
        ["ZERO", "FEW", "CoT", "INJECT"],
        center_top="REPLAY",
        center_main="n=24",
    )
    return k.hero(
        kicker="AILAB · PROMPTING",
        title="Change one prompt.",
        accent="Measure the cost.",
        lede_html=(
            "Three prompting strategies on one structured-extraction task, scored against a real "
            "<b style='color:#F4EFE6'>gemma3:27b</b> model recorded once and replayed. No network, "
            "no stub, one number that gates CI."
        ),
        rules=[
            ("One variable per run", "Swap the strategy, hold the task. The accuracy and cost delta is attributable."),
            ("Real token cost", "The cost axis is the vendor eval_count, never a whitespace estimate."),
            ("Honest when it cannot prove", "The injection probe reports INSUFFICIENT, not a fake defense win."),
        ],
        pill="REPLAY · OFFLINE · GATED",
        right_html=right,
        footer_left="AILAB-PROMPTING · HOW IT WORKS",
    )


def build_flow() -> str:
    y, h, w = 330, 150, 176
    xs = [56, 272, 488, 704, 920, 1136]
    boxes = (
        k.box(xs[0], y, w, h, "SUPPORT MESSAGE", ["untrusted text", "one ticket in"])
        + k.box(xs[1], y, w, h, "PROMPTSTRATEGY", ["zero_shot", "few_shot", "cot"], accent=True)
        + k.box(xs[2], y, w, h, "REPLAYPROVIDER", ["real gemma3:27b", "recorded cassette", "no network"])
        + k.box(xs[3], y, w, h, "PARSE + VALIDATE", ["JSON record", "schema + enum", "allowlist"])
        + k.box(xs[4], y, w, h, "HONEST GATE", ["floors, baselines", "saturation, delta", "injection"], accent=True)
        + k.box(xs[5], y, w, h, "EVAL_RESULTS", ["metrics per", "strategy", "PASS / FAIL"])
    )
    cy = y + h // 2
    specs = []
    for i in range(5):
        x1 = xs[i] + w
        x2 = xs[i + 1]
        specs.append((x1, cy, x2, cy))
    return k.flow(
        kicker="PIPELINE",
        title_html="One message, one record, one " + k.em("gate"),
        subline="src/ailab_prompting: strategies.py · providers.py · extract.py · schema.py · eval.py",
        boxes_html=boxes,
        arrow_specs=specs,
        footer_left="AILAB-PROMPTING · HOW IT WORKS",
    )


def build_anatomy(cap: dict) -> str:
    d = parse_eval(cap["output"])
    doc: list[tuple[str, str]] = [
        ("h1", "ailab-eval (real gemma3:27b, replayed)"),
        ("m", f"$ ./.venv/bin/ailab-eval   commit {cap['commit'][:7]}   {cap['captured_at'][:10]}"),
        ("h2", "Per-strategy score"),
        ("code", "strategy    field_acc  record_exact  format_valid"),
    ]
    for r in d["rows"]:
        tag = "  PASS" if r["notes"] == "PASS" else ""
        doc.append(
            (
                "code",
                f"{r['strategy']:<11} {r['field_acc']:>9}  {r['record_exact']:>12}  {r['format_valid']:>12}{tag}",
            )
        )
    doc += [
        ("h2", "Gate checks"),
        ("code", f"baselines    best {d['best_fa']} vs majority {d['majority_fa']}, margin {d['margin']}"),
        ("code", f"delta        {d['delta_val']} on record_exact (required 0.02)"),
        ("code", f"saturation   record_exact_max {d['sat_max']}  ->  ok"),
        ("code", f"injection    asr_off {d['asr_off']} over n={d['injection_n']}  ->  {d['injection_verdict']}"),
        ("code", f"cassette     {d['cassette_model']}, {d['cassette_entries']} entries, {d['cassette_calls']} calls"),
    ]
    notes = [
        (150, "few_shot wins field_accuracy for the fewest output tokens."),
        (300, "A blind majority guess scores far below every strategy, so the task is real."),
        (470, "The attack never fired even undefended, so the probe is INSUFFICIENT, not a win."),
        (560, "Real vendor token counts, replayed from a committed cassette."),
    ]
    return k.anatomy(
        kicker="ANATOMY OF A REAL RUN",
        doc_lines=doc,
        notes=notes,
        footer_left=f"AILAB-PROMPTING · REAL RUN {CAP_DATE}",
        doc_width=760,
    )


def build_catalog() -> str:
    cards = [
        ("SETUP", "Create the locked env", "make install", "uv sync --locked"),
        ("CHECK", "Lint, format, types", "make lint", "ruff + mypy strict"),
        ("TEST", "Suite with coverage floor", "make test", "fails under 90 percent"),
        ("EVAL", "Replay cassette, enforce gate", "make eval", "offline, no key"),
        ("DEMO", "Each prompt, then the eval", "make demo", "stub format path"),
        ("PROVE", "Each sabotage must fail", "make red", "names the metric"),
        ("COMPARE", "Diff two runs, one variable", "make compare", "bootstrap CI"),
        ("RECORD", "Re-record from a live model", "make record", "needs OLLAMA_HOST"),
    ]
    return k.catalog(
        kicker="MAKE TARGETS",
        title_html="The workflow is " + k.em("make targets"),
        sub_html="Every number in this lab comes from one of these, never typed by hand.",
        cards=cards,
        footer_left="AILAB-PROMPTING · HOW IT WORKS",
        cols=4,
        card_height=150,
    )


def main() -> None:
    cap = k.load_capture(HERE / "captures" / "eval_run.json")
    pages = {
        "hero": build_hero(),
        "architecture": build_flow(),
        "anatomy": build_anatomy(cap),
        "catalog": build_catalog(),
    }
    k.write_pages(HERE, pages)


if __name__ == "__main__":
    main()
