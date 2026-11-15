"""Render experiments/replication/results/summary.json (from analysis.py) as Markdown tables → RESULTS.md. Never retype numbers.

python3 experiments/replication/results_table.py experiments/replication/results/summary.json > experiments/replication/RESULTS.md
"""
import json
import sys

LEVEL_ORDER = ["L0", "L0r", "L0p", "L1", "L2", "L2s", "L3"]


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}"


def short(m):
    return m.split(":", 1)[1].split("@")[0].split(":")[0].split("/")[-1] if ":" in m else m


def main(path):
    d = json.load(open(path))
    cells = d["by_cell"]
    print("# Replication results (auto-generated from summary.json — do not edit by hand)\n")
    print(f"Files: {', '.join(d['files'])} · judged episodes: {d.get('n_judged', 0)}\n")
    for prompt in ("P0", "P2", "P1"):
        cs = [c for c in cells if c["prompt"] == prompt]
        if not cs:
            continue
        print(f"## Prompt {prompt} ({'app prompt only' if prompt == 'P0' else 'app prompt + golden preferences'})\n")
        print("| model | level | pairs | PP strict | PP judged | A informed | A harm | B correct | B needless ask | strict format |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for c in sorted(cs, key=lambda c: (short(c["model"]), LEVEL_ORDER.index(c["level"]) if c["level"] in LEVEL_ORDER else 9)):
            print(f"| {short(c['model'])} | {c['level']} | {c['n_pairs']} | {pct(c['PP_strict'])} | {pct(c['PP_judged'])} | "
                  f"{pct(c['A_success_strict'])} | {pct(c['harm'])} | {pct(c['B_success'])} | {pct(c['needless_ask'])} | "
                  f"{pct(c['strict_format'])} |")
        print()
    gaps = [g for g in d.get("gaps", []) if g["kind"] == "strict"]
    if gaps:
        print("## Gaps (strict paired pass, percentage points; 95 % source-post cluster bootstrap; exact McNemar)\n")
        print("| model | prompt | gap | from → to | pairs | value (pp) | 95 % CI | pass→fail | fail→pass | McNemar p |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for g in sorted(gaps, key=lambda g: (g["gap"], short(g["model"]), g["prompt"])):
            print(f"| {short(g['model'])} | {g['prompt']} | {g['gap']} | {g['from']} → {g['to']} | {g['n_pairs']} | {g['value_pp']} | "
                  f"[{g['ci95_pp'][0]}, {g['ci95_pp'][1]}] | {g['flip_pass_to_fail']} | {g['flip_fail_to_pass']} | {g['mcnemar_p']} |")


if __name__ == "__main__":
    main(sys.argv[1])
