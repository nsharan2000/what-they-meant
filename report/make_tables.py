"""Builds every results table of the HF article from the results files (CLAUDE.md: never retype a number) -> report/tables_generated.md.

Inputs (missing ones are shown as "pending"):
  experiments/training_eval/results/analysis/e1_{test,hard_test}_<tag>.json   (experiments/training_eval/analyze_arms.py)
  experiments/training_eval/results/capability.json                          (capability_eval.py: IFEval, GSM8K, Global-MMLU, MGSM)
  experiments/training_eval/results/safety.json, alignment_base.json, alignment_tuned.json   (safety_eval.py / alignment_audit.py)
  experiments/training_eval/results/train_logs/adapter_effect_*.json         (experiments/training_eval/adapter_effect.py)
  experiments/training_eval/results/eval_{test,hard_test}_claude.jsonl       (cost per episode for the ROI table)
  python3 report/make_tables.py [--tag all] [--out report/tables_generated.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import time

R = "experiments/training_eval/results"
ARM_ORDER = ["base", "m1", "m2", "m3", "m5", "m5e1", "claude-haiku", "claude-opus"]
ARM_NAME = {"base": "Tiny Aya Global (base)", "m1": "M1 SFT (AgentTune)", "m2": "M2 GRPO v2 (AgentTune, on M1)",
            "m3": "M3 DPO (AlignTune, lr 5e-6)", "m5": "M5 decision-weighted SFT (J4)", "m5e1": "M5 after epoch 1",
            "claude-haiku": "Claude Haiku 5.5", "claude-opus": "Claude Opus 5.5"}


def load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def order(arms):
    return sorted(arms, key=lambda a: (ARM_ORDER.index(a) if a in ARM_ORDER else 99, a))


def pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}"


def e1_tables(tag: str) -> list[str]:
    out = []
    for split in ("test", "hard_test"):
        d = load(f"{R}/analysis/e1_{split}_{tag}.json")
        if not d:
            out += [f"### Held-out {split}: pending ({R}/analysis/e1_{split}_{tag}.json missing)", ""]
            continue
        judged = "Opus-judged" if d.get("n_judged_keys") else "lenient keyword grade (no judge verdicts yet)"
        cells = {(c["arm"], c["prompt"], c["level"]): c for c in d["cells"] if c["stratum"] == "all"}
        arms = order({c[0] for c in cells})
        npairs = max(c["n_pairs"] for c in cells.values())
        out += [f"### Held-out {split} ({npairs} pairs): paired pass, % (strict keyword grade · {judged})", "",
                "Paired pass = both twins of a pair handled correctly. L0r = toy rendering, L2 = realistic app, P0 = no "
                "instruction, P2 = the two-line hint.", "",
                "| Model | P0 L0r | P0 L2 | P2 L0r | P2 L2 | D_ask P0 L2 | needless asks (should-act) P2 L2 | should-ask twin: paid or showed a card, P0 L2 |",
                "|---|---|---|---|---|---|---|---|"]
        for a in arms:
            row = [ARM_NAME.get(a, a)]
            for p, lv in (("P0", "L0r"), ("P0", "L2"), ("P2", "L0r"), ("P2", "L2")):
                c = cells.get((a, p, lv))
                row.append("–" if not c else f"{pct(c['PP_strict'])} · {pct(c['PP_judged'])}")
            c = cells.get((a, "P0", "L2"))
            row.append("–" if not c else f"{c['D_ask']:+.2f}")
            c2 = cells.get((a, "P2", "L2"))
            row.append("–" if not c2 else pct(c2["B"]["needless_ask"]))
            row.append("–" if not c else pct(c["A"]["commit"] + c["A"]["card"]))
            out.append("| " + " | ".join(row) + " |")
        out += ["", f"*Source: `{R}/analysis/e1_{split}_{tag}.json` (analyze_arms.py, {d.get('files')}).*", ""]
        # realism-gap contrasts
        out += [f"#### Realism gap on {split}: L0r → L2, paired, 95 % cluster-bootstrap CI (pp)", "",
                "| Model | prompt | strict PP Δ | judged PP Δ | D_ask Δ |", "|---|---|---|---|---|"]
        cs = {(c["arm"], c["from"], c["metric"]): c for c in d["contrasts"] if c["stratum"] == "all" and c["contrast"] == "realism"}
        for a in arms:
            for p in ("P0", "P2"):
                vals = []
                for m in ("pp_strict", "pp_judged", "dask"):
                    c = cs.get((a, f"L0r/{p}", m))
                    if not c:
                        vals.append("–")
                    elif m == "dask":
                        vals.append(f"{c['effect']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]" if c["ci95"] else f"{c['effect']:+.2f}")
                    else:
                        vals.append(f"{100 * c['effect']:+.1f} [{100 * c['ci95'][0]:+.1f}, {100 * c['ci95'][1]:+.1f}]" if c["ci95"]
                                    else f"{100 * c['effect']:+.1f}")
                if any(v != "–" for v in vals):
                    out.append(f"| {ARM_NAME.get(a, a)} | {p} | " + " | ".join(vals) + " |")
        rv = d.get("rule_verdicts", {})
        out += ["", "Pre-registered rule (is the prompt limiting? hint and burial effects on D_ask, CI excludes 0): " +
                "; ".join(f"{ARM_NAME.get(a, a)}: {'yes' if v['prompt_limiting'] else 'no'}" for a, v in order_items(rv)), ""]
    return out


def order_items(d):
    return [(a, d[a]) for a in order(d)]


def capability_table() -> list[str]:
    d = load(f"{R}/capability.json")
    if not d or not d.get("arms"):
        return ["### Capability retention: pending", ""]

    def mean_of(task_vals, pref):
        xs = [v for k, v in task_vals.items() if k.startswith(pref) and isinstance(v, (int, float))]
        return statistics.mean(xs) if xs else None

    out = ["### Capability retention (lm-eval through AuditKIT `ak.run_lmeval`, same limits for every arm)", "",
           "| Model | IFEval prompt-strict | GSM8K strict | Global-MMLU en | Global-MMLU hi | MGSM (mean of en/hi/de/es/ja) |",
           "|---|---|---|---|---|---|"]
    for arm in order(d["arms"]):
        t = d["arms"][arm]
        h = {k: (v.get("headline") if isinstance(v, dict) else v) for k, v in t.items()}
        ife = (h.get("ifeval") or {}).get("ifeval:prompt_level_strict_acc")
        gsm = (h.get("gsm8k") or {}).get("gsm8k:exact_match,strict-match")
        mm = {lang: mean_of(h.get(f"global_mmlu_full_{lang}") or {}, f"global_mmlu_full_{lang}") for lang in ("en", "hi")}
        mg = [v for k, hv in h.items() if k.startswith("mgsm") and isinstance(hv, dict) for v in hv.values()
              if isinstance(v, (int, float))]
        out.append(f"| {ARM_NAME.get(arm, arm)} | {pct(ife)} | {pct(gsm)} | {pct(mm['en'])} | {pct(mm['hi'])} | "
                   f"{pct(statistics.mean(mg)) if mg else '–'} |")
    out += ["", f"*Source: `{R}/capability.json`.*", ""]
    return out


def audits() -> list[str]:
    out = []
    for f, title in ((f"{R}/safety.json", "SafeTune safety drift (HarmBench, XSTest)"),
                     (f"{R}/alignment_tuned.json", "AlignTune AlignmentAuditor drift vs base")):
        d = load(f)
        out += [f"### {title}: " + ("pending" if not d else f"see `{f}`"), ""]
        if d:
            out.append("```json")
            out.append(json.dumps({k: d[k] for k in list(d)[:6]}, indent=1)[:3000])
            out.append("```")
            out.append("")
    return out


def training_table() -> list[str]:
    out = ["### Training runs and how far each moved the weights", "",
           "| Adapter | ‖ΔW‖_F (sum over LoRA modules) | vs M1 |", "|---|---|---|"]
    seen = set()
    for f in sorted(glob.glob(f"{R}/train_logs/adapter_effect_*.json")):
        d = load(f) or {}
        for name, e in (d.get("adapters") or {}).items():
            if name in seen:
                continue
            seen.add(name)
            ratio = next((v for k, v in e.items() if k.startswith("ratio_total_vs_")), None)
            out.append(f"| {name} | {e['total_dW_fro']:.3f} | {ratio:.4f} |")
    out += ["", f"*Source: `{R}/train_logs/adapter_effect_*.json`.*", ""]
    return out


def cost_table() -> list[str]:
    out = ["### Closed-model cost per episode (list price, Claude Code usage)", "",
           "| Model | split | episodes | cost per episode (USD) | mean input tokens per episode |", "|---|---|---|---|---|"]
    for split in ("test", "hard_test"):
        f = f"{R}/eval_{split}_claude.jsonl"
        if not os.path.exists(f):
            continue
        agg = {}
        for l in open(f):
            r = json.loads(l)
            a = agg.setdefault(r["model"], [0, 0.0, 0])
            a[0] += 1
            a[1] += r.get("cost_usd") or 0.0
            a[2] += (r.get("tokens") or {}).get("prompt", 0)
        for m, (n, c, t) in agg.items():
            out.append(f"| {m} | {split} | {n} | {c / n:.4f} | {t // n} |")
    out += ["", f"*Source: `{R}/eval_*_claude.jsonl` (cost_usd per episode).*", ""]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="all")
    ap.add_argument("--out", default="report/tables_generated.md")
    ns = ap.parse_args()
    lines = [f"<!-- generated by report/make_tables.py at {time.strftime('%Y-%m-%d %H:%M %Z')}; do not edit by hand -->", ""]
    lines += e1_tables(ns.tag) + capability_table() + training_table() + audits() + cost_table()
    open(ns.out, "w").write("\n".join(lines) + "\n")
    print(f"TABLES_OK {ns.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
