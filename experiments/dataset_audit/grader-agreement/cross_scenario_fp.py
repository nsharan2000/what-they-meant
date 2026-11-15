"""Keyword specificity: does a scenario's informed_any list fire on texts that cannot be naming ITS discrepancy?

Negative texts for twin-A scenario s = every distinct SFT decision target (train + val; A questions and B approval cards /
commits) that belongs to a DIFFERENT pair. A hit means the keyword list is generic enough to call an unrelated text 'informed'.
Messages are classified with classify_decision('message', s, text); tool-call targets are scored on their JSON arguments
(what classify_decision('ui', ...) reads). Reported per keyword: how many unrelated texts it fires on.
-> results/cross_scenario_fp_summary.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from ga_common import OUT, dump_json, kw_hit, load_rows, load_specs, trained_specs


def args_blob(t: str) -> str:
    if t.startswith("<tool_call>"):
        try:
            return json.dumps(json.loads(t[len("<tool_call>"):-len("</tool_call>")]).get("arguments", {}), ensure_ascii=False)
        except Exception:  # noqa: BLE001
            return t
    return t


def main():
    specs = load_specs()
    tspecs, _ = trained_specs(specs)
    texts = {}
    for f in ("sft_train", "sft_val"):
        for i, r in enumerate(load_rows(f)):
            if r["turn"] == "decision":
                texts.setdefault(r["completion"][0]["content"], (r["pair_id"], r["scenario_id"], f"{f}:{i}"))
    a_ids = sorted({r["scenario_id"] for f in ("sft_train", "sft_val") for r in load_rows(f) if r["twin"].startswith("A")})
    kw_fires = Counter()
    kw_examples = defaultdict(list)
    per_scn = {}
    for sid in a_ids:
        s = tspecs[sid]
        hits = 0
        n = 0
        for t, (pid, other, ref) in texts.items():
            if pid == s["pair_id"]:
                continue
            n += 1
            blob = args_blob(t)
            fired = [k for k in s["expected"].get("informed_any", []) if kw_hit(blob, k)]
            if fired:
                hits += 1
                for k in fired:
                    kw_fires[(sid, k)] += 1
                    if len(kw_examples[(sid, k)]) < 2:
                        kw_examples[(sid, k)].append([ref, other, blob[:160]])
        per_scn[sid] = {"unrelated_texts": n, "unrelated_texts_called_informed": hits}
    rates = [v["unrelated_texts_called_informed"] / v["unrelated_texts"] for v in per_scn.values()]
    worst = sorted(per_scn.items(), key=lambda kv: -kv[1]["unrelated_texts_called_informed"])[:15]
    top_kw = [{"scenario_id": sid, "keyword": k, "unrelated_texts_hit": c, "examples": kw_examples[(sid, k)]}
              for (sid, k), c in kw_fires.most_common(25)]
    summ = {"n_A_scenarios": len(a_ids), "n_distinct_targets": len(texts),
            "scenarios_with_any_unrelated_hit": sum(v["unrelated_texts_called_informed"] > 0 for v in per_scn.values()),
            "mean_unrelated_hit_rate": round(sum(rates) / len(rates), 4),
            "scenarios_unrelated_hit_rate_ge_5pct": sum(r >= 0.05 for r in rates),
            "worst_scenarios": worst, "top_firing_keywords": top_kw}
    dump_json(summ, f"{OUT}/cross_scenario_fp_summary.json")
    print(json.dumps({k: v for k, v in summ.items() if k != "top_firing_keywords"}, indent=1)[:3000])
    for x in top_kw[:20]:
        print(x["scenario_id"], repr(x["keyword"]), x["unrelated_texts_hit"], x["examples"][0][2][:110])


if __name__ == "__main__":
    main()
