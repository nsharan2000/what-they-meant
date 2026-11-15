#!/usr/bin/env python3
"""grpo-dynamics VERIFIER, step 11: the analyst's 'checked_ok' items + reward-vs-evaluation semantics + prompt-setting
confound across twins + strong-model d' on the sampled scenarios.

(a) checked_ok: logged reward_std = batch std (ddof 0 or 1?) vs within-group std; completion lengths; format bonus count;
    verbosity; truncation; end ids; clip ratios; grad_norm vs the HF default max_grad_norm 1.0.
(b) reward vs evaluation grader (wtm/reward.py vs wtm/episode.py run_episode/decision_grade): look-ups (reward -0.1, eval:
    free, the episode continues), narration messages (reward: 'statement'; eval: not a decision), adjudicated B rows whose
    expected.accept includes 'act' in evaluation (wtm/labels.py) while the reward table scores commit_gold -0.5.
(c) twin pairs whose two twins carry different prompt settings (P0 vs P2) at the same level, over the whole GRPO pool.
(d) d' of strong models (replication grids) on the sampled scenarios, for scale.
Outputs: results/v11_checks.json
"""
import collections
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from wtm.episode import is_narration, decision_grade  # noqa: E402
from wtm.labels import adjudications  # noqa: E402

OUT = os.path.join(HERE, "results")
TL = os.path.join(ROOT, "experiments", "training_eval", "results", "train_logs")
S = pd.read_csv(os.path.join(OUT, "v01_samples.csv"))
log = [e for e in json.load(open(os.path.join(TL, "m2_grpo", "result.json")))["log_history"] if "loss" in e]
grpo = [json.loads(l) for l in open(os.path.join(ROOT, "dataset", "curated_v121", "grpo_train.jsonl"))]
R = {}
# (a)
d0 = [abs(e["reward_std"] - S[S.step == k]["score"].std(ddof=0)) for k, e in enumerate(log)]
d1 = [abs(e["reward_std"] - S[S.step == k]["score"].std(ddof=1)) for k, e in enumerate(log)]
R["reward_std_vs_batch_std_ddof0_maxdiff"] = float(max(d0))
R["reward_std_vs_batch_std_ddof1_maxdiff"] = float(max(d1))
R["completion_tokens_max"] = int(S["n_tokens"].max())
R["format_bonus_count"] = int((S["fmt"] > 0).sum())
R["verbosity_nonzero"] = int((S["verb"] != 0).sum())
R["end_id_counts"] = {str(k): int(v) for k, v in S["end_id"].value_counts().items()}
R["clip_ratio_max"] = max(e.get("clip_ratio/region_mean", 0) for e in log)
R["grad_norm_max"] = max(e["grad_norm"] for e in log)
# (b)
R["eval_narration_example_is_nondecision"] = is_narration("Let me look that up first.", informed=False)
R["reward_statement_samples"] = int((S.decision == "statement").sum())
R["reward_lookup_samples"] = int((S.decision == "lookup").sum())
adj = adjudications()
adj_rows = [i for i, r in enumerate(grpo) if r["scenario_id"] in adj]
R["adjudicated_grpo_rows"] = len(adj_rows)
R["adjudicated_scenarios_in_grpo"] = sorted({grpo[i]["scenario_id"] for i in adj_rows})
ex = json.loads(grpo[adj_rows[0]]["scenario"]) if adj_rows else None
if ex:
    from wtm.labels import adjudicate
    R["eval_grade_commit_gold_on_adjudicated_example"] = decision_grade(adjudicate(ex), "commit_gold")["success"]
    R["grpo_scenario_has_accept_field"] = "accept" in ex["expected"]
    R["reward_base_commit_gold_on_it"] = -0.5
sa = S[S.scenario_id.isin(adj)]
R["sampled_adjudicated_prompts"] = int(sa.grpo_i.nunique())
R["sampled_adjudicated_commit_gold"] = int((sa.decision == "commit_gold").sum())
# (c) prompt settings across twins
cells = collections.defaultdict(dict)
for r in grpo:
    cells[(r["pair_id"], r["level"])][r["twin"][0]] = r["prompt_setting"]
both = [v for v in cells.values() if "A" in v and "B" in v]
R["cells_both_twins"] = len(both)
R["cells_twins_differ_in_prompt_setting"] = int(sum(v["A"] != v["B"] for v in both))
R["P2_share_by_twin_whole_pool"] = {t: round(float(np.mean([r["prompt_setting"] == "P2" for r in grpo if r["twin"][0] == t])), 4)
                                    for t in ("A", "B")}
# (d) strong models' d' on the sampled scenarios
v10 = json.load(open(os.path.join(OUT, "v10_extras.json")))


def z(p):
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


R["strong_dprime_sampled_scenarios"] = round(z(v10["strong_P_ask_A"]) - z(v10["strong_P_ask_B"]), 4)
R["M1_dprime"] = round(z(v10["M1_P_ask_A"]) - z(v10["M1_P_ask_B"]), 4)
R["base_tiny_aya_dprime"] = round(z(v10["base_tiny_aya_P_ask_A"]) - z(v10["base_tiny_aya_P_ask_B"]), 4)
json.dump(R, open(os.path.join(OUT, "v11_checks.json"), "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
print("OK_v11")
