"""Step 8: does DPO contrast ask vs act on BOTH twins? Decision class (wtm.reward.decision_of, the GRPO/eval grader) of the chosen
and rejected sides of every dpo_train row, by twin; plus how often the rejected text is the synthetic fallback string.
python3 -B experiments/dataset_audit/twin-contrast-shortcuts/s8_dpo_contrast.py -> results/s8_dpo_contrast.json
"""
import sys
sys.dont_write_bytecode = True
import json, os  # noqa: E401,E402
from collections import Counter  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, jdump, load_rows, load_specs, twin_letter  # noqa: E402
from wtm.reward import decision_of  # noqa: E402
from wtm.protocol import parse_action  # noqa: E402
from wtm.spec import load_env  # noqa: E402


def main():
    specs, _ = load_specs()
    kinds = {}
    out = {"by_twin": {}}
    tab = Counter()
    fmt = Counter()
    lens = []
    ex = {}
    for r in load_rows("dpo_train"):
        s = specs[r["scenario_id"]]
        k = kinds.setdefault(s["env_id"], {t["name"]: t["kind"] for t in load_env(s["env_id"])["tools"]})
        t = twin_letter(r["twin"])
        dc = decision_of(r["chosen"][0]["content"], s, k)[0]
        dr = decision_of(r["rejected"][0]["content"], s, k)[0]
        grp = "ask_like" if dr in ("ask_informed", "ask_generic", "tell_informed", "statement") else \
              "approve" if dr.startswith("approve") else "commit" if dr.startswith("commit") else dr
        tab[(t, dc, grp)] += 1
        sc_ = parse_action(r["chosen"][0]["content"], set(k)).strict_ok
        sr_ = parse_action(r["rejected"][0]["content"], set(k)).strict_ok
        fmt[(t, sc_, sr_)] += 1
        lens.append((t, len(r["chosen"][0]["content"]), len(r["rejected"][0]["content"])))
        ex.setdefault((t, grp), [["dpo_train", r["_i"], r["scenario_id"], r["level"], r["rejected"][0]["content"][:120]]])
    for t in "AB":
        out["by_twin"][t] = {f"chosen={k[1]} | rejected={k[2]}": v for k, v in sorted(tab.items()) if k[0] == t}
    out["examples"] = {f"{k[0]}|{k[1]}": v for k, v in ex.items()}
    out["strict_format_chosen_vs_rejected"] = {f"{k[0]}|chosen_strict={k[1]}|rejected_strict={k[2]}": v for k, v in sorted(fmt.items())}
    out["share_rows_format_alone_separates"] = round(sum(v for k, v in fmt.items() if k[1] and not k[2]) / sum(fmt.values()), 3)
    for t in "AB":
        sel = [x for x in lens if x[0] == t]
        out[f"chars_{t}"] = {"chosen_mean": round(sum(x[1] for x in sel) / len(sel), 1),
                             "rejected_mean": round(sum(x[2] for x in sel) / len(sel), 1),
                             "share_rejected_longer": round(sum(x[2] > x[1] for x in sel) / len(sel), 3)}
    jdump(out, os.path.join(RES, "s8_dpo_contrast.json"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
