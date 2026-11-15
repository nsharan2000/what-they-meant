"""Step 5: for twin pairs whose ONLY difference is the member request (loci == ['request'] in step 1), where does the evidence that
resolves the request sit? (e.g. S02 '315433' vs '3154': the usual amount is in the member's history inside the system prompt.)
For each such decision row: the decisive strings / decisive-fact fragments found in the prompt OUTSIDE the request, and their
distance from the end of the prompt in ESTIMATED tokens (chars / the row's exact chars-per-token from token_stats.jsonl).
python3 -B experiments/dataset_audit/twin-contrast-shortcuts/s5_request_only_evidence.py -> results/s5_request_only_evidence.json
"""
import sys
sys.dont_write_bytecode = True
import json, os, re  # noqa: E401,E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, jdump, load_jsonl, load_rows, load_specs, load_token_stats  # noqa: E402


def fragments(s):
    out = set(s.get("decisive_strings") or [])
    for f in s.get("decisive_facts") or []:
        out |= {m for m in re.findall(r"[\w.@+-]*\d[\w.,@+-]*", f) if len(m) >= 3}
    return sorted(out)


def main():
    specs, _ = load_specs()
    ts = load_token_stats()
    diffs = [r for r in load_jsonl(os.path.join(RES, "s1_twin_diff_rows.jsonl"))
             if r["file"] == "sft_train" and r["turn"] == "decision" and r["loci"] == ["request"]]
    rows = {r["_i"]: r for r in load_rows("sft_train")}
    out = []
    for d in diffs:
        for side in ("a_row", "b_row"):
            r = rows[d[side]]
            text = "".join(m["content"] for m in r["prompt"])
            req = r["prompt"][1]["content"]
            sys_len = len(r["prompt"][0]["content"])
            cpt = len(text) / ts[("sft_train", r["_i"])]["prompt_tokens"]
            hits = []
            for frag in fragments(specs[r["scenario_id"]]):
                for m in re.finditer(re.escape(frag), r["prompt"][0]["content"]):
                    hits.append({"fragment": frag, "in": "system", "tok_from_end_est": round((len(text) - m.start()) / cpt)})
            out.append({"pair_id": d["pair_id"], "scenario_id": r["scenario_id"], "level": r["level"], "row": ["sft_train", r["_i"]],
                        "request": req, "prompt_tokens": ts[("sft_train", r["_i"])]["prompt_tokens"],
                        "evidence_hits_in_system": hits[:6],
                        "nearest_evidence_tok_from_end_est": min([h["tok_from_end_est"] for h in hits], default=None)})
    summ = {"n_rows": len(out), "by_level": {}}
    for lv in ("L0r", "L1", "L2", "L3"):
        sel = [o for o in out if o["level"] == lv and o["nearest_evidence_tok_from_end_est"] is not None]
        if sel:
            v = sorted(o["nearest_evidence_tok_from_end_est"] for o in sel)
            summ["by_level"][lv] = {"n": len(sel), "median": v[len(v) // 2], "max": v[-1], "n_gt_4096": sum(x > 4096 for x in v)}
    summ["rows_without_evidence_hit"] = [[o["scenario_id"], o["level"]] for o in out if o["nearest_evidence_tok_from_end_est"] is None]
    jdump({"summary": summ, "rows": out}, os.path.join(RES, "s5_request_only_evidence.json"))
    print(json.dumps(summ, indent=1))
    for o in out[:12]:
        print(o["scenario_id"], o["level"], o["request"][:70], o["nearest_evidence_tok_from_end_est"], [h["fragment"] for h in o["evidence_hits_in_system"]][:4])


if __name__ == "__main__":
    main()
