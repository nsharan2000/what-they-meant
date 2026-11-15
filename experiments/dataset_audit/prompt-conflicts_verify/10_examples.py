"""V10 - check that every example row the analyst cites (file:line scenario level setting turn) is what it says it is.
    python3 -B experiments/dataset_audit/prompt-conflicts_verify/10_examples.py -> results/10_examples.json
"""
import json
import os

from vcommon import OUT, dump

T = {(r["file"], r["i"]): r for r in (json.loads(l) for l in open(os.path.join(OUT, "table.jsonl")))}
CITED = [("PC1", "sft_train", 0, "S42A", "L1", "P0"), ("PC1", "sft_train", 6, "V094A", "L1", "P0"), ("PC1", "sft_train", 53, "S09A", "L2", "P2"),
         ("PC2", "sft_train", 82, "S14B", "L2", "P2"), ("PC2", "sft_train", 100, "V104B", "L0r", "P2"), ("PC2", "sft_train", 121, "V152B", "L2", "P0"),
         ("PC3", "sft_train", 12, "V042B", "L2", "P0"), ("PC3", "sft_train", 18, "V043B", "L1", "P0"), ("PC3", "sft_train", 181, "S04B", "L2", "P0"),
         ("PC4", "sft_train", 72, "S02B", "L1", "P0"), ("PC4", "sft_train", 131, "S03B", "L3", "P0"), ("PC4", "sft_train", 183, "S03A", "L1", "P0"),
         ("PC5", "sft_train", 1, "V254A", "L2", "P0"), ("PC5", "sft_train", 7, "S07A", "L1", "P0"), ("PC5", "sft_train", 17, "V073A", "L2", "P0"),
         ("PC6", "sft_train", 2, "V532B", "L1", "P2"), ("PC6", "sft_train", 294, "V553B", "L0r", "P0"), ("PC6", "sft_train", 614, "S55B", "L1", "P0"),
         ("PC7", "sft_train", 750, "V183B", "L2", "P2"), ("PC7", "sft_train", 670, "V182A", "L2", "P0"), ("PC7", "sft_train", 782, "V183A", "L2", "P0"),
         ("PC8", "sft_train", 11, "V474A", "L0r", "P0"), ("PC9", "grpo_train", 372, "S02A", "L3", "P0"), ("PC9", "grpo_train", 163, "V104B", "L1", "P2"),
         ("PC9", "grpo_train", 288, "S15B", "L3", "P0"), ("PC10", "sft_train", 132, "S25B", "L0r", "P2"), ("PC10", "sft_train", 176, "S25A", "L0r", "P0"),
         ("PC10", "sft_train", 189, "V254A", "L0r", "P0")]
out = []
for fid, f, i, sid, lv, ps in CITED:
    r = T[(f, i)]
    ok = (r["sid"], r["level"], r["ps"], r["turn"]) == (sid, lv, ps, "decision")
    tgt = (r.get("completion") or r.get("chosen") or {}).get("text", "")
    out.append({"finding": fid, "file": f, "line": i, "claimed": [sid, lv, ps], "actual": [r["sid"], r["level"], r["ps"], r["turn"]],
                "match": ok, "target": tgt[:150]})
p = dump("10_examples.json", {"all_match": all(o["match"] for o in out), "rows": out})
print(json.dumps({"all_match": all(o["match"] for o in out), "mismatches": [o for o in out if not o["match"]]}, indent=1))
for o in out:
    print(o["finding"], o["line"], o["actual"], "|", o["target"][:110])
