"""Step 7: where does the ask/act choice sit inside the SFT target? From the exact token statistics
(experiments/dataset_audit/results/token_stats.jsonl, Tiny Aya tokenizer): the first target token of every sft_train / sft_val row by
twin and turn, and the mean target length. A completion-only loss averages over all target tokens, so the share of the loss that
comes from the decision-bearing first token is about 1 / (target tokens) per row.
python3 -B experiments/dataset_audit/twin-contrast-shortcuts/s7_decision_token.py -> results/s7_decision_token.json
"""
import sys
sys.dont_write_bytecode = True
import json, os, statistics as st  # noqa: E401,E402
from collections import Counter, defaultdict  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, ROOT, jdump, load_jsonl, twin_letter, hint_text, load_token_stats, load_rows  # noqa: E402


def main():
    ts = load_jsonl(os.path.join(ROOT, "experiments", "dataset_audit", "results", "token_stats.jsonl"))
    out = {}
    for f in ("sft_train.jsonl", "sft_val.jsonl"):
        rows = [r for r in ts if r["file"] == f]
        g = defaultdict(list)
        for r in rows:
            g[(twin_letter(r["twin"]), "decision" if r["turn"] == "decision" else "lookup")].append(r)
        tab = {}
        for k, v in sorted(g.items()):
            first = Counter(r["completion_first5"][0] if r["completion_first5"] else "" for r in v)
            tab[f"{k[0]}|{k[1]}"] = {"n": len(v), "share_first_token_lt": round(first.get("<", 0) / len(v), 3),
                                     "top_first_tokens": first.most_common(5),
                                     "mean_target_tokens": round(st.mean(r["completion_tokens"] for r in v), 1),
                                     "decision_token_share_of_row_loss_mass": round(st.mean(1 / r["completion_tokens"] for r in v), 4)}
        for k2 in list(tab):
            # what a coin-flip decision token (ln 2 nats) would add to the mean per-token completion loss of these rows
            tab[k2]["coin_flip_first_token_adds_nats_to_mean_loss"] = round(0.6931 * tab[k2]["decision_token_share_of_row_loss_mass"], 4)
        all_lt = sum(1 for r in rows if r["completion_first5"] and r["completion_first5"][0] == "<")
        tab["all_rows_first_token_lt_share"] = round(all_lt / len(rows), 3)
        out[f.replace(".jsonl", "")] = tab
    # P2 hint size (chars) and an estimated token count with the median chars-per-token of sft_train rows
    tsd = load_token_stats()
    cpts = []
    for r in load_rows("sft_train"):
        cpts.append(sum(len(m["content"]) for m in r["prompt"]) / tsd[("sft_train", r["_i"])]["prompt_tokens"])
    h = hint_text()
    out["p2_hint"] = {"chars_incl_separator": len(h) + 2, "median_chars_per_token_sft_train": round(st.median(cpts), 3),
                      "est_tokens": round((len(h) + 2) / st.median(cpts), 1),
                      "note": "prose tokenises at more chars/token than the JSON-heavy prompt median, so this over-estimates"}
    jdump(out, os.path.join(RES, "s7_decision_token.json"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
