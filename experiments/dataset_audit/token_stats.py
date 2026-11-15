"""Exact Tiny Aya token statistics for every training row (CPU only; no model weights loaded).

Per row: prompt tokens as the trainers see them (chat template + generation prompt), target tokens (completion / chosen /
rejected, add_special_tokens=False), and the first target tokens (the decision-bearing ones). Also records the model's context
settings from config.json. Output: <out>/token_stats.jsonl (one line per row) + <out>/token_stats_summary.json.

    CUDA_VISIBLE_DEVICES= python3 experiments/dataset_audit/token_stats.py --model <snapshot dir> --data <dir> --out <dir>
"""
import argparse
import json
import os
import statistics

from transformers import AutoTokenizer


def text_of(c):
    if isinstance(c, list):
        return c[-1]["content"]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ns = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(ns.model)
    cfg = json.load(open(os.path.join(ns.model, "config.json")))
    ctx = {k: cfg.get(k) for k in ("max_position_embeddings", "sliding_window", "sliding_window_pattern", "rope_theta",
                                   "rope_scaling", "model_type", "vocab_size")}
    os.makedirs(ns.out, exist_ok=True)
    out = open(os.path.join(ns.out, "token_stats.jsonl"), "w")
    summ = {"model_context": ctx, "files": {}}
    for fname in ("sft_train.jsonl", "sft_val.jsonl", "dpo_train.jsonl", "grpo_train.jsonl"):
        path = os.path.join(ns.data, fname)
        p_lens, t_lens = [], []
        for i, line in enumerate(open(path)):
            r = json.loads(line)
            ids = tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=True)
            if hasattr(ids, "keys"):          # BatchEncoding (a UserDict, not a dict) in recent transformers
                ids = ids["input_ids"]
            row = {"file": fname, "i": i, "scenario_id": r.get("scenario_id"), "level": r.get("level"),
                   "prompt_setting": r.get("prompt_setting"), "turn": r.get("turn"), "twin": r.get("twin"),
                   "prompt_tokens": len(ids)}
            p_lens.append(len(ids))
            for key in ("completion", "chosen", "rejected"):
                if key in r:
                    t = tok.encode(text_of(r[key]), add_special_tokens=False)
                    row[key + "_tokens"] = len(t)
                    row[key + "_first5"] = tok.convert_ids_to_tokens(t[:5])
                    t_lens.append(len(t))
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        def st(v):
            if not v:
                return None
            s = sorted(v)
            return {"n": len(v), "mean": round(statistics.mean(v), 1), "p50": s[len(s) // 2], "p95": s[int(0.95 * (len(s) - 1))],
                    "max": s[-1]}
        summ["files"][fname] = {"prompt_tokens": st(p_lens), "target_tokens": st(t_lens)}
    out.close()
    json.dump(summ, open(os.path.join(ns.out, "token_stats_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print("TOKEN_STATS_OK")


if __name__ == "__main__":
    main()
