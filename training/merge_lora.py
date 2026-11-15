"""Merge a LoRA adapter into its base model and save a full bf16 checkpoint (M1 → M2 warm start, M1 → M3 stacking, serving).

The merge is published only when it is verified, because a merged dir is consumed as a warm start by GRPO and DPO:
  * Fidelity: teacher-forced log-probs of a short known assistant reply (plus the generation position) must agree between the
    adapter model (unmerged) and the merged model: mean |Δ logprob| of the reply tokens ≤ --tol. Only those positions count:
    the ~390-token forced preamble of the Tiny Aya template has near-tied top-2 tokens whose argmax flips under one extra
    bf16 rounding, so an all-positions max/argmax check fails correct merges. The all-positions max |Δ logit| is recorded
    for information only.
  * Two-sided: the same metric between the adapter model and the base (adapter disabled) is the adapter's effect `eff`; the
    merge must stay below 0.1 · eff and eff must exceed --min-effect, so a merge that silently dropped the adapter fails.
    --min-effect 0 disables this part (smoke adapters are too weak to measure).
Everything is written to <out>.tmp and renamed to <out> only on a pass; a failure leaves <out>.tmp/merge_info.json with
"ok": false (and no weights) and exits non-zero. The merged generation_config stops on the tokenizer eos too (Tiny Aya: [6, 3];
generation_config.json alone says 3) so a plain HF generate() stops at <|END_OF_TURN_TOKEN|>.

python3 training/merge_lora.py --base <base path or id> --adapter /workspace/runs/m1_sft/final --out /workspace/runs/m1_sft/merged
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import guards  # noqa: E402

PROBE = [{"role": "system", "content": "You are the member-services assistant of a credit union. Use the tools to act on "
                                       "the member's requests."},
         {"role": "user", "content": "Pay my electricity bill of $82.40 from checking."}]
REPLY = "I'll pay the $82.40 electricity bill from your checking account now. Shall I go ahead?"
EFFECT_RATIO = 0.1


def probe_ids(tok, messages=PROBE, reply=REPLY):
    """(ids, start): the template-rendered conversation ending in a known assistant reply, and the length of its generation
    prompt. Positions start-1 … len(ids)-2 predict the reply tokens, the first of them being the generation position."""
    def render(msgs, gen):
        return list(tok.apply_chat_template(msgs, add_generation_prompt=gen, tokenize=True, return_dict=True)["input_ids"])
    prompt = render(messages, True)
    ids = render(messages + [{"role": "assistant", "content": reply}], False)
    if ids[:len(prompt)] != prompt or len(ids) <= len(prompt):
        raise SystemExit("PROBE_FAIL: the chat template does not render the reply as a continuation of its generation prompt")
    if tok.eos_token_id not in ids[len(prompt):]:
        raise SystemExit(f"PROBE_FAIL: the rendered reply never reaches the tokenizer eos {tok.eos_token}")
    return ids, len(prompt)


def forward(model, ids, start, dev, keep_logits=False):
    """{"lp": fp32 log-softmax at the reply positions [K, V], "logits": all positions [L, V] (model dtype) or None}."""
    import torch
    x = torch.tensor([ids], device=dev)
    with torch.no_grad():
        logits = model(input_ids=x).logits[0]
    return {"lp": logits[start - 1:len(ids) - 1].float().log_softmax(-1), "logits": logits if keep_logits else None}


def compare(a, b, ids, start):
    """Teacher-forced comparison of two forward() results on the reply positions."""
    import torch
    tgt = torch.tensor(ids[start:], device=a["lp"].device)[:, None]
    ta, tb = a["lp"].gather(-1, tgt)[:, 0], b["lp"].gather(-1, tgt)[:, 0]
    d = (ta - tb).abs()
    kl = (a["lp"].exp() * (a["lp"] - b["lp"])).sum(-1)
    return {"mean_abs_dlogprob": d.mean().item(), "max_abs_dlogprob": d.max().item(),
            "top1_agree": (a["lp"].argmax(-1) == b["lp"].argmax(-1)).float().mean().item(), "kl_mean": kl.mean().item()}


def max_logit_delta(a, b):
    return (a["logits"].float() - b["logits"].float()).abs().max().item()


def verdict(delta, eff, tol, min_effect):
    """Reasons to reject ([] = pass). Every test is written as `not (value within bound)` so a NaN fails."""
    d, e = delta["mean_abs_dlogprob"], eff["mean_abs_dlogprob"]
    reasons = []
    if not d <= tol:
        reasons.append(f"mean |Δ logprob| {d:.4g} > tol {tol}")
    if min_effect > 0:
        if not e > min_effect:
            reasons.append(f"adapter effect {e:.4g} ≤ --min-effect {min_effect}: adapter not applied, or too weak to verify")
        elif not d < EFFECT_RATIO * e:
            reasons.append(f"mean |Δ logprob| {d:.4g} ≥ {EFFECT_RATIO} × adapter effect {e:.4g}")
    return reasons


def rounded(m, nd=6):
    return {k: (round(v, nd) if isinstance(v, float) and math.isfinite(v) else v) for k, v in m.items()}


def fresh_tmp(out):
    tmp = out.rstrip("/") + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    return tmp


def publish(tmp, out):
    """Move the verified <out>.tmp to <out>, replacing an older <out> (rename needs an absent or empty target)."""
    if os.path.lexists(out):
        old = out.rstrip("/") + ".old"
        shutil.rmtree(old, ignore_errors=True)
        os.replace(out, old)
        os.replace(tmp, out)
        shutil.rmtree(old)
    else:
        os.replace(tmp, out)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stop_ids(tok, gen_eos):
    """The tokenizer eos first, then the generation_config eos ids (Tiny Aya: [6, 3])."""
    ids = [tok.eos_token_id] + (list(gen_eos) if isinstance(gen_eos, (list, tuple)) else [gen_eos])
    return [i for n, i in enumerate(ids) if i is not None and i not in ids[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=0.05, help="max mean |Δ logprob| of the probe reply tokens, adapter vs merged")
    ap.add_argument("--min-effect", type=float, default=1e-3,
                    help="adapter vs base mean |Δ logprob| must exceed this and 10× the merge delta; 0 disables (smoke)")
    ns = ap.parse_args()

    import torch
    guards.apply_mem_cap()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tmp = fresh_tmp(ns.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ns.adapter if os.path.exists(os.path.join(ns.adapter, "tokenizer_config.json"))
                                        else ns.base)
    ids, start = probe_ids(tok)
    base = AutoModelForCausalLM.from_pretrained(ns.base, dtype=torch.bfloat16, device_map=dev)
    model = PeftModel.from_pretrained(base, ns.adapter).eval()
    adapted = forward(model, ids, start, dev, keep_logits=True)
    with model.disable_adapter():   # before the merge: on a merged layer, disabling unmerges it
        plain = forward(model, ids, start, dev)
    merged = model.merge_and_unload().eval()
    after = forward(merged, ids, start, dev, keep_logits=True)
    delta, eff = compare(adapted, after, ids, start), compare(adapted, plain, ids, start)
    reasons = verdict(delta, eff, ns.tol, ns.min_effect)
    info = {"ok": not reasons, "base": ns.base, "adapter": ns.adapter,
            "adapter_sha256": sha256(os.path.join(ns.adapter, "adapter_model.safetensors")),
            "probe": {"tokens": len(ids), "reply_positions": len(ids) - start, "reply": REPLY},
            "merge_delta": rounded(delta), "adapter_effect": rounded(eff),
            "max_abs_logit_delta_all_positions": round(max_logit_delta(adapted, after), 4),
            "tol": ns.tol, "min_effect": ns.min_effect, "effect_ratio": EFFECT_RATIO, "reasons": reasons}
    del adapted, after, plain
    gc.collect()
    if not reasons:
        merged.generation_config.eos_token_id = stop_ids(tok, merged.generation_config.eos_token_id)
        info["generation_eos_token_id"] = merged.generation_config.eos_token_id
        merged.save_pretrained(tmp)
        tok.save_pretrained(tmp)
    info.update({"minutes": round((time.time() - t0) / 60, 2), "peak_mem": guards.peak_mem()})
    json.dump(info, open(os.path.join(tmp, "merge_info.json"), "w"), indent=1)
    print(json.dumps(info), flush=True)
    if reasons:
        stale = f"; the older {ns.out} was left untouched" if os.path.exists(ns.out) else ""
        raise SystemExit(f"MERGE_FAIL: {'; '.join(reasons)} (details in {tmp}/merge_info.json{stale})")
    publish(tmp, ns.out)
    print(f"MERGED -> {ns.out}", flush=True)


if __name__ == "__main__":
    main()
