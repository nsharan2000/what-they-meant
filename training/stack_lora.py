"""Stack two LoRA adapters into ONE adapter on the original base, so every arm is served the same way (run_eval.sh: one vLLM
server, base + LoRA adapters side by side).

M2 (GRPO) and stacked M3 (DPO) are trained as LoRA on the MERGED M1 model, i.e. W1 = W0 + s1·B1·A1 (bf16-rounded). Serving their
adapter on the base W0 would silently drop M1. LoRA deltas add, so
    W0 + s1·B1·A1 + s2·B2·A2 = W0 + [s1·B1 | s2·B2] · [A1 ; A2]
is a rank r1 + r2 adapter on W0 with scale 1 (lora_alpha = r). That identity holds only for plain LoRA on linear weights with a
single global scale, so adapters with rsLoRA, DoRA or another LoRA variant, rank/alpha patterns, modules_to_save, trainable
tokens or trained biases are refused.

Verification (with --first-merged; the same metrics as merge_lora.py): teacher-forced log-probs of a short known reply under
(merged M1 + adapter 2), the model the second adapter was trained on, vs (base + stacked adapter): mean |Δ logprob| ≤ --tol, and
two-sided: below 0.1 × the second adapter's own effect (merged M1 with the adapter disabled), which must exceed --min-effect
(0 disables this part, for smoke adapters). The unavoidable floor is the bf16 rounding of the M1 merge, recorded as
first_merge_residual. Without --first-merged the stack is published with "verified": false; a --first-merged path without
config.json is a STACK_FAIL. Output goes to <out>.tmp and is renamed to <out> only when it is published.

python3 training/stack_lora.py --base <base> --first /workspace/runs/m1_sft/final --first-merged /workspace/runs/m1_sft/merged \
    --second /workspace/runs/m2_grpo/final --out /workspace/runs/m2_grpo/stacked
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import guards  # noqa: E402
import merge_lora as ml  # noqa: E402

# LoraConfig fields that must be falsy (peft 0.20 tuners/lora/config.py; the variants are LoraLayer.lora_variants)
NOT_PLAIN = ("use_rslora", "use_dora", "use_qalora", "use_bdlora", "alora_invocation_tokens", "arrow_config", "velora_config",
             "monteclora_config", "lora_bias", "modules_to_save", "rank_pattern", "alpha_pattern", "trainable_token_indices",
             "target_parameters", "layer_replication", "fan_in_fan_out")


def fail(msg):
    raise SystemExit(f"STACK_FAIL: {msg}")


def check_config(cfg, d):
    """Return the adapter's scale lora_alpha / r after refusing anything but plain LoRA."""
    if cfg.get("peft_type", "LORA") != "LORA":
        fail(f"{d} is a {cfg.get('peft_type')} adapter, not LoRA")
    if cfg.get("use_rslora") or cfg.get("use_dora"):
        fail(f"{d} uses rsLoRA/DoRA — only plain LoRA can be stacked this way")
    bad = [k for k in NOT_PLAIN if cfg.get(k)]
    if cfg.get("init_lora_weights") == "mica":
        bad.append("init_lora_weights=mica")
    if bad:
        fail(f"{d} sets {', '.join(bad)} — only plain LoRA (one global scale, no extra trained modules) can be stacked this way")
    if cfg.get("bias", "none") != "none":
        fail(f"{d} trains biases (bias={cfg.get('bias')!r}); they are not part of B·A and cannot be stacked")
    return cfg["lora_alpha"] / cfg["r"]


def modules(w, r, d):
    """{module prefix: (A, B)} after checking that the file holds exactly one A [r, in] and one B [out, r] per module."""
    out = {}
    for k, t in w.items():
        for part in ("lora_A", "lora_B"):
            if k.endswith(f".{part}.weight"):
                out.setdefault(k[:-len(f".{part}.weight")], {})[part] = t
                break
        else:
            fail(f"{d}: unexpected adapter tensor {k}")
    for m, ab in out.items():
        if set(ab) != {"lora_A", "lora_B"}:
            fail(f"{d}: {m} lacks {({'lora_A', 'lora_B'} - set(ab)).pop()}")
        a, b = ab["lora_A"].shape, ab["lora_B"].shape
        if len(a) != 2 or len(b) != 2 or a[0] != r or b[1] != r:
            fail(f"{d}: {m} has A {tuple(a)} / B {tuple(b)}, expected rank {r} (r in adapter_config.json)")
    return {m: (ab["lora_A"], ab["lora_B"]) for m, ab in out.items()}


def stack(m1, m2, s1, s2):
    """Stacked state dict: A = [A1 ; A2] (rank r1 + r2 rows), B = [s1·B1 | s2·B2], in the wider of the two dtypes."""
    import torch
    if set(m1) != set(m2):
        fail(f"adapters target different modules ({len(set(m1) ^ set(m2))} differ)")
    out = {}
    for m, (a1, b1) in m1.items():
        a2, b2 = m2[m]
        if a1.shape[1] != a2.shape[1] or b1.shape[0] != b2.shape[0]:
            fail(f"{m}: in/out features differ between the adapters")
        dt = torch.promote_types(a1.dtype, a2.dtype)
        out[f"{m}.lora_A.weight"] = torch.cat([a1.float(), a2.float()], dim=0).to(dt)
        out[f"{m}.lora_B.weight"] = torch.cat([s1 * b1.float(), s2 * b2.float()], dim=1).to(dt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--first", required=True, help="adapter trained on --base (M1)")
    ap.add_argument("--second", required=True, help="adapter trained on the merged first model (M2 / stacked M3)")
    ap.add_argument("--first-merged", default=None, help="merged first model, for the verification (omitted: verified false)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--min-effect", type=float, default=1e-3, help="see merge_lora.py; 0 disables the two-sided check (smoke)")
    ns = ap.parse_args()
    if ns.first_merged is not None and not os.path.exists(os.path.join(ns.first_merged, "config.json")):
        fail(f"--first-merged {ns.first_merged} has no config.json (missing or failed merge)")

    import torch
    guards.apply_mem_cap()
    from safetensors.torch import load_file, save_file

    t0 = time.time()
    c1, c2 = (json.load(open(os.path.join(d, "adapter_config.json"))) for d in (ns.first, ns.second))
    s1, s2 = check_config(c1, ns.first), check_config(c2, ns.second)
    w1, w2 = (load_file(os.path.join(d, "adapter_model.safetensors")) for d in (ns.first, ns.second))
    out = stack(modules(w1, c1["r"], ns.first), modules(w2, c2["r"], ns.second), s1, s2)
    del w1, w2
    r = c1["r"] + c2["r"]
    cfg = dict(c1)
    cfg.update({"r": r, "lora_alpha": r, "lora_dropout": 0.0, "base_model_name_or_path": ns.base, "rank_pattern": {},
                "alpha_pattern": {}})
    tmp = ml.fresh_tmp(ns.out)
    save_file(out, os.path.join(tmp, "adapter_model.safetensors"), metadata={"format": "pt"})
    json.dump(cfg, open(os.path.join(tmp, "adapter_config.json"), "w"), indent=1)
    info = {"ok": True, "verified": False, "base": ns.base, "first": ns.first, "second": ns.second,
            "first_merged": ns.first_merged, "rank": r, "scales": [s1, s2], "tensors": len(out),
            "sha256": {k: ml.sha256(os.path.join(d, "adapter_model.safetensors"))
                       for k, d in (("first", ns.first), ("second", ns.second), ("stacked", tmp))}}
    del out
    reasons = []

    if ns.first_merged:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ids, start = ml.probe_ids(AutoTokenizer.from_pretrained(ns.base))
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(ns.first_merged, dtype=torch.bfloat16,
                                                                               device_map=dev), ns.second).eval()
        ref = ml.forward(model, ids, start, dev, keep_logits=True)
        with model.disable_adapter():
            first_merged = ml.forward(model, ids, start, dev)
        del model
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(ns.base, dtype=torch.bfloat16, device_map=dev),
                                          tmp, adapter_name="stacked")
        model.load_adapter(ns.first, adapter_name="first")
        model.eval()
        got = ml.forward(model, ids, start, dev, keep_logits=True)
        model.set_adapter("first", inference_mode=True)
        first_only = ml.forward(model, ids, start, dev)
        delta, eff = ml.compare(ref, got, ids, start), ml.compare(ref, first_merged, ids, start)
        reasons = ml.verdict(delta, eff, ns.tol, ns.min_effect)
        info.update({"ok": not reasons, "verified": not reasons, "probe": {"tokens": len(ids), "reply_positions": len(ids) - start},
                     "stack_delta": ml.rounded(delta), "second_effect": ml.rounded(eff),
                     "first_merge_residual": ml.rounded(ml.compare(first_merged, first_only, ids, start)),
                     "served_second_effect": ml.rounded(ml.compare(got, first_only, ids, start)),
                     "max_abs_logit_delta_all_positions": round(ml.max_logit_delta(ref, got), 4),
                     "tol": ns.tol, "min_effect": ns.min_effect, "effect_ratio": ml.EFFECT_RATIO, "reasons": reasons})
    info.update({"minutes": round((time.time() - t0) / 60, 2), "peak_mem": guards.peak_mem()})
    json.dump(info, open(os.path.join(tmp, "stack_info.json"), "w"), indent=1)
    print(json.dumps(info), flush=True)
    if reasons:
        res = info["first_merge_residual"]["mean_abs_dlogprob"]
        fail(f"{'; '.join(reasons)} (M1 merge residual {res:.4g}: if the delta is close to it, the stack is faithful but the "
             f"second adapter's effect is below the M1 bf16 merge rounding; details in {tmp}/stack_info.json)")
    ml.publish(tmp, ns.out)
    print(f"STACKED -> {ns.out}" + ("" if info["verified"] else " (NOT verified: no --first-merged)"), flush=True)


if __name__ == "__main__":
    main()
