"""Exact weight-space check of a stacked LoRA (training/stack_lora.py), independent of bf16 merge rounding (CPU only).

WHY (2026-11-15): stack_lora.py verifies a stacked M1+M2 adapter by comparing its log-probs on a probe against "merged M1 + M2
adapter". That reference carries the bf16 rounding of merging M1 into the weights (mean |Δ logprob| 0.032 on the probe), which is
larger than the GRPO v2 adapter's whole effect on that probe (0.018), so the check can neither pass nor fail meaningfully
(STACK_FAIL "mean |Δ logprob| 0.0223 ≥ 0.1 × adapter effect 0.0178"). This script checks what stacking is supposed to guarantee,
directly: for every LoRA module, ΔW_stack = s_stack · B_stack A_stack must equal s1 · B1 A1 + s2 · B2 A2 (fp32). The relative
Frobenius error over all modules must be below --tol (default 1e-2: only the bf16 storage of the stacked factors may differ);
the second adapter's own share must be reproduced too (it is the part that matters).

  CUDA_VISIBLE_DEVICES= python3 training/verify_stack_exact.py --first runs/m1_sft/final --second runs/m2_grpo/final \
      --stacked runs/m2_grpo/stacked.tmp --out runs/m2_grpo/stack_exact_check.json
Prints STACK_EXACT_OK or STACK_EXACT_FAIL with the numbers.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors.torch import load_file


def scale_of(cfg: dict) -> float:
    r, a = cfg["r"], cfg["lora_alpha"]
    return a / (r ** 0.5) if cfg.get("use_rslora") else a / r


def load_adapter(path: str):
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    sd = load_file(os.path.join(path, "adapter_model.safetensors"))
    mods = {}
    for k, v in sd.items():
        if "lora_A" in k:
            name = k.split(".lora_A")[0]
            mods.setdefault(name, {})["A"] = v.float()
        elif "lora_B" in k:
            name = k.split(".lora_B")[0]
            mods.setdefault(name, {})["B"] = v.float()
    return cfg, scale_of(cfg), mods


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", required=True)
    ap.add_argument("--second", required=True)
    ap.add_argument("--stacked", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-2)
    ns = ap.parse_args()
    c1, s1, m1 = load_adapter(ns.first)
    c2, s2, m2 = load_adapter(ns.second)
    cs, ss, mst = load_adapter(ns.stacked)
    names = sorted(set(m1) | set(m2))
    missing = [n for n in names if n not in mst]
    if missing:
        print(f"STACK_EXACT_FAIL {len(missing)} modules missing from the stacked adapter, e.g. {missing[:3]}")
        raise SystemExit(1)
    err2 = ref2 = sec2 = sec_err2 = 0.0
    worst = (0.0, None)
    for n in names:
        ref = torch.zeros(1)
        d1 = s1 * (m1[n]["B"] @ m1[n]["A"]) if n in m1 else 0
        d2 = s2 * (m2[n]["B"] @ m2[n]["A"]) if n in m2 else 0
        ref = d1 + d2
        st = ss * (mst[n]["B"] @ mst[n]["A"])
        e = (st - ref).pow(2).sum().item()
        r = ref.pow(2).sum().item()
        err2 += e
        ref2 += r
        if n in m2:   # does the stack reproduce the SECOND adapter's part? (stack minus first) vs second
            sec = d2.pow(2).sum().item()
            sec2 += sec
            sec_err2 += ((st - d1) - d2).pow(2).sum().item() if n in m1 else e
        rel = (e / r) ** 0.5 if r else 0.0
        if rel > worst[0]:
            worst = (rel, n)
    rel_total = (err2 / ref2) ** 0.5
    rel_second = (sec_err2 / sec2) ** 0.5 if sec2 else None
    ok = rel_total < ns.tol and (rel_second is None or rel_second < 10 * ns.tol)
    out = {"first": ns.first, "second": ns.second, "stacked": ns.stacked, "scales": {"first": s1, "second": s2, "stacked": ss},
           "ranks": {"first": c1["r"], "second": c2["r"], "stacked": cs["r"]}, "modules": len(names),
           "rel_frobenius_error_total": rel_total, "rel_frobenius_error_second_part": rel_second,
           "worst_module": {"name": worst[1], "rel_error": worst[0]}, "tol": ns.tol, "ok": ok,
           "norms": {"ref_total": ref2 ** 0.5, "second": sec2 ** 0.5}}
    json.dump(out, open(ns.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("STACK_EXACT_OK" if ok else "STACK_EXACT_FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
