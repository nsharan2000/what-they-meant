"""Weight-space size of LoRA adapters (CPU only, no model load).

For every LoRA module computes ||scale * B @ A||_F exactly via trace((B^T B)(A A^T)) (r x r matrices), with
scale = alpha / r (or alpha / sqrt(r) for rsLoRA), and reports the total and per-module statistics per adapter, plus
ratios against the first adapter given. Use it to tell whether an RL stage moved a fresh LoRA at all.

    CUDA_VISIBLE_DEVICES= python3 experiments/training_eval/adapter_effect.py \
        --adapter m1=runs/m1_sft/final --adapter m2=runs/m2_grpo/checkpoint-25 --out results/adapter_effect.json
"""
import argparse
import json
import os
import time

import torch
from safetensors.torch import load_file


def effect(path):
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    r, alpha, rs = cfg["r"], cfg["lora_alpha"], bool(cfg.get("use_rslora", False))
    scale = alpha / (r ** 0.5) if rs else alpha / r
    sd = load_file(os.path.join(path, "adapter_model.safetensors"))
    total2, norms, bmax = 0.0, [], 0.0
    for k, A in sd.items():
        if "lora_A" not in k:
            continue
        B = sd[k.replace("lora_A", "lora_B")].float()
        A = A.float()
        f2 = torch.trace((B.T @ B) @ (A @ A.T)).item() * scale * scale
        total2 += f2
        norms.append(f2 ** 0.5)
        bmax = max(bmax, B.abs().max().item())
    return {"path": path, "r": r, "alpha": alpha, "rslora": rs, "modules": len(norms),
            "total_dW_fro": total2 ** 0.5, "mean_module_dW_fro": sum(norms) / len(norms),
            "max_module_dW_fro": max(norms), "max_abs_B": bmax}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", action="append", required=True, help="name=path (the first one is the reference)")
    ap.add_argument("--out", required=True)
    ns = ap.parse_args()
    res = {}
    for spec in ns.adapter:
        name, path = spec.split("=", 1)
        res[name] = effect(path)
    ref = next(iter(res))
    for name, e in res.items():
        e["ratio_total_vs_" + ref] = e["total_dW_fro"] / res[ref]["total_dW_fro"]
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "args": vars(ns), "adapters": res}
    os.makedirs(os.path.dirname(os.path.abspath(ns.out)), exist_ok=True)
    json.dump(out, open(ns.out, "w"), indent=1)
    for name, e in res.items():
        print(f"{name:10s} total {e['total_dW_fro']:.5f}  mean/module {e['mean_module_dW_fro']:.6f}  "
              f"max|B| {e['max_abs_B']:.6f}  ratio_vs_{ref} {e['ratio_total_vs_' + ref]:.4f}")
    print("ADAPTER_EFFECT_OK ->", ns.out)


if __name__ == "__main__":
    main()
