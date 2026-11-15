"""Verify that an ungated HF mirror of Tiny Aya Global is bit-identical to Cohere's OFFICIAL ungated BF16 GGUF
(CohereLabs/tiny-aya-global-GGUF, tiny-aya-global-bf16.gguf). Insurance in case the official safetensors gate (Q1) is not accepted in time.

Compares every tensor (llama.cpp names ↔ HF Cohere2 names) bitwise on the raw bf16 bits; reports metadata (arch, context length,
chat template) from the GGUF so the official chat template can be used. Run inside lx-exp:
  python3 experiments/infra/verify_mirror.py --gguf <path/to/tiny-aya-global-bf16.gguf> --mirror <dir of 1-800-LLMs/tiny-aya-global>
"""
import argparse
import glob
import json
import os

import numpy as np
from gguf import GGUFReader
from safetensors.numpy import load_file

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--mirror", required=True)
ap.add_argument("--out", default="/workspace/experiments/infra/verify_mirror.json")
ns = ap.parse_args()

r = GGUFReader(ns.gguf)
meta = {}
for k, f in r.fields.items():
    if k.startswith(("general.", "cohere2.", "command-r.", "tokenizer.chat_template")) or "context_length" in k:
        try:
            v = f.parts[f.data[0]]
            meta[k] = bytes(v).decode("utf-8", "ignore") if v.dtype == np.uint8 else v.tolist()
        except Exception:  # noqa: BLE001
            pass

# HF safetensors may be stored as bf16; safetensors.numpy cannot load bf16 directly -> read raw bits via torch if needed
try:
    hf = {}
    for f in sorted(glob.glob(os.path.join(ns.mirror, "*.safetensors"))):
        hf.update(load_file(f))
    as_bits = {k: v.view(np.uint16) if v.dtype == np.dtype("V2") else v for k, v in hf.items()}
except Exception:  # noqa: BLE001
    import torch
    from safetensors.torch import load_file as tload
    hf = {}
    for f in sorted(glob.glob(os.path.join(ns.mirror, "*.safetensors"))):
        hf.update(tload(f))
    as_bits = {k: v.view(torch.int16).numpy().view(np.uint16) for k, v in hf.items()}

MAP = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
       "output_norm.bias": "model.norm.bias", "output.weight": "lm_head.weight"}
BLK = {"attn_norm.weight": "input_layernorm.weight", "attn_norm.bias": "input_layernorm.bias",
       "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
       "attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
       "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight", "ffn_down.weight": "mlp.down_proj.weight"}
res = {"identical": [], "different": [], "missing_in_mirror": [], "shape_mismatch": []}
for t in r.tensors:
    name = t.name
    if name.startswith("blk."):
        _, i, rest = name.split(".", 2)
        hf_name = f"model.layers.{i}.{BLK.get(rest, rest)}"
    else:
        hf_name = MAP.get(name, name)
    if hf_name not in as_bits:
        res["missing_in_mirror"].append(name)
        continue
    g = np.asarray(t.data)
    h = as_bits[hf_name]
    if g.dtype == np.float32:                       # llama.cpp keeps norms in f32; bf16 -> f32 is exact, compare values
        h_f32 = (h.reshape(-1).astype(np.uint32) << 16).view(np.float32)
        if g.size != h_f32.size:
            res["shape_mismatch"].append([name, list(g.shape), list(h.shape)])
            continue
        same = np.array_equal(g.reshape(-1), h_f32)
    else:
        g_bits = g.view(np.uint16) if g.dtype != np.uint16 else g
        if g_bits.size != h.size:
            res["shape_mismatch"].append([name, list(g.shape), list(h.shape)])
            continue
        same = np.array_equal(g_bits.reshape(-1), h.reshape(-1))
    (res["identical"] if same else res["different"]).append(name)
summary = {"gguf_tensors": len(r.tensors), "mirror_tensors": len(as_bits), "identical": len(res["identical"]),
           "different": res["different"][:20], "n_different": len(res["different"]), "missing_in_mirror": res["missing_in_mirror"][:20],
           "shape_mismatch": res["shape_mismatch"][:10], "gguf_meta": {k: (v[:300] if isinstance(v, str) else v) for k, v in meta.items()}}
json.dump(summary, open(ns.out, "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "gguf_meta"}, indent=1))
print("arch/context:", {k: v for k, v in meta.items() if "architecture" in k or "context_length" in k})
print("VERIFY_DONE")
