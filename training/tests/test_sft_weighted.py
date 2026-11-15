"""CPU test of training/sft_weighting.py (J4) with a tiny random Cohere2 model + the REAL Tiny Aya tokenizer and SFT rows.

Run inside lx-exp (CPU only, < 2 GB, a few minutes):
  CUDA_VISIBLE_DEVICES= PYTHONPATH=<dir holding sft_weighting.py> python3 training/tests/test_sft_weighted.py \
      --base <Tiny Aya snapshot> --data /workspace/training/data_v121_curated --work /workspace/scratch/chain2/sftw
Checks: (1) weighted positions = the first completion token of decision rows ("<" for tool calls, a word otherwise) and the
first token of the tool name; look-up rows get none; (2) W=1 equals TRL's own (chunked) loss; (3) W=10 equals a hand-computed
weighted CE from full logits; (4) two micro-batches with num_items_in_batch = their total equal the padded 2-row batch;
(5) EpochEndSaver writes epoch-1; (6) the decision weight share on the full SFT data at W=10. Prints TEST_SFT_WEIGHTED_OK.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import types

import torch

torch.manual_seed(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ns = ap.parse_args()
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    import sft_weighting as sw

    os.makedirs(ns.work, exist_ok=True)
    tiny = os.path.join(ns.work, "tiny_aya")
    tok = AutoTokenizer.from_pretrained(ns.base)
    if not os.path.exists(os.path.join(tiny, "config.json")):
        cfg = AutoConfig.from_pretrained(ns.base)
        for k, v in dict(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=32).items():
            setattr(cfg, k, v)
        if hasattr(cfg, "layer_types") and cfg.layer_types:
            cfg.layer_types = cfg.layer_types[:2]
        AutoModelForCausalLM.from_config(cfg).save_pretrained(tiny)
        tok.save_pretrained(tiny)
    rows_all = [json.loads(l) for l in open(os.path.join(ns.data, "sft_train.jsonl")) if l.strip()]
    pick, seen = [], {"A": 0, "Bcard": 0, "Bact": 0, "lookup": 0}
    for r in rows_all:   # a mix: twin-A decisions, B cards, B direct commits, look-ups
        t = r["completion"][0]["content"]
        kind = ("lookup" if r["turn"] != "decision" else "A" if r["twin"].startswith("A") else
                "Bcard" if '"request_approval"' in t else "Bact")
        if seen[kind] < 5:
            seen[kind] += 1
            pick.append(r)
    keep = ("prompt", "completion")

    def trainer_for(rows, out):
        model = AutoModelForCausalLM.from_pretrained(tiny, dtype=torch.float32)
        cfg = SFTConfig(output_dir=out, completion_only_loss=True, max_length=12288, per_device_train_batch_size=1,
                        report_to="none", bf16=False, use_cpu=True, num_train_epochs=1, logging_steps=1, save_strategy="no")
        peft = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        ds = Dataset.from_list([{k: r[k] for k in keep} for r in rows])
        return SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft)

    tr = trainer_for(pick, os.path.join(ns.work, "out"))
    table, stats = sw.build_decision_table(tr.train_dataset, tok, pick, 10.0)
    print("table stats (subset):", stats)
    # (1) positions
    for r, ex in zip(pick, tr.train_dataset):
        comp = (tuple(t for t in ex["labels"] if t != -100) if "labels" in ex else
                tuple(t for t, m in zip(ex["input_ids"], ex["completion_mask"]) if m))
        text = r["completion"][0]["content"]
        pos = table.get(comp)
        toks = [tok.decode([comp[p]]) for p in (pos or [])]
        if r["turn"] != "decision":
            assert pos is None, ("look-up row weighted", text[:60])
            continue
        assert pos and pos[0] == 0, pos
        if text.startswith("<tool_call>"):
            name = json.loads(text[len("<tool_call>"):text.index("</tool_call>")])["name"]
            assert toks[0] == "<" and len(pos) == 2 and name.startswith(toks[1].strip()), (toks, name)
        else:
            assert text.startswith(toks[0].strip()) or toks[0].strip() == "", (toks, text[:40])
        print(f"  {r['twin'][:1]} {r['scenario_id']:6s} weighted tokens {toks} <- {text[:50]!r}")
    model = tr.model
    coll = tr.data_collator
    # (2) W=1 == TRL's own loss (eval mode: no dropout, our weights are 1 anyway)
    model.eval()
    for i in (0, 7, 12, 17):
        batch = coll([tr.train_dataset[i]])
        with torch.no_grad():
            std = SFTTrainer.compute_loss(tr, model, dict(batch))
            ours, _ = sw.weighted_nll(model, dict(batch), table, 1.0)
        assert abs(std.item() - ours.item()) < 1e-4 * max(1.0, abs(std.item())), (i, std.item(), ours.item())
    print("W=1 equals TRL's chunked loss on 4 rows")
    # (3) W=10 == hand-computed weighted CE from full logits (train mode, LoRA dropout 0 -> deterministic)
    model.train()
    for i in range(len(pick)):
        batch = coll([tr.train_dataset[i]])
        with torch.no_grad():
            ours, _ = sw.weighted_nll(model, dict(batch), table, 10.0)
            full = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask")).logits[:, :-1].float()
        lbl = batch["labels"][:, 1:]
        m = lbl != -100
        ce = torch.nn.functional.cross_entropy(full[m], lbl[m], reduction="none")
        w = torch.ones_like(ce)
        rel = table.get(tuple(lbl[m].tolist()))
        if rel:
            w[rel] = 10.0
        hand = (ce * w).sum() / m.sum()
        assert abs(hand.item() - ours.item()) < 1e-4 * max(1.0, abs(hand.item())), (i, hand.item(), ours.item())
    print(f"W=10 equals the hand-computed weighted CE on {len(pick)} rows")
    # (4) accumulation: two micro-batches with num_items_in_batch = their total == one padded batch of both
    a, b = tr.train_dataset[0], tr.train_dataset[5]
    ba, bb, bab = coll([a]), coll([b]), coll([a, b])
    na, nb = int((ba["labels"][:, 1:] != -100).sum()), int((bb["labels"][:, 1:] != -100).sum())
    with torch.no_grad():
        la, _ = sw.weighted_nll(model, dict(ba), table, 10.0, num_items_in_batch=na + nb)
        lb, _ = sw.weighted_nll(model, dict(bb), table, 10.0, num_items_in_batch=na + nb)
        lab, _ = sw.weighted_nll(model, dict(bab), table, 10.0)
    assert abs((la + lb).item() - lab.item()) < 1e-4 * max(1.0, abs(lab.item())), (la.item(), lb.item(), lab.item())
    print("accumulation over 2 micro-batches equals the joint batch")
    # (5) EpochEndSaver
    ep = os.path.join(ns.work, "epochs")
    shutil.rmtree(ep, ignore_errors=True)
    sw.EpochEndSaver(ep).on_epoch_end(None, types.SimpleNamespace(epoch=1.0, global_step=3), None, model=model)
    assert os.path.exists(os.path.join(ep, "epoch-1", "adapter_model.safetensors")), os.listdir(ep)
    print("EpochEndSaver wrote", sorted(os.listdir(os.path.join(ep, "epoch-1"))))
    # (6) share on the full SFT data
    trf = trainer_for(rows_all, os.path.join(ns.work, "out_full"))
    _, full_stats = sw.build_decision_table(trf.train_dataset, tok, rows_all, 10.0)
    print("full SFT data at W=10:", full_stats)
    assert full_stats["decision_rows"] == sum(r["turn"] == "decision" for r in rows_all)
    assert 0.12 < full_stats["weight_share"] < 0.22, full_stats
    print("TEST_SFT_WEIGHTED_OK")


if __name__ == "__main__":
    main()
