"""J4 helpers for training/train_sft.py --decision-weight / --keep-epoch-ends (2026-11-15; experiments/dataset_audit/REPORT.md P1).

The twin decision is the first target token (a word = ask/tell, "<" = a tool call) and, for a tool call, the first token of
the tool name (request_approval vs a direct commit such as pay_bill); together 1.95 % of the trained tokens per epoch. With
weight W those positions get W x the cross-entropy of every other completion token; the loss is sum(w * CE) / (completion
tokens of the optimizer step) — HF/TRL's gradient-accumulation normalisation — so W shifts the relative weight only.
Positions are looked up from TRL's prepared dataset (trained completion ids -> positions): exactly the trained ids, no
re-tokenisation. Logits are computed for the completion positions only (logits_to_keep): TRL 1.13's chunked-CE patch falls
back to the model's original forward when no labels are passed (sft_trainer.py _chunked_ce_forward), which honours Cohere's
logit_scale and logits_to_keep. Validation loss stays the plain mean NLL. Imported by train_sft.py after torch.
"""
from __future__ import annotations

import collections
import json
import os
import re

import torch
from transformers import TrainerCallback
from trl import SFTTrainer

_TOOL_NAME = re.compile(r'"name"\s*:\s*"')


def decision_positions(tok, comp_ids: list[int], text: str) -> list[int]:
    """Relative indices (in the completion's trained ids) of the decision tokens: [0] for a chat reply, [0, k] for a tool
    call where k is the first token of the tool name. Raises ValueError if the ids do not decode back to the target."""
    pos = [0]
    if not text.lstrip().startswith("<tool_call>"):
        return pos
    m = _TOOL_NAME.search(text)
    if not m:
        raise ValueError(f"tool call without a name field: {text[:120]!r}")
    acc = ""
    for k, tid in enumerate(comp_ids):
        acc += tok.decode([tid])
        if len(acc) > m.end():
            if not acc.startswith(text[: m.end() + 1]):
                raise ValueError(f"trained ids decode to {acc[:80]!r}, target starts {text[:80]!r}")
            if k not in pos:
                pos.append(k)
            return pos
    raise ValueError(f"tool name not reached while decoding the trained ids of {text[:120]!r}")


def build_decision_table(train_dataset, tok, rows: list[dict], weight: float) -> tuple[dict, dict]:
    """tuple(trained completion ids) -> weighted positions for every decision row, from TRL's prepared dataset (input_ids +
    completion_mask, same order as `rows`). Raises ValueError when the dataset does not line up with the rows."""
    cols = train_dataset.column_names
    if len(train_dataset) != len(rows) or not ({"labels", "completion_mask"} & set(cols)):
        raise ValueError(f"prepared dataset has {len(train_dataset)} rows / columns {cols}; expected {len(rows)} rows with "
                         f"labels or completion_mask (packing must stay off)")
    table, roles, stats = {}, {}, collections.Counter()
    for i, (r, ex) in enumerate(zip(rows, train_dataset)):
        if "labels" in cols:   # TRL 1.13 "Building labels": labels = input_ids with -100 outside the completion
            comp = tuple(t for t in ex["labels"] if t != -100)
        else:
            comp = tuple(t for t, mk in zip(ex["input_ids"], ex["completion_mask"]) if mk)
        text = r["completion"][0]["content"]
        role = "decision" if r.get("turn") == "decision" else "other"
        if roles.setdefault(comp, role) != role:
            raise ValueError(f"row {i}: the same completion is a decision target in one row and not in another")
        stats["completion_tokens"] += len(comp)
        if role == "decision":
            pos = decision_positions(tok, list(comp), text)
            table[comp] = pos
            stats["decision_rows"] += 1
            stats["weighted_tokens"] += len(pos)
            stats["tool_call_rows"] += len(pos) > 1
    stats = dict(stats)
    denom = stats["completion_tokens"] + (weight - 1) * stats.get("weighted_tokens", 0)
    stats["weight_share"] = round(weight * stats.get("weighted_tokens", 0) / denom, 4) if denom else 0.0
    return table, stats


def weighted_nll(model, inputs: dict, table: dict, weight: float, num_items_in_batch=None, seen=None):
    """The J4 loss for one micro-batch -> (loss, model output). Plain mean NLL when the model is in eval mode."""
    labels = inputs["labels"]
    shift = labels[:, 1:]
    valid = shift != -100
    w = valid.to(torch.float32)
    if model.training:
        for b in range(shift.size(0)):
            idx = valid[b].nonzero(as_tuple=True)[0]
            rel = table.get(tuple(shift[b, idx].tolist()))
            if seen is not None:
                seen["rows"] += 1
            if rel:
                w[b, idx[rel]] = weight
                if seen is not None:
                    seen["weighted_rows"] += 1
    first = int(valid.any(dim=0).nonzero(as_tuple=True)[0].min())
    keep = shift.size(1) - first                         # shifted positions first..T-2 <- logits at positions first..T-2
    fwd = {k: inputs[k] for k in ("input_ids", "attention_mask", "position_ids") if k in inputs}
    out = model(**fwd, use_cache=False, logits_to_keep=keep + 1)   # no labels: original forward (logit_scale applied)
    logits = out.logits[:, :-1, :].float()
    lbl, ww = shift[:, first:], w[:, first:]
    mask = lbl != -100
    nll = -torch.log_softmax(logits, dim=-1).gather(-1, lbl.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    loss_sum = (nll * ww * mask).sum()
    denom = num_items_in_batch if num_items_in_batch is not None else mask.sum()
    if isinstance(denom, torch.Tensor):
        denom = denom.to(loss_sum.device)
    return loss_sum / denom, out


class DecisionWeightedSFTTrainer(SFTTrainer):
    """SFTTrainer with the J4 loss (class attributes set by train_sft.py before training)."""

    dw_weight = 1.0
    dw_table: dict = {}
    dw_seen = collections.Counter()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not bool((inputs["labels"][:, 1:] != -100).any()):
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        loss, out = weighted_nll(model, inputs, self.dw_table, self.dw_weight, num_items_in_batch, self.dw_seen)
        return (loss, out) if return_outputs else loss


class EpochEndSaver(TrainerCallback):
    """--keep-epoch-ends: the adapter at the end of every epoch -> <out>/epoch-<k> (adapter files + epoch_info.json)."""

    def __init__(self, out: str):
        self.out = out

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        k = int(round(state.epoch or 0))
        d = os.path.join(self.out, f"epoch-{k}")
        if model is not None and k > 0 and not os.path.exists(os.path.join(d, "adapter_model.safetensors")):
            model.save_pretrained(d)
            with open(os.path.join(d, "epoch_info.json"), "w") as fh:
                json.dump({"epoch": state.epoch, "global_step": state.global_step}, fh)
            print(f"EPOCH_END_SAVED {d} (step {state.global_step})", flush=True)
