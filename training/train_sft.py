"""M1 — supervised fine-tuning through AgentTune's distillation API.

AgentTune has no plain SFT entry point; `create_distill_trainer` orchestrates (Project, provenance) and calls a trainer factory.
Its built-in factory projects EventLogs without a system role and trains on every token, so we pass our own factory that trains
TRL SFTTrainer on our prompt-completion rows (completion-only loss, TRL's default chunked NLL; system prompt kept). Teacher
demonstrations recorded in the AgentTune Project are the same decision turns (for provenance / AuditKIT import).

Data: --data is the directory written by training/from_curated.py (sft_train.jsonl, sft_val.jsonl: the original rows that
survived CuratorKIT). CuratorKIT's own export flattens the prompt into one user message and must not be trained on directly.
training/guards.py checks every prepared row before the first step.

Memory (shared GB10 host, unified memory): guards.apply_mem_cap() caps this process's CUDA allocations (WTM_GPU_CAP_GB) so an
overrun raises an OOM here instead of draining host RAM; run the job under training/memguard.py. --batch 1 --grad-accum 16 gives
the same token-normalised loss as 2x8 without padding and with a lower peak.

Resume: a 2-epoch run takes hours and memguard may kill it on purpose, so a checkpoint is saved every --save-steps optimizer steps
(last 3 kept). Unless --fresh, the run resumes from the newest complete checkpoint-* in --out, and refuses to (GUARD_FAIL) when the
training-relevant arguments or the training rows differ from <out>/args.json (paths stored resolved, cwd-independent).
A checkpoint without a readable trainer_state.json (killed mid-save; Trainer writes that file last) is renamed incomplete_<name>.
Rerunning a finished run (last checkpoint at max_steps, final/ and result.json present) prints SFT_ALREADY_DONE and exits 0, so a
chained rerun never replaces the real result.json. Each process is one segment in <out>/progress.json and result.json["segments"]
(minutes, step times, peak memory), so a memguard kill followed by a resume keeps the killed segment's numbers. With --fresh, or
when there is nothing to resume, the previous run's checkpoints, final/, result.json, progress.json, guards.json and
step_times.jsonl are renamed stale_<name>.

J4 flags (2026-11-15, experiments/dataset_audit/REPORT.md P1; absent = everything above, unchanged — the default never touches
the code below and its args.json stays byte-compatible, so finished runs are still recognised):
  --decision-weight W  the twin decision is ~1 token of a ~40-token target (1.30 % of the 38,160 trained tokens per epoch;
                       1.95 % with the tool-name token), so SFT barely supervises it. W != 1 weights the cross-entropy of
                       (a) the first completion token of every decision row (turn == "decision": a word = ask/tell, "<" = a
                       tool call) and (b) for a tool call, the first token of the tool name (where request_approval and a direct
                       commit diverge) by W; every other completion token keeps weight 1; loss = sum(w * CE) / (completion
                       tokens of the optimizer step) — TRL/HF's gradient-accumulation normalisation, so W only shifts the
                       relative weight (W=10: decision tokens carry ~16.6 % of the loss weight). The weighted positions are
                       looked up from TRL's own prepared rows (completion token ids -> positions), so they are exactly the
                       trained ids. Logits are computed only for the completion positions (logits_to_keep), never the full
                       262K-vocab x prompt-length matrix. Validation loss stays unweighted (comparable with M1's).
  --keep-epoch-ends    after every epoch the adapter is also saved to <out>/epoch-<k> (M1 lost its epoch-1 weights to
                       save_total_limit; the audit asks whether epoch 2 helped held-out decisions).

Run inside lx-exp, under memguard:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 training/memguard.py --name sft --budget-gb 30 --gpu-cap-gb 26 -- \
    python3 training/train_sft.py --model <base> --data /workspace/training/data_v12_curated --out /workspace/runs/m1_sft
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PURE_TRL_MODE", "1")
import guards  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
ap.add_argument("--revision", default=None)
ap.add_argument("--data", required=True, help="training/from_curated.py output directory")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lora-r", type=int, default=16)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--grad-accum", type=int, default=16)
ap.add_argument("--max-length", type=int, default=12288)   # L2 decision rows reach ~8.5K Tiny Aya tokens; never truncate targets
ap.add_argument("--max-rows", type=int, default=None)
ap.add_argument("--save-steps", type=int, default=10)
ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints in --out (they are renamed stale_*)")
ap.add_argument("--smoke", action="store_true", help="first 16 train + 4 val rows, 1 epoch")
ap.add_argument("--decision-weight", type=float, default=1.0, help="loss weight of the decision tokens (J4); 1 = plain SFT")
ap.add_argument("--keep-epoch-ends", action="store_true", help="also save the adapter to <out>/epoch-<k> after every epoch")
ns = ap.parse_args()
if ns.decision_weight <= 0:
    guards.fail(f"--decision-weight must be > 0 (got {ns.decision_weight})")

import torch  # noqa: E402
guards.apply_mem_cap()
from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoTokenizer, TrainerCallback  # noqa: E402
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402
from agenttune.agentic.distill_trainer import create_distill_trainer  # noqa: E402


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


train_rows = load(os.path.join(ns.data, "sft_train.jsonl"))
val_rows = load(os.path.join(ns.data, "sft_val.jsonl"))
if ns.max_rows:
    train_rows = train_rows[: ns.max_rows]
if ns.smoke:
    train_rows, val_rows, ns.epochs = train_rows[:16], val_rows[:4], 1.0
if not train_rows:
    guards.fail(f"no SFT training rows in {os.path.join(ns.data, 'sft_train.jsonl')}")
keep = ("prompt", "completion")
if ns.decision_weight != 1.0 and not all("turn" in r for r in train_rows):
    guards.fail("--decision-weight needs the 'turn' field on every training row (training/from_curated.py output)")
train_ds = Dataset.from_list([{k: r[k] for k in keep} for r in train_rows])
val_ds = Dataset.from_list([{k: r[k] for k in keep} for r in val_rows]) if val_rows else None
print(f"M1 SFT rows: train={len(train_ds)} val={len(val_ds) if val_ds else 0}", flush=True)

# ---- run state: resume or start clean -------------------------------------------------------------------------------------
NOT_COMPARED = ("fresh", "smoke", "save_steps")   # smoke's effect is compared through epochs, row counts and the rows' hash
PATH_ARGS = ("model", "data", "out")
ARGS_PATH = os.path.join(ns.out, "args.json")
PROGRESS = os.path.join(ns.out, "progress.json")


def rows_sha(rows):
    return hashlib.sha256(json.dumps([[r["prompt"], r["completion"]] for r in rows], sort_keys=True).encode()).hexdigest()


def read_json(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def dump_atomic(obj, p, **kw):
    """A kill mid-write leaves the previous file intact; result.json is also the finished-run marker."""
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, **kw)
    os.replace(tmp, p)


def resolved(d):
    """Existing path arguments as realpaths now, so args.json written from one cwd compares correctly from another."""
    return {k: os.path.realpath(v) if k in PATH_ARGS and isinstance(v, str) and os.path.exists(v) else v for k, v in d.items()}


def comparable(d):
    return {k: v for k, v in d.items() if k not in NOT_COMPARED}


def set_aside(name, prefix):
    src = os.path.join(ns.out, name)
    dst = os.path.join(ns.out, prefix + name)
    if os.path.exists(dst):
        dst += time.strftime(".%Y%m%d-%H%M%S")
    os.rename(src, dst)
    print(f"SET_ASIDE {src} -> {dst}", flush=True)


def last_complete_checkpoint():
    while (ck := get_last_checkpoint(ns.out)) is not None:
        try:
            json.load(open(os.path.join(ck, "trainer_state.json")))
            return ck
        except (OSError, ValueError):
            set_aside(os.path.basename(ck), "incomplete_")
    return None


os.makedirs(ns.out, exist_ok=True)
run_args = resolved({**vars(ns), "n_train": len(train_rows), "n_val": len(val_rows), "train_sha256": rows_sha(train_rows),
                     "val_sha256": rows_sha(val_rows)})
for _k, _default in (("decision_weight", 1.0), ("keep_epoch_ends", False)):   # J4 flags: recorded only when set, so the
    if run_args.get(_k) == _default:                                           # args.json of earlier runs still compares equal
        run_args.pop(_k, None)
ck = None if ns.fresh else last_complete_checkpoint()
if ck:
    if not os.path.exists(ARGS_PATH):
        guards.fail(f"{ck} has no args.json beside it, so the resume cannot be checked; rerun with --fresh")
    saved = read_json(ARGS_PATH)
    if not isinstance(saved, dict):
        guards.fail(f"{ARGS_PATH} is unreadable or not a JSON object, so the resume from {ck} cannot be checked; "
                    f"rerun with --fresh")
    saved, cur = comparable(saved), comparable(json.loads(json.dumps(run_args)))
    diff = {k: (saved.get(k), cur.get(k)) for k in sorted(set(saved) | set(cur)) if saved.get(k) != cur.get(k)}
    if diff:
        guards.fail(f"refusing to resume from {ck}: arguments differ from {ARGS_PATH} (saved, now): {diff}; "
                    f"rerun with the saved arguments or with --fresh")
    state = read_json(os.path.join(ck, "trainer_state.json"))
    if (state.get("global_step", 0) >= state.get("max_steps", 0) > 0
            and all(os.path.exists(os.path.join(ns.out, f)) for f in ("final", "result.json"))):
        # Trainer also saves at the last step, so a finished run looks resumable: rerunning it would reload the model, train
        # 0 steps and replace result.json (the real wall time) with an empty segment.
        print(f"SFT_ALREADY_DONE {ns.out} (checkpoint step {state['global_step']} of {state['max_steps']})", flush=True)
        raise SystemExit(0)
    prior_segments = (read_json(PROGRESS) or {}).get("segments", [])
    print(f"RESUMING from {ck} ({len(prior_segments)} earlier segment(s))", flush=True)
else:
    prior_segments = []
    for name in sorted(os.listdir(ns.out)):
        if name.startswith("checkpoint-") or name in ("final", "result.json", "step_times.jsonl", "progress.json",
                                                      "guards.json"):
            set_aside(name, "stale_")
    dump_atomic(run_args, ARGS_PATH)

# One segment per process; earlier segments (a run killed by memguard, then resumed) are carried along, never dropped, so the
# total wall time stays measurable. progress.json is rewritten after every optimizer step.
t0 = time.time()
SEG = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "resumed_from": ck, "minutes": 0.0, "step_times_s": [], "peak_mem": None}


def save_progress():
    SEG["minutes"], SEG["peak_mem"] = round((time.time() - t0) / 60, 2), guards.peak_mem()
    dump_atomic({"segments": prior_segments + [SEG]}, PROGRESS)


save_progress()

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class StepTimer(TrainerCallback):
    """Wall seconds between consecutive optimizer steps (includes any save/eval in between: what an ETA needs), recorded in this
    segment and appended to <out>/step_times.jsonl as they happen, so a killed run still shows its rate."""

    def __init__(self):
        self.t = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        dt = round(now - self.t, 2)
        self.t = now
        SEG["step_times_s"].append(dt)
        with open(os.path.join(ns.out, "step_times.jsonl"), "a") as fh:
            fh.write(json.dumps({"segment": len(prior_segments), "step": state.global_step, "s": dt,
                                 "max_steps": state.max_steps}) + "\n")
        save_progress()


import sft_weighting  # noqa: E402  (J4 helpers: decision_positions, build_decision_table, DecisionWeightedSFTTrainer, EpochEndSaver)


class WtmSFTFactory:
    """trainer_factory(model=, train_dataset=, **kwargs) -> trainer. Ignores AgentTune's EventLog projection (no system role)."""

    def __init__(self, resume_from):
        self.resume_from, self.timer, self.dw_stats = resume_from, StepTimer(), None

    def __call__(self, model, train_dataset=None, **kwargs):
        tok = AutoTokenizer.from_pretrained(model, revision=ns.revision)
        cfg = SFTConfig(
            output_dir=ns.out, num_train_epochs=ns.epochs, learning_rate=ns.lr, lr_scheduler_type="cosine", warmup_steps=5,
            per_device_train_batch_size=ns.batch, gradient_accumulation_steps=ns.grad_accum, max_length=ns.max_length,
            per_device_eval_batch_size=1,   # default 8 × ~8.7K tokens × 262K-vocab logits would not fit in memory
            completion_only_loss=True, bf16=True, gradient_checkpointing=True, logging_steps=1,
            save_strategy="steps", save_steps=ns.save_steps, save_total_limit=3,
            eval_strategy="epoch" if val_ds is not None else "no", report_to="none", seed=0,
            model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa", "revision": ns.revision},
        )
        peft = LoraConfig(r=ns.lora_r, lora_alpha=2 * ns.lora_r, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                          target_modules=LORA_TARGETS)
        cls = SFTTrainer if ns.decision_weight == 1.0 else sft_weighting.DecisionWeightedSFTTrainer
        callbacks = [self.timer] + ([sft_weighting.EpochEndSaver(ns.out)] if ns.keep_epoch_ends else [])
        self.trainer = cls(model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tok,
                           peft_config=peft, callbacks=callbacks)
        if ns.decision_weight != 1.0:
            try:
                table, self.dw_stats = sft_weighting.build_decision_table(self.trainer.train_dataset, tok, train_rows,
                                                                          ns.decision_weight)
            except ValueError as e:
                guards.fail(f"--decision-weight: {e}")
            sft_weighting.DecisionWeightedSFTTrainer.dw_weight = ns.decision_weight
            sft_weighting.DecisionWeightedSFTTrainer.dw_table = table
            print(f"DECISION_WEIGHT {ns.decision_weight}: {json.dumps(self.dw_stats)}", flush=True)
        guards.write(ns.out, "sft_train", guards.check_sft(self.trainer, [r["completion"][0]["content"] for r in train_rows],
                                                           ns.max_length))
        if val_ds is not None:
            guards.write(ns.out, "sft_val", guards.check_sft(self.trainer, [r["completion"][0]["content"] for r in val_rows],
                                                             ns.max_length, dataset=self.trainer.eval_dataset, name="val"))
        if self.resume_from:
            # AgentTune's Project.distill calls .train() with no arguments on the object returned here
            self.trainer.train = functools.partial(self.trainer.train, resume_from_checkpoint=self.resume_from)
        return self.trainer   # Project.distill returns only the TrainOutput, so keep a handle for saving


demos = [{"prompt": json.dumps(r["prompt"][-1]["content"])[:2000], "response": r["completion"][0]["content"]}
         for r in train_rows[:64]]
factory = WtmSFTFactory(ck)
trainer = create_distill_trainer(student=ns.model, demonstrations=demos, trainer_factory=factory, output_dir=ns.out)
result = trainer.train()
final = os.path.join(ns.out, "final")
factory.trainer.save_model(final)
factory.trainer.processing_class.save_pretrained(final)
save_progress()
segments = prior_segments + [SEG]
# minutes / step_times_s / peak_mem are this process; minutes_all_segments adds the earlier (killed, then resumed) processes
dump_atomic({"train_output": str(result), "resumed_from": ck, "minutes": SEG["minutes"],
             "minutes_all_segments": round(sum(s.get("minutes") or 0 for s in segments), 2),
             "peak_mem": SEG["peak_mem"], "step_times_s": SEG["step_times_s"], "segments": segments,
             "log_history": factory.trainer.state.log_history,
             **({"decision_weight": {"weight": ns.decision_weight, "table": factory.dw_stats,
                                     "seen_this_segment": dict(sft_weighting.DecisionWeightedSFTTrainer.dw_seen)}}
                if ns.decision_weight != 1.0 else {})},
            os.path.join(ns.out, "result.json"), default=str)
print(f"SFT_DONE in {SEG['minutes']:.1f} min ({len(segments)} segment(s)) -> {final}", flush=True)
