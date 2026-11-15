"""M3 — DPO on decision-point pairs (chosen = ideal move, rejected = an observed base-model failure or a realistic wrong move).

Lexsi chain: CuratorKIT already ran upstream (dataset/curate.py: exact dedup + SecretsGate → training/from_curated.py maps the
surviving pairs back to our original rows), then AlignTune DPO (`create_rl_trainer(algorithm="dpo", backend="trl")`).
Fallback (`--backend agenttune`) uses AgentTune's offline-pairs DPO (`create_agentic_trainer("dpo", backend="trl")`).

Data: --data is the dpo_train.jsonl FILE written by training/from_curated.py. CuratorKIT's own dpo.jsonl is refused: its text is
whitespace-normalised. AlignTune's DataManager is bypassed (the pairs go straight to trainer.train_dataset) because it flattens
every message list into role-less plain text, which TRL then trains on silently. <out>/data_provenance.json records the file,
its sha256 and the CuratorKIT report, and the final adapter's lexsi_provenance.json embeds CuratorKIT's own provenance.

Guards (training/guards.py): TRL 1.13 DPOConfig defaults to max_length=1024 and drops every pair whose prompt reaches it, and
AlignTune's config.train defaults (truncation_mode keep_end, save_steps 500) leak into DPOConfig, so the effective DPOConfig and
every tokenized pair are checked: from a hook on TRL's reference-log-prob precompute, i.e. before that long pass (with
--no-precompute, right after setup), so a bad config or a dropped pair stops the run before any GPU-heavy work.

Memory (shared GB10 host, unified memory): TRL 1.13 DPO materialises full-vocabulary fp32 logits for the whole sequence, so
batch 1 × grad-accum 16 (16 pairs per optimizer step) and the reference log-probs are precomputed at batch 1 in bf16, before
accelerate wraps the model; the reference model is never held during training. Never use liger here (no Cohere logit_scale,
and TRL refuses it with precompute). guards.apply_mem_cap() caps CUDA allocations at WTM_GPU_CAP_GB (set by memguard).

Resume: a checkpoint every --save-steps optimizer steps (last 3 kept). Unless --fresh, the run resumes from the newest complete
checkpoint-* in --out and refuses to (GUARD_FAIL) when the arguments or pairs differ from <out>/args.json. The precomputed
reference log-probs are cached in <out>/ref_logps_train.json, keyed by the model files and the tokenized pairs, so a resume skips
that pass. Rerunning a finished run (last checkpoint at max_steps, final/ and result.json present) prints DPO_ALREADY_DONE and
exits 0, so a chained rerun never replaces the real result.json. Each process is one segment in <out>/progress.json and
result.json["segments"] (minutes, reference-pass seconds, step times, peak memory), so a memguard kill followed by a resume keeps
the killed segment's numbers. With --fresh, or when there is nothing to resume, the old checkpoints, final/, result.json,
progress.json and step_times.jsonl are renamed stale_*. The adapter is saved to <out>/final.

Run inside lx-exp, under memguard:
  python3 training/memguard.py --name dpo --budget-gb 70 --gpu-cap-gb 60 -- \
    python3 training/train_dpo.py --model <base or merged M1> --data /workspace/training/data_v12_curated/dpo_train.jsonl \
      --out /workspace/runs/m3_dpo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PURE_TRL_MODE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # must be set before torch initialises CUDA
import torch  # noqa: E402
import guards  # noqa: E402
guards.apply_mem_cap()

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True, help="dpo_train.jsonl written by training/from_curated.py (a file, not a directory)")
ap.add_argument("--out", required=True)
ap.add_argument("--backend", default="aligntune", choices=["aligntune", "agenttune"])
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--lr", type=float, default=5e-6)
ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--grad-accum", type=int, default=16)
ap.add_argument("--max-length", type=int, default=12288)   # never truncate targets (L2 rows ~8.5K tokens)
ap.add_argument("--lora-r", type=int, default=16)
ap.add_argument("--save-steps", type=int, default=8)
ap.add_argument("--no-precompute", action="store_true", help="compute reference log-probs on the fly (more memory per step)")
ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints in --out (they are renamed stale_*)")
ap.add_argument("--smoke", action="store_true", help="first 8 pairs, 1 epoch")
ns = ap.parse_args()
PRECOMPUTE = not ns.no_precompute
SAVE_TOTAL_LIMIT = 3
LORA = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
CURATED = "/workspace/dataset/curated"   # dataset/curate.py output on the Spark (curation_report.json, dpo/ export)
CURATED_DPO = os.path.join(CURATED, "dpo")
REF_CACHE = os.path.join(ns.out, "ref_logps_train.json")

# ---- data -------------------------------------------------------------------------------------------------------------------
src = os.path.abspath(ns.data)
if not os.path.isfile(src) or not src.endswith(".jsonl"):
    guards.fail(f"--data must be a .jsonl file (training/from_curated.py's dpo_train.jsonl), got {ns.data}")
if os.path.basename(src) == "dpo.jsonl":
    guards.fail("--data is CuratorKIT's dpo.jsonl export (whitespace-normalised text); use from_curated.py's dpo_train.jsonl")
data_sha = hashlib.sha256(open(src, "rb").read()).hexdigest()
rows = [json.loads(line) for line in open(src) if line.strip()]
n_file = len(rows)
if ns.smoke:
    rows, ns.epochs = rows[:8], 1
if not rows:
    guards.fail(f"no DPO pairs in {src}")


def is_messages(x, roles=None):
    return (isinstance(x, list) and x and all(isinstance(m, dict) and set(m) == {"role", "content"}
                                              and isinstance(m["content"], str) for m in x)
            and (roles is None or [m["role"] for m in x] == roles))


for i, r in enumerate(rows):   # message lists with role/content only, so the Arrow round-trip adds no None-valued keys
    if not is_messages(r.get("prompt")) or not all(is_messages(r.get(k), ["assistant"]) for k in ("chosen", "rejected")):
        guards.fail(f"pair {i} ({r.get('scenario_id')}) is not prompt=[messages], chosen/rejected=[one assistant message]")

from datasets import Dataset  # noqa: E402
ds = Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rows])
if not isinstance(ds[0]["prompt"], list):
    guards.fail(f"prompt column is {type(ds[0]['prompt']).__name__}, expected a message list")
pairs_sha = hashlib.sha256(json.dumps([[r["prompt"], r["chosen"], r["rejected"]] for r in rows],
                                      sort_keys=True).encode()).hexdigest()
print(f"M3 DPO pairs: {len(ds)} of {n_file} in {src}", flush=True)

# ---- run state: resume or start clean -------------------------------------------------------------------------------------
from transformers import TrainerCallback  # noqa: E402
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402

NOT_COMPARED = ("fresh", "save_steps", "data_sha256")   # the pairs themselves are compared through pairs_sha256
PATH_ARGS = ("model", "data", "out")
ARGS_PATH = os.path.join(ns.out, "args.json")
PROGRESS = os.path.join(ns.out, "progress.json")


def read_json(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def comparable(d):
    out = {k: v for k, v in d.items() if k not in NOT_COMPARED}
    for k in PATH_ARGS:
        if isinstance(out.get(k), str) and os.path.exists(out[k]):
            out[k] = os.path.realpath(out[k])
    return out


def set_aside(name, prefix):
    src_path = os.path.join(ns.out, name)
    dst = os.path.join(ns.out, prefix + name)
    if os.path.exists(dst):
        dst += time.strftime(".%Y%m%d-%H%M%S")
    os.rename(src_path, dst)
    print(f"SET_ASIDE {src_path} -> {dst}", flush=True)


def last_complete_checkpoint():
    while (ck := get_last_checkpoint(ns.out)) is not None:
        try:   # Trainer._save_checkpoint writes trainer_state.json last: without it the save was cut short
            json.load(open(os.path.join(ck, "trainer_state.json")))
            return ck
        except (OSError, ValueError):
            set_aside(os.path.basename(ck), "incomplete_")
    return None


run_args = {**vars(ns), "n_pairs": len(ds), "pairs_sha256": pairs_sha, "data_sha256": data_sha}
os.makedirs(ns.out, exist_ok=True)
ck = None if ns.fresh else last_complete_checkpoint()
if ck:
    if not os.path.exists(ARGS_PATH):
        guards.fail(f"{ck} has no args.json beside it, so the resume cannot be checked; rerun with --fresh")
    saved, cur = comparable(json.load(open(ARGS_PATH))), comparable(json.loads(json.dumps(run_args)))
    diff = {k: (saved.get(k), cur.get(k)) for k in sorted(set(saved) | set(cur)) if saved.get(k) != cur.get(k)}
    if diff:
        guards.fail(f"refusing to resume from {ck}: arguments differ from {ARGS_PATH} (saved, now): {diff}; "
                    f"rerun with the saved arguments or with --fresh")
    state = json.load(open(os.path.join(ck, "trainer_state.json")))
    if (state.get("global_step", 0) >= state.get("max_steps", 0) > 0
            and all(os.path.exists(os.path.join(ns.out, f)) for f in ("final", "result.json"))):
        # Trainer also saves at the last step, so a finished run looks resumable: rerunning it would reload the model, train
        # 0 steps and replace result.json (the real wall time) with an empty segment.
        print(f"DPO_ALREADY_DONE {ns.out} (checkpoint step {state['global_step']} of {state['max_steps']})", flush=True)
        raise SystemExit(0)
    prior_segments = (read_json(PROGRESS) or {}).get("segments", [])
    print(f"RESUMING from {ck} ({len(prior_segments)} earlier segment(s))", flush=True)
else:
    prior_segments = []
    for name in sorted(os.listdir(ns.out)):
        if name.startswith("checkpoint-") or name in ("final", "result.json", "step_times.jsonl", "progress.json"):
            set_aside(name, "stale_")
    json.dump(run_args, open(ARGS_PATH, "w"), indent=1)

# One segment per process; earlier segments (a run killed by memguard, then resumed) are carried along, never dropped, so the
# total wall time stays measurable. progress.json is rewritten after the reference pass and after every step.
t0 = time.time()
SEG = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "resumed_from": ck, "minutes": 0.0, "ref_precompute_s": None,
       "step_times_s": [], "peak_mem": None}


def save_progress():
    SEG["minutes"], SEG["peak_mem"] = round((time.time() - t0) / 60, 2), guards.peak_mem()
    tmp = PROGRESS + ".tmp"
    json.dump({"segments": prior_segments + [SEG]}, open(tmp, "w"), indent=1)
    os.replace(tmp, PROGRESS)


save_progress()

report_p = os.path.join(CURATED, "curation_report.json")
from_curated_p = os.path.join(os.path.dirname(src), "from_curated_report.json")
json.dump({"source": src, "sha256": data_sha, "n_pairs_file": n_file, "n_pairs": len(ds), "smoke": ns.smoke,
           "curation_report": report_p if os.path.exists(report_p) else None,
           "from_curated_report": from_curated_p if os.path.exists(from_curated_p) else None,
           "curatorkit_provenance": read_json(os.path.join(CURATED_DPO, "lexsi_provenance.json"))},
          open(os.path.join(ns.out, "data_provenance.json"), "w"), indent=1)


# ---- checks + reference log-prob cache, run inside TRL's precompute hook ------------------------------------------------------
def check_config(a):
    """The DPOConfig TRL actually uses. AlignTune applies extra_params with setattr after building DPOConfig and gap-fills from
    its own config.train defaults, so what we passed is not necessarily what is in effect."""
    want = {"output_dir": os.path.expanduser(ns.out), "max_length": ns.max_length, "truncation_mode": "keep_start",
            "precompute_ref_log_probs": PRECOMPUTE, "per_device_train_batch_size": ns.batch,
            "gradient_accumulation_steps": ns.grad_accum, "save_strategy": "steps", "save_steps": ns.save_steps,
            "save_total_limit": SAVE_TOTAL_LIMIT, "num_train_epochs": ns.epochs, "max_steps": -1, "learning_rate": ns.lr,
            "beta": ns.beta, "use_liger_kernel": False}
    if PRECOMPUTE:
        want["precompute_ref_batch_size"] = 1
    got = {k: getattr(a, k, None) for k in want}
    bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
    if bad:
        guards.fail(f"effective DPOConfig differs (got, want): {bad}")
    return got


CHECKED = {}


def check(inner):
    if not CHECKED:
        CHECKED["dpo_config"] = check_config(inner.args)
        guards.write(ns.out, "dpo_config", CHECKED["dpo_config"])
        CHECKED["dpo_train"] = guards.check_dpo(inner, len(ds), ns.max_length, source_rows=rows)
        guards.write(ns.out, "dpo_train", CHECKED["dpo_train"])


def ref_cache_key(dataset):
    """The reference is the --model weights (adapter disabled), so the key covers those files and the tokenized pairs."""
    h = hashlib.sha256(json.dumps({"model": os.path.realpath(ns.model) if os.path.exists(ns.model) else ns.model,
                                   "max_length": ns.max_length}).encode())
    if os.path.isdir(ns.model):
        for f in sorted(os.listdir(ns.model)):
            if f.endswith((".safetensors", ".bin", ".json")):
                st = os.stat(os.path.join(ns.model, f))
                h.update(f"{f}:{st.st_size}:{st.st_mtime_ns}".encode())
    for e in dataset:
        h.update(json.dumps([e["prompt_ids"], e["chosen_ids"], e["rejected_ids"]]).encode())
    return h.hexdigest()


def install_precompute_hook():
    """TRL's DPOTrainer.__init__ prepares the dataset, sets self.args / train_dataset / processing_class (Trainer.__init__) and only
    then precomputes the reference log-probs; hooking that call runs the checks before the long pass and caches its result."""
    from trl import DPOTrainer
    original = DPOTrainer._precompute_ref_logps

    def hooked(self, dataset, name, batch_size):
        if name != "train":
            return original(self, dataset, name, batch_size)
        check(self)
        key = ref_cache_key(dataset)
        cached = read_json(REF_CACHE)
        if cached and cached.get("key") == key:
            print(f"REF_LOGPS_CACHE hit {REF_CACHE}", flush=True)
            return dataset.add_column("ref_chosen_logps", cached["chosen"]).add_column("ref_rejected_logps",
                                                                                       cached["rejected"])
        t = time.time()
        out = original(self, dataset, name, batch_size)
        SEG["ref_precompute_s"] = round(time.time() - t, 1)
        tmp = REF_CACHE + ".tmp"
        json.dump({"key": key, "chosen": list(out["ref_chosen_logps"]), "rejected": list(out["ref_rejected_logps"])},
                  open(tmp, "w"))
        os.replace(tmp, REF_CACHE)
        save_progress()
        return out

    DPOTrainer._precompute_ref_logps = hooked


class StepTimer(TrainerCallback):
    """Wall seconds between consecutive optimizer steps (includes any save in between: what an ETA needs), recorded in this
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


# ---- train ------------------------------------------------------------------------------------------------------------------
COMMON = dict(max_length=ns.max_length, truncation_mode="keep_start", precompute_ref_log_probs=PRECOMPUTE,
              save_strategy="steps", save_steps=ns.save_steps, save_total_limit=SAVE_TOTAL_LIMIT, logging_steps=1,
              bf16=True, gradient_checkpointing=True, report_to="none")
if PRECOMPUTE:
    COMMON["precompute_ref_batch_size"] = 1
final = os.path.join(ns.out, "final")
timer = StepTimer()
install_precompute_hook()
if ns.backend == "aligntune":
    from aligntune.core.backend_factory import create_rl_trainer
    from aligntune.utils.provenance import provenance_input
    # every leftover kwarg also lands in config.train.extra_params, which AlignTune setattr()s onto its DPOConfig (always wins)
    trainer = create_rl_trainer(
        model_name=ns.model, dataset_name=ds, algorithm="dpo", backend="trl", output_dir=ns.out, num_epochs=ns.epochs,
        batch_size=ns.batch, learning_rate=ns.lr, max_seq_length=ns.max_length, beta=ns.beta,
        use_peft=True, lora_r=ns.lora_r, lora_alpha=2 * ns.lora_r, lora_target_modules=LORA,
        gradient_accumulation_steps=ns.grad_accum, **COMMON)
    # AlignTune's own train() = setup_model → setup_data → setup_trainer → callbacks → train → save_model. setup_data is replaced:
    # its DataManager would flatten the message lists; setup_trainer reads only self.train_dataset / self.eval_dataset.
    trainer.setup_model()
    trainer.train_dataset, trainer.eval_dataset = ds, None
    trainer.setup_trainer()
    inner = trainer.trainer
    for cb in trainer.get_hf_callbacks():
        inner.add_callback(cb)
else:
    from peft import LoraConfig
    from agenttune.core.backend_factory import create_agentic_trainer
    trainer = create_agentic_trainer(
        "dpo", backend="trl", model=ns.model, train_dataset=ds, output_dir=ns.out, num_train_epochs=ns.epochs,
        per_device_train_batch_size=ns.batch, gradient_accumulation_steps=ns.grad_accum, learning_rate=ns.lr, beta=ns.beta,
        peft_config=LoraConfig(r=ns.lora_r, lora_alpha=2 * ns.lora_r, task_type="CAUSAL_LM", target_modules=LORA),
        model_init_kwargs={"dtype": torch.bfloat16},
        # read only by AgentTune's save_model provenance: a folder with lexsi_provenance.json is embedded as the input
        dataset_name=CURATED_DPO if os.path.isdir(CURATED_DPO) else src, **COMMON)
    # TrlAgenticDPO.train() reruns setup_data + setup_trainer, which would build and train a second model: set up by hand and
    # train the inner TRL trainer instead.
    trainer.setup_data()
    trainer.setup_trainer()
    inner = trainer.trainer
    if inner is None or inner.train_dataset is None:
        guards.fail("AgentTune DPO set up no TRL trainer or no prepared dataset")
check(inner)   # no-op when the precompute hook already ran it
inner.add_callback(timer)
result = inner.train(resume_from_checkpoint=ck)
if ns.backend == "aligntune":
    trainer.save_model(final)
    inputs = [provenance_input(ns.model, "model"), provenance_input(src, "dataset")]
    if os.path.isdir(CURATED_DPO):
        inputs.append(provenance_input(CURATED_DPO, "dataset"))
    trainer.write_provenance(final, inputs=inputs)
else:
    trainer._train_result = result   # training_stats.json is filled from it
    trainer.save_model(final)
save_progress()
segments = prior_segments + [SEG]
# minutes / step_times_s / peak_mem are this process; *_all_segments add up the earlier (killed, then resumed) processes
json.dump({"train_output": str(result), "resumed_from": ck, "minutes": SEG["minutes"],
           "minutes_all_segments": round(sum(s.get("minutes") or 0 for s in segments), 2),
           "ref_precompute_s": SEG["ref_precompute_s"],
           "ref_precompute_s_all_segments": round(sum(s.get("ref_precompute_s") or 0 for s in segments), 1),
           "peak_mem": SEG["peak_mem"], "step_times_s": SEG["step_times_s"], "segments": segments,
           "log_history": inner.state.log_history},
          open(os.path.join(ns.out, "result.json"), "w"), indent=1, default=str)
print(f"DPO_DONE in {SEG['minutes']:.1f} min ({len(segments)} segment(s)) -> {final}", flush=True)
