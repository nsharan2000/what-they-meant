"""M2 — GRPO with a programmatic reward through AgentTune (create_agentic_trainer("grpo", backend="trl"), no `tools`:
batched TRL GRPO on decision turns; AgentTune's multi-turn tool path is broken on TRL 1.13 + vLLM — see research doc §3.3).

Reward = wtm.reward (same grader as evaluation) + small format term; every reward's reasons are logged through AgentTune's
score logger (AGENTTUNE_SCORE_LOG) for curves, each record tagged with meta.step (optimizer steps already applied to the policy
that generated it), meta.run_id, meta.end_id (last completion token), meta.n_tokens and meta.truncated. Start from the merged M1
checkpoint (warm start) or the base model (ablation).

Truncation is decided here, not by TRL. TRL 1.13 counts a completion as finished only if its last id is the tokenizer's eos or
pad (grpo_trainer.py:2324-2325 for completions/clipped_ratio, 2511-2512 for mask_truncated_completions), but its colocated vLLM
also stops on generation_config.json's eos ids (vllm_generation.py:662-683 passes no stop_token_ids, so vLLM's defaults apply).
Tiny Aya's tokenizer eos is <|END_OF_TURN_TOKEN|> (6) while generation_config.json says <EOS_TOKEN> (3), so a turn closed with 3
looked truncated to TRL: the 2026-11-14 smoke had clipped_ratio 1 at 15-166 tokens (max 256), mean_terminated_length 0, and with
mask_truncated_completions=True every completion was masked (loss 0, grad_norm 0, entropy 0: GRPO learned nothing). So the mask
is off, and the reward gives TRUNC_PENALTY (without grading) only to a completion that both lacks an END_ID (tokenizer eos,
<EOS_TOKEN>, <|END_RESPONSE|>, every generation_config eos of --model) and reached --max-completion tokens. The true fractions
are logged per step through TRL's log_metric as wtm/truncated_frac, wtm/end_eos<id>_frac and wtm/end_other_frac (no end token
yet shorter than the limit: vLLM stopped for another reason); TRL's completions/clipped_ratio stays misleading for Tiny Aya.

Step semantics (PLAN.md): one optimizer step = one generation round of --batch prompts × --num-generations completions
(gradient accumulation defaults to num_generations micro-batches of --batch completions, so every step is on-policy).
Prompt lengths are checked against --max-model-len before training (training/guards.py); TRL 1.13 GRPO never truncates prompts.

Memory (shared GB10, 121 GB unified memory; the host hung once): vLLM runs colocated, sleep mode off, so its pool
(--vllm-mem × total) stays allocated through every backward. Hence a 0.20 default, vLLM forced into eager mode (TRL 1.13 passes no
enforce_eager in colocate mode, so vLLM would torch.compile with one inductor worker per core and capture CUDA graphs), one
inductor compile thread, and guards.apply_mem_cap() (WTM_GPU_CAP_GB from memguard; it must exceed vllm_mem × total + the trainer
peak). Run it under training/memguard.py.

Long runs are resumable: unless --fresh, training resumes from the newest complete checkpoint-* in --out, refused (GUARD_FAIL)
if <out>/args.json is missing or unreadable or its training-relevant args differ, including data_sha256 (the grpo_train.jsonl
rows after the --levels filter) and warmup_steps (fixed when the run starts and inherited on resume, so a resume that raises
--max-steps keeps the learning-rate schedule). A rerun of a finished run exits 0 without loading anything, so a retried chain
never overwrites the real result.json. Each process is one segment in progress.json/result.json ("segments": run_id,
resumed_from, minutes, step_times, step_times_s, peak_mem), so a memguard kill followed by a resume keeps the killed segment's
numbers. A fresh start moves old checkpoints, final/, progress.json, result.json and guards.json aside and renames scores.jsonl.

GRPO v2 flags (2026-11-15, experiments/dataset_audit/REPORT.md P2/P4; absent = the behaviour above, unchanged):
  --no-shuffle         TRL shuffle_dataset=False: RepeatSampler takes prompts in file order, --batch per generation round =
                       one optimizer step (grpo_trainer.py _get_train_sampler); use with training/build_grpo_v2.py's ordered file
                       ([A1,B1,A2,B2] per step). On resume the Trainer skips the consumed batches, so the order continues.
  --reward-variant v2  wtm.reward variant v2 (twin-A approve_informed scored like approve_generic); default v1.
  --log-completions    adds the completion text to every scores.jsonl record (meta.completion) to read reward hacking later.

Run inside lx-exp:
  python3 training/memguard.py --name grpo --budget-gb 65 --gpu-cap-gb 60 -- \
      python3 training/train_grpo.py --model /workspace/runs/m1_sft/merged --data /workspace/training/data_v12_curated \
      --out /workspace/runs/m2_grpo --levels L1 L2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PURE_TRL_MODE", "1")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")   # cubin/flashinfer mismatch in this image (previous project)
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")        # top_p kernel signature mismatch on sm_121
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")      # no per-core compile worker pool on the shared host

import guards  # noqa: E402  (stdlib only; torch is imported later, right before apply_mem_cap)

SMOKE_DEFAULTS = {"max_steps": 2, "batch": 4, "num_generations": 4, "save_steps": 1}
# A resume must not silently change what is being trained; memory/runtime knobs (vllm_mem, save_steps, max_steps) may change.
RESUME_KEYS = ("model", "data", "data_sha256", "levels", "batch", "num_generations", "grad_accum", "lr", "beta", "lora_r",
               "max_completion", "max_model_len", "warmup_steps", "smoke", "no_shuffle", "reward_variant")
# keys added 2026-11-15: an args.json written before them compares as these defaults (no spurious refusal of an old resume)
RESUME_DEFAULTS = {"no_shuffle": False, "reward_variant": "v1"}
TRUNC_PENALTY = -1.0   # = wtm.reward's worst decision (invalid / harmful commit)
_CKPT = re.compile(r"^checkpoint-(\d+)$")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", default="training/data")
ap.add_argument("--out", required=True)
# default None = "not given": --smoke fills only these (SMOKE_DEFAULTS), so an explicit --batch 1 survives a smoke retry
ap.add_argument("--max-steps", type=int, default=None, help="default 40 (~4x8 rollouts/step; smoke 2)")
ap.add_argument("--num-generations", type=int, default=None, help="default 8 (smoke 4)")
ap.add_argument("--batch", type=int, default=None, help="prompts per step (x num_generations completions); default 4")
ap.add_argument("--grad-accum", type=int, default=None, help="default = num_generations (one generation round per step)")
ap.add_argument("--lr", type=float, default=5e-6)
ap.add_argument("--beta", type=float, default=0.0, help="PLAN.md: no KL term (LoRA + few steps bound the drift; measured)")
ap.add_argument("--lora-r", type=int, default=16)
ap.add_argument("--max-completion", type=int, default=256)
ap.add_argument("--vllm-mem", type=float, default=0.20, help="~24 GB: 6.7 GB weights + KV for 32 seqs x ~7.7K tokens")
ap.add_argument("--max-model-len", type=int, default=12288, help="L2 decision prompts reach ~8.5K Tiny Aya tokens (log 22:2x)")
ap.add_argument("--save-steps", type=int, default=None, help="default 5 (smoke 1)")
ap.add_argument("--levels", nargs="+", default=None, help="keep only GRPO rows at these levels, e.g. --levels L1 L2")
ap.add_argument("--fresh", action="store_true", help="ignore and move aside existing checkpoints and scores.jsonl")
ap.add_argument("--smoke", action="store_true", help="first 8 rows; fills max_steps 2, batch 4, num_generations 4, save_steps 1")
ap.add_argument("--no-shuffle", action="store_true", help="keep the file's prompt order (TRL shuffle_dataset=False); GRPO v2")
ap.add_argument("--reward-variant", default="v1", choices=["v1", "v2"], help="wtm.reward variant; v2 = no partial credit for "
                "echo approval cards on twin A")
ap.add_argument("--log-completions", action="store_true", help="store the completion text in scores.jsonl (meta.completion)")
ns = ap.parse_args()
for k, v in (SMOKE_DEFAULTS if ns.smoke else {"max_steps": 40, "batch": 4, "num_generations": 8, "save_steps": 5}).items():
    if getattr(ns, k) is None:
        setattr(ns, k, v)
if ns.grad_accum is None:
    ns.grad_accum = ns.num_generations
if ns.levels:
    ns.levels = sorted(set(ns.levels))
ns.warmup_steps = max(1, min(10, ns.max_steps // 10))   # a fixed 10 would keep a 2-step smoke at lr ~0
os.makedirs(ns.out, exist_ok=True)
run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
score_log = os.path.join(ns.out, "scores.jsonl")
progress_path = os.path.join(ns.out, "progress.json")
args_path = os.path.join(ns.out, "args.json")


def dump_atomic(path, obj, **kw):
    """A kill mid-write (memguard SIGKILL) must not leave a truncated JSON that blocks the next resume."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, **kw)
    os.replace(tmp, path)


def last_complete_checkpoint(out):
    """Newest checkpoint-N with trainer_state.json, which Trainer._save_checkpoint writes last (a kill mid-save leaves none)."""
    cks = sorted((int(m.group(1)), d) for d in os.listdir(out) if (m := _CKPT.match(d)) and os.path.isdir(os.path.join(out, d)))
    for _, d in reversed(cks):
        if os.path.isfile(os.path.join(out, d, "trainer_state.json")):
            return os.path.join(out, d)
        print(f"SKIP incomplete {d} (no trainer_state.json)", flush=True)
    return None


def comparable(key, value):
    return sorted(set(value)) if key == "levels" and value else value


rows = [json.loads(l) for l in open(os.path.join(ns.data, "grpo_train.jsonl")) if l.strip()]
if ns.levels:
    rows = [r for r in rows if r["level"] in ns.levels]
    if not rows:
        guards.fail(f"no GRPO rows at levels {ns.levels}")
ns.data_sha256 = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
if ns.smoke:
    rows = rows[:8]

resume_from = None if ns.fresh else last_complete_checkpoint(ns.out)
if resume_from:
    try:
        prev = json.load(open(args_path))
    except (OSError, ValueError) as e:
        guards.fail(f"{resume_from} exists but {args_path} is missing or unreadable ({e!r}), so the resume cannot be checked; "
                    f"rerun with --fresh")
    ns.warmup_steps = prev.get("warmup_steps", ns.warmup_steps)   # absent (older run) -> differs below -> refused
    diff = {k: (prev.get(k, RESUME_DEFAULTS.get(k)), getattr(ns, k)) for k in RESUME_KEYS
            if comparable(k, prev.get(k, RESUME_DEFAULTS.get(k))) != comparable(k, getattr(ns, k))}
    if diff:
        guards.fail(f"refusing to resume from {resume_from}: args differ from {args_path} "
                    f"(prev, now) {diff}; rerun with the same args or with --fresh")
    done_step = json.load(open(os.path.join(resume_from, "trainer_state.json")))["global_step"]
    if done_step >= ns.max_steps and all(os.path.exists(os.path.join(ns.out, f)) for f in ("final", "result.json")):
        # Trainer saves a checkpoint at max_steps (DefaultFlowCallback), so this is a finished run: rerunning would load the
        # model and vLLM, train 0 steps and replace result.json with an empty segment.
        print(f"GRPO_ALREADY_DONE {ns.out} (checkpoint step {done_step} >= max_steps {ns.max_steps})", flush=True)
        raise SystemExit(0)
    prior_segments = json.load(open(progress_path))["segments"] if os.path.isfile(progress_path) else []
    print(f"RESUMING from {resume_from} ({len(prior_segments)} earlier segment(s))", flush=True)
else:
    # Fresh start (asked for, or nothing to resume): earlier checkpoints, the final adapter, timings, guard summaries and score
    # records belong to another run.
    prior_segments = []
    stale = [d for d in os.listdir(ns.out)
             if _CKPT.match(d) or d in ("final", "progress.json", "result.json", "guards.json")]
    old_stale_log = os.path.join(ns.out, "scores.stale.jsonl")
    if stale or os.path.exists(old_stale_log):
        aside = os.path.join(ns.out, f"stale-{run_id}")
        os.makedirs(aside)
        for d in stale + (["scores.stale.jsonl"] if os.path.exists(old_stale_log) else []):
            shutil.move(os.path.join(ns.out, d), os.path.join(aside, d))
        print(f"MOVED aside to {aside}: {sorted(os.listdir(aside))}", flush=True)
    if os.path.exists(score_log):
        os.replace(score_log, old_stale_log)
os.environ["AGENTTUNE_SCORE_LOG"] = score_log   # read at import time by agenttune.utils.score_logger

import torch  # noqa: E402

guards.apply_mem_cap()

from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoTokenizer, GenerationConfig, TrainerCallback  # noqa: E402
import trl.generation.vllm_generation as _vg  # noqa: E402  (imports vllm, which configures the "vllm" logger)
from agenttune.core.backend_factory import create_agentic_trainer  # noqa: E402
from agenttune.utils.score_logger import log_score  # noqa: E402
from wtm.reward import reward as wtm_reward  # noqa: E402

# TRL 1.13 builds the colocated engine as `self.llm = LLM(...)` inside VLLMGeneration (vllm_generation.py:347) with LLM a module
# global bound by `from vllm import LLM` (:43); the name is resolved at call time, so rebinding it here reaches that call.
if not hasattr(_vg, "LLM"):
    guards.fail("trl.generation.vllm_generation has no LLM (vLLM not importable?)")
_TrlLLM = _vg.LLM


def _eager_llm(*a, **k):
    k.setdefault("enforce_eager", True)
    return _TrlLLM(*a, **k)


_vg.LLM = _eager_llm


class _VllmInitLines(logging.Handler):
    """Keeps vLLM's memory-sizing lines (weights, KV pool, KV tokens) for result.json; best effort, depends on vLLM's log text.
    Short lines are kept whole; a long one (the engine-config dump that carries enforce_eager) only as its matching fragments."""
    KEYS = ("Model loading took", "Available KV cache memory", "KV cache size", "Maximum concurrency", "enforce_eager")
    LONG = 400

    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        msg = record.getMessage()
        hits = [msg.find(k) for k in self.KEYS if k in msg]
        if not hits:
            return
        if len(msg) <= self.LONG:
            self.lines.append(msg)
        else:
            self.lines.append("[long line] " + " | ".join(msg[i:i + 80].split(", ")[0] for i in sorted(hits)))


vllm_lines = _VllmInitLines()
logging.getLogger("vllm").addHandler(vllm_lines)


class StepClock(TrainerCallback):
    """Wall seconds per optimizer step (generation + scoring + forward/backward); rewrites <out>/progress.json every step so a
    run killed by memguard still leaves its rate and memory peak behind (earlier segments are carried along, never dropped)."""

    def __init__(self):
        self.t0, self.t, self.steps = time.time(), None, []

    def segment(self):
        return {"run_id": run_id, "resumed_from": resume_from, "minutes": round((time.time() - self.t0) / 60, 2),
                "step_times": self.steps, "step_times_s": [s["seconds"] for s in self.steps], "peak_mem": guards.peak_mem()}

    def on_step_begin(self, args, state, control, **kw):
        self.t = time.time()

    def on_step_end(self, args, state, control, **kw):
        if self.t is not None:
            self.steps.append({"step": state.global_step, "seconds": round(time.time() - self.t, 2)})
        dump_atomic(progress_path, {"segments": prior_segments + [self.segment()]})


def resolve_end_ids(tok, model):
    """Every id that closes a turn for vLLM or for the Tiny Aya template -> its token string."""
    ids = [tok.eos_token_id]
    for t in ("<EOS_TOKEN>", guards.END_RESPONSE):
        i = tok.convert_tokens_to_ids(t)   # an unknown token maps to unk_token_id (tokenization_utils_tokenizers.py:832-836)
        if i is None or i == tok.unk_token_id:
            print(f"NOTE tokenizer has no {t}: not an end id for this model", flush=True)
        else:
            ids.append(i)
    try:
        eos = GenerationConfig.from_pretrained(model).eos_token_id
    except OSError:
        print(f"NOTE {model} has no generation_config.json: end ids from the tokenizer only", flush=True)
        eos = None
    ids += [eos] if isinstance(eos, int) else list(eos or [])
    return {int(i): tok.convert_ids_to_tokens(int(i)) for i in ids if i is not None}


ds = Dataset.from_list([{"prompt": r["prompt"], "scenario": r["scenario"], "tool_kinds": r["tool_kinds"],
                         "scenario_id": r["scenario_id"], "level": r["level"]} for r in rows])
levels = {lv: sum(r["level"] == lv for r in rows) for lv in sorted({r["level"] for r in rows})}
print(f"M2 GRPO prompts: {len(ds)} {levels}", flush=True)
tok = AutoTokenizer.from_pretrained(ns.model)
guards.write(ns.out, "grpo_prompts", guards.check_grpo_prompts(tok, [r["prompt"] for r in rows], ns.max_model_len,
                                                                ns.max_completion))
END_IDS = resolve_end_ids(tok, ns.model)
print(f"GRPO END_IDS {END_IDS} (a completion without one of these at >= {ns.max_completion} tokens scores {TRUNC_PENALTY})",
      flush=True)


def decision_reward(completions, scenario, tool_kinds, scenario_id, level, completion_ids=None, trainer_state=None,
                    log_metric=None, **kw):
    """TRL passes completion_ids (raw generated ids), trainer_state and log_metric to a synchronous reward function
    (grpo_trainer.py:1645-1651, 1683-1684); AgentTune's combine_rewards forwards them as keywords (composite.py:89-93)."""
    step = trainer_state.global_step if trainer_state is not None else None
    ids_list = completion_ids if completion_ids is not None else [None] * len(completions)
    out, ends, n_trunc = [], Counter(), 0
    for c, ids, sj, tk, sid, lv in zip(completions, ids_list, scenario, tool_kinds, scenario_id, level, strict=True):
        scn = json.loads(sj)
        end_id = int(ids[-1]) if ids is not None and len(ids) else None
        truncated = ids is not None and end_id not in END_IDS and len(ids) >= ns.max_completion
        if truncated:
            r, parts = TRUNC_PENALTY, {"decision": "truncated", "base": TRUNC_PENALTY}
            n_trunc += 1
        else:
            r, parts = wtm_reward(c[-1]["content"] if isinstance(c, list) else c, scn, json.loads(tk),
                                  variant=ns.reward_variant)
            ends[end_id if end_id in END_IDS else "other"] += 1
        meta = {"scenario_id": sid, "level": lv, "twin": scn["twin"], "step": step, "run_id": run_id,
                "end_id": end_id, "n_tokens": len(ids) if ids is not None else None, "truncated": truncated}
        if ns.log_completions:
            meta["completion"] = c[-1]["content"] if isinstance(c, list) else c
        log_score("wtm.decision", r, components=parts, meta=meta)
        out.append(r)
    if completion_ids is not None and log_metric is not None and out:
        # TRL's _log_metric(name, value) appends to _pending_metrics and averages per logging step (grpo_trainer.py:1622-1633,
        # 2875-2880); with one generation round per optimizer step that is exactly this round's fraction.
        log_metric("wtm/truncated_frac", n_trunc / len(out))
        for i in END_IDS:
            log_metric(f"wtm/end_eos{i}_frac", ends[i] / len(out))
        log_metric("wtm/end_other_frac", ends["other"] / len(out))
    return out


dump_atomic(args_path, vars(ns), indent=1)
clock = StepClock()   # segment minutes include model + vLLM load: AgentTune builds the GRPOTrainer inside train()
trainer = create_agentic_trainer(
    "grpo", backend="trl",
    model=ns.model, train_dataset=ds, reward_funcs=[decision_reward], callbacks=[clock],
    peft_config=LoraConfig(r=ns.lora_r, lora_alpha=2 * ns.lora_r, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
    output_dir=ns.out, max_steps=ns.max_steps, logging_steps=1, save_steps=ns.save_steps, save_total_limit=20,
    resume_from_checkpoint=resume_from,
    per_device_train_batch_size=ns.batch, gradient_accumulation_steps=ns.grad_accum,
    generation_batch_size=ns.batch * ns.num_generations, num_generations=ns.num_generations,
    max_completion_length=ns.max_completion, learning_rate=ns.lr, lr_scheduler_type="constant_with_warmup",
    warmup_steps=ns.warmup_steps,
    # False: TRL's truncation test knows only tokenizer eos/pad (grpo_trainer.py:2511-2512), so for Tiny Aya it masked every
    # completion that vLLM ended on <EOS_TOKEN> (id 3): the 2026-11-14 smoke trained on nothing. decision_reward penalises
    # the truly truncated ones instead (module docstring).
    beta=ns.beta, loss_type="dapo", scale_rewards="group", mask_truncated_completions=False,
    use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=ns.vllm_mem, vllm_max_model_length=ns.max_model_len,
    vllm_enable_sleep_mode=False, temperature=1.0, top_p=1.0,
    bf16=True, gradient_checkpointing=True, report_to="none", log_completions=False,
    shuffle_dataset=not ns.no_shuffle,   # TRL default True; GRPO v2 keeps build_grpo_v2.py's step order
    model_init_kwargs={"attn_implementation": "sdpa", "dtype": torch.bfloat16}, seed=0,
)
trainer.train()   # AgentTune: setup_data -> setup_trainer -> GRPOTrainer.train(resume_from_checkpoint) (agentic_grpo.py:612-622)
inner = getattr(trainer, "trainer", trainer)
inner.save_model(os.path.join(ns.out, "final"))
seg = clock.segment()
segments = prior_segments + [seg]
dump_atomic(os.path.join(ns.out, "result.json"),
            {"run_id": run_id, "resumed_from": resume_from, "minutes": seg["minutes"],
             "minutes_all_segments": round(sum(s.get("minutes") or 0 for s in segments), 2), "peak_mem": seg["peak_mem"],
             "step_times": seg["step_times"], "step_times_s": seg["step_times_s"], "segments": segments,
             "end_ids": END_IDS, "trunc_penalty": TRUNC_PENALTY, "vllm_init": vllm_lines.lines,
             "log_history": inner.state.log_history}, default=str)
print(f"GRPO_DONE in {seg['minutes']:.1f} min ({len(segments)} segment(s)) -> {ns.out}", flush=True)
