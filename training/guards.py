"""Pre-training guards: every trainer checks its OWN processed dataset before the first optimizer step and stops on a silent
data corruption; plus the in-process memory cap and peak-memory record every GPU script uses. Why (VERIFIED 2026-11-14 against
the installed TRL 1.13 / transformers 5.17 / AlignTune b6e924e):
  * TRL 1.13 DPOConfig defaults to max_length=1024 + truncation_mode="keep_start" and DROPS every pair whose prompt alone reaches
    max_length; AlignTune's DPO backend never maps its max_seq_length onto DPOConfig. Our L2 prompts are ~8.5K Tiny Aya tokens.
  * SFT/DPO truncate to max_length during preparation: a truncated row loses its target (keep_start keeps the prompt).
  * AlignTune's DataManager flattens conversational DPO rows into role-less text; TRL then tokenizes them with no chat template
    and appends eos, so the rows still end in eos and the count still matches. Only a template-level check catches it: every
    prompt must end with the generation prompt and every completion with <|END_RESPONSE|><|END_OF_TURN_TOKEN|>.
  * A trained span shifted across the prompt/completion boundary, or a prompt whose roles break the Tiny Aya template's
    alternation, trains silently on the wrong tokens; so every SFT row's trained span must be exactly its target. Token ids
    are the primary test (encode(target) + [<|END_RESPONSE|>, eos]): decode can normalise text and hide a wrong span. The
    decoded text is the fallback, because a BPE tokenizer may split the target differently inside the rendered template than
    standalone; such rows pass and are counted (n_text_only_matches). A row fails only when both differ.
  * GB10 memory is unified: an allocation past what the box has does not raise a CUDA OOM, it starves the host. A torch
    allocator cap (WTM_GPU_CAP_GB, set by training/memguard.py) turns an overrun into a catchable torch.OutOfMemoryError.
Each check returns a summary dict (written to <out>/guards.json by the caller) or raises SystemExit("GUARD_FAIL: …").
"""
from __future__ import annotations

import json
import os
import sys

GEN_PROMPT = "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>"   # Tiny Aya template, add_generation_prompt=True
END_RESPONSE = "<|END_RESPONSE|>"
TURN_END = END_RESPONSE + "<|END_OF_TURN_TOKEN|>"                       # how the template closes an assistant turn
GiB = 2 ** 30


def fail(msg):
    raise SystemExit(f"GUARD_FAIL: {msg}")


def apply_mem_cap():
    """WTM_GPU_CAP_GB (GiB, exported by memguard --gpu-cap-gb) -> torch caching-allocator cap. Call right after `import torch`.
    For GRPO colocate the cap must exceed vllm_gpu_memory_utilization x total + the trainer peak: vLLM's weights and KV cache
    are allocated through the same torch allocator in-process."""
    cap = os.environ.get("WTM_GPU_CAP_GB")
    if not cap:
        return None
    import torch
    if not torch.cuda.is_available():
        print(f"GPU_CAP {cap} GiB requested but CUDA is unavailable: not applied", flush=True)
        return None
    try:
        cap_gb = float(cap)
    except ValueError:
        fail(f"WTM_GPU_CAP_GB={cap!r} is not a number")
    if cap_gb <= 0:
        fail(f"WTM_GPU_CAP_GB={cap!r} must be > 0")
    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(0.95, cap_gb * GiB / total)
    torch.cuda.set_per_process_memory_fraction(frac)
    print(f"GPU_CAP {cap_gb:g} GiB = {frac:.3f} of {total / GiB:.1f} GiB", flush=True)
    return frac


def peak_mem():
    """Peaks since process start: torch allocator (allocated / reserved) and the process's max RSS. On unified memory the host
    cost is closer to reserved + RSS than to allocated; memguard's summary has the host-wide number."""
    out = {"torch_allocated_gb": None, "torch_reserved_gb": None, "rss_max_gb": None}
    torch = sys.modules.get("torch")   # a script that never imported torch never touched CUDA
    try:
        if torch is not None and torch.cuda.is_available():
            out["torch_allocated_gb"] = round(torch.cuda.max_memory_allocated() / GiB, 2)
            out["torch_reserved_gb"] = round(torch.cuda.max_memory_reserved() / GiB, 2)
    except Exception:
        pass
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out["rss_max_gb"] = round(rss * (1 if sys.platform == "darwin" else 1024) / GiB, 2)   # Linux reports KiB
    except Exception:
        pass
    return out


def peak_mem_gb():
    return peak_mem()["torch_allocated_gb"]


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def _ids(x):
    """apply_chat_template(tokenize=True) returns a BatchEncoding/dict under transformers 5 (return_dict defaults to True)."""
    if hasattr(x, "keys"):
        x = x["input_ids"]
    return list(x)


def _template_ids(tok):
    gen = tok.encode(GEN_PROMPT, add_special_tokens=False)
    end_resp = tok.convert_tokens_to_ids(END_RESPONSE)
    if not gen or end_resp is None or end_resp == tok.unk_token_id:
        fail(f"tokenizer has no {GEN_PROMPT!r} / {END_RESPONSE!r}: not the Tiny Aya chat template")
    return list(gen), end_resp


def _decode(tok, ids):
    return tok.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _role_error(prompt):
    """The Tiny Aya template takes an optional leading system turn, then raises unless user/assistant alternate from user;
    the prompt must end on a user turn so the completion is the assistant's reply."""
    if not isinstance(prompt, list) or not prompt or not all(isinstance(m, dict) for m in prompt):
        return f"prompt is not a list of messages ({type(prompt).__name__})"
    roles = [m.get("role") for m in prompt]
    body = roles[1:] if roles[0] == "system" else roles
    if not body:
        return "prompt has no user turn"
    for j, r in enumerate(body):
        if r != ("user" if j % 2 == 0 else "assistant"):
            return f"roles {roles} do not alternate user/assistant after an optional leading system turn"
    if body[-1] != "user":
        return f"last prompt role is {body[-1]!r}, not 'user'"
    return None


def check_sft(trainer, targets, max_length, dataset=None, name="train"):
    """targets[i] = the assistant text of input row i (TRL keeps row order: SFTConfig.shuffle_dataset defaults to False and
    preparation only maps/filters, and a dropped row already fails the count). Checks EVERY row."""
    tok = trainer.processing_class
    eos = tok.eos_token_id
    gen, end_resp = _template_ids(tok)
    ds = dataset if dataset is not None else trainer.train_dataset
    cols = ds.column_names
    if len(ds) != len(targets):
        fail(f"SFT {name} dataset has {len(ds)} rows after preparation, expected {len(targets)}")
    if "prompt" not in cols:
        fail(f"SFT {name}: prepared dataset has no 'prompt' column, so the roles cannot be checked: {cols}")
    lens, n_trained, problems, text_only = [], [], [], []
    longest, longest_span = None, []
    for i, e in enumerate(ds):
        ids = e["input_ids"]
        if "labels" in cols:
            pos = [k for k, l in enumerate(e["labels"]) if l != -100]
        elif "completion_mask" in cols:
            pos = [k for k, m in enumerate(e["completion_mask"]) if m]
        elif "assistant_masks" in cols:
            pos = [k for k, m in enumerate(e["assistant_masks"]) if m]
        else:
            fail(f"no label/mask column in the prepared SFT {name} dataset: {cols}")
        lens.append(len(ids))
        n_trained.append(len(pos))
        err = _role_error(e["prompt"])
        if err is None and "completion" in cols:
            c = e["completion"]
            if not (isinstance(c, list) and len(c) == 1 and isinstance(c[0], dict) and c[0].get("role") == "assistant"):
                err = f"completion is not one assistant message: {str(c)[:120]}"
        if err is None and not pos:
            err = "no trained tokens"
        if err is None and (pos != list(range(pos[0], pos[0] + len(pos))) or pos[-1] != len(ids) - 1):
            err = f"trained positions are not one span ending the sequence (first {pos[0]}, last {pos[-1]}, len {len(ids)})"
        if err is None and list(ids[max(0, pos[0] - len(gen)):pos[0]]) != gen:
            err = f"the untrained prefix does not end with {GEN_PROMPT}"
        if err is None and ids[-1] != eos:
            err = f"trained span does not end with eos {tok.convert_ids_to_tokens(eos)}"
        if err is None:
            span = [ids[k] for k in pos]
            if span != list(tok.encode(targets[i], add_special_tokens=False)) + [end_resp, eos]:
                decoded = _decode(tok, span)
                if decoded != targets[i] + TURN_END:
                    err = (f"trained span ids != encode(target) + [{END_RESPONSE}, eos] and its text differs too: "
                           f"{decoded[:200]!r} != target + {TURN_END} {(targets[i] + TURN_END)[:200]!r}")
                else:
                    text_only.append(i)
            if err is None and (longest is None or len(ids) > lens[longest]):
                longest, longest_span = i, span
        if err is not None:
            problems.append(f"row {i}: {err}")
    if max(lens) >= max_length:
        fail(f"an SFT {name} row reaches max_length={max_length} (truncated): longest {max(lens)}")
    if problems:
        fail(f"{len(problems)}/{len(ds)} SFT {name} rows fail: " + " | ".join(problems[:3]))
    longest_text = _decode(tok, longest_span)
    return {"rows": len(ds), "columns": cols, "max_len": max(lens), "p95_len": _pct(lens, 0.95),
            "trained_tokens_max": max(n_trained), "trained_tokens_mean": round(sum(n_trained) / len(n_trained), 1),
            "n_text_only_matches": len(text_only), "text_only_rows": text_only[:10],
            "longest_row_trained_span": longest_text[:400] + (" …" if len(longest_text) > 400 else "")}


def _content(side):
    if isinstance(side, list):
        return "".join(m.get("content", "") for m in side if isinstance(m, dict))
    return str(side)


def check_dpo(trainer, n_expected, max_length, source_rows=None):
    """Checks the prepared pairs of a TRL DPOTrainer. Extra columns (prompt/chosen/rejected kept by TRL's map,
    ref_chosen_logps/ref_rejected_logps from precompute_ref_log_probs) are fine. source_rows = the rows given to the trainer,
    in order (TRL keeps order; a dropped row fails the count) — used to check row 0 against a fresh template render."""
    a = trainer.args
    if a.max_length != max_length:
        fail(f"DPOConfig.max_length={a.max_length}, expected {max_length}")
    ds = trainer.train_dataset
    if len(ds) != n_expected:
        fail(f"TRL DPO kept {len(ds)}/{n_expected} pairs (it drops pairs whose prompt reaches max_length)")
    missing = {"prompt_ids", "chosen_ids", "rejected_ids"} - set(ds.column_names)
    if missing:
        fail(f"prepared DPO dataset lacks {sorted(missing)}: {ds.column_names}")
    tok = trainer.processing_class
    eos = tok.eos_token_id
    gen, end_resp = _template_ids(tok)
    tail = [end_resp, eos]
    lens, bad_prompt, bad_tail = [], [], []
    for i, e in enumerate(ds):
        p, c, r = list(e["prompt_ids"]), list(e["chosen_ids"]), list(e["rejected_ids"])
        lens.append(len(p) + max(len(c), len(r)))
        if p[-len(gen):] != gen:
            bad_prompt.append(i)
        if c[-2:] != tail or r[-2:] != tail or len(c) < 3 or len(r) < 3:
            bad_tail.append(i)
    if bad_prompt:
        fail(f"{len(bad_prompt)}/{len(ds)} DPO prompts do not end with the generation prompt {GEN_PROMPT} (flattened, "
             f"template-less rows?): rows {bad_prompt[:5]}, row {bad_prompt[0]} ends {_decode(tok, ds[bad_prompt[0]]['prompt_ids'][-12:])!r}")
    if bad_tail:
        fail(f"{len(bad_tail)}/{len(ds)} DPO pairs have a chosen/rejected side that does not end with {TURN_END}: rows "
             f"{bad_tail[:5]}, row {bad_tail[0]} chosen ends {_decode(tok, ds[bad_tail[0]]['chosen_ids'][-8:])!r}")
    if max(lens) > max_length:
        fail(f"a DPO pair exceeds max_length={max_length} (would be truncated): longest {max(lens)}")
    e = ds[0]
    if source_rows is not None:
        if len(source_rows) != n_expected:
            fail(f"check_dpo got {len(source_rows)} source rows for {n_expected} pairs")
        src = source_rows[0]
        want = _ids(tok.apply_chat_template(src["prompt"], add_generation_prompt=True, tokenize=True))
        if want != list(e["prompt_ids"]):
            fail(f"row 0 prompt_ids ({len(e['prompt_ids'])} tokens) != apply_chat_template(source prompt) ({len(want)} tokens): "
                 f"{_decode(tok, e['prompt_ids'][:40])!r} vs {_decode(tok, want[:40])!r}")
        for side in ("chosen", "rejected"):
            text, dec = _content(src[side]).strip(), _decode(tok, e[f"{side}_ids"])
            if text not in dec:
                fail(f"row 0 {side}_ids decode to {dec[:200]!r}, which does not contain the source {side} {text[:200]!r}")
    return {"pairs": len(ds), "columns": ds.column_names, "max_len": max(lens), "p95_len": _pct(lens, 0.95),
            "truncation_mode": a.truncation_mode, "beta": a.beta, "loss_type": a.loss_type,
            "precompute_ref_log_probs": getattr(a, "precompute_ref_log_probs", None),
            "per_device_train_batch_size": getattr(a, "per_device_train_batch_size", None),
            "row0_checked_against_source": source_rows is not None,
            "example_chosen": _decode(tok, e["chosen_ids"])[:300], "example_rejected": _decode(tok, e["rejected_ids"])[:300]}


def check_grpo_prompts(tok, prompts, max_model_len, max_completion):
    lens = [len(tok.apply_chat_template(p, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"])
            for p in prompts]
    if max(lens) + max_completion > max_model_len:
        fail(f"a GRPO prompt ({max(lens)} tokens) + max_completion {max_completion} exceeds max_model_len {max_model_len}")
    return {"prompts": len(lens), "max_prompt_len": max(lens), "p95_prompt_len": _pct(lens, 0.95),
            "mean_prompt_len": round(sum(lens) / len(lens), 1)}


def write(out, name, summary):
    p = os.path.join(out, "guards.json")
    d = json.load(open(p)) if os.path.exists(p) else {}
    d[name] = summary
    json.dump(d, open(p, "w"), indent=1, ensure_ascii=False, default=str)
    print(f"GUARD_OK {name}: " + json.dumps({k: v for k, v in summary.items() if not str(k).startswith(("example", "longest"))},
                                         default=str), flush=True)
