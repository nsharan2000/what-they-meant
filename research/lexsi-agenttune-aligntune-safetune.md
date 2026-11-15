# Lexsi AgentTune · AlignTune · SafeTune: hands-on study for training and auditing a Cohere Aya agent

Compiled 2026-11-12. **Method:** I cloned the repos into `third_party/` and read them without running any Lexsi code. I also fetched the docs sites, PyPI JSON, TRL and transformers sources at their tags, and the HF model API.
**Commits:** AgentTune `1dfd054`, aligntune `b6e924e`, SafeTune `cc3de5f` (all 2026-11-12).
**How to read the citations:** a path such as `AgentTune/src/…` means `github.com/Lexsi-Labs/<repo>/blob/<commit>/<path>`.
**Tags:** **VERIFIED** means I read the primary source. **CAVEATED** means it comes from reading code, is an estimate, or was not run.

## 0. Bottom line
- **AgentTune should run both training methods:**
  - **M1, SFT/distillation:** `create_distill_trainer` with our own `trainer_factory`.
  - **M2, GRPO:** `create_agentic_trainer("grpo", backend="trl")` on the single decision turn.
- **AlignTune is optional:** a DPO arm plus its alignment audit. Never use `aligntune compose`; it is a stub that fabricates metrics.
- **SafeTune `evaluate()`** measures safety drift between the base model and each tuned model.
- **Only AgentTune handles tool calls for Tiny Aya**, but its multi-turn tool rollouts have three problems:
  - they are slow;
  - turn 2 onwards is trained on the wrong context;
  - with vLLM on, they break under our TRL 1.13.
- **No Lexsi Claude skill, MCP server or plugin exists.** Only SafeTune has an `llms.txt`.
- **AgentTune does not create personas, member profiles or transaction histories.**

## 1. Versions, licences, docs
| | AgentTune | AlignTune | SafeTune |
|---|---|---|---|
| PyPI latest (VERIFIED, `pypi.org/pypi/<pkg>/json`) | 1.1.5 | 0.1.35 | 0.1.18 |
| Pins | py≥3.12, `torch==2.11.0`, `trl==1.7.1`, `transformers>=5.15,<6` | py≥3.11, `torch<2.14`, `trl==1.7.1`, `peft>=0.18` | py≥3.12, `torch>=2.7`, `trl<1.15`, `transformers>=5.15,<6` |
| Licence (VERIFIED, READMEs) | LSAL v1.1: non-commercial; no weakening of a model's safety | MIT, but vendors unsloth (AGPL parts) | LSAL v1.2: no production use of unrepaired drifted checkpoints |
| Docs (VERIFIED) | agenttune.lexsi.ai; `llms.txt` and `llms-full.txt` return 404 | aligntune.lexsi.ai; both 404 | safetune.lexsi.ai; `/llms.txt` 200, `llms-full.txt` 404 |

## 2. Claude skills, MCP, plugins, llms.txt (negative result)
- **Clones (VERIFIED):** no `SKILL.md`, `.claude/`, MCP server or plugin manifest.
- **Org-wide GitHub code search (CAVEATED):** `filename:SKILL.md`, `.mcp.json`, `plugin.json` and `FastMCP` all returned 0 hits. The index is incomplete: it also missed SafeTune's `llms.txt`.
- **Web search:** found nothing either.
- **Saved:** SafeTune's `llms.txt` is now at `third_party/lexsi-skills/SafeTune/llms.txt`. It is a link index over six pillars (harden, recover, unlearn, steer, interpret, evaluate) with no code or skills.

## 3. AgentTune
### 3.1 Algorithms and strategies (VERIFIED: `README.md`, `docs/algorithms/overview.md`, `src/agenttune/agentic/README.md`)
- **RL algorithms:** `create_agentic_trainer(algorithm=…)`, TRL backend.

  | Algorithm | Feed it | Notes |
  |---|---|---|
  | `grpo` | reward function | default choice |
  | `rloo` | reward function | leave-one-out baseline, no β |
  | `dpo` | pairs | three modes: offline pairs, reward-ranked on-policy pairs, tool rollouts |
  | `bco` | binary labels | |
  | `ppo` | reward model and value model | |

  There is no stand-alone SFT entry point; SFT exists only inside distillation.
- **The five strategies** are ReAct, Plan-and-Solve, Reflexion, MemoryReAct and Tree-of-Thoughts/LATS. Each is an injected `policy(state)` callable for the GPU-free "spine" (`Project.infer/evaluate`).
  - **The RL trainers do not use them.** RL rollouts run a built-in ReAct-style tool loop.
- **Backend default:** `backend="auto"` picks Unsloth whenever it is importable (`core/backend_factory.py:173`).

### 3.2 Tool calling without a native template (VERIFIED: `rollout_factory.py`, `tests/test_cohere_tool_calls.py`, CHANGELOG 1.1.0)
- **Detection:** AgentTune probes the chat template. Tiny Aya and Aya Expanse ignore `tools`, and Tiny Aya renders a `tool` turn as empty.
- **Fallback:** it appends the tool schemas to the system prompt and folds each tool result into a user turn. Both texts can be overridden with `tools_fallback_prompt=` and `tool_result_format=`:
```python
# AgentTune/src/agenttune/agentic/rollout_engines/rollout_factory.py:102,544
DEFAULT_TOOL_RESULT_FORMAT = "Tool result for {name}: {content}\nUse this result to answer the original question."
DEFAULT_TOOLS_FALLBACK_PROMPT = ("You can call these tools:\n{tools}\n\nTo call tools, reply with only a JSON list, ... "
  '[{{"tool_name": "<name>", "parameters": {{"<argument>": <value>}}}}]. ... answer the user\'s original question as plain text.')
```
- **Parser** (`tool_call_parse.py`) accepts XML, `<tool_call>`, Cohere `<|START_ACTION|>`, bare `{"tool_name"}` lists, ReAct `Action:/Action Input:`, Mistral, Harmony and fenced JSON.
- **API-engine gap:** `APIRolloutEngine` sets `_tokenizer=None` (`api_engine.py:283`), so the fallback never fires there. Aya via HF providers would receive native `tools`, which those endpoints ignore or reject.

### 3.3 Plugging in our environment (VERIFIED code; impact CAVEATED)
- **Tools:**
  - typed callables with `Args:` docstrings (schema built by `transformers.get_json_schema`), or
  - `BaseTool` subclasses with `name`, `description`, `_parameters()` and `execute(**kw) -> ToolResult`.
- **Per-episode state:** none; the rollout sees only the prompt.
  - `create_agentic_trainer` does not forward `pre_step_hook`.
  - Workaround: pass your own `rollout_func=create_rollout_fn(pre_step_hook=bind_scenario, on_trajectory_end=trajectory_writer(path))`. Episodes are serial, so a module-level "current scenario" variable is safe.
- **Simulated user:** none. A turn with no tool call ends the episode. Add an `ask_member(question)` tool that returns a scripted reply to continue.
- **Reward:** the TRL batch contract, `fn(completions, **kw)`.
  - `kw` holds every dataset column plus `trajectories`, `conversations` and `tool_call_counts`.
  - `agenttune.utils.score_logger.log_score(name, score, reasons, components, meta)` appends a "why" trail to `$AGENTTUNE_SCORE_LOG`.
- **Rollout limits (trainer path, `rollout_factory.py`):**
  - It generates one sequence at a time.
  - Each turn is capped at `max_new_tokens=getattr(trainer,"_max_new_tokens",256)` with T=0.7 (`:2049`). GRPO's own length and temperature settings are ignored here.
  - `completion_ids` are the generated tokens plus raw `str(result)` tokens (`:2377`), **with no chat-template tokens between turns**. Turns 2 onwards are optimised on a context the model never sampled from.
  - When over budget, truncation drops the oldest tool pairs, which can delete our buried fact.
- **Output:**
  - With `tools`: `trajectories.jsonl` (one `Trajectory.to_dict()` per rollout, `reward: null`) plus `lexsi_provenance.json`.
  - Without `tools`: no trajectory file.

### 3.4 EventLog and distillation (VERIFIED: `agentic/events.py`, `distill_trainer.py`)
- **Schema:** `Event{kind ∈ text|reasoning|tool_call|tool_result|observation|turn_complete|reward(scope)|memory_op, payload, token_span?, logprobs?}` inside `EventLog{events, tier light|full, id}`. There is no JSON serialiser.
- **An offline teacher works:** `create_distill_trainer(student, trajectories=[EventLog…])` accepts our own teacher logs. The gaps:
  - `as_dataset_rows` has **no system role**.
  - `TOOL_CALL` events become `<tool_call>{json}</tool_call>`, which is not the fallback format.
  - `segment_weights` and `loss_mask` are never read. The built-in factory trains `SFTTrainer` on `messages`, which means **loss on every token**: TRL 1.13 applies completion-only loss only to prompt-completion data (VERIFIED: `trl/trainer/sft_config.py@v1.13.0`).
  - `EventLog.from_trajectory` duplicates each call and emits `<tool_call>{}</tool_call>` for the final step.
- **Our workaround:** build logs only from OBSERVATION and TEXT events, written in our protocol, and pass a custom `trainer_factory`.

### 3.5 DECIDE, self-healing, synthesis, AuditKIT (VERIFIED)
- **DECIDE:**
  - YAML stages `llm_call`, `rules`, `router`, `parallel`, `llm_judge`, `tool_call`, `human_review` (a prototype) and `output` compile to LangGraph with an audit log. There are 12 BFSI templates.
  - Our ask/act policy fits as extract → lookup → `rules` → `output`, but as a hand-coded pipeline it serves only as a baseline.
  - `AuditReader.extract_dpo_pairs` always returns `[]` (known-issues).
- **Self-healing:** it detects `loop_collapse`, `tool_crash` and `low_judge_score`, fixes only `wrong_tool` and `loop_collapse` (via litellm), and cannot detect "acted when it should have asked".
- **Synthesis:**
  - The RAG pipeline turns documents into graded QA.
  - The undocumented `agenttune.scenarios.generate_scenarios(tools)` turns tool schemas into `{task, difficulty}` user requests, through a local model or any OpenAI-compatible endpoint.
  - **None of the three repos has a persona, profile or transaction-history generator.**
  - Standard alternatives (VERIFIED, HF API): `nvidia/Nemotron-Personas-USA` (CC-BY-4.0) and `proj-persona/PersonaHub` (CC-BY-NC-SA-4.0).
- **AuditKIT:**
  - `episodes_from_agenttune(run_dir)` reads every `*.jsonl` plus the provenance file.
  - `episode_from_eventlog(log)` duck-types EventLog objects (`AuditKIT/src/auditkit/agent_eval/importers.py:515,572`).

## 4. AlignTune
- **API (VERIFIED, `core/backend_factory.py`):** `create_sft_trainer`, `create_rl_trainer(algorithm=…)` and `create_distill_trainer`.
  - **Algorithms:** SFT, DPO, Online-DPO, PPO, GRPO, GSPO, DAPO, Dr.GRPO, GBMPO, Counterfactual-GRPO, PACE, ORPO, SPIN, RAFT, KD/SDFT.
  - **Distillation needs a local teacher's logits**, so it cannot distil from Claude transcripts.
  - **LoRA:** `use_peft` and `lora_*`. **QLoRA:** `quantization={"load_in_4bit":True}`.
- **Defaults to override:**
  - `backend="auto"` prefers Unsloth.
  - SFT defaults to `use_peft=False` and `max_seq_length=512`.
  - RL LoRA targets only q/k/v/o.
  - `create_rl_trainer` defaults to `learning_rate=2e-4` (far too high for GRPO), `num_epochs=3`, `beta=0.1` and `num_generations=batch_size`.
  - Importing aligntune patches TRL's `is_vllm_available` to `False` unless `ALIGNTUNE_ENABLE_VLLM=1` (`__init__.py:40`).
- **Cohere support:** it registers `Cohere2ForSequenceClassification` (`core/hf_compat.py`).
- **Rewards:** **52 registered names** (`rewards/registry.py`), against the docs landing page and pyproject's "27+", the guide's "50+" and the paper's "43".
  - Usable as small terms: `json_schema_verifiable`, `regex_verifiable`, `length`/`brevity`.
  - `rubric` is an LLM judge; keep it out of the RL loop.
  - The rest do not apply: math, code, SQL, BLEU, classifiers, keyword heuristics (e.g. `financial_accuracy` pays for the words "cpa" and "audited").
- **Data (CAVEATED, code reading):**
  - SFT drops `tool`-role turns (`data/processors.py:59`).
  - It stringifies conversational `prompt` lists (line 117).
  - It trains the full sequence when `messages` exists.
  - GRPO keeps extra columns for rewards.
- **Multi-stage pipelines:** `aligntune compose run` is a **placeholder**. `_create_trainer` returns `None`, `_run_trainer` returns hard-coded `{'final_loss': 0.5, 'training_steps': 1000,…}`, and every stage reports ✅ (`core/composition/runner.py:311–350`). Chain stages by hand: SFT → merge → pass the merged path as `model_name`.
- **Alignment audit** (`eval/alignment_auditor.py`):
  - **Run it:** `AlignmentAuditor(judge=None).score(adapter, probe_set=json.load(open(".../recipes/probe_sets/default_alignment_probes.json")))`. The file holds 18 refusal, 12 sycophancy, 20 verbosity and 12 reward-hacking probes.
  - **Compare:** `AlignmentDriftTracker.set_baseline(base)` then `check_drift(tuned)` returns `*_delta` values and warns above 0.1.
  - **Output:** an `AuditReport` with `reward_hacking`, `sycophancy`, `refusal_collapse`, `verbosity_gain`, `avg_response_tokens`, `repetition_ratio`, `unique_token_ratio` and `degenerate_rate`.
  - **What the metrics really are:**
    - "reward hacking" = `0.5·trigram-repetition + 0.3·(1−unique-token ratio) + 0.2·degenerate-length`, a degeneracy score;
    - sycophancy = the keyword-affirmation gap between biased and neutral prompts;
    - refusal collapse = the keyword-refusal rate.
  - **Adapter flaws:** both adapters send raw strings with **no chat template**, and the vLLM adapter maps T=0 to T=1.0 (`model_adapters.py:275`).
  - **One-command alternative:** `aligntune verify-export run <base> -a tuned:hf:<dir> --probe-set … --eval-config …`.
- **Utility advisor** (`core/advisor.py`): **no GB10 profile** (unknown GPUs fall back to A100-40GB).
  - `infer_model_size` matches digit substrings. My local re-implementation gives `Qwen2.5-7B` → 2B (contradicting its own docstring), `aya-expanse-32b` → 2B, and `tiny-aya-global` → assumed 7B, so GRPO at seq 4096 → 973 GB.
  - Unusable as ROI evidence.
- **Hub:** `trainer.push_to_hub(repo_id, private=False…)`, GGUF and Ollama exporters; Hub-upload fixes are only in the Unreleased changelog.
- **Paper** (arXiv 2602.09621v2, VERIFIED): "43 built-in reward functions"; no audit features, which came later.

## 5. SafeTune
- **Measuring drift** (VERIFIED: `docs/user-guide/audit-your-model.md`, `instrumentation/evaluate/suite/evaluate.py`):
  - **Call:** `evaluate(model, tokenizer=tok, benchmarks=["harmbench","xstest"], judge="wildguard", max_prompts=…)` on the base model and each tuned model.
  - **Returns, per benchmark:** `asr`, `refusal_rate`, `harmfulness_score`, `n_total`, `n_evaluated` and `headline_metric`. For XSTest the headline is over-refusal on its 250 safe prompts.
  - **Default suite:** HarmBench (400), WildJailbreak (500), AdvBench, SORRY-Bench, HEx-PHI, OR-Bench and AILuminate.
  - **Judges:** `allenai/wildguard` (7.25B, gated "auto") or HarmBench-Mistral-7B (ungated) (VERIFIED, HF API).
  - **Prompting:** a chat template with a single user turn; there is no system-prompt hook. `max_prompts` keeps the *first* k rows.
  - **Cost (CAVEATED):** about 850 generations of up to 512 tokens plus about 15 GB for the judge, roughly 0.5–1 GPU-hour per model with HF generation. `run_safety_eval(backend="vllm")` is faster.
- **Harden:** 26 trainers (SafeGrad, Lisa, Vaccine…; `docs/user-guide/harden.md`) that **replace** the SFT loop (LoRA, merge, save). They do not plug into AlignTune, AgentTune or GRPO. Raw-row tokenisation defaults to ≤2048 tokens, so pass an explicit `max_len` for our prompts.
- **Recover:** ReSta is documented on Tiny Aya with α≈0.25. The README notes Tiny Aya's chat preamble is about 366 tokens.

## 6. Compatibility with our stack (torch 2.13, transformers 5.17, TRL 1.13, vLLM 0.29, aarch64)
- **Pins:** a plain `pip install agenttune` would downgrade torch and TRL, and `aligntune` pins TRL 1.7.1. SafeTune accepts our stack.
- **TRL 1.13 differences (VERIFIED, GitHub trees):**
  - `trl/experimental/openenv` is gone. AgentTune imports it for tools with `use_vllm=True` (`rollout_factory.py:1963`), so that combination fails.
  - `trl/experimental/ppo` is gone too.
  - The no-tools GRPO path, `rollout_func`, `trl.experimental.bco` and `unwrap_model_for_generation` all still exist.
  - TRL 1.13 *warns* about vLLM outside 0.19.1–0.28.0. Our previous project ran TRL 1.13 with vLLM colocate on this Spark (CAVEATED).
- **Model facts (VERIFIED, HF API and model card):**
  - Tiny Aya Global is `Cohere2ForCausalLM` with 3,349,227,520 BF16 parameters, about 6.7 GB.
  - **Its context is 8K.**
  - peft has no default LoRA targets for cohere2. AgentTune sets `all-linear` itself; AlignTune's RL default targets q/k/v/o.
- **Memory (CAVEATED):** LoRA SFT at ≤8K tokens needs about 15–25 GB; GRPO (G=8, vLLM colocate 0.25) about 45–60 GB. Both fit in about 111 GB.

## 7. Recommended split and glue
| Method | Library and entry point |
|---|---|
| **M1 SFT/distillation** of Claude-pure + seed `ideal_message` demonstrations | AgentTune `create_distill_trainer(student, trajectories=logs, trainer_factory=WtmSFT(systems))`. LoRA all-linear r=16, loss only on the decision turn. |
| **M2 GRPO**, programmatic reward | AgentTune `create_agentic_trainer("grpo", backend="trl")`, no `tools` (batched, vLLM OK). Prompt = conversation up to `decision_point_index`. Start from merged M1, plus an ablation from base. |
| M3 DPO (optional) | AlignTune `create_rl_trainer(algorithm="dpo", backend="trl")` on twin pairs. |
| Stretch: live-tool GRPO | AgentTune with a custom `rollout_func` (`pre_step_hook` + `ask_member`), `use_vllm=False`. Time 3 steps first. |

```python
# M2 sketch (CAVEATED)
from agenttune.core.backend_factory import create_agentic_trainer
from agenttune.agentic.rollout_engines.tool_call_parse import extract_tool_calls_from_text
from agenttune.utils.score_logger import log_score
def wtm_reward(completions, expected_move, discrepancy_terms, scenario_id, **kw):
    out = []
    for c, mv, terms, sid in zip(completions, expected_move, discrepancy_terms, scenario_id):
        text = c[-1]["content"] if isinstance(c, list) else c
        r, parts = grade(text, extract_tool_calls_from_text(text) or [], mv, terms)   # our grader
        log_score("wtm.decision", r, components=parts, meta={"scenario_id": sid}); out.append(r)
    return out
trainer = create_agentic_trainer("grpo", backend="trl", model=SFT_MERGED, train_dataset=ds, reward_funcs=[wtm_reward],
    peft_config=dict(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"), num_generations=8,
    max_completion_length=192, beta=0.02, use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.25)
```
**Glue to write:**
1. `wtm/protocol.py`: one text tool protocol shared by eval, SFT, GRPO and the Claude arm.
2. `wtm/render.py`: scenario → messages, cut at the decision point, with tool results folded in.
3. `wtm/reward.py`: the reward function.
4. `training/sft_factory.py`: EventLog rows → conversational prompt + completion pairs.
5. `training/eventlogs.py`: teacher transcripts → logs, plus JSON for AuditKIT.
6. `audit/chat_adapter.py`: an AlignTune `ModelAdapter` with `generate`, `close` and `backend_name` that applies the chat template.

## 8. Doc gaps and bugs (feedback material)
1. **AlignTune `compose` is a stub** with fabricated metrics (§4).
2. **AlignTune's in-training audit never runs:**
   - `AlignmentAuditCallback` reads `kwargs["tokenizer"]`, but transformers 5.x passes `processing_class` (VERIFIED, `trainer_callback.py@v5.17.0`);
   - its default `eval/probes/*.jsonl` files do not exist;
   - custom probes sit under a `"custom"` key that `score()` ignores.
3. **AgentTune multi-turn GRPO:** missing tokens between turns, a hidden 256-token/T=0.7 per-turn setting, serial generation, and a removed TRL import (§3.3, §6).
4. **AgentTune distillation:** no system role, ignored loss masks, no Cohere fallback on API engines (§3.2, §3.4).
5. **`backend="auto"` → Unsloth** in both libraries; AlignTune's vendored `unsloth` flips AgentTune too.
6. **Audit and advisor validity:** raw prompts with no chat template (SafeTune's own comment says this "systematically distort[s] ASR"), keyword metrics, the size bug.
7. **Agent tooling:** reward counts disagree; no `llms.txt` for AgentTune or AlignTune; no Claude Code skill or MCP server for any of the three.

## Implications for What They Meant
1. **Install** in `lx-exp`:
   - Check `python3 --version` is ≥3.12.
   - Run `pip install --no-deps -e third_party/{AgentTune,aligntune,SafeTune}` and add pure-Python dependencies until all three import.
   - Export `PURE_TRL_MODE=1`, always pass `backend="trl"`, and run each library in its own process.
2. **Freeze one protocol** (AgentTune's Cohere JSON list plus folded tool results) for every model arm.
   - Put compact tool docs in our system prompt.
   - Pass our own `tools_fallback_prompt` (no `{tools}` placeholder) and `tool_result_format`, so training prompts equal eval prompts.
3. **Respect the 8K window:**
   - Assert ≤7K prompt tokens before every run, so truncation never silently deletes the buried fact.
   - Report that realism had to fit in 8K.
4. **Train M1 then M2:**
   - Mix A/B twins in every GRPO batch.
   - Log reward components through `AGENTTUNE_SCORE_LOG`.
   - Evaluate the *saved* adapters.
   - Add DPO via AlignTune only if time allows.
   - Training still waits for the user's dataset approval.
5. **Audit base, M1 and M2:**
   - AuditKIT on our eval trajectories;
   - SafeTune HarmBench + XSTest, about 1.5–3 GPU-h for three models; the user must accept the WildGuard and Aya licences;
   - AlignTune `AlignmentAuditor` through our chat adapter, labelled as heuristic.
   - Measure pushback sycophancy (the P variants) with our own grader.
6. **ROI:** use measured tokens/s and GPU-hours. Show `aligntune advise` output only as a contrast.
7. **Personas and histories:** generate them with our own seeded generator, optionally seeding names from Nemotron-Personas-USA.
8. **`lexsi-feedback.md` candidates:** (a) the `compose` stub plus the silent audit callback; (b) the multi-turn token mismatch plus TRL-module drift; (c) no Claude Code skill or `llms.txt`.
9. **Publishing:** Lexsi `push_*` helpers default to `private=False`. Pass `private=True` until the user approves.
