# CuratorKIT + AuditKIT: hands-on study for "What They Meant"

*2026-11-12. I cloned CuratorKIT @ `ed45ae3` and AuditKIT @ `556eb48` into `third_party/`, then read their docs, examples and source. I also fetched the live docs sites, PyPI, the GitHub and HF APIs, and arXiv 2606.21631. Per project rules, **nothing cloned was executed**: behaviour claims come from reading the code ("code") or a template. CK = `https://github.com/Lexsi-Labs/CuratorKIT/blob/main/`, AK = `https://github.com/Lexsi-Labs/AuditKIT/blob/main/`. A tag in a heading, or closing a group of bullets or a table, covers every line in that group.*

## TL;DR
1. **CuratorKIT has no tool-call or agent-trajectory support.** Its exporters keep only `role`/`content`. **AuditKIT has strong agent-eval support**: tool-call metrics, a documented `ask_clarification` route example, recorded episodes and an AgentTune importer.
2. Several CuratorKIT defaults would **silently corrupt minimal-pair agent data**: `strip_html`, a dedup key without the system prompt, MinHash or diversity filters merging twins, PII pseudonymisation rewriting near-miss names, and a random per-sample split.
3. **The Aya chat templates have no tool role.** A `tool` message raises an error or disappears. We must use only system/user/assistant, with tool calls written as `<tool_call>{json}</tool_call>` text.
4. **Never install `auditkit[vllm]` or `auditkit[all]` on the Spark.** They force `vllm>=0.30`, and the newest vLLM pins torch 2.14.
5. **Lexsi publishes no Claude skills, MCP servers or plugins.** The only agent-facing files are SafeTune's `llms.txt` and a stale AuditKIT `AGENTS.md`.

## 1. Versions and licences
- **CuratorKIT:** PyPI `curatorkit` **0.1.27** (2026-11-12), Python ≥ 3.11, licence LSAL v1.1 (non-commercial). `main` says `__version__="0.1.21"`. The 0.1.27 release tag `7c72ee1` is an unmerged child of `ed45ae3` that changes only the version string, so the code we read is the code pip installs. **VERIFIED** https://pypi.org/pypi/curatorkit/json and git history.
- **AuditKIT:** PyPI `auditkit` **1.1.11** (2026-11-12), the same as HEAD. Python ≥ 3.10, licence LSAL v1.2. **VERIFIED** https://pypi.org/pypi/auditkit/json

## 2. Claude skills, MCP and llms.txt across Lexsi (for our feedback to Lexsi)
- I listed the trees of all **17 public repos** in the `Lexsi-Labs` GitHub org via the API. Result: **0** `SKILL.md`, `.claude/`, `.mcp.json`, MCP servers or plugins. **VERIFIED.**
- Only two agent-facing files exist. Both are saved as files only in `third_party/lexsi-skills/`:
  - `AuditKIT/AGENTS.md`: contributor notes, stale ("release v1.0.0", "9 backends").
  - `SafeTune/docs/llms.txt`: a 43-line index of doc links, served at `safetune.lexsi.ai/llms.txt`.
- `/llms.txt` and `/llms-full.txt` return **404** on the curatorkit, auditkit, agenttune, aligntune and circuitkit `.lexsi.ai` sites. **VERIFIED.**
- The HF orgs have no Spaces, and a web search found no Lexsi MCP server or skill. **VERIFIED** (negative result).

## 3. CuratorKIT

### 3.1 Install and backends
- **Core install:** `pip install curatorkit` (pydantic, typer, tqdm, pyyaml).
- **Extras:**
  - `hf`: datasets
  - `generation`: litellm ≥ 1.60
  - `embedding`: sentence-transformers (+ faiss)
  - `hygiene`: presidio, spacy, detoxify, detect-secrets
  - `pdf`: MinerU, which is AGPL
  - `trl`: tests only; requires transformers ≥ 5.15,<6
  - `all`: everything except `pdf` and `trl`

  **VERIFIED** CK:pyproject.toml
- `SecretsGate` needs only `detect-secrets`, because the hygiene modules import lazily. **VERIFIED** (code).
- **LLM calls** go through LiteLLM or Ollama, with a separate judge model and per-role overrides. **VERIFIED** CK:docs/reference/configuration.md
- **A local OpenAI-compatible endpoint is documented** (CK:docs/reference/cli.md):
  ```yaml
  llm: {model: openai/Qwen/Qwen3-8B, api_base: http://localhost:8000/v1, api_key: token-abc123}
  ```
- The HF router should work the same way: `openai/CohereLabs/aya-expanse-32b:cohere` with `api_base=https://router.huggingface.co/v1` and the HF token. **CAVEATED**: not run.
- Judges **fail closed**: a failed or unparseable judge call becomes `judge_error:<Type>`.
- **There is no LLM response cache.** **VERIFIED** (code: CK:curatorkit/gates/_judge.py, CK:curatorkit/llm/base.py).
- A custom `BaseLLM` needs only `_call(messages)`. It can be wired into gates through the `Pipeline` API, not through `CuratorConfig`. **VERIFIED** (code: CK:curatorkit/llm/base.py).

### 3.2 Fixed stage order (the built-in selection logic for our flowchart)
**VERIFIED** (code, CK:curatorkit/curator.py `_build_steps`):

Reader (format detection) → `SchemaGate` → `ExactDeduplicator` (+ `MinHash`) → `TextCleaner` → `SecretsGate` → `PIIPseudonymizer` → `ToxicityGate` → generation → `HallucinationGate` (+ probe) → `RewardGate` (+ probe and refiner; DPO dual-scoring) → `DiversityGate` → cross-run `EmbeddingDeduplicator` → `StratifiedSampler` → `max_samples` → exporters (each applies a task-type gate).

Every run writes `manifest.json` (per-stage counts), `rejected.jsonl` (reason strings, e.g. `dpo_pair_failed:rejected_above_threshold:0.8`), `dataset_card.md`, `README.md` (so `load_dataset(dir,"dpo")` works), `lexsi_provenance.json` and `checksums.txt`. Caveats (**VERIFIED**, code: CK:curatorkit/pipeline.py, CK:curatorkit/curator.py):
- Dedup and other normalizer removals appear **only as counts**.
- Exports **drop ids and metadata**, so kept rows have no per-row lineage.
- `checksums.txt` skips the split sub-folders.
- The run hash covers only 8 config fields.

### 3.3 Ingesting our JSONL: gotchas (all **VERIFIED**, code)
| Gotcha | Consequence |
|---|---|
| Columns are detected by alias: `turns`/`history` count as conversation, `system_prompt` as system, and the alias is picked by iterating a frozenset (string-hash order varies per process). CK:curatorkit/detection/detector.py | Our seed rows have `turns: 8` and `system_prompt: {dict}`. Detection could be **nondeterministic**, or the dict could be stored as its Python repr. |
| `preprocessing_fn` may return a **`DataSample`**, which skips detection. CK:curatorkit/connectors/base.py; undocumented. | This is the robust ingest path. |
| `strip_html` (on by default) deletes `<[^>]+>`, but only in instruction/input/output. CK:curatorkit/normalizers/clean.py | It would strip `<tool_call>` in the first exchange only, and mangle text like "<$500 … >". |
| The messages exporter writes only `{role, content}`. CK:curatorkit/exporters/messages.py | `tool_calls`, `tool_call_id` and `name` are dropped, so tool calls must be text. |
| The `preference` reader runs `str()` on a list-valued prompt. | Store the prompt as `json.dumps(list)`. The DPO exporter then re-parses it into a turn list. |
| In YAML, the `schema` gate defaults to `required_fields: [instruction, output]`. CK:curatorkit/config.py | DPO rows would fail with `missing_field:output`. Use the Python API. |

Our recipe, built from documented API calls:
```python
import json; from curatorkit import Curator, CuratorConfig, DataSample
def to_pair(r):   # one decision point: gold move vs observed failure
    return DataSample(source_uri=f"wtm://{r['scenario_id']}", task_type="preference",
        instruction=json.dumps(r["prompt_msgs"]),   # system/user/assistant only
        chosen=json.dumps([{"role": "assistant", "content": r["gold"]}]),
        rejected=json.dumps([{"role": "assistant", "content": r["failure"]}]),
        input=r["evidence"],                         # read by HallucinationGate, not exported
        metadata={"wtm_scenario": r["scenario_id"], "wtm_twin": r["twin"]})
res = Curator(CuratorConfig(dataset="data/rendered/pairs.jsonl", preprocessing_fn=to_pair,
        clean_transforms={"strip_html": False}, max_tokens=32768, dedup="exact",
        secrets_gate=True, export_formats=["dpo"], output_dir="dataset/curated/dpo")).run()
```

### 3.4 Which stages fit agent trajectories (**VERIFIED**, code)
| Stage | Use? | Why |
|---|---|---|
| Exact dedup | Yes, plus our own check | The conversational key is instruction + output + later turns, with **no system prompt**, so variants that differ only in member history get merged. |
| MinHash (0.85), `DiversityGate` (0.92), embedding dedup | **No, not across twins** | `DiversityGate` embeds only instruction + first output, so twin A and twin B score cosine 1.0. MiniLM also truncates long contexts (**CAVEATED**). |
| `SecretsGate` | Yes, but review every reject | The paper's 184-sample test (four batches, one of them a clean control, combined) had SecretsGate reject **all 184, so the clean control batch was rejected too** (**VERIFIED** https://arxiv.org/html/2606.21631, notebook 08 passage; section number not re-checked). Measure false positives on a planted clean set. |
| `PIIPseudonymizer` | **No** | It maps per sample and per exact string, so `Jon Smith` and `John Smith` get unrelated fakes. That destroys the near-miss signal. Our data is synthetic already. |
| `ToxicityGate` | Audit only | It may flag scam-pressure language. |
| `HallucinationGate` | Yes, repurposed | Checks that a twin-A question cites facts present in `input` (the evidence). A custom template must emit `grounding_score`. |
| `RewardGate` | Yes, for DPO | Dual scoring requires chosen ≥ t **and** rejected < t, a built-in contrast filter. A template must emit **`overall_score`**. Conversational rows are judged on the first exchange only. |
| `output_split` | **No** | It shuffles samples individually, so twins leak across splits. Split by `source_post_id` ourselves, then call `curatorkit.exporters.EXPORTERS[fmt]().export(...)` per split. |

### 3.5 Export schemas (**VERIFIED**, CK:curatorkit/exporters/)
| File | Row | Accepted task types |
|---|---|---|
| `sft_alpaca.jsonl` | `{"instruction","input","output"}` | instruction_following, unpaired_preference |
| `sft_sharegpt.jsonl` | `{"conversations":[{"from":"system"\|"human"\|"gpt","value"}]}` | + conversational |
| `sft_messages.jsonl` | `{"messages":[{"role","content"}]}`, i.e. system + user(+input) + assistant + `metadata.turns` | same |
| `dpo.jsonl` | `{"prompt","chosen","rejected"}`; each a string, or a `{role,content}` list when stored as a JSON list | preference, implicit_preference |
| `grpo.jsonl` | `{"prompt": str, "responses", "rewards", "reference"?}`; the prompt is never a message list | grpo |

## 4. AuditKIT

### 4.1 Install
- The core has zero dependencies.
- Our set: `pip install "auditkit[requests,interop]"`.
- `[transformers]` requires transformers ≥ 5.15,<5.22 (ours is 5.17). `[lmeval]` adds lm-eval.
- `[vllm]` requires `vllm>=0.30`, and `[all]` includes it. **VERIFIED** AK:pyproject.toml

### 4.2 Our dataset with our ask-vs-act grader (**VERIFIED**: AK:docs/agents_and_rag.md, AK:src/auditkit/loaders.py)
- `ak.load_jsonl` reads `input`, `tools`, `expected_tool_calls`, `actual_output`, `actual_trace` and `metadata`.
- `ToolCallAdapter` sends **`Sample.metadata["messages"]`** (the full system prompt and history). `mode="prompt"` asks for `<tool_call>` blocks. **VERIFIED** AK:src/auditkit/adapter.py
- The docs already treat asking as a valid route (AK:docs/agents_and_rag.md):
  ```python
  ASK = {"name": "ask_clarification", "arguments": {"question": "Which Paris booking?"}}
  expected_tool_calls = {"any_of": [[], [[ASK]]]}
  ```
- Custom metric, following AK:docs/examples/custom_scorer.md:
  ```python
  # from auditkit.metric import Metric; from auditkit.types import ScoreKind; COMMIT = {"show_approval", ...}
  class AskActCorrect(Metric):
      direction, name = ak.Direction.MAXIMIZE, "ask_act_correct"
      def identity(self): return {"name": self.name, "v": 1}
      def score(self, sample, output, context=None):
          acted = any(c.name in COMMIT for c in ak.parse_tool_calls(output))
          ok = acted if sample.metadata["twin"] == "B" else not acted
          return ak.Score(name=self.name, value=float(ok), kind=ScoreKind.CODE)
  ```
- Score recorded runs with `model="precomputed"`.

### 4.3 Models (**VERIFIED**: AK:docs/model_backends.md)
- Specs: `hf:`, `vllm:`, `api:` (OpenAI-compatible: our `vllm serve`, or `https://router.huggingface.co/v1`), `openai:`, `anthropic:`, `litellm:`, `agent:` (HTTP), a callable, or `precomputed`.
- AuditKIT itself states that **no Cohere template renders tools; use `ToolCallAdapter(mode="prompt")`**. **VERIFIED** AK:docs/known_issues.md
- Always pass `api_key`. The code falls back only to the `API_KEY` environment variable (AK:src/auditkit/model/api_gen.py). AK:docs/SGLANG.md still describes an older `OPENAI_API_KEY` fallback.

### 4.4 LLM judge (**VERIFIED**, code)
- `judge_model` is either a spec string plus `judge_model_args={"api_base","api_key"}` (so a local vLLM judge works), or **any object with `.generate()`**. **VERIFIED** AK:src/auditkit/metrics/judge.py
- `LLMJudge` sends one flattened prompt, so `ak.model.CallableModel(claude_pure_fn, name="claude-opus-5-5-pure-v1")` satisfies our Claude-only rule.
- `choices={"names_discrepancy":1,"generic":0}` gives classifier mode, and templates can read any `metadata` key.
- A reply that cannot be parsed is **excluded and logged in `RunResult.errors`**, never scored as 0.

### 4.5 Benchmarks and safety
- **Native:** MMLU, GSM8K, ARC-Challenge, HellaSwag, TruthfulQA and HumanEval (no execution).
- **IFEval and multilingual tasks:** only via `ak.run_lmeval` (`api:` maps to lm-eval's `local-completions` with `base_url`). Task names are **CAVEATED**.
- **BFCL v3:** `ak.load_bfcl` covers single-turn and irrelevance. **Multi-turn and missing-parameter categories are refused.**
- **Safety:** `auditkit redteam` has 4 small probe sets (5–8 prompts each), plus `GuardJudge` (Llama Guard and others), `ToxicityScore` and `BiasJudge`. **VERIFIED** AK:docs/scenarios.md, AK:src/auditkit/bfcl.py

### 4.6 Fingerprint, cache, compare
- **Fingerprint:** sha256[:16] over model identity (name + checkpoint + `api_base`), content-hashed samples, adapter, metrics, `RunConfig` and AuditKIT version. Results are cached in `~/.cache/auditkit/runs/` with `cache=True` as the default. **VERIFIED** AK:src/auditkit/runspec.py
- **Stale-cache trap:** a LoRA swapped behind the same served name gets the cached result. Use unique names or `cache=False`. **VERIFIED** (code: AK:src/auditkit/model/__init__.py `identity`)
- **Comparing runs:** `ak.compare(base, tuned)` gives deltas, retention, a paired bootstrap **aligned by `sample_id`**, and a regressed/improved browser. **VERIFIED** AK:src/auditkit/comparison.py
- The minimum reportable p is about 0.001, so use McNemar's test for 0/1 outcomes. **VERIFIED** AK:docs/known_issues.md

### 4.7 "Reads an AgentTune run directly"
- `ak.load_agenttune` detects run_eval rows, TrajectoryDataset JSONL, `trace.jsonl` and report JSON. `episode_from_eventlog`, `auditkit agent import-agenttune --inspect`, verified-outcome oracles (`FinalStateAssertion`, `CustomPredicate(fn, version=)`) and `write_sidecar` are also available. **VERIFIED** AK:docs/agent_eval.md
- **Caveat 1:** the bridge is pinned to AgentTune commit `36d4724`, which is **absent from the public history** ("not our ref"; the public repo starts on 2026-10-01). The CuratorKIT importer is pinned 49 commits behind HEAD. **VERIFIED** (git).
- **Caveat 2:** harness mode starts with a single user turn and **no system prompt**. **VERIFIED** AK:src/auditkit/agent_eval/runner.py

## 5. Compatibility risks on the Spark (aarch64, CUDA 13, torch 2.13, transformers 5.17, TRL 1.13, vLLM 0.29)
| Risk | Evidence |
|---|---|
| `auditkit[vllm]` or `[all]` replaces vLLM 0.29. The latest vLLM, 0.33.0, pins `torch==2.14.0`. AuditKIT's Cohere matrix was measured on vLLM 0.30.0. | **VERIFIED** https://pypi.org/pypi/vllm/json, AK:docs/model_backends.md |
| `curatorkit[generation]` pulls litellm 1.109.1 (`openai>=2.20,<4`, `tokenizers<1`). | **VERIFIED** PyPI. Use a separate venv. |
| `[hygiene]` and `[embedding]` pull spacy, detoxify and sentence-transformers, which could disturb torch. | **CAVEATED**. Skip them. |
| **Aya templates** (Tiny Aya ×4, Aya Expanse 8B/32B): no tools and strict user/assistant alternation. In Tiny Aya a `tool` turn renders empty (its content is lost), and the sequence assistant→tool→assistant raises "roles must alternate". In Expanse a `tool` turn raises or is dropped, depending on its position. Tiny Aya also prepends a fixed safety preamble and nests our prompt as "Developer Preamble". | **VERIFIED** (template text) https://huggingface.co/api/models/CohereLabs/tiny-aya-global, identical to AK:tests/fixtures/chat_templates/. Not executed. |

## 6. Doc bugs and gaps (for `lexsi-feedback.md`)
1. **Wrong reward key in the docs.** The live customisation guide says a reward template must return `"score"`, but `RewardGate` parses only `overall_score`. Following the docs, every sample would fail as `judge_error`.
2. **No tool or agent data path in CuratorKIT.** The paper (arXiv 2606.21631) never mentions tool calling (its only "tool" hits are generic "existing tools"/"toolkit"), and kept rows carry no lineage.
3. **AuditKIT docs site is stale.** auditkit.lexsi.ai shows **v1.0.0**, and its `agent_eval/`, `agents_and_rag/` and `lexsi_integration/` pages return 404.
4. **The `[vllm]` extra forces a vLLM upgrade.**
5. **Version drift.** CuratorKIT's repo says 0.1.21, its changelog says 0.1.20, the docs badge says v1.0, and PyPI has 0.1.27.
6. **Unsafe default.** AuditKIT's `push_to_hub` defaults to `private=False`; CuratorKIT's defaults to private.
7. **Agent support is minimal.** There are no skills, MCP servers or `llms.txt` (except SafeTune's), and `AGENTS.md` is stale.

## Implications for What They Meant
1. **Tool protocol.**
   - Use only system/user/assistant roles.
   - Write tool calls as `<tool_call>{"name","arguments"}</tool_call>` in assistant turns. This is AuditKIT's prompt format, so its parser and metrics work unchanged.
   - Write tool results as user turns: `<tool_result name=…>`.
   - Make the approval widget a `show_approval` call.
   - Render one sample per Aya template before training.
2. **CuratorKIT through the Python API** (§3.3 recipe):
   - Use `strip_html: False`, `max_tokens=32768`, `dedup="exact"` and `secrets_gate=True`.
   - Export `messages` (SFT) and `dpo` (conversational pairs).
   - Do **not** use MinHash, diversity, embedding dedup, PII pseudonymisation or `output_split`.
   - Add our own full-content dedup check (system prompt + turns).
3. **Gates on decision-point views.**
   - `HallucinationGate` checks that twin-A questions are grounded.
   - `RewardGate` dual-scores DPO pairs with an ask-vs-act rubric that emits `overall_score`; its contrast filter goes in the flowchart.
   - Provide the Claude-only judge as a custom `BaseLLM`, or as one localhost OpenAI-compatible shim around `claude_pure.py` that serves both CuratorKIT (`openai/…` + `judge_llm_api_base`) and AuditKIT (`api:`). Cache inside the shim.
4. **Flowchart and provenance.**
   - Cite §3.2 and use the `manifest.json` stage counts and `rejected.jsonl` reasons for the box numbers.
   - Dump `result.passed` with its `provenance_chain` to `dataset/curated/passed.jsonl`.
   - Split by `source_post_id` ourselves.
5. **Spark install.** `pip install "auditkit[requests,interop]==1.1.11"` plus `lm-eval`, under a constraints file pinning torch 2.13, transformers 5.17, vLLM 0.29 and TRL 1.13. Run `--dry-run` first. Never install `[vllm]` or `[all]`.
6. **Evaluation flow.**
   - Run every arm (Aya via HF router or vLLM, other open models, Claude-pure) through our harness.
   - Score transcripts with `model="precomputed"`.
   - Metrics: `AskActCorrect` + `ToolCallF1(arg_mode="subset")` with `any_of` routes + `ToolPermission` + `LLMJudge` (names the discrepancy).
   - Skip harness mode: it has no system prompt.
7. **Realism gap.** Give the toy and realistic versions the same `sample_id`, then run `ak.compare(toy_run, real_run)` to get paired deltas and a list of decisions passed as toy and failed as realistic. Confirm with McNemar's test.
8. **Capability retention before and after training.** GSM8K/MMLU, IFEval and multilingual tasks via `run_lmeval`; BFCL single-turn and irrelevance via `load_bfcl` in prompt mode; `auditkit redteam` as a smoke test.
9. **Caching.** Serve each checkpoint under a unique name (e.g. `wtm-aya-dpo-s500-<sha8>`), and set `XDG_CACHE_HOME=experiments/<name>/cache`.
10. **AgentTune.** Run `auditkit agent import-agenttune --inspect` on AgentTune output first (the importer is pinned to an old commit). Use `CustomPredicate(version=…)` oracles and `write_sidecar`.
11. **Publishing.** Never call AuditKIT's `push_to_hub` before the user approves: it defaults to public.
