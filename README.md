# What They Meant

A realism-graded benchmark, a curated training set and trained Cohere Tiny Aya agents for **payment tools that should act on what
the customer means**. When a request clashes with the records (look-alike payee, invoice already paid, a "free" trial that renews,
scam pressure, …) the agent should ask **one** question that names the clash; when everything matches it should show a plain
approval card and proceed — never ask needlessly. Built for the Lexsi Labs AI Safety & Alignment Hackathon (with Cohere Labs and
Hugging Face).

- **Article (results, methods, caveats):** `report/hf_article_draft.md` → published on Hugging Face (link in the article header).
- **Models and data:** Hugging Face collections under `sharan-nagarajan` (links in the article).
- **Headline finding — the realism gap:** frontier models that handle a decision on a short, clean, benchmark-style card fail the
  same decision inside a realistic app (long policy prompt, many tools, real-looking history). On 41 held-out twin pairs with a
  two-line intent hint, Claude Opus 5.5 goes from 65.8 % to 26.8 % Opus-judged paired pass (−39.0 pp, 95 % CI [−63.0, −13.5]),
  mostly through needless questions (`experiments/training_eval/results/analysis/e1_test_all.json`).

## Repository map

| Path | What it holds |
|---|---|
| `envs/` | The 4 app environments (wallet "Kestrel", shopping "Errand", bills "Ledgerly", bizops "Northwind Ops"): system prompt, tool schemas, member profiles + histories (synthetic; members grounded in NVIDIA Nemotron-Personas, CC-BY-4.0) |
| `dataset/` | Seed twin scenarios (from first-person Reddit incidents, provenance only), synthetic variants, splits by source post, hard pairs, label adjudications, curated training views (`curated_v121/`), the curation flowchart (`curation-flowchart.md`) |
| `wtm/` | Harness: text tool protocol, realism renderer (L0r/L1/L2/L2s/L3 × P0/P1/P2), scripted simulator, keyword grader, programmatic reward, Opus judge, pure Claude Code driver, OpenAI-compatible driver |
| `experiments/replication/` | Realism-gap study across 13 models (preregistered: `README.md`), analysis + judge files |
| `experiments/hardset/` | Hard pairs that big models fail even with a two-line hint |
| `experiments/dataset_audit/` | 8-lens audit of the training data and prompts (`REPORT.md`) |
| `experiments/training_eval/` | Held-out evaluation of base / trained / closed models, capability, safety and alignment audits, `analyze_arms.py` |
| `experiments/roi/` | Self-hosted Tiny Aya vs Claude cost per decision, from measured inputs |
| `training/` | Training views, CuratorKIT mapping, SFT / GRPO / DPO trainers (AgentTune, AlignTune), memory guard, the self-driving pipelines (`run_pipeline.sh`, `run_pipeline_v2.sh`, `run_chain2.py` + `chain2_steps/`), offline tests |
| `explorer/` | The dataset explorer (static site builder) |
| `report/` | The HF article, generated tables (`make_tables.py`), the HF publisher (`publish_hf.py`) |
| `research/` | Background research, one topic per file, every claim marked VERIFIED (source fetched) or CAVEATED |

## Lexsi Labs stack used

CuratorKIT (twin-safe selection: exact dedup + secrets gate) → AgentTune (M1 SFT via the distillation API; M2 GRPO via
`create_agentic_trainer`) → AlignTune (M3 DPO via `create_rl_trainer`; AlignmentAuditor) → AuditKIT (precomputed scoring,
`ak.compare`, `ak.run_lmeval` capability retention) → SafeTune (HarmBench / XSTest drift). Pinned commits and install notes:
`.claude/skills/lexsi-stack/SKILL.md`.

## Reproduce

Python ≥ 3.12. GPU work ran on an NVIDIA DGX Spark (GB10, 121 GB unified memory) inside a vLLM container; every GPU job runs under
`training/memguard.py`. Closed models run through Claude Code in pure mode (`wtm/drivers/claude_pure.py`); open models through vLLM
or Hugging Face Inference Providers. No other API keys are used.

```bash
# 1. Realism-gap replication tables (rebuilds experiments/replication/results/summary.json)
python3 experiments/replication/analysis.py --results experiments/replication/results/grid_*.jsonl \
  --judge experiments/replication/results/judge_*_v3.jsonl --scenarios 'dataset/frozen_replication_v1/*.jsonl' --out summary.json

# 2. Training views and curation (deterministic)
python3 training/build_datasets.py ...            # see training/PLAN.md
python3 training/from_curated.py ...              # CuratorKIT decides which rows survive; text untouched

# 3. Training + evaluation, self-driving on the GPU box (inside the container)
bash training/run_pipeline.sh                     # M1 SFT → merge → M2 GRPO → stack → M3 DPO → eval → audits
python3 training/run_chain2.py                    # the remaining experiments (GRPO v2, decision-weighted SFT, evals, audits)

# 4. Held-out analysis (any machine)
python3 experiments/training_eval/analyze_arms.py --results experiments/training_eval/results/eval_test.jsonl \
  experiments/training_eval/results/eval_test_claude.jsonl --scenarios 'dataset/eval/test_*.jsonl' --split test \
  --judge experiments/training_eval/results/judge/judge_eval_test_v3.jsonl --out e1_test.json --md e1_test.md
python3 report/make_tables.py                     # every article table, from the results files

# Offline tests (no GPU)
python3 training/tests/test_grpo_v2.py && python3 training/tests/test_chain2_driver.py
```

## Licences

To be confirmed by the authors (proposed: code MIT; data CC-BY-4.0). Trained adapters inherit the Tiny Aya Global licence
(CC-BY-NC-4.0). Member personas are derived from NVIDIA Nemotron-Personas (CC-BY-4.0); Reddit posts are cited as provenance only,
never copied. Third-party Lexsi repositories are used as installed packages and are not redistributed here. Claude transcripts
had the Claude Code account e-mail redacted (it is injected into every session; 7 of 864 held-out Claude episodes mention it).
