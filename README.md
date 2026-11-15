# What They Meant

An RL environment for AI agents that hold the payment keys. When a customer's request does not match what the records show (a look-alike payee, an invoice that was already paid, a "free" trial that renews at £90 a year, a supplier's bank details "changed" in an email), the agent should ask one question that names the problem. When everything matches, it should show a plain approval card and move on, without a needless question. We use the environment to measure models, to curate training data, and to fine-tune Cohere's Tiny Aya Global (3.35B) with the Lexsi Labs stack.

Built for the Lexsi Labs AI Safety & Alignment Hackathon (with Cohere Labs and Hugging Face) by Sharan Nagarajan and GLKAR.

- Article, with results and the reasoning behind each choice: [report/ARTICLE.md](report/ARTICLE.md)
- Models: https://huggingface.co/collections/sharan-nagarajan/what-they-meant-models-6af9e567bdba59d91ddfc5c3
- Data, every transcript and every judge verdict: https://huggingface.co/collections/sharan-nagarajan/what-they-meant-data-6af9eb020a621b28fe22ecc8

![One request, two outcomes](report/figures/twins.png)

Every scenario comes as near-identical twins. In most pairs the request is word for word the same and only the records differ. A model scores only when it gets both twins right, so an agent that always asks, or always pays, scores zero. Every scenario is also rendered from a short benchmark-style version up to the full app (long policy, 26 to 28 tools, months of history), which exposes a realism gap. With a two-line hint, Claude Opus 5.5 gets 66% of held-out pairs right in the benchmark version but 27% in the app. Fine-tuning took Tiny Aya from paying on 73% of risky requests to 10%, and from checking the records first 5% of the time to 95%.

## What is where

| Path | Contents |
|---|---|
| `envs/` | The four app environments (wallet, shopping, bills, small-business finance): system prompts, tool schemas, synthetic customers and histories |
| `dataset/` | Seed scenarios and variants, the split by source story, harder test pairs, curated training data (`curated_v121/`), the curation flowchart |
| `wtm/` | The harness: tool-call protocol, realism renderer, scripted customer, grader, training reward, Opus judge, model drivers |
| `training/` | Training data builders, the AgentTune and AlignTune trainers, the GPU memory guard, the self-driving training chain, offline tests |
| `experiments/` | The realism-gap study, the harder pairs, the dataset audit, the held-out evaluation and its analysis, the cost comparison |
| `report/` | The article, its figures, auto-generated result tables, model and dataset cards, the Hugging Face publisher |

## Lexsi Labs stack

CuratorKIT selects the training data, AgentTune runs supervised fine-tuning and reinforcement learning (GRPO), AlignTune runs preference training (DPO), and AuditKIT scores transcripts and general skills. Pinned versions and install notes: `.claude/skills/lexsi-stack/SKILL.md`.

## Reproduce

Python 3.12 or newer. Training and evaluation ran on one NVIDIA DGX Spark inside a vLLM container (the repository root is `/workspace` there), with every GPU job wrapped in `training/memguard.py`. Open models are served by vLLM or Hugging Face Inference Providers. Claude runs through Claude Code (`wtm/drivers/claude_pure.py`). No other API keys are needed.

```bash
# training data (deterministic), then CuratorKIT selection (inside the container)
python3 training/build_datasets.py --scenarios 'dataset/scenarios/*.jsonl' 'dataset/scenarios_synth/*.jsonl' \
  --splits dataset/splits.json --failures 'experiments/replication/results/grid_*.jsonl' --out training/data
python3 dataset/curate.py --data training/data --out dataset/curated
python3 training/from_curated.py --data training/data --curated dataset/curated --out training/data_v121_curated

# training and evaluation, on the GPU box
bash training/run_pipeline.sh            # supervised fine-tuning, GRPO, DPO, evaluation
python3 training/run_chain2.py           # second round: the GRPO rerun and its evaluation (runs training/chain2_steps/)

# analysis on any machine; every article table is rebuilt from the result files
R=experiments/training_eval/results
for split in test hard_test; do
  python3 experiments/training_eval/analyze_arms.py --split $split \
    --results $R/eval_$split.jsonl $R/eval_${split}_claude.jsonl --scenarios "dataset/eval/${split}_*.jsonl" \
    --judge $R/judge/judge_eval_${split}_v3.jsonl $R/judge/judge_eval_${split}_m2_v3.jsonl \
    --out $R/analysis/e1_${split}_all.json
done
python3 report/make_tables.py

# offline tests (no GPU)
python3 training/tests/test_grpo_v2.py && python3 training/tests/test_chain2_driver.py
```

## Licences

Code: MIT (see `LICENSE`). Data: CC-BY-4.0; customer profiles derive from NVIDIA Nemotron-Personas (CC-BY-4.0). Each scenario links to the Reddit post that inspired it and keeps a short quote from it as a provenance record; those quotes remain their authors' and are not covered by our licence. Trained adapters: CC-BY-NC-4.0, inherited from Tiny Aya Global. Claude transcripts have the Claude Code account e-mail redacted.
