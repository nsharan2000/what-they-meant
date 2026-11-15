# What They Meant: an RL environment for AI agents that hold the payment keys

Sharan Nagarajan ([@sharan-nagarajan](https://huggingface.co/sharan-nagarajan)) and [@GLKAR](https://huggingface.co/GLKAR) · Lexsi Labs AI Safety & Alignment Hackathon, with Cohere Labs and Hugging Face

Models and data, in one Hugging Face collection: https://huggingface.co/collections/sharan-nagarajan/what-they-meant-6af9f4e1a81fe397ad7536fd

## 1. The problem and our approach

AI assistants can now pay bills, send money and start subscriptions for us. Most of the risk comes from ordinary customers who are slightly wrong, and the evidence is usually one lookup away. A careful assistant would notice the mismatch and ask one short question that names it. When nothing is wrong, it would simply show the payment for approval, because an assistant that questions everything teaches people to click "yes".

We built an RL environment that tests and trains exactly this judgement, and used it to fine-tune Cohere's Tiny Aya Global (3.35B). Every scenario comes as a pair of near-identical twins. A model only scores on a pair if it gets both twins right, so an agent that always asks, or always pays, scores zero.

![One request, two outcomes](https://raw.githubusercontent.com/nsharan2000/what-they-meant/main/report/figures/twins.png)

The environment is built to feel like production rather than a benchmark. It has four complete finance apps, each with its own written policy and 26 to 28 tools. Their customers are synthetic, built from NVIDIA's Nemotron-Personas, and each one comes with about ten months of transactions, saved payees, subscriptions and standing instructions.

![The four apps](https://raw.githubusercontent.com/nsharan2000/what-they-meant/main/report/figures/apps.png)

Benchmarks usually put the deciding fact right next to the question. A real app buries it. We render every scenario both ways, with the same request, the same tool answers and the same correct move, so any change in behaviour comes from realism alone. We call that difference the realism gap.

![What the agent reads before it decides](https://raw.githubusercontent.com/nsharan2000/what-they-meant/main/report/figures/realism.png)

The 134 scenario pairs are modelled on real incidents that people posted on Reddit, with every name, account and amount invented, and 13 harder pairs are kept for testing. Seven more of them:

![Seven scenarios](https://raw.githubusercontent.com/nsharan2000/what-they-meant/main/report/figures/scenarios.png)

In short: fine-tuning on our data turned Tiny Aya from a model that pays on 73% of risky requests into one that moves money on its own in only 10% of them, checks the records first 95% of the time (up from 5%) and turns 83% of legitimate payments into an approval card (up from 5%). Frontier models fall into the realism gap: with a two-line hint, Claude Opus 5.5 gets 66% of held-out pairs right in the benchmark version but only 27% inside the realistic app.

## 2. Hackathon repos we used, and what for

CuratorKIT selects the training data. We kept its exact deduplication and secrets filter and switched off its fuzzy deduplication, because our twins are near-duplicates by design and fuzzy deduplication would delete half of every pair. AgentTune runs supervised fine-tuning, through its distillation interface, and reinforcement learning (GRPO) with our own programmatic reward. AlignTune runs preference training (DPO), where the right move is set against a real mistake an untrained model made. AuditKIT scores our evaluation transcripts and runs the general-skill benchmarks.

## 3. Models and licences

Tiny Aya Global (3.35B, Cohere Labs, CC-BY-NC-4.0) is the model we train. It is the general-purpose chat model of its family (Base has no chat training, and Fire, Water and Earth are tuned for particular regions), and it is small enough to run on one rented GPU. It has no built-in tool calling, so every model, Tiny Aya and Claude alike, calls tools in the same plain-text format, which keeps the comparisons fair.

Claude Opus 5.5 and Claude Haiku 5.5 (Anthropic, proprietary) are the closed-model comparison, run through Claude Code with our app's system prompt. Opus also judges the answers our programmatic grader cannot settle; we tested it first on transcripts with known right and wrong answers, and it was never fooled. Claude Sonnet 5.5 gave our first realism-gap result, and Claude helped turn the seed scenarios into app episodes and wrote the harder test pairs, which are never trained on.

Qwen3-235B-A22B-Instruct-2507 (Apache-2.0) wrote scenario variants and DeepSeek-V3.2 (MIT) reviewed their labels. No closed model ever wrote a training target. Our first realism-gap study also ran Tiny Aya Fire, Water and Earth, Aya Expanse 32B and Command R7B (CC-BY-NC-4.0), Llama 4 Scout and Llama 3.1 8B (Llama community licences), and Qwen3-4B and Mistral 7B (Apache-2.0). Some of the Cohere models' mistakes serve as rejected answers in our preference pairs.

## 4. Datasets, benchmarks and licences

![How the data was built](https://raw.githubusercontent.com/nsharan2000/what-they-meant/main/report/figures/curation.png)

The dataset is ours, under CC-BY-4.0, and every name, account and amount in it is synthetic. Each scenario keeps a link to the Reddit post that inspired it and a short quote from it, as a record of where it came from. Customer profiles come from NVIDIA's Nemotron-Personas (CC-BY-4.0).

Three choices matter most. We split the data by source story, so no incident appears in both training and test; otherwise a model could pass by remembering a story instead of reading the records. Nothing enters unchecked: code verifies every scenario (the evidence is visible, the twins differ only in the deciding fact, the request is identical at every realism level) and an independent model reviews every label. And the twins' tool outputs share one format, so formatting gives nothing away.

The filtering matters because raw generated data is unreliable. Of the 180 variant pairs the writer model produced, only 89 survived: 32 leaked the answer through their wording, which would teach a model to spot words instead of reading records; 29 carried a label that a second model disagreed with; 10 broke the tool format or gave away the verdict; and 20 more drifted into a different kind of mistake or failed a final label check.

The result is 904 worked examples for fine-tuning, 496 preference pairs and 496 practice prompts for reinforcement learning, a held-out test of 41 twin pairs from 13 stories that no training step saw, and 13 harder held-out pairs written to trip up large models. General skills are measured on IFEval (Apache-2.0), GSM8K (MIT), Global-MMLU (Apache-2.0) and MGSM (CC-BY-SA-4.0).

## 5. Model artefacts

The collection holds three LoRA adapters for Tiny Aya Global, each with a model card: the fine-tuned agent, that agent after reinforcement learning (merged into one adapter), and the preference-trained variant, which we keep as a comparison point.

## 6. Data artefacts

The dataset (https://huggingface.co/datasets/sharan-nagarajan/what-they-meant) sits in the same collection. It holds the four apps, every scenario and split, the training data, every evaluation transcript and judge verdict, and the full training logs, including every answer produced during reinforcement learning. The code, and how to reproduce every number, are at the end of this page.

## 7. Experiments and findings

![How the model was trained and tested](https://raw.githubusercontent.com/nsharan2000/what-they-meant/main/report/figures/training.png)

Every model faces the same held-out test, from the benchmark version to the full app, with and without the two-line hint. The headline metric is "both twins right", because no habit can game it. We also track the two ways to fail: moving money on a risky request, and asking a needless question on a harmless one.

Here is what training changed in Tiny Aya, on the 41 held-out pairs inside the realistic app, with no hint:

| | Base model | After fine-tuning | After reinforcement learning |
|---|---|---|---|
| Pays a risky request without approval | 73% | 10% | 7% |
| Checks the records before deciding | 5% | 95% | 90% |
| Legitimate payment shown as an approval card | 5% | 83% | 56% |
| Names the problem in a question on the risky twin | 0% | 0% | 7% (27% in the benchmark version) |
| Asks a needless question on the harmless twin | 12% | 7% | 37% |
| Both twins right (judged) | 0% | 2% | 5% |

Fine-tuning turned a model that acts on the literal words into a careful one: it looks things up, rarely moves money on its own, and puts most payments in front of the customer as an approval card, which keeps a human in the loop. What it did not learn is telling the twins apart: on risky requests it still shows the approval card (85% of the time) instead of asking.

Reinforcement learning was aimed at that last step. It gives full reward only to the right move, puts both twins of a pair in every training step, and gives no credit for an approval card that merely repeats the problem. It taught the model to ask good, specific questions, from 5% to 27% of risky requests in the benchmark version. One it wrote on a held-out pair: "The UPI ID 'namma.daily.recharge@ybl' belongs to V. Suresh Kumar, not Namma Daily Mart. This looks like a scam. Do you still want to pay?" Inside the realistic app that skill shrank to 7%, and needless questions rose to 37%: the realism gap, inside our own trained model. Our dataset audit suggests why. In more than half of the pair renderings the twins differ in a single tool result, by a median of 28 characters, about 0.3% of the prompt.

Frontier models show the same pattern. With the two-line hint, on the 41 held-out pairs, judged by Opus:

| | Benchmark version | Realistic app | Change (95% confidence interval) |
|---|---|---|---|
| Claude Opus 5.5 | 66% | 27% | −39 points (−63 to −14) |
| Claude Haiku 5.5 | 46% | 10% | −37 points (−51 to −19) |

Inside the app the hint makes them see problems everywhere: Opus asks a needless question on 73% of harmless requests. Without the hint, Opus scores about the same in both settings (49% and 44%). We first saw the gap with Claude Sonnet 5.5 on our 45 seed pairs and a longer version of the hint (78% versus 47%, p = 0.003). On the 13 harder pairs, Opus scores 92% in the benchmark version and 46% in the app.

Two checks keep these numbers honest. We measured how far each training run moved the model's weights, which caught a first reinforcement-learning run that had barely changed the model (2% of the change fine-tuning made) because its learning rate was meant for full fine-tuning. The rerun with a corrected rate moved the weights ten times further, and the results above come from it. The preference run had the same problem; we spent the remaining compute on reinforcement learning instead, since it trains the ask-or-act decision directly. A dataset audit with eight independent checks also confirmed that the two-line hint does not leak the answer, and every training reward is computed by code from the agent's actions, so no answer can talk its way into a higher score.

General skills held where this job needs them: instruction following (IFEval) stayed at 67.5% and general knowledge (Global-MMLU, English) went from 49.8% to 51.3%. Maths dropped (GSM8K from 30% to 26%, multilingual MGSM from 45% to 22%), since our training data has no maths problems in it. Running the model ourselves is cheap. On our DGX Spark it processed about 11,000 tokens a second, and a decision takes about 7,400 tokens, so a million decisions a month need about 185 GPU-hours. One NVIDIA L4 rented at $0.80 an hour, always on, costs about $584 a month and covers that even at half our measured speed. At that volume Claude Haiku would cost about $8,400 a month and Claude Opus about $90,400, at list price.

## 8. With more credits

The next step is to make the deciding signal stronger rather than to train longer. We would weight the decision more heavily during fine-tuning, so the one word that chooses between asking and acting counts for more than the rest of the reply; add harder twins that keep the risky wording while the request is actually fine; and move to Aya Expanse 8B, a larger model in the same family with more room to hold the evidence. We would also mix general instruction data into fine-tuning to protect maths, and replace the single scripted customer with simulated customers who push back in different ways. Every change would be measured against the numbers above, on the same held-out test.

## 9. Team

Sharan Nagarajan ([@sharan-nagarajan](https://huggingface.co/sharan-nagarajan)) and [@GLKAR](https://huggingface.co/GLKAR).

## Code and reproduction

| Path | Contents |
|---|---|
| `envs/` | The four app environments (wallet, shopping, bills, small-business finance): system prompts, tool schemas, synthetic customers and histories |
| `dataset/` | Seed scenarios and variants, the split by source story, harder test pairs, curated training data (`curated_v121/`), the curation flowchart |
| `wtm/` | The harness: tool-call protocol, realism renderer, scripted customer, grader, training reward, Opus judge, model drivers |
| `training/` | Training data builders, the AgentTune and AlignTune trainers, the GPU memory guard, the self-driving training chain, offline tests |
| `experiments/` | The realism-gap study, the harder pairs, the dataset audit, the held-out evaluation and its analysis, the cost comparison |
| `report/` | The figures on this page, auto-generated result tables, model and dataset cards, the Hugging Face publisher |

Training and evaluation ran on one NVIDIA DGX Spark inside a vLLM container (the repository root is `/workspace` there), with every GPU job wrapped in `training/memguard.py`. Open models are served by vLLM or Hugging Face Inference Providers, and Claude runs through Claude Code (`wtm/drivers/claude_pure.py`); no other API keys are needed. Python 3.12 or newer. Pinned versions of the Lexsi Labs libraries and install notes are in `.claude/skills/lexsi-stack/SKILL.md`.

```bash
# training data (deterministic), then CuratorKIT selection (inside the container)
python3 training/build_datasets.py --scenarios 'dataset/scenarios/*.jsonl' 'dataset/scenarios_synth/*.jsonl' \
  --splits dataset/splits.json --failures 'experiments/replication/results/grid_*.jsonl' --out training/data
python3 dataset/curate.py --data training/data --out dataset/curated
python3 training/from_curated.py --data training/data --curated dataset/curated --out training/data_v121_curated

# training and evaluation, on the GPU box
bash training/run_pipeline.sh            # supervised fine-tuning, GRPO, DPO, evaluation
python3 training/run_chain2.py           # second round: the GRPO rerun and its evaluation (runs training/chain2_steps/)

# analysis on any machine; every table on this page is rebuilt from the result files
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

Licences. Code: MIT (see `LICENSE`). Data: CC-BY-4.0; customer profiles derive from NVIDIA Nemotron-Personas (CC-BY-4.0). Each scenario links to the Reddit post that inspired it and keeps a short quote from it as a provenance record; those quotes remain their authors' and are not covered by our licence. Trained adapters: CC-BY-NC-4.0, inherited from Tiny Aya Global. Claude transcripts have the Claude Code account e-mail redacted.
