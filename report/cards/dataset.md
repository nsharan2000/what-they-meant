---
license: cc-by-4.0
task_categories: [text-generation]
tags: [agents, tool-use, payments, benchmark, realism-gap, lexsi-labs-hackathon]
---
# What They Meant: scenarios, environments, training data, transcripts and verdicts

Everything behind *What They Meant*, an RL environment for AI agents that handle payments. Every scenario is a pair of twins
with the same customer request. In one twin the records contradict the request and the agent should ask one question that
names the problem. In the other everything matches and it should go ahead without asking.

What is in this repository:

- `environments/`: the four fictional apps (a wallet, a shopping assistant, a bills app and a small-business finance tool) with
  their policies, tools and synthetic customers.
- `scenarios/`: the seed pairs (based on real incidents posted on Reddit, used as inspiration only; every detail is synthetic),
  the variants written by an open model, the harder test pairs, and the split by source story (`splits.json`).
- `training_views/v1.2.1/`: the curated training data after CuratorKIT, for fine-tuning, preference training and
  reinforcement learning.
- `evaluation/`: every evaluation transcript for the base, trained and Claude models, every Opus judge verdict, the analysis,
  the general-skill scores and the training logs, including every answer produced during reinforcement learning.
- `replication/`, `hardset/`, `audit/`, `roi/`: the first realism-gap study, the selection of the harder pairs, the dataset audit
  and the cost comparison.

Field names you will see in the files: the realism level runs from `L0r` (a short benchmark-style card) through `L1` (the real
app with recent history) to `L2` (the full app with the complete history); `L2s` is the full app with the deciding records
repeated at the end. The prompt is `P0` (the app's own prompt) or `P2` (the same prompt plus a two-line hint).

Code, reproduction steps and the article: https://github.com/nsharan2000/what-they-meant

Licence: CC-BY-4.0. Customer profiles derive from NVIDIA Nemotron-Personas (CC-BY-4.0). The Claude transcripts have the
Claude Code account e-mail redacted.
