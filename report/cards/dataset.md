---
license: cc-by-4.0
task_categories: [text-generation]
tags: [agents, tool-use, payments, benchmark, realism-gap, lexsi-labs-hackathon]
---
# What They Meant — twin-pair scenarios, environments, curated training views, rollouts and verdicts

Everything behind the project *What They Meant*: four fictional app environments (wallet, shopping, bills, small-business ops) with
synthetic members and histories; twin-pair scenarios grounded in first-person Reddit incidents (provenance only; all details
synthetic) — in twin A the request clashes with the records and the agent should ask one specific question, in twin B everything
matches and it should approve or act; synthetic variants and hard pairs; group splits by source post; the curated training views
(SFT, DPO, GRPO; dataset v1.2.1 after CuratorKIT); every evaluation rollout (base, trained and closed models, realism levels L0r /
L1 / L2 / L2s, prompts P0 / P2); every Opus-judge verdict; training logs (including every GRPO completion); the dataset audit.

Code, reproduction and the curation flowchart: https://github.com/nsharan2000/what-they-meant.
Licence: CC-BY-4.0 (to be confirmed by the authors). Member personas derive from NVIDIA Nemotron-Personas (CC-BY-4.0).
Claude transcripts had the Claude Code account e-mail redacted.
