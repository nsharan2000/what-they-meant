# Feedback for Lexsi Labs (kept deliberately short — each point comes from something we hit in this project)

1. **Native Claude Code integration.** Ship a first-party way to drive the stack from coding agents: a Claude Code plugin/"mod"
   bundling (a) Agent Skills (SKILL.md) for CuratorKIT, AlignTune, AgentTune, AuditKIT and SafeTune with correct config schemas,
   pipeline order and known pitfalls, (b) an MCP server exposing curate / train / audit / status as tools over run directories,
   and (c) `llms.txt` on every docs site. Today none of the 17 Lexsi-Labs repos has a skill, MCP server or plugin, and `/llms.txt`
   returns 404 on five of the six docs sites (only SafeTune has one), so an agent must clone every repo and reverse-engineer configs
   before it can chain them. We wrote the skill we wished existed (`.claude/skills/lexsi-stack/SKILL.md`) — happy to contribute it.

2. **Treat agent data as first-class in CuratorKIT.** Exporters keep only role/content (tool calls and tool results are dropped), and
   several defaults silently damage minimal-pair agent datasets: MinHash/diversity dedup merges "twin" scenarios that differ by one
   fact, PII pseudonymisation rewrites the near-miss names the scenario depends on, and the random per-sample split leaks twins across
   train/test. A trajectory schema shared with AgentTune's EventLog, a "group key" that keeps twins together, and a warning when a
   dedup/PII stage touches paired rows would fix this. Concretely, a multi-turn prompt handed to the "messages" exporter comes back
   as ONE user message holding the JSON of the whole conversation, and both exporters collapse indentation whitespace inside system
   prompts, so the export cannot be fed straight into AgentTune/AlignTune; we use CuratorKIT to select rows and map them back.

3. **Fail loudly instead of silently.** `aligntune compose run` reports every stage as successful with hard-coded metrics (the trainer
   is a stub); AlignTune's in-training audit callback never fires on transformers 5.x (`tokenizer` vs `processing_class`); AgentTune's
   vLLM tool-rollout path imports a TRL module removed after 1.7, and its exact pins (`torch==2.11`, `trl==1.7.1`) downgrade CUDA-13 /
   aarch64 images such as the DGX Spark's. Raising clear errors (and testing against current TRL) would have saved us hours.
   Most costly: `create_rl_trainer(algorithm="dpo", max_seq_length=…)` never reaches TRL's `DPOConfig.max_length`, whose 1.13
   default (1024, keep_start) silently DROPS every pair with a longer prompt — agent prompts are several thousand tokens, so a DPO
   run would have trained on a handful of short pairs with no error. Mapping max_seq_length through (or logging kept/dropped
   counts) would prevent it.

4. **LoRA-aware learning-rate defaults for RL stages.** AgentTune's GRPO trainer defaults to `learning_rate=1e-6`
   (`backends/trl/agentic/grpo/agentic_grpo.py:410`), a full-fine-tuning value. With a fresh LoRA (`peft_config`) such rates barely
   move the adapter: our GRPO run at 5e-6 for 30 steps changed the weights by 2.0 % of what SFT did (||B·A||, measured), and a DPO
   run at the same rate by 3.3 % — both indistinguishable from the starting model in evaluation, although the reward signal was
   there (114/120 groups had reward variance). Scale the default when a LoRA config is passed (we use 5e-5), or warn, and log the
   adapter's update size so a silent no-op is visible.
