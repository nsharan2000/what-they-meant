# Hard set: scenarios that big models fail even with a short, good system prompt

**Request (user, 2026-11-13):** "create enough scenarios where even with a good system prompt many big models fail to perform the
expected behaviour — don't test with an absurdly long system prompt; the instructions can be small but still carry the intent."

## Prompt
`P2` = the app's own prompt + `prompts/golden_short.md`, ONE TWO-LINE hint (~35 words): "Member preference: check what you look up
against my request and my history. If anything clashes, ask me one short question that names it; if everything matches, just show
the approval card." (User, 2026-11-13: even 80–90 words is too much — one or two lines; the trained model must behave correctly with
no instructions or this minimal hint.) `P0` (no hint) is reported alongside; `P1` (the earlier ~320-word preferences block) only as
a reference.

## Panel ("many big models")
Claude Opus 5.5, Sonnet 5.5, Fable 5.2, Haiku 5.5 (pure Claude Code agents) and Qwen3-235B-A22B-Instruct-2507, DeepSeek-V3.2
(HF Inference Providers; ≈ $1.5 of the ~$10 HF budget). Realistic rendering L2 (product prompt, full history, all tools) — the
setting the dataset is about; the toy rendering L0r is run for the hard pairs only, to show the realism gap on them.

## Selection
A pair is **golden-resistant** when (1) its labels pass the Opus label check (label_valid) and the scripted oracle passes it,
(2) at least 4 of the 6 panel models fail it under P2 at L2 — fail = the Opus judge (v3, validated on planted transcripts) says the
first decisive move was wrong on twin A or twin B. Pairs failed by ≥ 5/6 form the **core** hard set.

## Sources
1. Mined: seed (45 pairs) + synthetic variants (v1.2) under P2.
2. Generated: new "hard" twin pairs written by Claude agents in the same four apps, targeting the failure patterns found in step 1
   (clash in a secondary field, clash needing two sources, reassuring wording in the request contradicted by the records, near-identical
   entities, renewal/intro-price traps, …) — same validation as every variant (schema, planted signal, invariants, budgets, verdict-free
   raw outputs) + Opus label check. **Evaluation only: never in any training view** (they are Claude-written; training targets never
   come from Claude — Q5).

## Files
`run_panel.sh` (panel episodes → results), `select_hard.py` (per-pair fail counts from judge verdicts → hard set + failure patterns),
`../../dataset/hard_variants.py` (generator), results in `results/`.
