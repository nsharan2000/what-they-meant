# Dataset pipeline: stages, counts, and who did each step
Actor key: HUMAN = team member; MODEL = LLM (named); CODE = deterministic script.

| # | Stage | Actor | In -> Out | Dropped (why) |
|---|-------|-------|-----------|---------------|
| 1 | Scope: 10 finance sub-domains, core mistake tags | HUMAN + MODEL (Claude) | - | - |
| 2 | Source selection: 19 subreddits (P2P apps, merchants, accounting, AI-agent, personal finance US/India, investing) | HUMAN + MODEL (Claude) | - | - |
| 3 | Collection via Arctic Shift API (785 requests, windowed retries) + one subreddit dump | CODE | -> 3,001 posts (Jan 2024 - Nov 2026) | - |
| 4 | Rule filter: mistake term AND "it happened" cue AND score >= 4 | CODE | 3,001 -> 358 | 2,643 (no incident signal) |
| 5 | Triage of all 358 (yes / maybe / no) | MODEL (Claude) | 358 -> 115 (53 yes, 62 maybe) | 243 (advice, promo, disputes, no money action) |
| 6 | Record extraction: one structured record per post (said, believed, was true, check, quote) | MODEL (Gemini) | 115 -> 115 records | - |
| 7 | Gate: verbatim-quote + schema check; failures re-run | CODE (loop) | 104 first pass -> 115/115 pass | 0 after re-runs |
| 8 | Adjudication: Claude vs Gemini agree 75% (Cohen's kappa 0.50); 11 yes/no conflicts decided by a human | HUMAN | 115 -> 66 usable | 49 (Gemini "no", or human drop) |
| 9 | Scenario authoring: twin pairs (A should ask / B should act), multi-turn, invented details listed per scenario | MODEL (Claude) | 66 -> 58 pairs | 8 (near-duplicates, incl. 1 exact duplicate) |
| 10 | Human review: full pair / variant / park | HUMAN | 58 -> 45 pairs (38 core, 7 variant) | 13 parked (repetitive, thin) |
| 11 | Augmentation: pushback variants; one scenario simplified | MODEL + HUMAN | +3 conversations | - |
| 12 | Semantic grouping on 4 axes (what is wrong, where evidence is, whose mistake, who acts) | MODEL (Claude), pending HUMAN check | 45 pairs | - |
| - | **Final** | | **45 pairs, 93 conversations** | |
