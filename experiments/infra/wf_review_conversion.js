export const meta = {
  name: 'wtm-review-conversion',
  description: 'Cheap per-app review of the programmatic seed conversion: clash keywords, tool-rule consistency under alternative lookups, member parity, label concerns',
  phases: [{ title: 'Review', detail: 'one Sonnet agent per app, fixes in place, re-runs planted signal + invariants' }],
}
const ROOT = '/Users/sharan-home/Desktop/Projects/Lexsi-what-they-meant'
const ENVS = ['wallet', 'shopping', 'bills', 'bizops']
const SCHEMA = { type: 'object', properties: { env: { type: 'string' }, fixes: { type: 'array', items: { type: 'string' } },
  label_concerns: { type: 'array', items: { type: 'string' } }, planted: { type: 'string' }, invariants: { type: 'string' } },
  required: ['env', 'fixes', 'label_concerns', 'planted', 'invariants'] }
phase('Review')
const out = await parallel(ENVS.map(env => () => agent(`Repo ${ROOT}. Read CLAUDE.md. The seed rows of app "${env}" were converted PROGRAMMATICALLY
(dataset/convert_seed.py + dataset/manual_overrides.json) because Claude agents were rate-limited; raw tool outputs (rule.result_raw) were written
by an open model (dataset/raw_results_open.py). File: ${ROOT}/dataset/scenarios/${env}.jsonl (schema: wtm/spec.py).
Other agents are reviewing the other apps IN PARALLEL: edit ONLY ${ROOT}/dataset/scenarios/${env}.jsonl and write a record of every change
(row id, field, old → new, why) to ${ROOT}/dataset/review_${env}.json. Do NOT run dataset/convert_seed.py, raw_results_open.py or noisy_open.py
and do NOT edit dataset/manual_overrides.json (they touch every app). Checks:
1. expected.informed_any for every A/P row: distinctive (a plain approval for the literal request must NOT hit it — test with
   wtm.spec.any_hit), complete enough that natural paraphrases naming the clash would hit at least one item (add 2-4 paraphrase cues if thin).
2. tool_rules consistent under plausible alternative lookups (other argument names/values, history/contact queries): simulate with
   wtm.sim.Simulator at L1 for 6+ A rows; specific rules before the broad fallback.
3. Do NOT rewrite existing result_raw objects (a programmatic open-model pass harmonises twin schemas after you). Any NEW rule you add must reuse the
   field names of the existing rules for that tool.
4. Rows whose A/B twins use different members (envs/wallet/scenario_members.json, "_b" members): confirm the difference is decision-relevant.
5. Flag any seed label that looks wrong (label_concerns) — do not change labels.
Finally run: python3 -m wtm.policies dataset/scenarios/${env}.jsonl (oracle 1.0, others 0.0) and python3 -m wtm.invariants dataset/scenarios/${env}.jsonl
(0 hard failures).${env === 'wallet' ? ` ALSO write ${ROOT}/envs/wallet/README.md (missing; model it on envs/shopping/README.md, shorter: product,
tools by kind, members and what makes their histories distinct, which seed pairs live here).` : ''}
Be economical: no web searches, no long essays.`, { label: `review:${env}`, schema: SCHEMA, model: 'sonnet' })))
return out
