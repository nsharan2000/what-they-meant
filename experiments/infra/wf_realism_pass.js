export const meta = {
  name: 'wtm-realism-pass',
  description: 'Add the realism-ladder fields to every scenario (raw tool results, decisive strings, near-miss records, noisy request) and verify invariants per environment',
  phases: [
    { title: 'Enrich', detail: 'one agent per environment adds result_raw / decisive_strings / l2_extra_records / user_message_noisy' },
    { title: 'Check', detail: 'independent fact-equivalence and near-miss review, invariants + planted signal' },
  ],
}

const ROOT = '/Users/sharan-home/Desktop/Projects/Lexsi-what-they-meant'
const ENVS = (args && args.envs) || ['wallet', 'shopping', 'bills', 'bizops']

const COMMON = `PROJECT: "What They Meant" (Lexsi hackathon). Repo ${ROOT}. Read ${ROOT}/CLAUDE.md first, then ${ROOT}/research/realism-gap.md §3-§5
(the realism ladder and its invariants), ${ROOT}/wtm/render.py (levels L0, L0r, L0p, L1, L2, L2s, L3), ${ROOT}/wtm/sim.py (L0 serves rule
"result", every other level serves rule "result_raw"), ${ROOT}/wtm/invariants.py and ${ROOT}/wtm/policies.py.
Scenario rows live in ${ROOT}/dataset/scenarios/<env>.jsonl (schema in ${ROOT}/wtm/spec.py); environments in ${ROOT}/envs/<env>/.
Tiny Aya (our core model) has an 8K context: L1 must stay <= 5.5K and L2/L3 <= 7.2K estimated tokens (the checker estimates chars/3.0).
Never call any LLM API. Python is python3.`

const ENRICH = (env) => `${COMMON}

TASK: enrich every row of ${ROOT}/dataset/scenarios/${env}.jsonl IN PLACE with the realism-ladder fields (keep every existing field):
1. For EVERY entry in tool_rules add "result_raw": the same facts as structured JSON, the way a real API returns them (ids, names,
   handles/emails/phones, amounts, currencies, dates/times, statuses, counts, booleans such as "photo_url": null or
   "previous_payments_from_member": 0). NO evaluative or verdict words anywhere (not "matches", "mismatch", "no photo", "never paid",
   "already paid", "duplicate", "suspicious", "scam", "risk", "verified", "warning", "wrong", "correct"...): state facts and let the
   agent infer. Twin A and twin B results must differ only in the facts, never in tone. Keep "result" (the seed's verbatim text,
   used at L0) unchanged.
2. "decisive_strings": 1-4 exact strings the correct decision depends on that the model can see (in the member context or in a tool
   rule result), e.g. the e-mail on the earlier payment and the e-mail the search returns.
3. "l2_extra_records": 2-3 near-miss records for the buried level, as [{"field": "transactions"|"contacts"|<other member list field>,
   "record": {...}}]: realistic rows that mimic the decisive fact's semantics (a payee with a similar name or handle, a past payment
   of a similar amount to someone else, an older invoice from the same vendor that was legitimately paid...) WITHOUT changing the
   correct answer for either twin. Dates must be older than the decisive record. Same records for both twins of a pair.
4. "user_message_noisy": the L3 variant of the request — casual, lowercase, 1-2 typos or fillers; for India/UPI members natural
   Hinglish code-mixing, for the German member occasional German words — with every entity, amount, handle, number and date unchanged.
4b. "prefix_calls": the canonical read-only lookup call(s) that lead to the decision point, as
   [{"tool": "<env tool name>", "arguments": {<named JSON args>}}], mirroring the seed's tool calls before decision_point_index
   (e.g. search_recipient with {"query": "Sam Carter"}). Each must hit one of the row's tool_rules (check with wtm.sim.Simulator).
   Use [] when the seed decision comes before any lookup.
4c. "ideal_action": the exact ideal decision turn as structured data: {"type": "message", "text": <the seed ideal_message>} for
   A/P rows; {"type": "tool_call", "name": "request_approval", "arguments": {"summary": <the seed widget message>}} for B rows
   expecting an approval; {"type": "tool_call", "name": <commit tool>, "arguments": {<full realistic args that match a
   gold_commits signature>}} for "act" rows. These become supervised targets, so they must be exactly right.
5. Run: python3 -m wtm.invariants dataset/scenarios/${env}.jsonl  → zero hard failures, and no soft warnings about missing
   result_raw or verdict words (twin-parity warnings are OK only when the difference is decision-relevant; explain them).
   If a budget fails, shorten the environment's L1 policy text or tool descriptions in envs/${env}/env.json (never drop tools a seed
   row uses) or reduce history verbosity — re-run until it passes.
6. Run: python3 -m wtm.policies dataset/scenarios/${env}.jsonl → oracle paired_acc 1.0 and every other policy 0.0 (it now tests
   L0, L0r, L1 and L2).
Return the structured summary.`

const CHECK = (env, prev) => `${COMMON}

TASK: independent, skeptical check of the realism fields just added to ${ROOT}/dataset/scenarios/${env}.jsonl (enricher summary:
${JSON.stringify(prev).slice(0, 2500)}). Fix problems in place.
1. Fact equivalence: for every tool rule, "result_raw" carries exactly the facts of "result" (nothing added that changes the decision,
   nothing decision-relevant removed) and contains no verdict words.
2. Near-miss records: for every pair, reason explicitly that each l2_extra_records row does not change the correct answer for twin A
   or twin B, and that it does not collide with gold/harmful commit signatures or informed_any keywords in a way that would let a
   plain approval look "informed".
3. Noisy requests: entities, amounts, handles, numbers and dates unchanged; still understandable; locale-appropriate.
3b. prefix_calls hit a tool rule (run them through wtm.sim.Simulator at L1); ideal_action is exactly the seed's ideal move, and for
   "act"/"approve" rows its arguments match a gold_commits signature (wtm.episode._target must return "gold" for A-twin-free B rows).
4. Render 3 rows at every level (python3 -c "from wtm.render import render, LEVELS ...") and read them: L1 must read like a real app
   context, L2 must bury the decisive record among realistic rows, L0p must only add neutral filler.
5. Re-run python3 -m wtm.invariants dataset/scenarios/${env}.jsonl and python3 -m wtm.policies dataset/scenarios/${env}.jsonl.
Report checks, fixes and remaining issues.`

const SCHEMA = {
  type: 'object',
  properties: {
    env_id: { type: 'string' },
    n_rows: { type: 'integer' },
    invariants: { type: 'string', description: 'summary of the last python3 -m wtm.invariants output' },
    planted_signal: { type: 'string' },
    est_tokens_by_level: { type: 'string' },
    fixes_or_notes: { type: 'array', items: { type: 'string' } },
  },
  required: ['env_id', 'n_rows', 'invariants', 'planted_signal', 'fixes_or_notes'],
}

phase('Enrich')
const out = await pipeline(
  ENVS,
  (env) => agent(ENRICH(env), { label: `enrich:${env}`, phase: 'Enrich', schema: SCHEMA }),
  (prev, env) => prev && agent(CHECK(env, prev), { label: `check:${env}`, phase: 'Check', schema: SCHEMA, model: 'sonnet' }),
)
return out
