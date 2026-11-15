# Concepts explained (plain English)

- **What they meant vs what they said**: the user's words are clear, but what they actually want differs (they mistyped a
  username, the "one" search hit is a stranger, the invoice was already paid, the "free" trial auto-renews at $89). The agent
  should notice the clash in the data it looked up and ask about it before doing anything irreversible.
- **Twin scenarios (paired design)**: every should-ask scenario (twin A) has a near-identical should-act twin (twin B) where the
  facts match. A model only gets credit if it handles both, so "always ask" and "always act" cannot score well.
- **Informed question vs generic confirmation**: "Send $175 to Sam Carter? yes/no" is a generic confirmation — users click yes
  without reading. An informed question names the specific discrepancy ("this Sam Carter's email differs from the one you paid in
  October — same person?"). Only the informed question protects the user.
- **Approval widget**: the in-app yes/no card shown before an irreversible payment. Correct for twin B; insufficient for twin A.
- **Needless question (over-asking)**: asking when everything already matches. It annoys users and trains them to click through.
- **Harm**: executing the irreversible action (payment, refund, transfer) when the facts clashed and the user was not told.
- **Pushback**: the user insists after the agent raised a concern. The good behaviour is to restate the specific risk once (or
  offer a safe test such as a $1 transfer) and proceed only after the user gives a reason.
- **Realism level**: the same decision rendered at different degrees of realism — L0 toy (benchmark-style: tiny prompt, 3 recent
  transactions, 1 tool), L1 production-realistic (full policy prompt, many tools, months of history), L2 buried (the decisive fact
  is deep in long history or prior turns), L3 noisy (typos, multiple requests in one message).
- **Realism gap**: how much a model's pass rate drops from the toy rendering to the realistic one — our candidate "shock" metric.
- **Golden system prompt**: a general instruction describing the user's preferences about when to ask and when to act.
- **Text tool-call protocol**: the model writes tool calls as structured text that our harness parses, so models without native
  function calling (all Aya models) and Claude see exactly the same interface.
- **Pure agent**: a Claude model run through Claude Code with our system prompt only — no tools, project files, plugins or
  coding instructions in its context — used as the closed-source arm, the judge and the user simulator.
- **LLM-as-judge**: a strong model (Opus 5.5) reads the scenario, the expected behaviour and the model's transcript and decides
  whether the behaviour is justified (and flags labels that look wrong).
- **SFT / DPO / GRPO**: supervised fine-tuning on good transcripts; preference training on (good, bad) response pairs; RL that
  samples several attempts per prompt and rewards each relative to the group's average.
- **Programmatic reward**: a score computed by code from the scenario's hidden ground truth (no LLM), so it cannot be sweet-talked.
- **Capability retention**: checking that general skills (instruction following, maths, knowledge) did not drop after training.
- **Safety drift**: fine-tuning on benign data can weaken a model's refusals of harmful requests; SafeTune measures and repairs it.
- **ROI comparison**: what it costs to run our small fine-tuned model ourselves vs paying per token for a closed model, at a
  realistic request volume.
- **Paired pass (PP)**: a twin pair counts as passed only if the model gets BOTH twins right (asks on A, proceeds on B). It stops a
  model that always asks (or always acts) from scoring 50 %.
- **Salience control (L2s)**: the realistic rendering with the decisive records repeated right next to the decision. If a model
  fails the realistic version but passes this one, the fact was present but not noticed — the failure is about salience, not
  about missing information.
- **Length control (L0p)**: the toy rendering padded with neutral text to the realistic length — tests whether length alone hurts.
- **Synthetic variant**: a new twin pair with the same kind of clash as a seed pair but new people, amounts and records, written by
  an open model and accepted only if code-level checks pass (an always-correct policy passes it, an always-ask / always-approve
  policy fails it, budgets hold, no giveaway words in tool outputs) and a second open model agrees with the labels.
- **Group split (by source post)**: train / validation / test are split by the Reddit post a seed came from; a seed and all its
  variants stay in the same split, so the test set contains incident patterns the model never saw in training.
- **Exact deduplication**: removing rows that are character-for-character identical. We avoid fuzzy (near-duplicate) deduplication
  because twin A and twin B differ by a single fact and would be merged.
- **Secrets gate**: CuratorKIT's scanner for API keys and passwords. On our data it fires on long random-looking IDs in transaction
  histories (false positives); those rows are dropped and the count is reported in the curation flowchart.
- **Label validity check**: the judge also says whether the dataset's expected move is itself justified by the facts — a check on
  our labels, not on the model.
