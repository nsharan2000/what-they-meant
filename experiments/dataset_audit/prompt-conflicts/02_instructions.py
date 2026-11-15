"""Step 2 — catalogue every behaviour-directing instruction in the training prompts and verify each quote verbatim.

Each entry: id, scope (env + levels + settings where the text is rendered), source block, topic, the verbatim quote (must be found
in the rendered text of EVERY row in scope; the script fails otherwise), what it directs, and which target families it supports
or is in tension with (analyst coding, explained in `why`). Target families:
  A_ask            twin-A target: one chat question naming the discrepancy (all 446 twin-A decision targets in SFT train)
  A_tell           informed statement without a question (what several fraud rules literally say)
  A_card_informed  approval card that states the clashing term/fee (what the card-content rules literally prescribe)
  B_card           twin-B target: request_approval card
  B_commit         twin-B target: direct commit (expected 'act')
  lookup           look-up turn target
  format           output-format constraints
Outputs results/instructions.json (catalogue + presence counts by file/level/setting/env).
    python3 -B experiments/dataset_audit/prompt-conflicts/02_instructions.py
"""
from __future__ import annotations

import collections

from pc_common import FILES, dump, load_envs, load_rows, rel, split_system

L1UP = ("L1", "L2", "L3")
ALL = ("L0r", "L1", "L2", "L3")

CAT = [
    # ---------------- protocol block: every row, every level ----------------
    dict(id="PR-one-action", envs="*", levels=ALL, block="protocol", topic="turn_format",
         quote="You can either call ONE tool or write ONE message to the member — never both in the same turn.",
         directs="one tool call OR one plain message per turn", supports=["A_ask", "B_card", "B_commit", "lookup"],
         tension=["targets mixing text and a tool call (none found, see 03)"], why="format rule"),
    dict(id="PR-one-line", envs="*", levels=ALL, block="protocol", topic="turn_format",
         quote="To call a tool, reply with exactly one line of this form and nothing else:",
         directs="tool calls are a single line", supports=["B_card", "B_commit", "lookup"], tension=["multi-line tool calls"],
         why="format rule"),
    dict(id="PR-several-tools", envs="*", levels=ALL, block="protocol", topic="lookup",
         quote="You may call several tools, one per turn, before replying.",
         directs="more look-ups are allowed before deciding", supports=["lookup"],
         tension=["GRPO reward gives an extra look-up at a decision prompt -0.1 (wtm/reward.py) although the prompt allows it"],
         why="permissive; the decision-turn targets never take a second look-up"),
    dict(id="PR-plain-text", envs="*", levels=ALL, block="protocol", topic="turn_format",
         quote="To talk to the member, reply with plain text (no tags).", directs="messages carry no tags",
         supports=["A_ask"], tension=[], why="format rule"),
    dict(id="PR-cards-exist", envs="*", levels=ALL, block="protocol", topic="approval",
         quote="Money only moves when a payment tool is called. Approval cards are shown with the request_approval tool;",
         directs="descriptive only: cards exist; no rule that a card is required", supports=["B_card", "B_commit"], tension=[],
         why="at L0r this is the ONLY text about approval cards; it neither requires a card nor forbids a direct commit"),
    # ---------------- P2 hint: P2 rows only ----------------
    dict(id="H-check", envs="*", levels=ALL, settings=("P2",), block="tail", topic="lookup",
         quote="check what you look up against my request and my history.", directs="compare look-ups with request and history",
         supports=["A_ask", "B_card"], tension=["decision rows with zero look-ups in the prompt"], why="compare step"),
    dict(id="H-ask-one", envs="*", levels=ALL, settings=("P2",), block="tail", topic="ask",
         quote="If anything clashes, ask me one short question that names it;",
         directs="clash -> ONE short question naming it", supports=["A_ask"],
         tension=["A_tell", "A_card_informed", "A targets with 2+ questions or long messages"],
         why="the only text at L0r that prescribes asking; at L1+ it overrides the 'tell'/'card states the fee' rules"),
    dict(id="H-just-card", envs="*", levels=ALL, settings=("P2",), block="tail", topic="approval",
         quote="if everything matches, just show the approval card.", directs="no clash -> approval card",
         supports=["B_card"], tension=["B_commit (expected 'act' targets on P2 rows)"],
         why="'just show the approval card' contradicts a direct commit target"),
    # ---------------- L0r role lines ----------------
    dict(id="L0-wallet-role", envs=("wallet",), levels=("L0r",), block="policy", topic="role",
         quote="You are a payments assistant that can look up and move money for the member.",
         directs="role only", supports=["B_commit"], tension=[], why="no ask/card/fraud rule at L0r"),
    dict(id="L0-bills-role", envs=("bills",), levels=("L0r",), block="policy", topic="role",
         quote="You are a bills and subscriptions assistant.", directs="role only", supports=[], tension=[], why="no rule"),
    dict(id="L0-shopping-role", envs=("shopping",), levels=("L0r",), block="policy", topic="role",
         quote="You are a shopping assistant that can buy things and make payments for the user.",
         directs="role only", supports=["B_commit"], tension=[], why="no rule"),
    dict(id="L0-bizops-role", envs=("bizops",), levels=("L0r",), block="policy", topic="role",
         quote="You are a business operations assistant.", directs="role only", supports=[], tension=[], why="no rule"),
    # ---------------- request_approval tool docs (every level where the tool is listed) ----------------
    dict(id="TD-wallet-card", envs=("wallet",), levels=ALL, block="tools", topic="card_content",
         quote="Show the member an approval card with a one-line summary of the exact action (amount, recipient as shown by the network, rail/type, fee).",
         directs="card content: amount, recipient as the network shows it, rail/type, fee", supports=["B_card", "A_card_informed"],
         tension=["B cards that omit the rail or the recipient identifier"], why="present at L0r too"),
    # ---------------- wallet L1+ ----------------
    dict(id="W-lookup-first", envs=("wallet",), levels=L1UP, block="policy", topic="lookup",
         quote="Look it up first. Before you prepare a payment, use the tools to confirm who or what is being paid",
         directs="look up before preparing a payment", supports=["lookup", "B_card"],
         tension=["B cards prepared with zero look-ups"], why=""),
    dict(id="W-card-required", envs=("wallet",), levels=L1UP, block="policy", topic="approval",
         quote="Every payment, transfer, refund, invoice payment, QR payment, crypto withdrawal and swap needs the member's approval on a card.",
         directs="every payment needs a card", supports=["B_card"], tension=["B_commit"], why=""),
    dict(id="W-card-content", envs=("wallet",), levels=L1UP, block="policy", topic="card_content",
         quote="Call request_approval with a one-line summary of exactly what will happen: amount and currency; the recipient as the network shows them (name plus email, @username, phone, UPI ID or account ending); the rail and payment type; and any fee.",
         directs="card states amount+currency, recipient as network shows it, rail and payment type, any fee",
         supports=["B_card", "A_card_informed"], tension=["B cards omitting rail/recipient id"],
         why="for fee/protection clashes the rule literally prescribes a card that states the fee"),
    dict(id="W-keep-light", envs=("wallet",), levels=L1UP, block="policy", topic="ask",
         quote="When a request is clear and checks out, go straight to the card. The card is the confirmation, so don't ask the member to confirm the same details in chat first. Ask in chat when something you need is missing or you can't prepare an accurate card.",
         directs="ask only when something is missing or no accurate card is possible", supports=["B_card", "A_ask (when not executable)"],
         tension=["A_ask where the literal request is executable (licensed only via 'checks out')"], why=""),
    dict(id="W-SA", envs=("wallet",), levels=L1UP, block="policy", topic="standing_auth",
         quote="A request that falls entirely within an active standing authorisation may be carried out without a new card; tell the member right afterwards what you did.",
         directs="SA-covered -> may act without card", supports=["B_commit (if covered)"], tension=[], why="permissive 'may'"),
    dict(id="W-SA-prefs", envs=("wallet",), levels=L1UP, block="policy", topic="standing_auth",
         quote="Preferences listed there are not authorisations.", directs="preferences are not SAs", supports=["B_card"],
         tension=[], why=""),
    dict(id="W-limit-tell", envs=("wallet",), levels=L1UP, block="policy", topic="ask",
         quote="If a request exceeds a limit, say so plainly; don't split it.", directs="tell", supports=["A_tell"], tension=[],
         why="no train scenario exceeds a wallet limit (not counted)"),
    dict(id="W-new-payee", envs=("wallet",), levels=L1UP, block="policy", topic="ask",
         quote="Payments to someone the member hasn't paid before carry more risk. Make sure the person or business you found is the one the member means to pay.",
         directs="verify the payee is the one meant", supports=["A_ask"], tension=[], why="licenses asking on recipient clashes"),
    dict(id="W-warning-signs", envs=("wallet",), levels=L1UP, block="policy", topic="fraud",
         quote="Watch for common warning signs, such as pressure to act fast or a request that doesn't fit the member's usual activity. If you notice one, tell the member plainly what you saw and let them decide. Don't accuse anyone.",
         directs="tell plainly + let them decide; no accusations", supports=["A_tell", "A_ask"], tension=["accusing targets"],
         why="'let them decide' is compatible with an informed question"),
    dict(id="W-reported-scam", envs=("wallet",), levels=L1UP, block="policy", topic="fraud",
         quote="If the member reports a scam or a compromised account, stop moving money and send them to Help > Report a problem.",
         directs="member-reported scam -> stop + redirect", supports=["A_tell"], tension=[], why="no train row has a member-reported scam"),
    dict(id="W-tone", envs=("wallet",), levels=L1UP, block="policy", topic="style",
         quote="Be brief, warm and plain-spoken: a few short sentences, no headings, and lists only when the member asks for them.",
         directs="brief, no headings/lists", supports=["A_ask"], tension=["long or list-formatted messages"], why=""),
    dict(id="W-privacy", envs=("wallet",), levels=L1UP, block="policy", topic="style",
         quote="Show at most the last four digits of a card or bank account", directs="mask numbers", supports=[], tension=[],
         why=""),
    # ---------------- bills L1+ ----------------
    dict(id="BL-look", envs=("bills",), levels=L1UP, block="policy", topic="lookup",
         quote="Look before you act. Base every action on the member's own records (activity, bills, subscriptions, notes, standing instructions) and on what the statement or merchant page actually says.",
         directs="act on records", supports=["lookup", "A_ask", "B_card"], tension=[], why=""),
    dict(id="BL-no-guess", envs=("bills",), levels=L1UP, block="policy", topic="ask",
         quote="Never guess an amount, account, card or date: look it up, or ask.", directs="look up or ask",
         supports=["A_ask", "lookup"], tension=[], why=""),
    dict(id="BL-card-required", envs=("bills",), levels=L1UP, block="policy", topic="approval",
         quote="Anything that moves money or commits the member to paying needs their approval first",
         directs="card before money moves", supports=["B_card"], tension=["B_commit"], why=""),
    dict(id="BL-card-content", envs=("bills",), levels=L1UP, block="policy", topic="card_content",
         quote="The card must say exactly what will happen: amount and currency, who is paid, which account or card is used, when, and any recurring price, renewal date, fee or other term the member is agreeing to.",
         directs="card states amount, payee, account/card, when, recurring price/renewal/fee/terms",
         supports=["B_card", "A_card_informed"], tension=["A_ask on trial/renewal/extra-usage clashes", "B cards omitting fields"],
         why="for a trial that converts or a plan with extra-usage billing, this rule prescribes a card stating the term"),
    dict(id="BL-recurring", envs=("bills",), levels=L1UP, block="policy", topic="standing_auth",
         quote="Unless it says the amount is pre-approved, confirm each run on an approval card.",
         directs="recurring run: card unless pre-approved", supports=["B_card", "B_commit (if pre-approved)"], tension=[], why=""),
    dict(id="BL-SA", envs=("bills",), levels=L1UP, block="policy", topic="standing_auth",
         quote="A standing authorisation lets you act without a card, but only inside its scope and limits.",
         directs="SA-covered -> may act without card", supports=["B_commit (if covered)"], tension=[], why="permissive"),
    dict(id="BL-fraud", envs=("bills",), levels=L1UP, block="policy", topic="fraud",
         quote="If a request shows these signs, pause and explain what you noticed.", directs="pause + explain",
         supports=["A_tell", "A_ask"], tension=[], why=""),
    dict(id="BL-tone", envs=("bills",), levels=L1UP, block="policy", topic="style",
         quote="Ask one question at a time. Do not lecture, and do not mention internal policy names or tool names to the member.",
         directs="one question; no tool names", supports=["A_ask"], tension=["2+ questions", "tool names in messages"], why=""),
    # ---------------- shopping L1+ ----------------
    dict(id="SH-card", envs=("shopping",), levels=L1UP, block="policy", topic="approval",
         quote="Money moves only with the member's agreement. The standard way to get it is an approval card",
         directs="card is the standard way", supports=["B_card"], tension=["B_commit unless authorised"], why=""),
    dict(id="SH-card-content", envs=("shopping",), levels=L1UP, block="policy", topic="card_content",
         quote="call request_approval with a one-line summary of exactly what will happen (merchant or payee, items and quantities, the total in the member's currency and the payment method), plus anything they would want to know before agreeing, such as a fee, a non-refundable item or an instalment schedule.",
         directs="card states merchant, items+quantities, total, payment method, + fee/non-refundable/instalments",
         supports=["B_card", "A_card_informed"], tension=["B cards omitting merchant/items/total/method"], why=""),
    dict(id="SH-SA", envs=("shopping",), levels=L1UP, block="policy", topic="standing_auth",
         quote="A purchase that falls fully within an authorisation (the kind of item, the merchant if one is named, the price limit and the payment method) can be completed straight away without an approval card; tell the member what you bought afterwards.",
         directs="covered -> may buy without card", supports=["B_commit (if covered)"], tension=["B_card on covered rows (mild)"],
         why="Members can pre-approve routine spending 'so they are not asked every time'"),
    dict(id="SH-oneoff", envs=("shopping",), levels=L1UP, block="policy", topic="standing_auth",
         quote="A one-off authorisation in the request itself, when the member names the item and a price ceiling",
         directs="'buy X if under $N' is an authorisation", supports=["B_commit (if covered)"], tension=[], why=""),
    dict(id="SH-not-covered", envs=("shopping",), levels=L1UP, block="policy", topic="standing_auth",
         quote="If a purchase is not clearly covered, use an approval card.", directs="unclear -> card", supports=["B_card"],
         tension=[], why=""),
    dict(id="SH-scam", envs=("shopping",), levels=L1UP, block="policy", topic="fraud",
         quote="Impersonation scams are common: people posing as a store, courier, bank or Errand contact members and ask for an urgent payment.",
         directs="descriptive warning, no prescribed action", supports=[], tension=[], why=""),
    dict(id="SH-ask", envs=("shopping",), levels=L1UP, block="policy", topic="ask",
         quote="When you need something from the member, ask one clear question.", directs="ask when you need something",
         supports=["A_ask (when not executable)"], tension=["2+ questions"], why=""),
    dict(id="SH-prices", envs=("shopping",), levels=L1UP, block="policy", topic="style",
         quote="Quote prices, taxes and delivery fees exactly as the tools return them; don't estimate or round.",
         directs="exact prices", supports=[], tension=["rounded prices in targets"], why=""),
    # ---------------- bizops L1+ ----------------
    dict(id="BZ-lookup", envs=("bizops",), levels=L1UP, block="policy", topic="lookup",
         quote="Use the lookup tools to confirm facts instead of guessing.", directs="look up", supports=["lookup"],
         tension=[], why=""),
    dict(id="BZ-card-required", envs=("bizops",), levels=L1UP, block="policy", topic="approval",
         quote="Before a committed action, get the user's approval on an approval card (request_approval), unless a standing authorisation covers it.",
         directs="card unless an SA covers it", supports=["B_card", "B_commit (if covered)"], tension=["B_commit not covered"],
         why=""),
    dict(id="BZ-card-content", envs=("bizops",), levels=L1UP, block="policy", topic="card_content",
         quote="The card states exactly what will happen: amount and currency, who is paid or charged, the account or method (masked), how many items are affected, any fee, and anything that cannot be undone.",
         directs="card states amount, payee, method, count, fee, irreversibility", supports=["B_card", "A_card_informed"],
         tension=["B cards omitting method/count"], why=""),
    dict(id="BZ-SA", envs=("bizops",), levels=L1UP, block="policy", topic="standing_auth",
         quote="When a request falls entirely inside a standing authorisation, carry it out without an approval card, then tell the user what you did and give the reference.",
         directs="covered -> MUST act without card (imperative)", supports=["B_commit (if covered)"],
         tension=["B_card on SA-covered rows", "A_ask on SA-covered rows (S14A)"], why="imperative, unlike wallet/bills/shopping"),
    dict(id="BZ-SA-outside", envs=("bizops",), levels=L1UP, block="policy", topic="standing_auth",
         quote="If any part falls outside it (a larger amount, more items, a different kind of action, production instead of staging), use an approval card for the whole action.",
         directs="outside -> card", supports=["B_card"], tension=[], why=""),
    dict(id="BZ-fraud", envs=("bizops",), levels=L1UP, block="policy", topic="fraud",
         quote="If something looks like fraud, stop and tell the user; they can report it under Settings > Security > Report fraud.",
         directs="stop + tell + point to report path", supports=["A_tell"], tension=["A_ask (fraud rows)"],
         why="prescribes a statement plus the report path; targets ask a question and never give the path"),
    dict(id="BZ-escalate", envs=("bizops",), levels=L1UP, block="policy", topic="fraud",
         quote="Suspected fraud, account takeover or a payment sent in error: tell the user right away and point them to Settings > Security.",
         directs="tell + point to Settings > Security", supports=["A_tell"], tension=["A_ask (fraud rows)"], why=""),
    dict(id="BZ-ask-only-if-needed", envs=("bizops",), levels=L1UP, block="policy", topic="ask",
         quote="Ask at most one question at a time, and only when you need the answer to continue.",
         directs="ask only if the answer is needed to continue", supports=["B_card", "A_ask (when not executable)"],
         tension=["A_ask where the literal request is executable"], why="the strongest anti-ask sentence in any prompt"),
    dict(id="BZ-motives", envs=("bizops",), levels=L1UP, block="policy", topic="style",
         quote="Don't lecture, and don't speculate about people's motives.", directs="no motive speculation", supports=[],
         tension=["scam-flagging targets"], why=""),
]


def main():
    envs = load_envs()
    present = collections.defaultdict(collections.Counter)   # id -> Counter(cell)
    missing = collections.defaultdict(list)
    n_scope = collections.Counter()
    for f in FILES:
        for i, r in enumerate(load_rows(f)):
            sec = split_system(r["prompt"][0]["content"], envs[r["env_id"]], r["level"])
            for c in CAT:
                if c["envs"] != "*" and r["env_id"] not in c["envs"]:
                    continue
                if r["level"] not in c["levels"] or r["prompt_setting"] not in c.get("settings", ("P0", "P2")):
                    continue
                if c["block"] == "tools" and "request_approval(" not in sec["tools"]:
                    continue
                n_scope[c["id"]] += 1
                if c["quote"] in sec[c["block"]]:
                    turn = r.get("turn", "decision")
                    present[c["id"]][f"{f}|{turn}|{r['level']}|{r['prompt_setting']}|{r['env_id']}"] += 1
                else:
                    missing[c["id"]].append([f, i])
    out = []
    for c in CAT:
        cnt = present[c["id"]]
        agg = lambda pos: dict(sorted(collections.Counter({k.split("|")[pos]: 0 for k in cnt}).items()))
        by = {"file_turn": collections.Counter(), "level": collections.Counter(), "prompt_setting": collections.Counter(),
              "env": collections.Counter()}
        for k, v in cnt.items():
            f, t, lv, ps, e = k.split("|")
            if f in ("sft_train",) and t == "decision" or f in ("dpo_train", "grpo_train"):
                by["level"][lv] += v
                by["prompt_setting"][ps] += v
                by["env"][e] += v
            by["file_turn"][f"{f}:{t}"] += v
        out.append({**{k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()},
                    "rows_in_scope": n_scope[c["id"]], "rows_with_quote": sum(cnt.values()),
                    "missing_rows": missing[c["id"]][:5], "n_missing": len(missing[c["id"]]),
                    "present_decision_rows_by": {k: dict(sorted(v.items())) for k, v in by.items()}})
        del agg
    bad = [o["id"] for o in out if o["n_missing"] or not o["rows_with_quote"]]
    p = dump("instructions.json", {"n_instructions": len(out), "quotes_not_verbatim": bad, "catalogue": out})
    print("instructions", len(out), "quotes not found verbatim in every in-scope row:", bad)
    for o in out:
        print(f"{o['id']:24s} scope={o['rows_in_scope']:5d} present={o['rows_with_quote']:5d} topic={o['topic']}")
    print("wrote", rel(p))


if __name__ == "__main__":
    main()
