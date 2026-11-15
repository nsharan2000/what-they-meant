"""Render one scenario at a realism level and prompt setting -> (system_prompt, first_user_message, allowed_tools).

Realism levels (same decision point, same facts — only the surrounding context changes). See research/realism-gap.md §3.
  L0  (= L0-seed) benchmark-style toy: minimal prompt, the seed's own member card, only the seed's tools (+ commit + approval card),
      tool results exactly as written in the seed (they sometimes contain verdict hints such as "matches last payment")
  L0r L0-raw: same as L0 but tool results are raw API fields (rule["result_raw"]), no verdict words  ← the anchor for the gap
  L0p L0-pad: L0r + neutral filler appended to the member card up to the L2 length (isolates length alone)
  L1  production-realistic: the environment's full policy prompt, all environment tools, member profile + accounts +
      15 most recent transactions + standing instructions + notes (contacts, linked accounts, subscriptions via tools)
  L2  buried: L1 + the member's full history (every transaction) and all records injected, so the decisive fact sits
      among many similar rows (the decisive tool results are unchanged)
  L2s L2-salient: L2 with the decisive facts restated in a "Most relevant records" block at the end of the prompt (isolates burial)
  L3  noisy: L2 with the scenario's noisy user message (typos / casual / locale code-mixing) when the row provides one
Budget: Tiny Aya / Aya Expanse 8B have an 8K window → L2 history is filled newest-first until the system prompt reaches
L2_CHAR_BUDGET characters (≈ 6.5K Aya tokens at a conservative 3.0 chars/token; JSON-heavy text is ~2.7, prose ~3.7–4.1).
Prompt settings:
  P0  the environment's own prompt (what a real app ships)
  P1  P0 + the golden "how this member wants you to work" block (prompts/golden.md, ~320 words)
  P2  P0 + a ONE-OR-TWO-LINE intent hint (prompts/golden_short.md, ~35 words) — user, 2026-11-13: the model must behave
      correctly with no instructions (P0) or this minimal hint; P2 is the hard-set mining and the main evaluation prompt
"""
from __future__ import annotations

import json
import os

from .protocol import PROTOCOL_INSTRUCTIONS, render_tool_docs
from .spec import ROOT

TODAY = "2026-11-12"


def _fmt_money(amount, currency="USD") -> str:
    try:
        a = float(amount)
    except (TypeError, ValueError):
        return f"{amount} {currency}".strip()
    sym = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}.get(currency, "")
    s = f"{abs(a):,.2f}"
    return f"{'-' if a < 0 else ''}{sym}{s}" + ("" if sym else f" {currency}")


def _txn_line(t: dict) -> str:
    sign = "-" if t.get("direction") == "out" else "+"
    who = t.get("counterparty", "")
    if t.get("handle"):
        who += f" ({t['handle']})"
    extra = " · ".join(x for x in [t.get("type", ""), t.get("memo", ""), t.get("status", "")] if x)
    return f"{t.get('date', '')} | {sign}{_fmt_money(t.get('amount'), t.get('currency', 'USD'))} | {who}" + (f" | {extra}" if extra else "")


def _list_block(title: str, rows: list, fmt=None) -> str:
    if not rows:
        return ""
    lines = [f"### {title}"]
    for r in rows:
        lines.append("- " + (fmt(r) if fmt else (r if isinstance(r, str) else json.dumps(r, ensure_ascii=False))))
    return "\n".join(lines)


L2_CHAR_BUDGET = 19500     # total system-prompt characters at L2 (≈ 6.5K Aya tokens incl. tool docs and policy)
FILLER = ("Service notice: scheduled maintenance windows are announced in the app inbox at least 48 hours in advance. "
          "Statements are generated on the first business day of each month and kept for seven years. ")


def _with_extra(member: dict, extra: list[dict] | None) -> dict:
    """L2 near-miss records (scenario['l2_extra_records'] = [{"field": "transactions", "record": {...}}]) merged into a copy."""
    if not extra:
        return member
    m = json.loads(json.dumps(member))
    for x in extra:
        m.setdefault(x["field"], []).append(x["record"])
    return m


def _cap(rows: list, k: int | None) -> list:
    """newest-first cap for record lists that carry dates; plain lists keep their order."""
    if k is None or not isinstance(rows, list) or len(rows) <= k:
        return rows
    if rows and isinstance(rows[0], dict) and any(x in rows[0] for x in ("date", "created_at", "timestamp", "ts", "due_date")):
        key = next(x for x in ("date", "created_at", "timestamp", "ts", "due_date") if x in rows[0])
        rows = sorted(rows, key=lambda r: str(r.get(key, "")), reverse=True)
    return rows[:k]


def member_block(member: dict, n_recent: int | None, everything: bool = False, max_other: int | None = None) -> str:
    p = member.get("profile", {})
    head = ", ".join(str(x) for x in [p.get("name"), f"age {p['age']}" if p.get("age") else None, p.get("city"),
                                      p.get("occupation")] if x)
    parts = [f"## Member\n{head}"]
    if p.get("summary"):
        parts.append(p["summary"])
    parts.append(_list_block("Accounts", member.get("accounts", []),
                             lambda a: f"{a.get('name', a.get('type', 'account'))}: {_fmt_money(a.get('balance'), a.get('currency', 'USD'))}"
                                       + (f" ({a['id']})" if a.get('id') else "")))
    tx = sorted(member.get("transactions", []), key=lambda t: t.get("date", ""), reverse=True)
    shown = tx if n_recent is None else tx[:n_recent]
    title = (f"Transaction history (newest first, {len(shown)} of {len(tx)})" if everything
             else f"Recent activity (last {len(shown)})")
    parts.append(_list_block(title, shown, _txn_line))
    # L1 injects what a real app typically puts in context (standing instructions + notes); contacts, linked accounts and
    # subscriptions are reachable through tools. L2 injects every record (the "everything in context" buried condition).
    keys = [("standing_instructions", "Standing instructions"), ("notes", "Notes on file")]
    everything = everything or n_recent is None
    if everything:
        keys = [("contacts", "Saved contacts / payees"), ("linked_accounts", "Linked accounts"),
                ("subscriptions", "Subscriptions")] + keys
    for key, title in keys:
        cap = None if key in ("standing_instructions", "notes") else max_other
        parts.append(_list_block(title, _cap(member.get(key, []), cap)))
    if everything:
        for key in [k for k in member if k not in ("member_id", "profile", "accounts", "transactions", "contacts",
                                                   "linked_accounts", "subscriptions", "standing_instructions", "notes")]:
            if isinstance(member[key], list):
                parts.append(_list_block(key.replace("_", " ").title(), _cap(member[key], max_other)))
    return "\n\n".join(x for x in parts if x)


_GOLDEN = {}
GOLDEN_FILES = {"P1": "golden.md", "P2": "golden_short.md"}   # P1 = full preferences (~320 words); P2 = a 1-2 line hint (~35 words)


def golden_text(prompt: str = "P1") -> str:
    if prompt not in _GOLDEN:
        path = os.path.join(ROOT, "prompts", GOLDEN_FILES.get(prompt, "golden.md"))
        _GOLDEN[prompt] = open(path).read().strip() if os.path.exists(path) else ""
    return _GOLDEN[prompt]


LEVELS = ["L0", "L0r", "L0p", "L1", "L2", "L2s", "L3"]


def _fill(tmpl: str, mblock: str, tools: list[dict]) -> str:
    return (tmpl.replace("{{member_block}}", mblock)
                .replace("{{tool_docs}}", render_tool_docs(tools))
                .replace("{{protocol}}", PROTOCOL_INSTRUCTIONS)
                .replace("{{today}}", TODAY))


def _l2_system(env: dict, member: dict, scn: dict) -> str:
    """Largest history (newest first) that keeps the L2 system prompt within L2_CHAR_BUDGET; decisive rows are recent so
    they are always included; near-miss rows from the scenario are merged in at their natural dates."""
    m = _with_extra(member, scn.get("l2_extra_records"))
    n_total = len(m.get("transactions", []))
    for max_other in (12, 8, 5, 3, 1):    # cap other record lists first, then binary-search the history length
        lo, hi, best = 15, max(15, n_total), None
        while lo <= hi:
            mid = (lo + hi) // 2
            sys_txt = _fill(env["system_prompt"]["L1"], member_block(m, n_recent=mid, everything=True, max_other=max_other),
                            env["tools"])
            if len(sys_txt) <= L2_CHAR_BUDGET:
                best, lo = sys_txt, mid + 1
            else:
                hi = mid - 1
        if best:
            return best
    return _fill(env["system_prompt"]["L1"], member_block(m, n_recent=15, everything=True, max_other=1), env["tools"])


def scenario_member(member: dict, scn: dict, level: str) -> dict:
    """Member as seen in this scenario: + scenario_records (decisive records of synthetic variants, every level) and
    + l2_extra_records (near-miss records, buried levels only)."""
    m = _with_extra(member, scn.get("scenario_records"))
    if level in ("L2", "L2s", "L3"):
        m = _with_extra(m, scn.get("l2_extra_records"))
    return m


def render(env: dict, member: dict, scn: dict, level: str = "L1", prompt: str = "P0") -> tuple[str, str, list[str]]:
    member = _with_extra(member, scn.get("scenario_records"))
    if level in ("L0", "L0r", "L0p"):
        tools = [t for t in env["tools"] if t["name"] in set(scn["l0_tools"])]
        mblock = scn["l0_member_block"]
        system = _fill(env["system_prompt"]["L0"], mblock, tools)
        if level == "L0p":
            target = len(_l2_system(env, member, scn))
            pad = ""
            while len(system) + len(pad) < target:
                pad += FILLER
            system = _fill(env["system_prompt"]["L0"], mblock + "\n\n" + pad.strip(), tools)
    elif level in ("L2", "L2s", "L3"):
        tools = env["tools"]
        system = _l2_system(env, member, scn)
        if level == "L2s" and scn.get("decisive_facts"):
            system = system.rstrip() + "\n\n## Most relevant records\n" + "\n".join(f"- {f}" for f in scn["decisive_facts"])
    else:
        tools = env["tools"]
        system = _fill(env["system_prompt"]["L1"], member_block(member, n_recent=15), tools)
    if prompt in GOLDEN_FILES:
        g = golden_text(prompt)
        if g:
            system = system.rstrip() + "\n\n" + g
    user = scn.get("user_message_noisy") if level == "L3" and scn.get("user_message_noisy") else scn["user_message"]
    return system, user, [t["name"] for t in tools]
