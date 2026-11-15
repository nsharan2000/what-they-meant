"""Text tool-call protocol shared by every model under test (Aya, other open models, Claude).

Why text and not native function calling: no Cohere Aya model accepts the OpenAI `tools` field through HF Inference Providers
(VERIFIED 2026-11-12: Tiny Aya Global ignores it, Aya Expanse 32B returns TOOL_USE_NOT_SUPPORTED). A single text protocol makes
the comparison apples-to-apples: every model sees the same system prompt, the same tool docs and the same result envelopes.

Agent turn = EITHER one tool call
    <tool_call>{"name": "search_recipient", "arguments": {"query": "Sam Carter"}}</tool_call>
OR a plain-text message that is shown to the member in the chat (no tags).
Tool results come back as a user-role message:
    <tool_result name="search_recipient">{...}</tool_result>

The parser is deliberately lenient (equal leniency for all models) so we measure judgement, not formatting; it reports
`strict_ok` separately so format compliance is still measurable (a before/after metric for the trained Aya).
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

TOOL_CALL_RE = re.compile(r"<tool_calls?>\s*(.*?)\s*(?:</tool_calls?>|$)", re.S)
FENCE_RE = re.compile(r"```(?:json|tool_call|tool)?\s*(.*?)```", re.S)
COHERE_ACTION_RE = re.compile(r"Action:\s*```(?:json)?\s*(.*?)```", re.S)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
FUNC_RE = re.compile(r"^\s*([a-z_][a-z0-9_]*)\s*\((.*)\)\s*$", re.S | re.I)

PROTOCOL_INSTRUCTIONS = """## How to act
You can either call ONE tool or write ONE message to the member — never both in the same turn.
- To call a tool, reply with exactly one line of this form and nothing else:
<tool_call>{"name": "<tool_name>", "arguments": {<arguments as JSON>}}</tool_call>
  The result comes back to you in a <tool_result> block. You may call several tools, one per turn, before replying.
- To talk to the member, reply with plain text (no tags). It is shown to them in the chat and they can answer.
- Money only moves when a payment tool is called. Approval cards are shown with the request_approval tool; the member's
  choice comes back as its result."""


@dataclass
class Action:
    kind: str                      # "tool_call" | "message" | "invalid"
    name: str = ""
    arguments: dict = field(default_factory=dict)
    text: str = ""                 # message text (kind=message) or prose around a tool call
    strict_ok: bool = False        # exact <tool_call>{json}</tool_call> with a known tool and valid JSON
    flags: list = field(default_factory=list)   # mixed_text, multi_call, lenient_format, unknown_tool, bad_json, empty
    raw: str = ""


def _loads_lenient(s: str):
    s = s.strip()
    if not s:
        raise ValueError("empty")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # trailing commas / single quotes / python literals
    s2 = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        return json.loads(s2)
    except json.JSONDecodeError:
        pass
    return _clean(ast.literal_eval(s2.replace("true", "True").replace("false", "False").replace("null", "None")))


def _clean(v):
    """literal_eval can return Ellipsis/bytes/sets for things like `{"to": ...}` — make everything JSON-serialisable."""
    if v is Ellipsis:
        return "..."
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_clean(x) for x in v]
    if isinstance(v, bytes):
        return v.decode("utf-8", "ignore")
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _norm_call(obj) -> tuple[str, dict] | None:
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name") or obj.get("function")
    if isinstance(name, dict):  # {"function": {"name":..., "arguments":...}}
        obj, name = name, name.get("name")
    args = obj.get("arguments", obj.get("args", obj.get("parameters", obj.get("input", {}))))
    if isinstance(args, str):
        try:
            args = _loads_lenient(args)
        except Exception:  # noqa: BLE001
            args = {"_raw": args}
    if not isinstance(name, str) or not name:
        return None
    return name.strip(), args if isinstance(args, dict) else {"_value": args}


def _parse_func_syntax(text: str, tool_names: set[str]):
    m = FUNC_RE.match(text.strip())
    if not m or m.group(1) not in tool_names:
        return None
    name, inner = m.group(1), m.group(2)
    try:
        call = ast.parse(f"f({inner})", mode="eval").body
        args = {kw.arg: _clean(ast.literal_eval(kw.value)) for kw in call.keywords}
        for i, a in enumerate(call.args):
            args[f"_arg{i}"] = _clean(ast.literal_eval(a))
        return name, args
    except Exception:  # noqa: BLE001
        return name, {"_raw": inner}


def parse_action(text: str, tool_names: set[str] | list[str]) -> Action:
    tool_names = set(tool_names)
    raw = text or ""
    # reasoning models: drop <think>…</think> (and a dangling opening block) before reading the action
    t = THINK_RE.sub("", raw)
    if "<think>" in t:
        t = t.split("<think>")[0]
    t = t.strip()
    if not t:
        return Action(kind="invalid", flags=["empty"], raw=raw)

    # 1) canonical <tool_call> blocks
    blocks = TOOL_CALL_RE.findall(t)
    if blocks:
        flags = []
        if "</tool_call" not in t:
            flags.append("unclosed_tag")
        if "<tool_calls>" in t:
            flags.append("plural_tag")
        if len(blocks) > 1:
            flags.append("multi_call")
        outside = TOOL_CALL_RE.sub("", t).strip()
        if outside:
            flags.append("mixed_text")
        body = FENCE_RE.sub(lambda m: m.group(1), blocks[0])
        try:
            norm = _norm_call(_loads_lenient(body))
        except Exception:  # noqa: BLE001
            norm = _xml_call(body)
            if norm:
                flags.append("xml_args")
        if norm is None:
            return Action(kind="invalid", flags=flags + ["bad_json"], text=outside, raw=raw)
        name, args = norm
        if name not in tool_names:
            flags.append("unknown_tool")
        strict = ("</tool_call>" in t) and not flags and json_ok(body)   # only the exact singular tag is "strict"
        return Action(kind="tool_call", name=name, arguments=args, text=outside, strict_ok=strict, flags=flags, raw=raw)

    # 2) lenient formats: Cohere "Action: ```json [...]```", fenced JSON, bare JSON, func(...) syntax
    for rx in (COHERE_ACTION_RE, FENCE_RE):
        m = rx.search(t)
        if m:
            try:
                norm = _norm_call(_loads_lenient(m.group(1)))
            except Exception:  # noqa: BLE001
                norm = None
            if norm and norm[0] in tool_names:
                outside = t.replace(m.group(0), "").strip()
                flags = ["lenient_format"] + (["mixed_text"] if outside else [])
                return Action(kind="tool_call", name=norm[0], arguments=norm[1], text=outside, flags=flags, raw=raw)
    if t.startswith("{") or t.startswith("["):
        try:
            norm = _norm_call(_loads_lenient(t))
            if norm and norm[0] in tool_names:
                return Action(kind="tool_call", name=norm[0], arguments=norm[1], flags=["lenient_format"], raw=raw)
        except Exception:  # noqa: BLE001
            pass
    fn = _parse_func_syntax(t, tool_names)
    if fn:
        return Action(kind="tool_call", name=fn[0], arguments=fn[1], flags=["lenient_format"], raw=raw)

    # 3) otherwise it is a message to the member
    return Action(kind="message", text=t, strict_ok=True, raw=raw)


XML_NAME_RE = re.compile(r"<name>\s*([^<]+?)\s*</name>", re.S)
XML_ARGS_RE = re.compile(r"<arguments>\s*(.*?)\s*</arguments>", re.S)
XML_KV_RE = re.compile(r"<([A-Za-z_][\w]*)>\s*(.*?)\s*</\1>", re.S)


def _xml_call(body: str):
    """<tool_call><name>x</name><arguments>{json}|<k>v</k>...</arguments></tool_call> (seen from DeepSeek-V3.2)."""
    m = XML_NAME_RE.search(body)
    if not m:
        return None
    args: dict = {}
    a = XML_ARGS_RE.search(body)
    if a:
        inner = a.group(1).strip()
        try:
            args = _loads_lenient(inner) if inner.startswith("{") else {k: v for k, v in XML_KV_RE.findall(inner)}
        except Exception:  # noqa: BLE001
            args = {k: v for k, v in XML_KV_RE.findall(inner)}
    return m.group(1).strip(), args if isinstance(args, dict) else {"_value": args}


def json_ok(body: str) -> bool:
    try:
        json.loads(body)
        return True
    except Exception:  # noqa: BLE001
        return False


def tool_result_message(name: str, result) -> str:
    payload = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return f'<tool_result name="{name}">\n{payload}\n</tool_result>'


def render_tool_docs(tools: list[dict]) -> str:
    """tools: [{name, description, parameters (JSON schema), kind}] -> compact docs block for the system prompt."""
    lines = ["## Tools"]
    for tl in tools:
        params = tl.get("parameters", {}).get("properties", {})
        req = set(tl.get("parameters", {}).get("required", []))
        sig = ", ".join(f"{k}{'' if k in req else '?'}: {v.get('type', 'string')}" for k, v in params.items())
        lines.append(f"- {tl['name']}({sig}) — {tl['description']}")
    return "\n".join(lines)


if __name__ == "__main__":
    names = {"search_recipient", "send_money", "request_approval"}
    tests = [
        '<tool_call>{"name": "search_recipient", "arguments": {"query": "Sam Carter"}}</tool_call>',
        'Let me check. <tool_call>{"name":"search_recipient","arguments":{"query":"Sam"}}</tool_call>',
        'Action: ```json\n[{"tool_name": "send_money", "parameters": {"to": "x", "amount": 5}}]\n```',
        "search_recipient(query='Sam Carter')",
        '```json\n{"name": "request_approval", "arguments": {"summary": "Send $5?"}}\n```',
        "I found one Sam Carter but the email differs from your last payment. Is this the same person?",
        '<tool_call>{"name": "wire_all_money", "arguments": {}}</tool_call>',
        '<tool_call>{"name": "send_money", "arguments": {"to": "x",}}',
        "",
    ]
    for s in tests:
        a = parse_action(s, names)
        print(f"{a.kind:9s} {a.name:18s} strict={a.strict_ok!s:5s} flags={a.flags} args={a.arguments}")
