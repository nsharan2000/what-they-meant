"""Pure Claude driver: run Claude models through Claude Code with NO coding/plugin context.

Why this exists: the user forbids external AI API keys, and wants the closed-source arm, the LLM judge and the user simulator
to be "pure" (no code, repo, plugin or tool text in context). Recipe (VERIFIED 2026-11-12, see log.md 18:12):
  * bundled CLI >= 2.1.290 (global ~/.local/bin/claude 2.1.268 rejects claude-opus-5-5 / claude-fable-5-2)
  * neutral working directory /tmp/assistant-runtime (the cwd path is visible to the model)
  * `env -i`-style clean environment + CLAUDE_CODE_SIMPLE_SYSTEM_PROMPT=1 (removes the SDK identity prefix, so the system
    prompt is exactly ours) + CLAUDE_CODE_DISABLE_{CLAUDE_MDS,AUTO_MEMORY,ATTACHMENTS,GIT_INSTRUCTIONS,NONESSENTIAL_TRAFFIC}
  * --system-prompt <ours> --tools "" --strict-mcp-config --setting-sources "" --disable-slash-commands
Residual context the model still sees (cannot be removed without an API key): a ~110-word environment block and an
account-context block. Record this caveat with every result.

Multi-turn: Claude Code cannot be handed fabricated assistant turns (editing session transcripts is blocked by the permission
classifier and we do not work around it). So conversations are genuine: each new user-role message (a tool result or a user
reply) is sent with `--resume <session_id>`, and the session only ever contains the model's own outputs.
`ClaudePure.chat(system, messages)` hides this: it maps the conversation prefix to the session that produced it.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field

NEUTRAL_CWD = "/tmp/assistant-runtime"
MODEL_IDS = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
    "fable": "claude-fable-5-2",
}
_DISABLE_FLAGS = {
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_ATTACHMENTS": "1",
    "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_SIMPLE_SYSTEM_PROMPT": "1",
}
_TRANSIENT = re.compile(r"overloaded|rate.?limit|529|503|502|timeout|ECONNRESET|temporarily", re.I)
_USAGE_LIMIT = re.compile(r"hit your .{0,20}limit|usage limit|limit .{0,10}resets|resets \d", re.I)


class UsageLimitError(RuntimeError):
    """The account's usage window is exhausted: abort the run instead of recording failed episodes (they would never be retried)."""


def _version_tuple(path: str) -> tuple:
    m = re.search(r"claude-code-(\d+)\.(\d+)\.(\d+)", path)
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)


def find_claude_bin() -> str:
    """$CLAUDE_BIN, else the newest VS Code-bundled native binary (>= 2.1.290 needed for Opus 5.5 / Fable 5.2)."""
    if os.environ.get("CLAUDE_BIN"):
        return os.environ["CLAUDE_BIN"]
    cands = glob.glob(os.path.expanduser(
        "~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude"))
    cands = [c for c in cands if _version_tuple(c) >= (2, 1, 290)]
    if not cands:
        raise RuntimeError("No Claude Code binary >= 2.1.290 found; set CLAUDE_BIN")
    return max(cands, key=_version_tuple)


def clean_env() -> dict:
    home = os.path.expanduser("~")
    user = os.environ.get("USER", os.path.basename(home))
    env = {"HOME": home, "USER": user, "LOGNAME": user, "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
           "TERM": "xterm-256color", "LANG": "en_US.UTF-8"}
    env.update(_DISABLE_FLAGS)
    return env


def _prefix_key(system: str, messages: list[dict]) -> str:
    blob = json.dumps({"s": system, "m": messages}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class CallResult:
    text: str
    session_id: str
    cost_usd: float
    usage: dict
    model: str
    is_error: bool
    raw_error: str = ""
    duration_s: float = 0.0


@dataclass
class ClaudePure:
    model: str = "claude-haiku-5-5"          # full model ID or alias key of MODEL_IDS
    timeout_s: int = 600
    max_retries: int = 4
    effort: str | None = None                 # None = model default (recorded in results)
    bin_path: str = field(default_factory=find_claude_bin)
    _sessions: dict = field(default_factory=dict)   # prefix-hash -> session_id
    _lock: threading.Lock = field(default_factory=threading.Lock)
    total_cost_usd: float = 0.0

    def __post_init__(self):
        self.model = MODEL_IDS.get(self.model, self.model)
        os.makedirs(NEUTRAL_CWD, exist_ok=True)

    # ---- low level ---------------------------------------------------------------------------------------------------
    def _run(self, system: str, prompt: str, session_id: str, resume: bool) -> CallResult:
        base = [self.bin_path, "-p", prompt,
                "--system-prompt", system,
                "--tools", "",
                "--strict-mcp-config",
                "--setting-sources", "",
                "--disable-slash-commands",
                "--model", self.model,
                "--output-format", "json"]
        if self.effort:
            base += ["--effort", self.effort]
        last_err = ""
        for attempt in range(self.max_retries + 1):
            # a failed first attempt may already have registered its session id ("Session ID … is already in use"),
            # so a NEW conversation gets a fresh id on every retry; a resumed conversation must keep its id
            sid = session_id if (resume or attempt == 0) else str(uuid.uuid4())
            cmd = base + (["--resume", sid] if resume else ["--session-id", sid])
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, cwd=NEUTRAL_CWD, env=clean_env(), capture_output=True, text=True,
                                      timeout=self.timeout_s)
                raw = proc.stdout or ""
                if _USAGE_LIMIT.search(raw[:2000] + "\n" + (proc.stderr or "")[:2000]) and '"is_error":false' not in raw.replace(" ", ""):
                    raise UsageLimitError((raw + " " + (proc.stderr or "")).strip()[:300])
                i = raw.find("{")
                if i < 0:
                    raise RuntimeError(f"no JSON in output (rc={proc.returncode}): {raw[:200]!r} {proc.stderr[:200]!r}")
                d = json.loads(raw[i:], strict=False)
                text = d.get("result") or ""
                is_err = bool(d.get("is_error"))
                if is_err and _USAGE_LIMIT.search(text):
                    raise UsageLimitError(text[:200])
                if is_err and _TRANSIENT.search(text) and attempt < self.max_retries:
                    last_err = text
                    time.sleep(min(60, 5 * 2 ** attempt))
                    continue
                cost = float(d.get("total_cost_usd") or 0.0)
                with self._lock:
                    self.total_cost_usd += cost
                return CallResult(text=text, session_id=d.get("session_id", sid), cost_usd=cost,
                                  usage=d.get("usage") or {}, model=self.model, is_error=is_err,
                                  raw_error=text if is_err else "", duration_s=time.time() - t0)
            except UsageLimitError:
                raise                                   # never retried: the caller aborts and resumes after the reset
            except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as e:
                last_err = str(e)
                if attempt < self.max_retries:
                    time.sleep(min(60, 5 * 2 ** attempt))
                    continue
        return CallResult(text="", session_id=session_id, cost_usd=0.0, usage={}, model=self.model, is_error=True,
                          raw_error=last_err)

    # ---- chat interface ----------------------------------------------------------------------------------------------
    def chat(self, system: str, messages: list[dict]) -> CallResult:
        """messages: alternating [{'role':'user'|'assistant','content':str}], last one must be 'user'.
        Every assistant message must have been produced by this driver (genuine multi-turn via --resume)."""
        assert messages and messages[-1]["role"] == "user", "last message must be from the user"
        if len(messages) == 1:
            res = self._run(system, messages[0]["content"], str(uuid.uuid4()), resume=False)
        else:
            key = _prefix_key(system, messages[:-1])
            with self._lock:
                sid = self._sessions.get(key)
            if sid is None:
                raise RuntimeError("conversation prefix was not produced by this driver; cannot resume "
                                   "(Claude Code does not accept fabricated assistant turns)")
            res = self._run(system, messages[-1]["content"], sid, resume=True)
        if not res.is_error:
            new_prefix = messages + [{"role": "assistant", "content": res.text}]
            with self._lock:
                self._sessions[_prefix_key(system, new_prefix)] = res.session_id
        return res

    def ask(self, system: str, prompt: str) -> CallResult:
        """Single-shot call (judge, user simulator, data generation)."""
        return self.chat(system, [{"role": "user", "content": prompt}])


if __name__ == "__main__":  # smoke test: python3 -m wtm.drivers.claude_pure haiku
    import sys
    d = ClaudePure(model=sys.argv[1] if len(sys.argv) > 1 else "haiku")
    sysmsg = "You are a helpful assistant."
    msgs = [{"role": "user", "content": "Remember the number 41. Reply only: ok"}]
    r1 = d.chat(sysmsg, msgs)
    msgs += [{"role": "assistant", "content": r1.text}, {"role": "user", "content": "What number? Reply with the number only."}]
    r2 = d.chat(sysmsg, msgs)
    print(json.dumps({"bin": d.bin_path, "model": d.model, "turn1": r1.text, "turn2": r2.text,
                      "cost": round(d.total_cost_usd, 6), "err": [r1.raw_error, r2.raw_error]}, indent=1))
