"""OpenAI-compatible chat driver for open-weight models.

Two backends, same interface as ClaudePure.chat(system, messages):
  * HF Inference Providers router  — base_url https://router.huggingface.co/v1, api key = HF token (user-approved credits),
    model ids like "CohereLabs/tiny-aya-global:cohere".
  * local vLLM on the DGX Spark     — base_url http://localhost:<port>/v1 (inside lx-exp or via ssh tunnel).
Responses are cached on disk (sha256 of model+params+messages) so reruns are free and every number is reproducible.
No `tools` field is ever sent: all models see the same text tool-call protocol (Aya models have no native tool calling).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field

from openai import OpenAI

HF_ROUTER = "https://router.huggingface.co/v1"


def hf_token() -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    # repo .env has a non-shell-safe "HF-TOKEN=" line; parse it without printing
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith("HF-TOKEN=") or line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError("HF token not found (HF_TOKEN env or repo .env)")


@dataclass
class CallResult:
    text: str
    usage: dict
    model: str
    is_error: bool
    raw_error: str = ""
    cached: bool = False
    duration_s: float = 0.0
    session_id: str = ""
    cost_usd: float = 0.0


@dataclass
class OpenAICompat:
    model: str
    base_url: str = HF_ROUTER
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float | None = None
    cache_dir: str | None = None
    timeout_s: int = 180
    max_retries: int = 5
    extra_body: dict | None = None
    _client: OpenAI | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def __post_init__(self):
        key = self.api_key or (hf_token() if "huggingface" in self.base_url else "EMPTY")
        self._client = OpenAI(base_url=self.base_url, api_key=key, timeout=self.timeout_s, max_retries=0)
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _key(self, system: str, messages: list[dict]) -> str:
        blob = json.dumps({"m": self.model, "b": self.base_url, "t": self.temperature, "mt": self.max_tokens,
                           "tp": self.top_p, "x": self.extra_body, "s": system, "msgs": messages},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, system: str, messages: list[dict]) -> CallResult:
        key = self._key(system, messages)
        path = os.path.join(self.cache_dir, key[:2], key + ".json") if self.cache_dir else None
        if path and os.path.exists(path):
            d = json.load(open(path))
            return CallResult(text=d["text"], usage=d.get("usage", {}), model=self.model, is_error=False, cached=True)
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        last = ""
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                kw = dict(model=self.model, messages=msgs, temperature=self.temperature, max_tokens=self.max_tokens)
                if self.top_p is not None:
                    kw["top_p"] = self.top_p
                if self.extra_body:
                    kw["extra_body"] = self.extra_body
                r = self._client.chat.completions.create(**kw)
                msg = r.choices[0].message
                text = msg.content or ""
                # Some providers run their own tool-call parser on the model's text (e.g. Qwen's <tool_call> blocks) and move
                # the calls into message.tool_calls even though we never send `tools`. Put them back into the shared protocol
                # so every model is read the same way.
                for tc in (getattr(msg, "tool_calls", None) or []):
                    fn = getattr(tc, "function", None)
                    if not fn:
                        continue
                    try:
                        targs = json.loads(fn.arguments) if isinstance(fn.arguments, str) else fn.arguments
                    except json.JSONDecodeError:
                        targs = {"_raw": fn.arguments}
                    text += ("\n" if text else "") + "<tool_call>" + json.dumps({"name": fn.name, "arguments": targs},
                                                                                ensure_ascii=False) + "</tool_call>"
                usage = r.usage.model_dump() if r.usage else {}
                with self._lock:
                    self.total_prompt_tokens += usage.get("prompt_tokens", 0) or 0
                    self.total_completion_tokens += usage.get("completion_tokens", 0) or 0
                if path:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    json.dump({"text": text, "usage": usage, "finish": r.choices[0].finish_reason}, open(path, "w"))
                return CallResult(text=text, usage=usage, model=self.model, is_error=False, duration_s=time.time() - t0)
            except Exception as e:  # noqa: BLE001 — retry transient provider errors, surface the rest
                last = f"{type(e).__name__}: {str(e)[:300]}"
                status = getattr(e, "status_code", None)
                if status in (400, 401, 403, 404, 422) or attempt == self.max_retries:
                    break
                time.sleep(min(60, 3 * 2 ** attempt))
        return CallResult(text="", usage={}, model=self.model, is_error=True, raw_error=last)
