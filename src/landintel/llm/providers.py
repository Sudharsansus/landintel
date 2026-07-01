"""The LLM engine -- LOCAL-FIRST provider chain (Qwen -> Claude -> Manus), fail-safe.

Consolidated here (out of the agent layer) so the whole product has ONE LLM engine. Qwen runs
LOCALLY by default (free/offline, via Ollama or any OpenAI-compatible server); Claude escalates
for hard cases; Manus is an optional third backend. If every provider is down, callers get None
and fall back to deterministic behaviour -- the product never breaks (or changes a placement)
when the LLM is unavailable.

DEPLOYMENT: LOCAL-HOST, FULLY OFFLINE. The default provider order is "local" ONLY -- the brain
never reaches the internet. The cloud backends (claude, manus) still exist but are STRICTLY
OPT-IN: set ``LANDINTEL_LLM_ORDER="local,claude"`` to allow a cloud fallback. Out of the box,
everything runs offline on the local Qwen.

Provider chain (env ``LANDINTEL_LLM_ORDER``, default "local" = offline-only):
  * local  -- Qwen (default ``LANDINTEL_LOCAL_LLM``="qwen2.5:7b"; also "qwen2.5:14b",
              "llama3.1:8b") via Ollama (``OLLAMA_HOST`` default http://localhost:11434) or an
              OpenAI-compatible endpoint (``LANDINTEL_LLM_BASE_URL`` -- vLLM / LM Studio).
  * claude -- Anthropic API (``ANTHROPIC_API_KEY``), model ``claude-opus-4-8``.
  * manus  -- any OpenAI-compatible API (``MANUS_BASE_URL`` + ``MANUS_API_KEY`` + ``MANUS_MODEL``;
              skipped if base unset -- no hard-coded endpoint).
"""

from __future__ import annotations

import json
import os
import urllib.request

_CLAUDE_MODEL = "claude-opus-4-8"
DEFAULT_LOCAL_MODEL = "qwen2.5:7b"


def _openai_chat(base: str, key: str, model: str, prompt: str, max_tokens: int,
                 system: str | None = None) -> str:
    """One call to any OpenAI-compatible /chat/completions endpoint (stdlib)."""
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": model, "max_tokens": max_tokens,
                         "messages": messages}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    d = json.loads(urllib.request.urlopen(req, timeout=60).read())
    return d["choices"][0]["message"]["content"].strip()


def _local_llm(prompt: str, max_tokens: int, system: str | None = None) -> tuple[str, str] | None:
    """Call the LOCAL open-source LLM -- Qwen by default (OpenAI-compatible or Ollama)."""
    model = os.environ.get("LANDINTEL_LOCAL_LLM", DEFAULT_LOCAL_MODEL)
    base = os.environ.get("LANDINTEL_LLM_BASE_URL")          # vLLM / LM Studio / llama.cpp
    try:
        if base:
            return _openai_chat(base, os.environ.get("LANDINTEL_LLM_KEY", "x"),
                                model, prompt, max_tokens, system), f"local:{model}"
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        # keep_alive: how long Ollama keeps the model resident in VRAM after a call. Short by
        # default so the LLM RELEASES the GPU when idle -- PaddleOCR (CPU, but may use light
        # GPU det) and any other GPU work always get the card back promptly. Set to "0" to
        # unload immediately after every call, or a longer value (e.g. "5m") for latency.
        req = urllib.request.Request(
            host + "/api/chat",
            data=json.dumps({"model": model, "stream": False, "messages": messages,
                             "keep_alive": os.environ.get(
                                 "LANDINTEL_OLLAMA_KEEP_ALIVE", "30s")}).encode(),
            headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=60).read())
        return d["message"]["content"].strip(), f"local:{model}"
    except Exception:  # noqa: BLE001 - not deployed / unreachable -> next provider
        return None


def _manus(prompt: str, max_tokens: int, system: str | None = None) -> tuple[str, str] | None:
    base = os.environ.get("MANUS_BASE_URL")
    if not base:
        return None
    model = os.environ.get("MANUS_MODEL", "manus")
    try:
        return _openai_chat(base, os.environ.get("MANUS_API_KEY", "x"),
                            model, prompt, max_tokens, system), f"manus:{model}"
    except Exception:  # noqa: BLE001
        return None


def _claude(prompt: str, max_tokens: int, system: str | None = None) -> tuple[str, str] | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        c = anthropic.Anthropic()
        kwargs = {"model": _CLAUDE_MODEL, "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        m = c.messages.create(**kwargs)
        return "".join(b.text for b in m.content
                       if getattr(b, "type", "") == "text").strip(), _CLAUDE_MODEL
    except Exception:  # noqa: BLE001
        return None


_PROVIDERS = {"local": _local_llm, "claude": _claude, "manus": _manus}


def llm_call(prompt: str, max_tokens: int = 400,
             system: str | None = None) -> tuple[str, str] | None:
    """Try providers in LANDINTEL_LLM_ORDER (default "local" = OFFLINE-ONLY local Qwen;
    cloud is opt-in via the env var); return (text, provider) or None if unavailable."""
    order = os.environ.get("LANDINTEL_LLM_ORDER", "local").split(",")
    for prov in (p.strip() for p in order):
        fn = _PROVIDERS.get(prov)
        if fn is None:
            continue
        out = fn(prompt, max_tokens, system)
        if out:
            return out
    return None


def local_llm_status() -> dict:
    """Is the local Qwen reachable? (for setup/health checks -- never raises)."""
    model = os.environ.get("LANDINTEL_LOCAL_LLM", DEFAULT_LOCAL_MODEL)
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        d = json.loads(urllib.request.urlopen(host + "/api/tags", timeout=3).read())
        models = [m.get("name", "") for m in d.get("models", [])]
        return {"reachable": True, "host": host, "wanted_model": model,
                "models": models, "model_present": any(model in m for m in models)}
    except Exception as exc:  # noqa: BLE001
        return {"reachable": False, "host": host, "wanted_model": model, "error": str(exc)}
