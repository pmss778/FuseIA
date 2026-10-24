"""Thin OpenAI-compatible client for FuseIA (SU8 / DeepSeek official).

Credentials resolve from the environment first, then the legacy owner-only
config file, so FuseIA runs with the same contract as the legacy projects.

Wire conventions that matter:

- **Streaming.** Non-streaming responses emit no bytes while the model thinks,
  so generation past the gateway's Cloudflare 120 s proxy window dies as 524.
  Streaming survives as long as *some* byte flows; if first-token latency
  itself exceeds the window (long reasoning that the gateway buffers), no
  client-side trick helps.
- **reasoning_content.** DeepSeek thinking models stream their chain of
  thought in ``delta.reasoning_content`` and the answer in ``delta.content``.
  Both are collected; an answer that ends with empty content is an error, not
  an empty success.
- **No max_tokens to remote providers.** A token cap truncates the reasoning
  chain and the surface content comes back empty. Only loopback (local
  llama.cpp) endpoints accept the env cap.
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import openai
from openai import OpenAI

log = logging.getLogger("FuseIA.llm")

_LEGACY_ENV = Path.home() / ".config" / "llm_index_selection" / "remote_api.env"
_THINKING_OFF = {"0", "false", "disabled", "off"}
_EXPORT_RE = re.compile(r"^export\s+")
_INLINE_COMMENT_RE = re.compile(r"\s+#.*$")
# Retry only these; 4xx contract errors and programming errors must surface.
_RETRYABLE = (openai.APIConnectionError, openai.InternalServerError, openai.RateLimitError)


def _private() -> Dict[str, str]:
    values: Dict[str, str] = {}
    if _LEGACY_ENV.exists():
        for raw in _LEGACY_ENV.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            line = _EXPORT_RE.sub("", line)
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = _INLINE_COMMENT_RE.sub("", value).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[key.strip()] = value
    return values


def _is_loopbase(base_url: str) -> bool:
    return any(h in base_url for h in ("127.0.0.1", "localhost", "::1"))


def resolve_config() -> Dict[str, Any]:
    private = _private()
    provider = os.getenv("LLM_PROVIDER", "fuseia").strip().lower()
    if provider == "fuseia":
        # FuseIA cloud inference server (MUSA 3B + role adapters), reached via
        # the local forward tunnel. Greedy decoding server-side; non-streaming.
        return {
            "provider": "fuseia",
            "base_url": os.getenv("FUSEIA_LLM_BASE_URL", "http://127.0.0.1:18000/v1"),
            "model": os.getenv("FUSEIA_LLM_MODEL", "FuseIA-trained"),
            "api_key": "local",
            "timeout": float(os.getenv("OPENAI_TIMEOUT", "600")),
        }
    if provider not in {"su8", "deepseek"}:
        log.warning("unknown LLM_PROVIDER=%r, falling back to su8", provider)
        provider = "su8"
    if provider == "deepseek":
        return {
            "provider": "deepseek",
            "base_url": (
                os.getenv("DEEPSEEK_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or private.get("DEEPSEEK_BASE_URL")
                or private.get("OPENAI_BASE_URL")
                or "https://api.deepseek.com"
            ),
            "model": os.getenv("OPENAI_MODEL") or private.get("DEEPSEEK_MODEL", "deepseek-chat"),
            "api_key": os.getenv("DEEPSEEK_API_KEY") or private.get("DEEPSEEK_API_KEY", ""),
            "timeout": float(os.getenv("OPENAI_TIMEOUT", "600")),
        }
    return {
        "provider": "su8",
        # Standard entry, proven stable across TPC-H and JOB runs (2026-09-26).
        # Deliberately NOT read from the legacy config file, whose www entry
        # stalls on JOB-scale prompts. SU8_BASE_URL env stays as the override.
        "base_url": os.getenv("SU8_BASE_URL") or "https://us2.su8.codes/v1",
        "model": os.getenv("SU8_MODEL") or private.get("SU8_MODEL", "deepseek-v4-flash"),
        "api_key": os.getenv("SU8_API_KEY") or private.get("SU8_API_KEY", ""),
        "timeout": float(os.getenv("OPENAI_TIMEOUT", "600")),
    }


@lru_cache(maxsize=4)
def _get_client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)


def describe_backend() -> str:
    cfg = resolve_config()
    return f'{cfg["provider"]}:{cfg["model"]}@{cfg["base_url"]}'


def chat(messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    """One streaming chat completion; returns the assistant content.

    ``reasoning_content`` deltas are collected for diagnostics but not returned.
    Raises when the completion carries no content (truncated reasoning, gateway
    buffering), which callers must treat as a failed call rather than an empty
    answer.
    """

    cfg = resolve_config()
    if not cfg["api_key"]:
        raise RuntimeError(
            f"no API key for provider={cfg['provider']}; "
            f"set SU8_API_KEY/DEEPSEEK_API_KEY or {_LEGACY_ENV}"
        )
    client = _get_client(cfg["base_url"], cfg["api_key"], cfg["timeout"])

    # Streaming is the default for su8/deepseek (bytes flowing keeps proxy
    # windows alive). The FuseIA cloud server always answers non-streaming.
    non_stream = os.getenv("FUSEIA_STREAM", "1").strip().lower() in {"0", "false", "off"}
    kwargs: Dict[str, Any] = {"stream": not non_stream}
    if cfg["provider"] == "fuseia":
        kwargs["stream"] = False
    if _is_loopbase(cfg["base_url"]):
        max_tokens = os.getenv("LLM_MAX_COMPLETION_TOKENS", "").strip()
        if max_tokens:
            kwargs["max_tokens"] = int(max_tokens)
    thinking_disabled = "deepseek" in cfg["model"].lower() and os.getenv(
        "SU8_THINKING" if cfg["provider"] == "su8" else "DEEPSEEK_THINKING", ""
    ).strip().lower() in _THINKING_OFF
    if thinking_disabled:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    else:
        # Reasoning-mode endpoints may reject temperature; omit it by default.
        kwargs["temperature"] = temperature

    retries = max(0, int(os.getenv("LLM_CALL_MAX_RETRIES", "3")))
    backoff = float(os.getenv("LLM_CALL_BACKOFF_S", "5"))
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        content_parts: List[str] = []
        reasoning_len = 0
        try:
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=messages,
                **kwargs,
            )
            if not kwargs.get("stream"):
                message = response.choices[0].message
                content_parts.append(message.content or "")
                reasoning_len = len(
                    getattr(message, "reasoning_content", "") or "")
            else:
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta is None:
                        continue
                    if delta.content:
                        content_parts.append(delta.content)
                    elif getattr(delta, "reasoning_content", None):
                        reasoning_len += len(delta.reasoning_content)
        except _RETRYABLE as exc:
            last_error = exc
            if content_parts:
                log.warning("stream broke after %d chars of content: %s",
                            sum(map(len, content_parts)), str(exc)[:200])
            if attempt < retries:
                wait = min(backoff * (2 ** attempt) + random.uniform(0, backoff), 120)
                log.warning("attempt %d/%d failed (%s); retry in %.0fs",
                            attempt + 1, retries + 1, str(exc)[:200], wait)
                time.sleep(wait)
                continue
            raise
        content = "".join(content_parts)
        if not content:
            # Empty content is transient (stalled/truncated generation at the
            # origin): retry like any other attempt failure instead of raising.
            last_error = RuntimeError(
                "completion returned empty content "
                f"(reasoning chars streamed: {reasoning_len}); "
                "the answer was likely truncated inside the reasoning phase")
            if attempt < retries:
                wait = min(backoff * (2 ** attempt) + random.uniform(0, backoff), 120)
                log.warning("attempt %d/%d returned empty content; retry in %.0fs",
                            attempt + 1, retries + 1, wait)
                time.sleep(wait)
            continue
        log.debug("chat ok: %d content chars, %d reasoning chars",
                  len(content), reasoning_len)
        return content
    raise last_error or RuntimeError("unreachable retry state")
