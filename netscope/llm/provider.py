"""Provider-agnostic LLM client — any OpenAI-compatible /chat/completions endpoint.

One thin client serves OpenRouter (the default gateway -> many cheap models like
Gemini Flash), OpenAI, Together, Groq, or a local server: they all speak the same
`POST {base_url}/chat/completions` shape. No SDK, no new dependency — stdlib
urllib only. Config is read from env (`Provider.from_env()`); with no key the
provider is None and the LLM layer is simply unavailable (the rest of netscope
keeps working offline).

Env (first non-empty wins for the key):
    NETSCOPE_LLM_API_KEY | OPENROUTER_API_KEY | OPENAI_API_KEY
    NETSCOPE_LLM_BASE_URL  (default: https://openrouter.ai/api/v1)
    NETSCOPE_LLM_MODEL     (default: a cheap, capable model)
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# a cheap, fast, widely-available default on OpenRouter; override via env/config.
DEFAULT_MODEL = "google/gemini-2.0-flash-001"

# message = {"role": "system"|"user"|"assistant", "content": str}
Message = Dict[str, str]
# a transport maps (url, headers, json-body-bytes) -> parsed response dict.
# Injectable so tests never hit the network.
Transport = Callable[[str, Dict[str, str], bytes], dict]


def _http_transport(url: str, headers: Dict[str, str], body: bytes) -> dict:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


class Provider:
    """An OpenAI-compatible chat endpoint + the model to call on it."""

    def __init__(self, *, base_url: str, model: str, api_key: str,
                 extra_headers: Optional[Dict[str, str]] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.extra_headers = extra_headers or {}

    @classmethod
    def from_env(cls) -> Optional["Provider"]:
        key = (
            os.environ.get("NETSCOPE_LLM_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not key:
            return None
        base = os.environ.get("NETSCOPE_LLM_BASE_URL") or DEFAULT_BASE_URL
        model = os.environ.get("NETSCOPE_LLM_MODEL") or DEFAULT_MODEL
        # OpenRouter likes (optional) attribution headers; harmless elsewhere.
        extra = {
            "HTTP-Referer": "https://github.com/eren23/netscope",
            "X-Title": "netscope",
        }
        return cls(base_url=base, model=model, api_key=key, extra_headers=extra)

    def complete(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 700,
        _transport: Optional[Transport] = None,
    ) -> str:
        """POST the chat-completion and return the assistant's text. Raises
        RuntimeError on a transport/format failure (callers gate on available())."""
        transport = _transport or _http_transport
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        try:
            data = transport(url, headers, body)
        except urllib.error.HTTPError as e:  # surface a readable error
            detail = e.read().decode("utf-8", "replace")[:300] if hasattr(e, "read") else str(e)
            raise RuntimeError(f"LLM request failed ({e.code}): {detail}") from e
        except Exception as e:
            raise RuntimeError(f"LLM request failed: {e}") from e
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"unexpected LLM response shape: {str(data)[:200]}") from e
