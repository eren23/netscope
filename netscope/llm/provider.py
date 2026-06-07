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
import json
import os
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# a cheap, fast, widely-available default on OpenRouter; override via env/config.
DEFAULT_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_TIMEOUT = 30          # seconds per attempt (was an un-tunable 60)
DEFAULT_RETRIES = 2          # extra attempts after the first, on transient errors

# transient server-side / network conditions worth retrying. A 4xx client error
# (400 bad request, 401 bad key) is NOT retryable — it'll fail the same way again.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

# message = {"role": "system"|"user"|"assistant", "content": str}
Message = Dict[str, str]
# a transport maps (url, headers, json-body-bytes) -> parsed response dict.
# Injectable so tests never hit the network.
Transport = Callable[[str, Dict[str, str], bytes], dict]


def _make_http_transport(timeout: float) -> Transport:
    def _http_transport(url: str, headers: Dict[str, str], body: bytes) -> dict:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    return _http_transport


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
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        backoff_base: float = 0.8,
        _transport: Optional[Transport] = None,
        _sleep: Callable[[float], None] = time.sleep,
    ) -> str:
        """POST the chat-completion and return the assistant's text.

        Retries transient failures (429/503/timeout/…) up to `retries` times with
        exponential backoff; a client error (400/401) is NOT retried. Raises
        RuntimeError after the final failure (callers gate on available())."""
        transport = _transport or _make_http_transport(timeout)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        body = json.dumps({
            "model": self.model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens,
        }).encode("utf-8")

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                data = transport(url, headers, body)
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:300]
                except Exception:
                    detail = str(e)
                last_err = RuntimeError(f"LLM request failed ({e.code}): {detail}")
                if e.code in RETRYABLE_STATUS and attempt < retries:
                    _sleep(backoff_base * (2 ** attempt))
                    continue
                raise last_err from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                # network / timeout — transient, retry.
                last_err = RuntimeError(f"LLM request failed: {e}")
                if attempt < retries:
                    _sleep(backoff_base * (2 ** attempt))
                    continue
                raise last_err from e
            except Exception as e:
                raise RuntimeError(f"LLM request failed: {e}") from e
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                raise RuntimeError(f"unexpected LLM response shape: {str(data)[:200]}") from e
        raise last_err or RuntimeError("LLM request failed")
