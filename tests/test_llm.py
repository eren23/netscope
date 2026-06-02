"""M3: the LLM assistant layer — provider-agnostic, OpenAI-compatible, gated.

netscope talks to ANY OpenAI-compatible chat endpoint (OpenRouter by default,
or OpenAI / Together / Groq / a local server) over one thin client. Config comes
from env; with no key the layer is simply unavailable and the rest of netscope
keeps working offline. Tests never touch the network — a fake transport is
injected, so we assert the request shape + response parsing deterministically.
"""
from __future__ import annotations

import json

import pytest

from netscope.core.ir import NVGraph
from netscope.llm import available, explain, LLMUnavailable
from netscope.llm.provider import Provider
from netscope.llm.prompts import build_messages


# ---- config / availability ------------------------------------------------
_KEYS = ["NETSCOPE_LLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY",
         "NETSCOPE_LLM_BASE_URL", "NETSCOPE_LLM_MODEL"]


def _clear(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)


def test_unavailable_without_any_key(monkeypatch):
    _clear(monkeypatch)
    assert available() is False
    assert Provider.from_env() is None


def test_from_env_reads_openrouter_key_with_defaults(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    p = Provider.from_env()
    assert p is not None
    assert p.api_key == "sk-or-test"
    assert "openrouter.ai" in p.base_url        # default gateway
    assert p.model                               # a default cheap model is set
    assert available() is True


def test_env_precedence_and_overrides(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    monkeypatch.setenv("NETSCOPE_LLM_API_KEY", "sk-explicit")  # wins
    monkeypatch.setenv("NETSCOPE_LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("NETSCOPE_LLM_MODEL", "gpt-4o-mini")
    p = Provider.from_env()
    assert p.api_key == "sk-explicit"
    assert p.base_url == "https://api.openai.com/v1"
    assert p.model == "gpt-4o-mini"


# ---- the client posts the OpenAI shape + parses the reply -----------------
def test_complete_posts_openai_chat_shape(monkeypatch):
    seen = {}

    def fake_transport(url, headers, body):
        seen["url"] = url
        seen["headers"] = headers
        seen["payload"] = json.loads(body.decode())
        return {"choices": [{"message": {"role": "assistant", "content": "hello!"}}]}

    p = Provider(base_url="https://openrouter.ai/api/v1", model="x/y",
                 api_key="sk-test", extra_headers={})
    out = p.complete([{"role": "user", "content": "hi"}], _transport=fake_transport)

    assert out == "hello!"
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["payload"]["model"] == "x/y"
    assert seen["payload"]["messages"][0]["content"] == "hi"
    assert seen["headers"]["Authorization"] == "Bearer sk-test"


# ---- the prompt is GROUNDED in the IR + source ----------------------------
def _graph_with_mismatch():
    g = NVGraph("m")
    g.add_node("a", kind="module", name="Encoder", source="runtime",
               loc={"file": "model.py", "line": 5}, meta={"out_shape": [1, 256], "qualname": "encoder"})
    g.add_node("b", kind="module", name="Head", source="runtime",
               loc={"file": "model.py", "line": 9}, meta={"in_shape": [1, 128], "qualname": "head"})
    g.add_edge("a", "b", kind="dataflow", source="runtime")
    return g


def test_build_messages_grounds_in_node_and_warning():
    g = _graph_with_mismatch()
    msgs = build_messages(g, "b", question="why_warn")
    blob = "\n".join(m["content"] for m in msgs)
    # mentions the node, its declared shape, and the concrete warning
    assert "Head" in blob or "head" in blob
    assert "128" in blob
    assert any(m["role"] == "system" for m in msgs)
    # references the mismatch the detector found
    assert "256" in blob and "128" in blob


def test_build_messages_explain_includes_source_when_readable(tmp_path):
    src = tmp_path / "model.py"
    src.write_text("import torch.nn as nn\n\nclass Net(nn.Module):\n    def __init__(self):\n        self.encoder = nn.Linear(8, 256)\n")
    g = NVGraph("m")
    g.add_node("a", kind="module", name="Linear", source="runtime",
               loc={"file": str(src), "line": 5}, meta={"out_shape": [1, 256], "qualname": "encoder"})
    msgs = build_messages(g, "a", question="explain")
    blob = "\n".join(m["content"] for m in msgs)
    assert "nn.Linear(8, 256)" in blob          # the real source line is quoted


# ---- public explain(): uses the provider, gates when unavailable ----------
def test_explain_returns_model_text_with_injected_provider():
    g = _graph_with_mismatch()
    p = Provider(base_url="http://x/v1", model="m", api_key="k", extra_headers={})

    def fake_transport(url, headers, body):
        return {"choices": [{"message": {"content": "The head expects 128 but gets 256."}}]}

    out = explain(g, "b", question="why_warn", provider=p, _transport=fake_transport)
    assert "128" in out


def test_explain_raises_when_unavailable(monkeypatch):
    _clear(monkeypatch)
    g = _graph_with_mismatch()
    with pytest.raises(LLMUnavailable):
        explain(g, "b")
