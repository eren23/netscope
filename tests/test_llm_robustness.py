"""Phase E: LLM client + prompt robustness.

The assistant must not hang the editor, silently fail on a transient hiccup, or
blow the model's context budget on a huge graph. These cover retry/backoff,
configurable timeout, neighbour/source truncation, and the readable-error edge
cases (missing node, relocated source, malformed CLI input).
"""
from __future__ import annotations

import json

import pytest

from netscope.core.ir import NVGraph
from netscope.llm.provider import Provider, RETRYABLE_STATUS
from netscope.llm.prompts import build_messages


def _provider():
    return Provider(base_url="http://x/v1", model="m", api_key="k", extra_headers={})


def _graph():
    g = NVGraph("g")
    g.add_node("a", kind="module", name="Encoder", source="runtime",
               loc={"file": "model.py", "line": 5},
               meta={"out_shape": [1, 256], "qualname": "encoder"})
    g.add_node("b", kind="module", name="Head", source="runtime",
               loc={"file": "model.py", "line": 9},
               meta={"in_shape": [1, 128], "qualname": "head"})
    g.add_edge("a", "b", kind="dataflow", source="runtime")
    return g


# ---- retry / backoff on transient failures -------------------------------
def test_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def flaky_transport(url, headers, body):
        calls["n"] += 1
        if calls["n"] < 3:
            import urllib.error, io
            raise urllib.error.HTTPError(url, 429, "rate limited", {}, io.BytesIO(b"slow down"))
        return {"choices": [{"message": {"content": "ok"}}]}

    p = _provider()
    out = p.complete([{"role": "user", "content": "hi"}],
                     _transport=flaky_transport, retries=3, backoff_base=0)
    assert out == "ok"
    assert calls["n"] == 3   # two 429s, third succeeds


def test_does_not_retry_on_client_error_400():
    calls = {"n": 0}

    def bad_request(url, headers, body):
        calls["n"] += 1
        import urllib.error, io
        raise urllib.error.HTTPError(url, 400, "bad", {}, io.BytesIO(b"bad request"))

    p = _provider()
    with pytest.raises(RuntimeError):
        p.complete([{"role": "user", "content": "hi"}],
                   _transport=bad_request, retries=3, backoff_base=0)
    assert calls["n"] == 1   # 400 is a client error -> no retry


def test_retryable_status_set_is_sane():
    assert 429 in RETRYABLE_STATUS and 503 in RETRYABLE_STATUS
    assert 400 not in RETRYABLE_STATUS and 401 not in RETRYABLE_STATUS


def test_gives_up_after_max_retries():
    calls = {"n": 0}

    def always_503(url, headers, body):
        calls["n"] += 1
        import urllib.error, io
        raise urllib.error.HTTPError(url, 503, "down", {}, io.BytesIO(b"unavailable"))

    p = _provider()
    with pytest.raises(RuntimeError):
        p.complete([{"role": "user", "content": "hi"}],
                   _transport=always_503, retries=2, backoff_base=0)
    assert calls["n"] == 3   # initial + 2 retries


# ---- prompt truncation: big graphs don't blow the context ----------------
def test_neighbours_block_is_truncated():
    g = NVGraph("big")
    g.add_node("hub", kind="module", name="Hub", source="runtime",
               meta={"qualname": "hub"})
    for i in range(200):
        g.add_node(f"u{i}", kind="module", name=f"Up{i}", source="runtime",
                   meta={"qualname": f"up{i}"})
        g.add_edge(f"u{i}", "hub", kind="dataflow", source="runtime")
    msgs = build_messages(g, "hub", question="explain")
    blob = "\n".join(m["content"] for m in msgs)
    # all 200 upstream names must NOT be dumped verbatim
    assert "up199" not in blob or "more" in blob.lower()
    # a sane cap: the prompt shouldn't balloon past a few KB for one node
    assert len(blob) < 6000, f"prompt too large: {len(blob)} chars"


def test_source_block_caps_huge_lines(tmp_path):
    f = tmp_path / "big.py"
    f.write_text("x = 1\n" + ("# " + "z" * 5000 + "\n") * 3 + "y = 2\n")
    g = NVGraph("s")
    g.add_node("n", kind="module", name="X", source="runtime",
               loc={"file": str(f), "line": 3}, meta={"qualname": "x"})
    msgs = build_messages(g, "n", question="explain")
    blob = "\n".join(m["content"] for m in msgs)
    assert len(blob) < 4000   # the 5000-char comment lines are clipped


# ---- readable errors -----------------------------------------------------
def test_explain_unknown_node_raises_keyerror_with_name():
    g = _graph()
    with pytest.raises(KeyError):
        build_messages(g, "does-not-exist", question="explain")


def test_build_messages_handles_missing_source_file():
    """A node whose loc points at a moved/deleted file still builds a prompt —
    the source block is just omitted, no crash."""
    g = NVGraph("g")
    g.add_node("a", kind="module", name="Enc", source="runtime",
               loc={"file": "/nonexistent/gone.py", "line": 5},
               meta={"qualname": "enc", "out_shape": [1, 8]})
    msgs = build_messages(g, "a", question="explain")
    blob = "\n".join(m["content"] for m in msgs)
    assert "Enc" in blob
    # the "source near …" quoted-lines block is omitted (file unreadable)
    assert "source near" not in blob


def test_cli_reports_missing_node_clearly(tmp_path):
    from netscope.llm.__main__ import main
    gp = tmp_path / "g.json"
    gp.write_text(json.dumps(NVGraph("g").to_dict()))
    # no key set, but the node-not-found path is independent of the provider
    import os
    for k in ("NETSCOPE_LLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        os.environ.pop(k, None)
    rc = main([str(gp), "nonexistent-node", "explain"])
    assert rc != 0   # nonzero exit on a bad node


def test_cli_reports_malformed_json(tmp_path):
    from netscope.llm.__main__ import main
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    rc = main([str(bad), "n", "explain"])
    assert rc != 0
