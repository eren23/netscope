"""M0: the capture session + contextvars nesting.

`with netscope.graph(name) as g:` opens a capture session. While it is open,
`netscope.active_capture()` returns the live Capture; instrumentors (M1) and the
hints API (M3) emit nodes via `cap.span(...)`, which nests by a contextvars
parent stack. Outside a session, capture is inactive (the zero-overhead gate).
"""
from __future__ import annotations

import netscope
from netscope.core import context as ctx


def test_inactive_by_default():
    assert netscope.active_capture() is None
    assert ctx.is_capturing() is False


def test_graph_session_activates_then_deactivates():
    with netscope.graph("m"):
        assert ctx.is_capturing() is True
        assert netscope.active_capture() is not None
    assert ctx.is_capturing() is False
    assert netscope.active_capture() is None


def test_spans_collected_into_graph():
    with netscope.graph("m") as g:
        cap = netscope.active_capture()
        with cap.span("plan", kind="stage"):
            with cap.span("qwen", kind="model"):
                pass
    names = {n["name"] for n in g.nodes()}
    assert {"plan", "qwen"} <= names


def test_span_nesting_sets_parent_and_contains_edge():
    with netscope.graph("m") as g:
        cap = netscope.active_capture()
        with cap.span("plan", kind="stage"):
            with cap.span("qwen", kind="model"):
                pass
    qwen = next(n for n in g.nodes() if n["name"] == "qwen")
    plan = next(n for n in g.nodes() if n["name"] == "plan")
    assert qwen["parent"] == plan["id"]
    assert plan["parent"] is None
    assert any(
        e["src"] == plan["id"] and e["dst"] == qwen["id"] and e["kind"] == "contains"
        for e in g.edges()
    )


def test_sibling_spans_share_parent():
    with netscope.graph("m") as g:
        cap = netscope.active_capture()
        with cap.span("root", kind="stage"):
            with cap.span("a", kind="op"):
                pass
            with cap.span("b", kind="op"):
                pass
    a = next(n for n in g.nodes() if n["name"] == "a")
    b = next(n for n in g.nodes() if n["name"] == "b")
    root = next(n for n in g.nodes() if n["name"] == "root")
    assert a["parent"] == root["id"] == b["parent"]


def test_span_records_loc_and_meta():
    with netscope.graph("m") as g:
        cap = netscope.active_capture()
        with cap.span("conv", kind="module", loc={"file": "m.py", "line": 7},
                      meta={"shape": [1, 3, 224, 224]}):
            pass
    conv = next(n for n in g.nodes() if n["name"] == "conv")
    assert conv["loc"] == {"file": "m.py", "line": 7}
    assert conv["meta"]["shape"] == [1, 3, 224, 224]
    assert conv["source"] == "runtime"
