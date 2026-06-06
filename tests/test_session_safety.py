"""Phase A5: session robustness.

Nested `netscope.graph()` is almost always a mistake — and worse, it used to
double-instrument (both sessions' global torch hooks fire, so the inner graph
captured every module twice). We now raise a clear error instead of silently
producing a corrupt graph. We also verify global hooks are fully removed after a
session (no leak across sessions).
"""
from __future__ import annotations

import torch
import torch.nn as nn

import pytest

import netscope


def test_nested_graph_raises_clear_error():
    with netscope.graph("outer"):
        with pytest.raises(RuntimeError, match="already capturing|nested"):
            with netscope.graph("inner"):
                pass


def test_session_leaves_no_global_hooks():
    """After a session, torch's global forward hooks return to baseline — no
    leak that would double-capture or fire with a dead capture next time."""
    import torch.nn.modules.module as M
    base_pre = len(M._global_forward_pre_hooks)
    base_post = len(M._global_forward_hooks)
    with netscope.graph("g"):
        nn.Linear(4, 4)(torch.randn(1, 4))
        assert len(M._global_forward_pre_hooks) > base_pre   # installed during
    assert len(M._global_forward_pre_hooks) == base_pre       # removed after
    assert len(M._global_forward_hooks) == base_post


def test_sequential_sessions_each_capture_cleanly():
    """Two SEQUENTIAL (not nested) sessions each capture independently — the
    common case must keep working."""
    m = nn.Linear(4, 4).train(False)
    with netscope.graph("a") as ga:
        m(torch.randn(1, 4))
    with netscope.graph("b") as gb:
        m(torch.randn(1, 4))
    assert len(ga.nodes()) == 1
    assert len(gb.nodes()) == 1


def test_exception_in_forward_does_not_corrupt_the_next_trace():
    """A forward that RAISES mid-run must not leak a half-open span that
    mis-parents / mislabels the next model in the same session (always_call=True
    on the post-hook unwinds it)."""
    class Boom(nn.Module):
        def __init__(s):
            super().__init__(); s.lin = nn.Linear(4, 4)
        def forward(s, x):
            s.lin(x); raise RuntimeError("boom")

    class Clean(nn.Module):
        def __init__(s):
            super().__init__(); s.head = nn.Linear(4, 2)
        def forward(s, x):
            return s.head(x)

    with netscope.graph("g", profile=True) as g, torch.no_grad():
        try:
            Boom()(torch.randn(2, 4))
        except RuntimeError:
            pass
        Clean()(torch.randn(2, 4))

    by_id = {n["id"]: n for n in g.nodes()}
    head = [n for n in g.nodes() if (n.get("meta") or {}).get("qualname") == "head"]
    assert head, "Clean.head should still be captured + correctly labeled"
    ancestry, n = [], head[0]
    while n.get("parent"):
        n = by_id[n["parent"]]; ancestry.append(n["name"])
    assert "Boom" not in ancestry and "Clean" in ancestry   # parented under Clean, not Boom
