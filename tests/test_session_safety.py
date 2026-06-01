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
