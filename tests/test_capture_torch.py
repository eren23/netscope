"""M1: torch auto-instrumentation.

Importing netscope registers (via wrapt post-import hooks) a session-scoped
instrumentor that adds torch's GLOBAL module forward hooks while a capture
session is open, and removes them on exit (zero hooks => zero overhead outside
a session). Each nn.Module forward becomes a node nested by module hierarchy,
annotated with real input/output tensor shapes — no decorators, no model edits.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import netscope


def test_traces_nested_module_forwards():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    x = torch.randn(3, 4)
    with netscope.graph("m") as g:
        model(x)
    names = [n["name"] for n in g.nodes()]
    assert any("Sequential" in nm for nm in names)
    assert sum("Linear" in nm for nm in names) == 2
    assert any("ReLU" in nm for nm in names)


def test_records_output_shape_and_kind():
    model = nn.Linear(4, 8)
    x = torch.randn(3, 4)
    with netscope.graph("m") as g:
        model(x)
    lin = next(n for n in g.nodes() if "Linear" in n["name"])
    assert lin["kind"] == "module"
    assert lin["meta"]["out_shape"] == [3, 8]


def test_hierarchy_parent_is_container():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
    x = torch.randn(2, 4)
    with netscope.graph("m") as g:
        model(x)
    seq = next(n for n in g.nodes() if "Sequential" in n["name"])
    lin = next(n for n in g.nodes() if "Linear" in n["name"])
    assert lin["parent"] == seq["id"]


def test_hooks_removed_after_session():
    """A forward outside any session must record nothing (hooks removed)."""
    model = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    with netscope.graph("a") as g1:
        model(x)
    n_after_session = len(g1.nodes())
    assert n_after_session > 0
    # forward outside any session: no active capture, closed graph untouched
    y = model(x)
    assert tuple(y.shape) == (2, 4)
    assert netscope.active_capture() is None
    assert len(g1.nodes()) == n_after_session
