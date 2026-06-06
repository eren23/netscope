"""Phase B4: torch.fx fallback producer.

Some models give a near-empty static AST graph (built via from_config / factories
with no literal layers in user source). When we have a model INSTANCE, torch.fx
can recover real structure WITHOUT running a forward — for the models fx can
trace. fx fails on dynamic control flow (many LLMs, TransformerEncoderLayer), so
this is best-effort: it returns an IR graph on success, None on failure, and the
caller falls back to runtime tracing.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import pytest

from netscope.static.fx_trace import trace_model


def test_fx_traces_a_simple_mlp():
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 16)
            self.act = nn.ReLU()
            self.b = nn.Linear(16, 4)
        def forward(self, x):
            return self.b(self.act(self.a(x)))

    g = trace_model(MLP())
    assert g is not None
    names = [n["name"] for n in g.nodes()]
    assert any("Linear" in nm for nm in names)
    # the module nodes carry their qualified name (from fx's call_module target)
    quals = {(n.get("meta") or {}).get("qualname") for n in g.nodes()}
    assert "a" in quals and "b" in quals
    # dataflow edges follow fx's def-use chain: a -> act -> b
    assert len(g.edges()) >= 2


def test_fx_traces_resnet18():
    pytest.importorskip("torchvision")
    from torchvision.models import resnet18
    g = trace_model(resnet18(weights=None))
    assert g is not None
    convs = sum(1 for n in g.nodes() if "Conv2d" in n["name"])
    assert convs >= 8, f"expected resnet's conv layers, got {convs}"


def test_fx_returns_none_on_untraceable_model():
    """A model fx can't symbolically trace (dynamic control flow) must return
    None, not raise — the caller falls back to runtime tracing."""
    layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64,
                                       batch_first=True)
    g = trace_model(layer)
    assert g is None


def test_fx_marks_source_fx():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)
        def forward(self, x):
            return self.lin(x)
    g = trace_model(Tiny())
    assert g is not None
    assert all(n["source"] == "static" for n in g.nodes())
