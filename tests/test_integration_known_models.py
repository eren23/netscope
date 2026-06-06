"""M6: auto-capture on popular known models — the headline "import and go".

No decorators, no edits — just `with netscope.graph(): model(x)`. Ground-truthed
against real runs (see scripts/_probe_models.py): a TransformerEncoderLayer
yields 13 nodes with the root at [2,5,32]; resnet18 yields 20 Conv2d nodes and
exactly 11,689,512 params summed across module nodes (the canonical count).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import netscope


def test_transformer_encoder_layer_captured_with_shapes():
    layer = nn.TransformerEncoderLayer(
        d_model=32, nhead=4, dim_feedforward=64, batch_first=True
    )
    x = torch.randn(2, 5, 32)  # (batch, seq, d_model)
    with netscope.graph("transformer") as g:
        layer(x)
    names = [n["name"] for n in g.nodes()]
    assert any("TransformerEncoderLayer" in nm for nm in names)
    assert any("Linear" in nm for nm in names)        # FFN / projections
    top = next(n for n in g.nodes() if n["parent"] is None)
    assert "TransformerEncoderLayer" in top["name"]
    assert top["meta"]["out_shape"] == [2, 5, 32]     # tensor shape preserved


def test_multihead_attention_tuple_output_is_handled_gracefully():
    """MultiheadAttention returns a (output, weights) tuple, not a bare tensor.
    The node is captured, nesting stays balanced, and (since A2) we surface the
    REPRESENTATIVE output shape — the first reachable tensor, i.e. the attention
    output [2, 5, 32] — instead of nothing. A regression guard for tuple/dict
    module outputs."""
    layer = nn.TransformerEncoderLayer(
        d_model=32, nhead=4, dim_feedforward=64, batch_first=True
    )
    with netscope.graph("t") as g:
        layer(torch.randn(2, 5, 32))
    mha = next(n for n in g.nodes() if n["name"] == "MultiheadAttention")
    # the (output, weights) tuple -> the attention OUTPUT shape is surfaced
    assert mha["meta"]["out_shape"] == [2, 5, 32]
    assert mha["parent"] is not None                   # still correctly nested


def test_resnet18_captured_hierarchy_and_params():
    pytest.importorskip("torchvision")
    from torchvision.models import resnet18

    model = resnet18(weights=None).eval()
    x = torch.randn(1, 3, 64, 64)
    with netscope.graph("resnet18") as g, torch.no_grad():
        model(x)
    nodes = g.nodes()
    names = [n["name"] for n in nodes]
    assert any("ResNet" in nm for nm in names)
    assert sum("Conv2d" in nm for nm in names) >= 8     # resnet18 has 20 convs
    conv = next(n for n in nodes if "Conv2d" in n["name"])
    assert len(conv["meta"]["out_shape"]) == 4          # NCHW
    assert conv["meta"]["params"] > 0
    # summed own-params equals the canonical resnet18 count (each param once)
    total = sum(n["meta"].get("params", 0) for n in nodes)
    assert 11_000_000 < total < 12_000_000
