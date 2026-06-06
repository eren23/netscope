"""The transformer role-lens: classify nodes by architectural component.

Verified against a REAL traced block + the standard HF naming, so the classifier
matches what models actually emit, not a guess.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import netscope
from netscope.enrich.roles import node_role, role_counts


def test_classifies_a_real_transformer_block():
    # train mode so the submodule hooks fire (eval fuses torch's fast path).
    m = nn.TransformerEncoderLayer(32, 4, batch_first=True)
    with netscope.graph("t") as g, torch.no_grad():
        m(torch.randn(2, 5, 32))
    c = role_counts(g)
    assert c.get("attention", 0) >= 1     # self_attn
    assert c.get("norm", 0) >= 2          # norm1, norm2
    assert c.get("linear", 0) >= 2        # the feed-forward linears


def test_qualname_path_classifies_children_by_their_block():
    # a leaf's own name is generic; the parent path decides the block (HF naming).
    role = lambda q, name="Linear": node_role({"name": name, "meta": {"qualname": q}})
    assert role("model.layers.0.self_attn.q_proj") == "attention"
    assert role("model.layers.0.mlp.gate_proj") == "mlp"
    assert role("model.layers.0.input_layernorm", "RMSNorm") == "norm"
    assert role("model.embed_tokens", "Embedding") == "embedding"


def test_block_role_wins_over_generic_linear():
    # q_proj also matches "proj" (linear), but attention is checked first.
    assert node_role({"name": "Linear", "meta": {"qualname": "h.3.attn.c_proj"}}) == "attention"
    # a plain head Linear, no block context -> linear
    assert node_role({"name": "Linear", "meta": {"qualname": "classifier"}}) == "linear"
    assert node_role({"name": "GELU", "meta": {}}) == "activation"


def test_roles_is_public():
    m = nn.TransformerEncoderLayer(32, 4, batch_first=True)
    with netscope.graph("t") as g, torch.no_grad():
        m(torch.randn(2, 5, 32))
    assert netscope.roles(g) == role_counts(g)
    assert "attention" in netscope.roles(g)
