"""M7: tensor-identity dataflow edges.

Hierarchy (`contains`) shows what's-inside-what; dataflow shows what-flows-where.
The torch instrumentor records each module output tensor's id, and when a later
module receives a tensor with that id (and matching shape, to guard against
id() reuse), draws a `dataflow` edge producer -> consumer. This turns a bag of
nested boxes into an actual graph: Linear -> ReLU -> Linear.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import netscope


def _name_pairs(g, kind):
    id2name = {n["id"]: n["name"] for n in g.nodes()}
    return {(id2name[e["src"]], id2name[e["dst"]]) for e in g.edges() if e["kind"] == kind}


def test_sequential_has_leaf_to_leaf_dataflow_chain():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    with netscope.graph("g") as g:
        model(torch.randn(3, 4))
    pairs = _name_pairs(g, "dataflow")
    assert ("Linear", "ReLU") in pairs
    assert ("ReLU", "Linear") in pairs


def test_no_dataflow_edge_from_container_to_its_child():
    """Parent/child are already linked by `contains`; the input tensor a parent
    and its first child share must NOT create a spurious parent->child dataflow."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
    with netscope.graph("g") as g:
        model(torch.randn(3, 4))
    id2name = {n["id"]: n["name"] for n in g.nodes()}
    for e in g.edges():
        if e["kind"] == "dataflow":
            assert not (id2name[e["src"]] == "Sequential")  # container is never a df source here


def test_dataflow_edge_carries_tensor_shape():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
    with netscope.graph("g") as g:
        model(torch.randn(3, 4))
    df = [e for e in g.edges() if e["kind"] == "dataflow"]
    assert df, "expected at least one dataflow edge"
    assert any(e.get("tensor_meta", {}).get("shape") == [3, 8] for e in df)


def test_two_independent_inputs_do_not_cross_link():
    """Two separate forwards in the same session must not link via stale ids."""
    a = nn.Linear(4, 4)
    b = nn.Linear(4, 4)
    with netscope.graph("g") as g:
        a(torch.randn(2, 4))
        b(torch.randn(2, 4))
    # a and b consumed graph-external inputs; no dataflow edge between them
    pairs = _name_pairs(g, "dataflow")
    assert ("Linear", "Linear") not in pairs
