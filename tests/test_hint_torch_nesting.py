"""M6 regression: hints and torch hooks share ONE parent stack.

When a real torch module is called inside a `with netscope.stage(...)` block, the
module node must nest under that stage, and the module's own children must nest
under it — not leak to a sibling. This interleaving is what makes the sfumato
demo's `plan > ARPlanner > Linear` and `diffuse[b] > Refiner > Linear` hierarchy
correct. Guards against a shared-stack ordering regression.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import netscope


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x):
        return self.lin(x)


def _by_id(g, node_id):
    return next((n for n in g.nodes() if n["id"] == node_id), None)


def test_module_nests_under_enclosing_hint_stage():
    m = Inner().eval()
    with netscope.graph("g") as g, torch.no_grad():
        with netscope.stage("plan"):
            m(torch.randn(2, 8))
    inner = next(n for n in g.nodes() if n["name"] == "Inner")
    plan = next(n for n in g.nodes() if n["name"] == "plan")
    assert inner["parent"] == plan["id"]                 # module under the stage
    lin = next(n for n in g.nodes() if n["name"] == "Linear")
    assert lin["parent"] == inner["id"]                  # its child under it


def test_two_stages_keep_their_modules_separate():
    a = Inner().eval()
    b = Inner().eval()
    with netscope.graph("g") as g, torch.no_grad():
        with netscope.stage("plan"):
            a(torch.randn(2, 8))
        with netscope.branch("diffuse[0]"):
            b(torch.randn(2, 8))
    plan = next(n for n in g.nodes() if n["name"] == "plan")
    diffuse = next(n for n in g.nodes() if n["name"] == "diffuse[0]")
    inners = [n for n in g.nodes() if n["name"] == "Inner"]
    assert {n["parent"] for n in inners} == {plan["id"], diffuse["id"]}
    # every Linear nests under an Inner, never directly under a stage
    for lin in [n for n in g.nodes() if n["name"] == "Linear"]:
        assert _by_id(g, lin["parent"])["name"] == "Inner"


def test_repeated_branch_each_gets_its_own_module_subtree():
    r = Inner().eval()
    with netscope.graph("g") as g, torch.no_grad():
        for i in range(5):
            with netscope.branch(f"diffuse[{i}]"):
                r(torch.randn(2, 8))
    branches = [n for n in g.nodes() if n["attrs"].get("branch")]
    assert len(branches) == 5
    inners = [n for n in g.nodes() if n["name"] == "Inner"]
    assert len(inners) == 5
    assert len({n["parent"] for n in inners}) == 5      # 5 distinct branch parents
