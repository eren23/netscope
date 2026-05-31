"""M7: hint-flow dataflow edges between stage siblings.

Tensor-identity links modules whose tensors literally flow; but across hinted
stages the tensors are often rebuilt (e.g. `plan + noise`), so we ALSO infer
flow from hint semantics at session finalize: plain stages chain in declaration
order; a run of `branch`es fans OUT from the preceding frontier and fans IN to
the following stage (the `reduce`). This yields sfumato's plan -> diffuse[b] ->
vote without the user wiring anything.
"""
from __future__ import annotations

import netscope


def _pairs(g):
    id2name = {n["id"]: n["name"] for n in g.nodes()}
    return {(id2name[e["src"]], id2name[e["dst"]]) for e in g.edges() if e["kind"] == "dataflow"}


def test_cmajc_plan_branches_vote_flow():
    with netscope.graph("g") as g:
        with netscope.stage("plan"):
            pass
        for b in range(3):
            with netscope.branch(f"d{b}"):
                pass
        with netscope.reduce("vote"):
            pass
    pairs = _pairs(g)
    for b in range(3):
        assert ("plan", f"d{b}") in pairs       # fan out
        assert (f"d{b}", "vote") in pairs        # fan in
    assert ("plan", "vote") not in pairs         # plan does not skip the branches


def test_plain_stages_chain_in_order():
    with netscope.graph("g") as g:
        with netscope.stage("a"):
            pass
        with netscope.stage("b"):
            pass
        with netscope.stage("c"):
            pass
    pairs = _pairs(g)
    assert ("a", "b") in pairs
    assert ("b", "c") in pairs
    assert ("a", "c") not in pairs               # sequential, not skip


def test_branches_without_reduce_fan_out_only():
    with netscope.graph("g") as g:
        with netscope.stage("plan"):
            pass
        for b in range(2):
            with netscope.branch(f"d{b}"):
                pass
    pairs = _pairs(g)
    assert ("plan", "d0") in pairs
    assert ("plan", "d1") in pairs
    assert ("d0", "d1") not in pairs             # branches are parallel, not chained


def test_single_stage_has_no_flow_edges():
    with netscope.graph("g") as g:
        with netscope.stage("solo"):
            pass
    assert _pairs(g) == set()
