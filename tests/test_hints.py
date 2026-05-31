"""M3: optional hint markers.

Auto-tracing captures *calls*; it cannot know that five loop iterations are
"branches" that get majority-voted, or that a block is the "plan" stage. The
hints API lets a user name those semantic regions. Each marker is BOTH a
decorator (`@nv.stage("plan")`) and a context manager (`with nv.stage(...)`),
is reentrancy-safe (sfumato calls the diffuse stage 5x), records the decorated
function's source loc for free, and is a pure pass-through outside a session.
"""
from __future__ import annotations

import netscope


def test_stage_decorator_creates_named_stage():
    @netscope.stage("plan")
    def plan(x):
        return x + 1

    with netscope.graph("g") as g:
        assert plan(1) == 2
    assert any(n["name"] == "plan" and n["kind"] == "stage" for n in g.nodes())


def test_stage_decorator_records_function_loc():
    @netscope.stage("plan")
    def plan():
        return 0

    with netscope.graph("g") as g:
        plan()
    node = next(n for n in g.nodes() if n["name"] == "plan")
    assert node["loc"]["file"].endswith("test_hints.py")
    assert isinstance(node["loc"]["line"], int)


def test_stage_context_manager_with_reduce_attr():
    with netscope.graph("g") as g:
        with netscope.stage("vote", reduce=True):
            pass
    vote = next(n for n in g.nodes() if n["name"] == "vote")
    assert vote["attrs"].get("reduce") is True


def test_decorator_is_reentrancy_safe_across_repeated_calls():
    @netscope.stage("diffuse")
    def diffuse(i):
        return i

    with netscope.graph("g") as g:
        for i in range(5):
            diffuse(i)
    assert sum(n["name"] == "diffuse" for n in g.nodes()) == 5


def test_markers_passthrough_outside_session():
    @netscope.stage("plan")
    def plan(x):
        return x * 3

    assert plan(2) == 6  # no active capture -> just runs
    with netscope.stage("vote"):
        pass  # must not raise


def test_branch_marker_sets_branch_attr():
    with netscope.graph("g") as g:
        with netscope.branch("b0"):
            pass
    b = next(n for n in g.nodes() if n["name"] == "b0")
    assert b["attrs"].get("branch") is True
