"""M3: static AST producer + merge-by-loc.

The AST producer reads a source file WITHOUT running it and recovers structure
that runtime tracing is blind to: `for _ in range(N)` loops (the branch fan-out)
and `Counter(...).most_common(...)` consensus (the vote) — exactly sfumato's
cmajc shape. `merge` then fuses a static graph onto a runtime graph by matching
source `loc`, so a node gets the runtime's real shapes AND the static structure.
"""
from __future__ import annotations

import textwrap

from netscope.core.ir import NVGraph
from netscope.core.merge import merge
from netscope.static.ast_producer import analyze_source

SFUMATO_LIKE = textwrap.dedent(
    '''
    def run(problem):
        branches = []
        for b in range(5):
            branches.append(diffuse(problem, seed=b))
        answers = [extract(b) for b in branches]
        winner = Counter(answers).most_common(1)[0][0]
        return winner
    '''
)


def test_detects_range_loop_with_count():
    g = analyze_source(SFUMATO_LIKE, filename="runner.py")
    loops = [n for n in g.nodes() if n["attrs"].get("repeat")]
    assert len(loops) == 1
    assert loops[0]["attrs"]["repeat"] == 5
    assert loops[0]["loc"]["file"] == "runner.py"


# --- real sfumato patterns (variable range, bound Counter) -------------------
SFUMATO_REAL = textwrap.dedent(
    '''
    def run_condition(problem, n_branches):
        from collections import Counter
        branches = []
        for b in range(n_branches):
            branches.append(diffuse(problem, seed=b))
        answers = [extract(b) for b in branches]
        counts = Counter(a for a in answers if a)
        winner = counts.most_common(1)[0][0] if counts else ""
        return winner
    '''
)


def test_detects_branch_loop_with_variable_range():
    """`for b in range(n_branches)` is still a branch fan-out even though the
    count is a runtime variable (this is the actual sfumato cmaj/cmajc form)."""
    g = analyze_source(SFUMATO_REAL, filename="runner.py")
    branches = [n for n in g.nodes() if n["attrs"].get("branch")]
    assert len(branches) == 1
    # count is unknown at parse time -> no concrete repeat, but still a branch
    assert "repeat" not in branches[0]["attrs"]


def test_detects_vote_when_counter_bound_to_variable():
    """`counts = Counter(...); counts.most_common(1)` — the vote is on a bound
    variable, not an inlined Counter(...).most_common() chain."""
    g = analyze_source(SFUMATO_REAL, filename="runner.py")
    votes = [n for n in g.nodes() if n["attrs"].get("reduce")]
    assert len(votes) == 1
    assert votes[0]["kind"] == "stage"


def test_detects_counter_vote_as_reduce():
    g = analyze_source(SFUMATO_LIKE, filename="runner.py")
    votes = [n for n in g.nodes() if n["attrs"].get("reduce")]
    assert len(votes) == 1
    assert votes[0]["kind"] == "stage"


def test_static_nodes_marked_static():
    g = analyze_source(SFUMATO_LIKE, filename="runner.py")
    assert g.nodes(), "expected at least one static node"
    assert all(n["source"] == "static" for n in g.nodes())


def test_detects_stage_decorated_function():
    src = textwrap.dedent(
        '''
        @nv.stage("plan")
        def plan(q):
            return q
        '''
    )
    g = analyze_source(src, filename="m.py")
    plan = next(n for n in g.nodes() if n["name"] == "plan")
    assert plan["kind"] == "stage"
    assert plan["loc"]["line"] == 3  # the def line


def test_merge_fuses_runtime_and_static_by_loc():
    runtime = NVGraph("r")
    runtime.add_node("rt", kind="model", name="Qwen",
                     source="runtime", loc={"file": "m.py", "line": 12},
                     meta={"out_shape": [1, 32]})
    static = NVGraph("s")
    static.add_node("st", kind="stage", name="plan",
                    source="static", loc={"file": "m.py", "line": 12},
                    attrs={"declared": True})

    fused = merge(runtime, static)
    node = next(n for n in fused.nodes() if n["loc"] == {"file": "m.py", "line": 12})
    assert node["source"] == "fused"
    assert node["meta"]["out_shape"] == [1, 32]   # kept from runtime
    assert node["attrs"].get("declared") is True   # gained from static


def test_merge_adds_static_only_nodes():
    runtime = NVGraph("r")
    runtime.add_node("rt", kind="model", name="Qwen",
                     source="runtime", loc={"file": "m.py", "line": 12})
    static = NVGraph("s")
    static.add_node("vote", kind="stage", name="vote",
                    source="static", loc={"file": "m.py", "line": 20},
                    attrs={"reduce": True})

    fused = merge(runtime, static)
    names = {n["name"] for n in fused.nodes()}
    assert names == {"Qwen", "vote"}   # vote (runtime-invisible) is carried in
    vote = next(n for n in fused.nodes() if n["name"] == "vote")
    assert vote["source"] == "static"
