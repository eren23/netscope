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


# --- declared-dim nodes must not pollute a fused runtime graph ----------------
def test_merge_drops_unmatched_declared_dim_nodes():
    """An M2 declared-dim node that doesn't loc-match a runtime node is a layer
    that never ran (e.g. inside an unused fallback class). It's redundant with
    runtime module nodes, so merge must drop it — otherwise it floats free in the
    real trace (the 'stray Linear' bug)."""
    runtime = NVGraph("r")
    runtime.add_node("rt", kind="module", name="Embedding", source="runtime",
                     loc={"file": "m.py", "line": 5}, meta={"out_shape": [1, 8, 1024]})
    static = NVGraph("s")
    # a declared-dim node for a layer that the runtime never executed
    static.add_node("dim#1", kind="module", name="Linear", source="static",
                    loc={"file": "m.py", "line": 58}, meta={"qualname": "lm_head",
                    "in_shape": [1, 256], "out_shape": [1, 8000]},
                    attrs={"declared_dim": True})
    # ...and a genuine static-only branch node, which MUST survive
    static.add_node("vote", kind="stage", name="vote", source="static",
                    loc={"file": "m.py", "line": 20}, attrs={"reduce": True})

    fused = merge(runtime, static)
    names = {n["name"] for n in fused.nodes()}
    assert "Embedding" in names           # runtime node kept
    assert "vote" in names                # genuine static-only structure kept
    assert "Linear" not in names          # the orphan declared-dim node dropped


def test_merge_keeps_declared_dim_node_when_it_matches_runtime():
    """If a declared-dim node DOES loc-match a runtime node, the runtime node wins
    (and gains the static attrs) — it isn't dropped, just fused normally."""
    runtime = NVGraph("r")
    runtime.add_node("rt", kind="module", name="Linear", source="runtime",
                     loc={"file": "m.py", "line": 7}, meta={"out_shape": [1, 128]})
    static = NVGraph("s")
    static.add_node("dim#1", kind="module", name="Linear", source="static",
                    loc={"file": "m.py", "line": 7}, meta={"in_shape": [1, 64]},
                    attrs={"declared_dim": True})
    fused = merge(runtime, static)
    assert len(fused.nodes()) == 1
    node = fused.nodes()[0]
    assert node["source"] == "fused"
    assert node["meta"]["out_shape"] == [1, 128]   # runtime shape kept


# --- merge loc-collision: a static node fuses into AT MOST ONE runtime node ----
def test_merge_static_node_fuses_into_at_most_one_runtime_node():
    """Two runtime nodes can share a loc (a submodule called twice, or a loop
    body). A single static node at that loc must fuse into only ONE of them —
    otherwise its attrs get duplicated across unrelated runtime nodes."""
    rt = NVGraph("r")
    rt.add_node("a", kind="module", name="Linear", source="runtime",
                loc={"file": "m.py", "line": 5}, meta={"out_shape": [1, 8]})
    rt.add_node("b", kind="module", name="Linear", source="runtime",
                loc={"file": "m.py", "line": 5}, meta={"out_shape": [1, 8]})
    st = NVGraph("s")
    st.add_node("s1", kind="stage", name="plan", source="static",
                loc={"file": "m.py", "line": 5}, attrs={"declared": True})

    fused = merge(rt, st)
    fused_count = sum(1 for n in fused.nodes() if n["source"] == "fused")
    assert fused_count == 1, f"a static node fused into {fused_count} runtime nodes (should be 1)"
    # both runtime nodes survive; only one carries the static attrs
    with_attr = [n for n in fused.nodes() if (n.get("attrs") or {}).get("declared")]
    assert len(with_attr) == 1
    assert len([n for n in fused.nodes()]) == 2   # no node lost


# --- static branch/vote nodes must not duplicate runtime ones (the sfumato strays)
def test_merge_drops_static_branch_when_runtime_has_branches():
    """The static AST producer recovers a `for range(n)` as a 'branch loop' stage
    and a Counter.most_common as a 'vote'. But if the runtime trace ALREADY has
    branch/reduce stages (the user's netscope.branch()/reduce() markers), the
    static ones are redundant duplicates — and since the runtime markers carry no
    loc, they don't loc-match, so they used to float in as disconnected strays
    (the 'branch loop' + second 'vote' seen in the sfumato fused view)."""
    runtime = NVGraph("rt")
    # runtime branch + reduce stages (loc=None, like the hint markers)
    runtime.add_node("b0", kind="stage", name="diffuse[0]", source="runtime",
                     loc=None, attrs={"branch": True})
    runtime.add_node("v", kind="stage", name="vote", source="runtime",
                     loc=None, attrs={"reduce": True})
    runtime.add_edge("b0", "v", kind="dataflow", source="runtime")

    static = NVGraph("st")
    static.add_node("s_loop", kind="stage", name="branch loop", source="static",
                    loc={"file": "m.py", "line": 63}, attrs={"branch": True})
    static.add_node("s_vote", kind="stage", name="vote", source="static",
                    loc={"file": "m.py", "line": 70}, attrs={"reduce": True})

    fused = merge(runtime, static)
    names = [n["name"] for n in fused.nodes()]
    # the redundant static branch/vote are dropped; runtime ones remain
    assert "branch loop" not in names
    assert names.count("vote") == 1
    assert "diffuse[0]" in names


def test_merge_keeps_static_branch_when_runtime_has_none():
    """If the runtime has NO branch/reduce stages (pure auto-trace, no hints), the
    static branch/vote ARE the only structure for them — keep them."""
    runtime = NVGraph("rt")
    runtime.add_node("m", kind="module", name="Net", source="runtime", loc=None)
    static = NVGraph("st")
    static.add_node("s_vote", kind="stage", name="vote", source="static",
                    loc={"file": "m.py", "line": 70}, attrs={"reduce": True})
    fused = merge(runtime, static)
    assert "vote" in [n["name"] for n in fused.nodes()]
