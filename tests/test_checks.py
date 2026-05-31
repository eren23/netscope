"""M8: mismatch detection — the "show errors" half of the vision.

A dataflow edge A->B is suspicious when A's output shape is incompatible with
B's input shape. `detect_mismatches` scans the IR and returns structured
warnings; capture attaches them so sinks can render red edges + a warnings list.

Shape-compat rule (feature-dim oriented, batch-agnostic): compare the LAST dim
of A.out_shape with the LAST dim of B.in_shape — a classic Linear(in!=out) /
channel-mismatch bug. Missing shapes => no warning (can't tell).
"""
from __future__ import annotations

from netscope.core.checks import detect_mismatches
from netscope.core.ir import NVGraph


def _edge_graph(a_out, b_in):
    g = NVGraph("m")
    g.add_node("a", kind="module", name="A", meta={"out_shape": a_out})
    g.add_node("b", kind="module", name="B", meta={"in_shape": b_in})
    g.add_edge("a", "b", kind="dataflow")
    return g


def test_flags_feature_dim_mismatch():
    g = _edge_graph([4, 8], [4, 16])   # A emits ..x8, B expects ..x16
    w = detect_mismatches(g)
    assert len(w) == 1
    assert w[0]["kind"] == "shape_mismatch"
    assert w[0]["src"] == "a" and w[0]["dst"] == "b"
    assert "8" in w[0]["detail"] and "16" in w[0]["detail"]


def test_no_warning_when_feature_dims_match():
    g = _edge_graph([4, 8], [32, 8])   # different batch, same feature dim -> ok
    assert detect_mismatches(g) == []


def test_no_warning_when_a_shape_missing():
    g = NVGraph("m")
    g.add_node("a", kind="module", name="A", meta={})
    g.add_node("b", kind="module", name="B", meta={"in_shape": [4, 16]})
    g.add_edge("a", "b", kind="dataflow")
    assert detect_mismatches(g) == []


def test_contains_edges_are_not_checked():
    g = NVGraph("m")
    g.add_node("p", kind="stage", name="P", meta={"out_shape": [4, 8]})
    g.add_node("c", kind="module", name="C", meta={"in_shape": [4, 16]})
    g.add_edge("p", "c", kind="contains")   # hierarchy, not dataflow
    assert detect_mismatches(g) == []


def test_multiple_mismatches_each_reported():
    g = NVGraph("m")
    for i, (ao, bi) in enumerate([([1, 8], [1, 9]), ([1, 4], [1, 5])]):
        g.add_node(f"a{i}", kind="module", name=f"A{i}", meta={"out_shape": ao})
        g.add_node(f"b{i}", kind="module", name=f"B{i}", meta={"in_shape": bi})
        g.add_edge(f"a{i}", f"b{i}", kind="dataflow")
    assert len(detect_mismatches(g)) == 2


def test_flags_rank_mismatch_with_flatten_hint():
    # classic Conv2d (N,C,H,W) -> Linear (N,F): ranks differ, "forgot flatten()"
    g = _edge_graph([1, 64, 8, 8], [1, 4096])
    w = detect_mismatches(g)
    assert len(w) == 1
    assert w[0]["kind"] == "rank_mismatch"
    assert "4" in w[0]["detail"] and "2" in w[0]["detail"]   # 4-D vs 2-D
    assert "flatten" in w[0]["detail"].lower()


def test_rank_mismatch_takes_precedence_over_feature_dim():
    # ranks differ AND last dims differ -> ONE rank warning, not also a feature one
    g = _edge_graph([1, 64, 8, 8], [1, 16])
    w = detect_mismatches(g)
    assert len(w) == 1
    assert w[0]["kind"] == "rank_mismatch"


def test_same_rank_different_feature_is_still_shape_mismatch():
    g = _edge_graph([1, 3, 8, 8], [1, 5, 8, 8])   # both 4-D, channel 3 vs 5
    w = detect_mismatches(g)
    assert len(w) == 1
    assert w[0]["kind"] == "shape_mismatch"


def test_capture_attaches_warnings_to_graph_dict():
    import netscope

    with netscope.graph("m") as g:
        cap = netscope.active_capture()
        with cap.span("A", kind="module", meta={"out_shape": [1, 8]}):
            pass
        with cap.span("B", kind="module", meta={"in_shape": [1, 16]}):
            pass
        ids = [n["id"] for n in g.nodes()]
        g.add_edge(ids[0], ids[1], kind="dataflow")
    d = g.to_dict()
    assert "warnings" in d
    assert any(w["kind"] == "shape_mismatch" for w in d["warnings"])
