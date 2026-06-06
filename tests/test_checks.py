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


def test_seq_len_difference_is_not_flagged():
    # encoder->decoder cross-attention: src_len 5, tgt_len 6, SAME feature dim 32.
    # Axis 1 is the sequence length and legitimately differs — flagging it is a
    # false alarm (the real dogfood bug on nn.Transformer). Feature (last) matches.
    g = _edge_graph([2, 5, 32], [2, 6, 32])
    assert detect_mismatches(g) == []


def test_seq_models_still_flag_feature_dim_clash():
    # but a genuine embedding-dim clash on a 3-D tensor IS a real bug -> flag it,
    # regardless of the sequence axis also differing.
    g = _edge_graph([2, 5, 32], [2, 6, 64])
    w = detect_mismatches(g)
    assert len(w) == 1
    assert w[0]["kind"] == "shape_mismatch"
    assert "32" in w[0]["detail"] and "64" in w[0]["detail"]


def test_conv_spatial_difference_is_not_flagged():
    # NCHW with matching channels (64) but different spatial size (56->28, a
    # stride-2/pool downsample) is normal — only the channel axis matters for conv.
    g = _edge_graph([1, 64, 56, 56], [1, 64, 28, 28])
    assert detect_mismatches(g) == []


def _conv_edge(a_out, b_in):
    g = NVGraph("m")
    g.add_node("a", kind="module", name="Conv1d", meta={"out_shape": a_out})
    g.add_node("b", kind="module", name="Conv1d", meta={"in_shape": b_in})
    g.add_edge("a", "b", kind="dataflow")
    return g


def test_conv1d_length_change_is_not_flagged():
    # Conv1d (N,C,L): channels match (8), only length changes (18->9 under a stride)
    # — legit, must not flag (the old rank-3=last-axis rule false-flagged this).
    assert detect_mismatches(_conv_edge([2, 8, 18], [2, 8, 9])) == []


def test_conv1d_channel_change_is_flagged():
    # a real Conv1d channel clash (8 -> 16 on axis 1) must be caught (the old rule
    # silently missed it by comparing the last axis).
    w = detect_mismatches(_conv_edge([2, 8, 18], [2, 16, 18]))
    assert len(w) == 1 and w[0]["kind"] == "shape_mismatch"


def test_multi_output_producer_compares_per_edge_not_node_shape():
    # a multi-scale backbone (FPN / RT-DETR) has ONE representative out_shape but
    # feeds different-dim feature maps down different edges (carried in tensor_meta).
    # Each edge must be checked against the tensor that actually flowed on it.
    g = NVGraph("m")
    g.add_node("bb", kind="module", name="backbone", meta={"out_shape": [1, 512, 28, 28]})
    g.add_node("p0", kind="module", name="proj0", meta={"in_shape": [1, 512, 28, 28]})
    g.add_node("p1", kind="module", name="proj1", meta={"in_shape": [1, 1024, 14, 14]})
    g.add_edge("bb", "p0", kind="dataflow", tensor_meta={"shape": [1, 512, 28, 28]})
    g.add_edge("bb", "p1", kind="dataflow", tensor_meta={"shape": [1, 1024, 14, 14]})
    assert detect_mismatches(g) == []   # each scale matches its own proj — no false alarm


def test_edge_shape_still_catches_a_real_clash():
    g = NVGraph("m")
    g.add_node("a", kind="module", name="A", meta={"out_shape": [1, 256]})
    g.add_node("b", kind="module", name="B", meta={"in_shape": [1, 128]})
    g.add_edge("a", "b", kind="dataflow", tensor_meta={"shape": [1, 256]})  # 256 -> 128
    w = detect_mismatches(g)
    assert len(w) == 1 and "256" in w[0]["detail"] and "128" in w[0]["detail"]


def test_concat_fanin_consumer_is_not_false_flagged():
    # a consumer fed by TWO producers (a concat in an FPN/PAN neck — YOLO, U-Net):
    # its in_shape is the COMBINED channels (64+64=128); each producer's edge is a
    # part, so a per-edge feature check is invalid and must be skipped.
    g = NVGraph("m")
    g.add_node("p1", kind="module", name="P1", meta={"out_shape": [1, 64, 80, 80]})
    g.add_node("p2", kind="module", name="P2", meta={"out_shape": [1, 64, 80, 80]})
    g.add_node("c", kind="module", name="C2f", meta={"in_shape": [1, 128, 80, 80]})
    g.add_edge("p1", "c", kind="dataflow", tensor_meta={"shape": [1, 64, 80, 80]})
    g.add_edge("p2", "c", kind="dataflow", tensor_meta={"shape": [1, 64, 80, 80]})
    assert detect_mismatches(g) == []   # fan-in merge — per-edge check skipped


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
