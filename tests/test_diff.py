"""Trace diffing — compare two traces (before/after an edit, or two variants).

The iteration superpower: edit a model, re-trace, and see exactly what changed —
nodes added/removed, and shape/param deltas on the ones that stayed. Keyed by a
STABLE identity (qualname > loc > name) because node ids are counter-based and
shift the moment you add a layer.

`diff_graphs(before, after)` returns a structured diff; `annotate_diff` returns a
single graph tagged `attrs.diff` per node so the existing renderer can paint it.
"""
from __future__ import annotations

from netscope.core.diff import annotate_diff, diff_graphs
from netscope.core.ir import NVGraph


def _g(nodes):
    """nodes: list of (qualname, out_shape, params) -> a graph of module nodes."""
    g = NVGraph("m")
    for i, (q, out, params) in enumerate(nodes):
        g.add_node(f"n{i}", kind="module", name=q.split(".")[-1],
                   meta={"qualname": q, "out_shape": out, "params": params})
    return g


def test_identical_graphs_have_no_changes():
    a = _g([("enc", [2, 16], 100), ("head", [2, 4], 68)])
    b = _g([("enc", [2, 16], 100), ("head", [2, 4], 68)])
    d = diff_graphs(a, b)
    assert d["summary"] == {"added": 0, "removed": 0, "changed": 0, "same": 2}


def test_added_node_detected():
    a = _g([("enc", [2, 16], 100)])
    b = _g([("enc", [2, 16], 100), ("head", [2, 4], 68)])
    d = diff_graphs(a, b)
    assert d["summary"]["added"] == 1
    assert d["added"][0]["qualname"] == "head"


def test_removed_node_detected():
    a = _g([("enc", [2, 16], 100), ("head", [2, 4], 68)])
    b = _g([("enc", [2, 16], 100)])
    d = diff_graphs(a, b)
    assert d["summary"]["removed"] == 1
    assert d["removed"][0]["qualname"] == "head"


def test_changed_shape_and_params_detected():
    a = _g([("enc", [2, 16], 100)])
    b = _g([("enc", [2, 32], 200)])
    d = diff_graphs(a, b)
    assert d["summary"]["changed"] == 1
    c = d["changed"][0]
    assert c["qualname"] == "enc"
    assert "out_shape" in c["fields"] and "params" in c["fields"]
    assert c["before"]["out_shape"] == [2, 16]
    assert c["after"]["out_shape"] == [2, 32]


def test_matching_is_stable_across_id_shifts():
    # inserting a node shifts counter-based ids; qualname keeps enc/head matched.
    a = _g([("enc", [2, 16], 100), ("head", [2, 4], 68)])
    b = _g([("enc", [2, 16], 100), ("mid", [2, 16], 50), ("head", [2, 4], 68)])
    d = diff_graphs(a, b)
    assert d["summary"]["added"] == 1 and d["added"][0]["qualname"] == "mid"
    assert d["summary"]["changed"] == 0 and d["summary"]["same"] == 2


def test_loc_keys_when_no_qualname():
    a = NVGraph("m"); b = NVGraph("m")
    a.add_node("x0", kind="op", name="relu", loc={"file": "m.py", "line": 7},
               meta={"out_shape": [2, 8]})
    b.add_node("y9", kind="op", name="relu", loc={"file": "m.py", "line": 7},
               meta={"out_shape": [2, 16]})
    d = diff_graphs(a, b)
    assert d["summary"]["changed"] == 1   # same loc, different shape


def test_annotate_tags_each_node_for_the_renderer():
    a = _g([("enc", [2, 16], 100), ("old", [2, 8], 10)])
    b = _g([("enc", [2, 32], 200), ("new", [2, 4], 5)])
    g = annotate_diff(a, b)
    tag = {(n.get("meta") or {}).get("qualname"): (n.get("attrs") or {}).get("diff")
           for n in g.nodes()}
    assert tag["enc"] == "changed"
    assert tag["new"] == "added"
    assert tag["old"] == "removed"        # removed nodes kept as ghosts


def test_annotate_changed_carries_detail():
    a = _g([("enc", [2, 16], 100)])
    b = _g([("enc", [2, 32], 100)])
    g = annotate_diff(a, b)
    enc = next(n for n in g.nodes() if (n.get("meta") or {}).get("qualname") == "enc")
    assert "out_shape" in (enc.get("attrs") or {}).get("diff_detail", "")


def test_end_to_end_diff_of_two_real_traced_models():
    """Trace two real variants — a Linear widened 16->32 and a layer added — and
    confirm the diff reflects reality (the actual iteration story)."""
    import torch
    import torch.nn as nn

    import netscope

    class V1(nn.Module):
        def __init__(s):
            super().__init__(); s.enc = nn.Linear(8, 16); s.head = nn.Linear(16, 4)
        def forward(s, x):
            return s.head(torch.relu(s.enc(x)))

    class V2(nn.Module):  # enc widened 16->32, an extra `mid` layer inserted
        def __init__(s):
            super().__init__()
            s.enc = nn.Linear(8, 32); s.mid = nn.Linear(32, 16); s.head = nn.Linear(16, 4)
        def forward(s, x):
            return s.head(torch.relu(s.mid(torch.relu(s.enc(x)))))

    with netscope.graph("v1") as g1, torch.no_grad():
        V1().train(False)(torch.randn(2, 8))
    with netscope.graph("v2") as g2, torch.no_grad():
        V2().train(False)(torch.randn(2, 8))

    d = netscope.diff(g1, g2)
    quals = lambda lst: {x["qualname"] for x in lst}
    assert "mid" in quals(d["added"])                 # the inserted layer
    assert "enc" in quals(d["changed"])               # widened 16 -> 32
    enc = next(c for c in d["changed"] if c["qualname"] == "enc")
    assert enc["before"]["out_shape"] == [2, 16]
    assert enc["after"]["out_shape"] == [2, 32]


def test_cli_graph_json_writes_annotated_ir(tmp_path):
    # the extension shells out to `--graph-json` and loads the result into its
    # webview, so the annotated IR must round-trip with diff tags intact.
    import json

    from netscope.core.diff import _main

    a = _g([("enc", [2, 16], 100)])
    b = _g([("enc", [2, 32], 200)])
    pa, pb, pg = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "g.json"
    pa.write_text(a.to_json()); pb.write_text(b.to_json())
    _main([str(pa), str(pb), "--graph-json", str(pg)])
    d = json.loads(pg.read_text())
    tags = {(n.get("attrs") or {}).get("diff") for n in d["nodes"]}
    assert "changed" in tags


def test_sibling_key_collision_is_not_collapsed():
    # two ops sharing name+parent with no qualname/loc must stay distinct, so a
    # genuine removal isn't hidden by the index collapsing them.
    a, b = NVGraph("m"), NVGraph("m")
    for g, n in [(a, 2), (b, 1)]:
        g.add_node("p", kind="stage", name="P")
        for i in range(n):
            g.add_node(f"add{i}", kind="op", name="add", parent="p")
    assert diff_graphs(a, b)["summary"]["removed"] == 1


def test_removed_subtree_ghost_keeps_its_parent():
    # when a parent AND its child are removed, the child ghost must attach to the
    # parent's ghost, not flatten to root.
    a, b = NVGraph("m"), NVGraph("m")
    a.add_node("blk", kind="module", name="Block", meta={"qualname": "block"})
    a.add_node("lin", kind="module", name="Linear", parent="blk",
               meta={"qualname": "block.lin"})
    b.add_node("keep", kind="module", name="Keep", meta={"qualname": "keep"})
    g = annotate_diff(a, b)
    ghost = {(n.get("meta") or {}).get("qualname"): n for n in g.nodes()
             if (n.get("attrs") or {}).get("diff") == "removed"}
    assert "block" in ghost and "block.lin" in ghost
    assert ghost["block.lin"]["parent"] == "removed::blk"


def test_from_dict_skips_dangling_edges():
    a = NVGraph.from_dict({
        "nodes": [{"id": "a", "kind": "module", "name": "A"}],
        "edges": [{"src": "a", "dst": "GHOST", "kind": "dataflow"}],
    })
    assert a.has_node("a") and not a.has_node("GHOST")   # no junk node auto-created
    assert a.edges() == []


def test_diff_survives_a_to_dict_round_trip():
    """Two SAVED traces (JSON dumps) diff the same as live graphs — the path the
    extension/CLI takes."""
    from netscope.core.ir import NVGraph

    a = _g([("enc", [2, 16], 100), ("head", [2, 4], 68)])
    b = _g([("enc", [2, 32], 200), ("head", [2, 4], 68)])
    a2 = NVGraph.from_dict(a.to_dict())
    b2 = NVGraph.from_dict(b.to_dict())
    assert diff_graphs(a2, b2)["summary"] == diff_graphs(a, b)["summary"]
    assert diff_graphs(a2, b2)["summary"]["changed"] == 1
