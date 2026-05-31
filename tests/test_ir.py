"""M0: the IR data model — typed nodes/edges over a networkx DiGraph.

The IR is the stable contract every producer (runtime trace, static AST) and
every sink (HTML, JSON, websocket) speaks. Every node carries `loc` + `source`
so the later static<->runtime fusion can merge by source location.
"""
from __future__ import annotations

import pytest

from netscope.core.ir import NVGraph, SCHEMA_VERSION


def test_add_and_get_node_preserves_attrs():
    g = NVGraph(name="m")
    g.add_node(
        "n1", kind="model", name="ResNet",
        loc={"file": "a.py", "line": 10}, meta={"params": 100},
    )
    n = g.get_node("n1")
    assert n["id"] == "n1"
    assert n["kind"] == "model"
    assert n["name"] == "ResNet"
    assert n["loc"] == {"file": "a.py", "line": 10}
    assert n["meta"]["params"] == 100


def test_node_defaults():
    g = NVGraph(name="m")
    g.add_node("n1", kind="op", name="add")
    n = g.get_node("n1")
    assert n["source"] == "runtime"   # default producer
    assert n["loc"] is None
    assert n["parent"] is None
    assert n["meta"] == {}
    assert n["attrs"] == {}


def test_add_edge_with_tensor_meta():
    g = NVGraph(name="m")
    g.add_node("a", kind="op", name="a")
    g.add_node("b", kind="op", name="b")
    g.add_edge("a", "b", kind="dataflow",
               tensor_meta={"shape": [4, 8], "dtype": "float32"})
    e = g.get_edge("a", "b")
    assert e["src"] == "a" and e["dst"] == "b"
    assert e["kind"] == "dataflow"
    assert e["tensor_meta"]["shape"] == [4, 8]


def test_children_reflect_parent():
    g = NVGraph(name="m")
    g.add_node("p", kind="stage", name="plan")
    g.add_node("c1", kind="model", name="qwen", parent="p")
    g.add_node("c2", kind="model", name="llada", parent="p")
    assert set(g.children("p")) == {"c1", "c2"}
    assert g.children("c1") == []


def test_to_dict_is_jsonable_and_has_schema():
    import json

    g = NVGraph(name="sfumato-cmajc")
    g.add_node("p", kind="stage", name="plan")
    g.add_node("c", kind="model", name="qwen", parent="p")
    g.add_edge("p", "c", kind="contains")
    d = g.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["name"] == "sfumato-cmajc"
    assert {n["id"] for n in d["nodes"]} == {"p", "c"}
    assert any(e["src"] == "p" and e["dst"] == "c" for e in d["edges"])
    json.dumps(d)  # must be serializable end-to-end


def test_get_missing_node_raises():
    g = NVGraph(name="m")
    with pytest.raises(KeyError):
        g.get_node("nope")
