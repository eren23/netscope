"""M2: sinks — JSON, mermaid, and the interactive Cytoscape HTML.

The IR's `parent` field maps to Cytoscape *compound nodes* (the hierarchy boxes),
so `contains` edges are implied by nesting and must NOT be drawn as edges. The
same self-contained HTML is what the M4 VSCode webview will reuse.
"""
from __future__ import annotations

import json
import os

import netscope
from netscope.core.ir import NVGraph


def _sample() -> NVGraph:
    g = NVGraph("demo")
    g.add_node("p", kind="stage", name="plan")
    g.add_node("m", kind="model", name="Qwen", parent="p", meta={"out_shape": [1, 32]})
    g.add_node("v", kind="stage", name="vote")
    g.add_edge("p", "m", kind="contains")
    g.add_edge("p", "v", kind="dataflow")
    return g


def test_to_json_roundtrips():
    g = _sample()
    d = json.loads(g.to_json())
    assert d["schema_version"] == netscope.SCHEMA_VERSION
    assert {n["id"] for n in d["nodes"]} == {"p", "m", "v"}


def test_to_cytoscape_maps_parent_and_skips_contains_edges():
    from netscope.sinks.html_sink import to_cytoscape

    el = to_cytoscape(_sample())
    nodes = {n["data"]["id"]: n["data"] for n in el["nodes"]}
    assert nodes["m"]["parent"] == "p"          # compound nesting
    assert "parent" not in nodes["p"]            # root has no parent key
    kinds = [e["data"]["kind"] for e in el["edges"]]
    assert "contains" not in kinds               # implied by nesting
    assert "dataflow" in kinds


def test_to_html_is_self_contained_and_embeds_data():
    html = _sample().to_html(title="demo")
    low = html.lower()
    assert "<html" in low
    assert "cytoscape" in low
    assert "Qwen" in html        # node label embedded
    assert "demo" in html        # title embedded
    assert "__NETSCOPE_ELEMENTS__" not in html  # placeholder fully replaced


def test_to_mermaid_lists_nodes_and_edges():
    m = _sample().to_mermaid()
    assert m.strip().startswith("flowchart")
    assert "plan" in m and "Qwen" in m and "vote" in m


def test_show_writes_openable_file(tmp_path):
    out = _sample().show(path=str(tmp_path / "g.netscope.html"), open_browser=False)
    assert out.endswith(".netscope.html")
    assert os.path.exists(out)
    with open(out) as f:
        assert "cytoscape" in f.read().lower()


def test_html_has_attention_overlay_control():
    from netscope.core.ir import NVGraph
    g = NVGraph(name="t")
    g.add_node("a", kind="module", name="self_attn",
               meta={"attn_heads": [{"entropy": 1.0, "dist": 2.0, "last": 0.1}]})
    html = g.to_html()
    assert "btn-attention" in html          # the new overlay button id
    assert "attn_heads" in html             # per-head data reaches the webview
