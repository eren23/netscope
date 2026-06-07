"""Augmented inference: the LLM fills structure the static AST can't recover.

When source-AST analysis yields almost nothing (a from_config model, a custom
forward with dynamic ops), the LLM reads the SOURCE and proposes a likely module
graph — returned as PROVISIONAL nodes/edges, each marked source="inferred" with a
confidence score, so the renderer can draw them distinctly (dashed) and never
present a guess as fact. Grounded: the model only ever annotates real source; the
result is schema-validated, not trusted blindly. Tests inject a fake transport.
"""
from __future__ import annotations

import json


from netscope.core.ir import NVGraph
from netscope.llm.infer import infer_structure, INFER_SCHEMA
from netscope.llm.provider import Provider


def _provider():
    return Provider(base_url="http://x/v1", model="m", api_key="k", extra_headers={})


SRC = '''
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n)])
        self.head = nn.Linear(cfg.d, cfg.vocab)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x)
'''


def _fake_reply(nodes, edges):
    """A transport returning a well-formed inference payload."""
    payload = {"nodes": nodes, "edges": edges}

    def transport(url, headers, body):
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}
    return transport


def test_infer_adds_provisional_nodes_marked_inferred():
    g = NVGraph("net")
    g.add_node("root", kind="model", name="Net", source="static",
               loc={"file": "m.py", "line": 4})
    transport = _fake_reply(
        nodes=[{"id": "blocks", "name": "ModuleList", "kind": "module",
                "qualname": "blocks", "confidence": 0.8},
               {"id": "head", "name": "Linear", "kind": "module",
                "qualname": "head", "confidence": 0.9}],
        edges=[{"src": "blocks", "dst": "head", "confidence": 0.7}],
    )
    out = infer_structure(g, SRC, "m.py", provider=_provider(), _transport=transport)
    inferred = [n for n in out.nodes() if n["source"] == "inferred"]
    assert len(inferred) == 2
    assert all(n["attrs"].get("inferred") for n in inferred)
    assert all("confidence" in n["attrs"] for n in inferred)
    # the original real node is preserved + NOT marked inferred
    root = next(n for n in out.nodes() if n["id"] == "root")
    assert root["source"] == "static"


def test_inferred_edges_marked_and_confidence_carried():
    g = NVGraph("net")
    transport = _fake_reply(
        nodes=[{"id": "a", "name": "A", "kind": "module", "confidence": 0.9},
               {"id": "b", "name": "B", "kind": "module", "confidence": 0.9}],
        edges=[{"src": "a", "dst": "b", "confidence": 0.6}],
    )
    out = infer_structure(g, SRC, "m.py", provider=_provider(), _transport=transport)
    edges = out.edges()
    assert len(edges) == 1
    assert edges[0]["source"] == "inferred"


def test_infer_validates_and_drops_garbage_nodes():
    """A node missing required fields is dropped, not crashed on."""
    g = NVGraph("net")

    def transport(url, headers, body):
        bad = {"nodes": [{"id": "ok", "name": "OK", "kind": "module", "confidence": 0.8},
                         {"name": "no-id"},          # missing id -> dropped
                         {"id": "x"}],               # missing name/kind -> dropped
               "edges": [{"src": "ok", "dst": "missing", "confidence": 0.5}]}  # dangling -> dropped
        return {"choices": [{"message": {"content": json.dumps(bad)}}]}
    out = infer_structure(g, SRC, "m.py", provider=_provider(), _transport=transport)
    inferred = [n for n in out.nodes() if n["source"] == "inferred"]
    assert len(inferred) == 1 and inferred[0]["id"] == "inferred:ok"
    # the dangling edge (dst not a real node) is dropped
    assert out.edges() == []


def test_infer_handles_nonjson_reply_gracefully():
    g = NVGraph("net")

    def transport(url, headers, body):
        return {"choices": [{"message": {"content": "I think there are some blocks."}}]}
    # a non-JSON reply -> no inferred structure added, no crash, original intact
    g.add_node("root", kind="model", name="Net", source="static")
    out = infer_structure(g, SRC, "m.py", provider=_provider(), _transport=transport)
    assert [n["id"] for n in out.nodes()] == ["root"]


def test_infer_strips_markdown_fenced_json():
    """LLMs often wrap JSON in ```json fences — we must still parse it."""
    g = NVGraph("net")

    def transport(url, headers, body):
        fenced = "```json\n" + json.dumps(
            {"nodes": [{"id": "h", "name": "Linear", "kind": "module", "confidence": 0.9}],
             "edges": []}) + "\n```"
        return {"choices": [{"message": {"content": fenced}}]}
    out = infer_structure(g, SRC, "m.py", provider=_provider(), _transport=transport)
    assert any(n["source"] == "inferred" for n in out.nodes())


def test_schema_is_well_formed():
    assert INFER_SCHEMA["type"] == "object"
    assert "nodes" in INFER_SCHEMA["properties"]
