"""Augmented inference — the LLM fills structure the static AST can't recover.

Static source analysis is blind to models built dynamically (`from_config`,
factory functions, ModuleList comprehensions, custom forwards with data-dependent
control flow). When you have the SOURCE but no runtime trace, this asks the LLM to
read it and propose the likely module graph — returned as PROVISIONAL nodes/edges,
each `source="inferred"` with a confidence in `attrs`, so the renderer draws them
distinctly (dashed) and never presents a guess as a fact.

Grounded + guarded: the model only sees the real source; its reply is parsed and
SCHEMA-VALIDATED (bad/dangling entries dropped), never trusted blindly. A non-JSON
or empty reply adds nothing — the original graph is returned unchanged. Gated on
an LLM key like the rest of the layer.
"""
from __future__ import annotations

import json
from typing import Optional

from netscope.core.ir import NVGraph
from netscope.llm.provider import Provider, Transport

# the structured delta we ask the model for (advertised in the prompt).
INFER_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "kind": {"type": "string",
                             "enum": ["pipeline", "stage", "model", "module", "op"]},
                    "qualname": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["id", "name", "kind"],
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "src": {"type": "string"},
                    "dst": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["src", "dst"],
            },
        },
    },
    "required": ["nodes"],
}

_VALID_KINDS = {"pipeline", "stage", "model", "module", "op"}

_SYSTEM = (
    "You are netscope's structure-inference assistant. You are given the SOURCE of "
    "a PyTorch model that static analysis could not fully parse (e.g. built via "
    "from_config, factories, or ModuleList comprehensions). Infer the model's "
    "likely module graph. Respond with ONLY a JSON object matching this schema:\n"
    + json.dumps(INFER_SCHEMA) +
    "\nRules: ids are short unique strings; kind is one of pipeline/stage/model/"
    "module/op; qualname is the attribute path if you can tell (e.g. blocks.0, "
    "head); confidence is 0..1 for how sure you are. Edges are dataflow "
    "(producer src -> consumer dst) using the node ids you defined. Infer only what "
    "the source supports — set a LOW confidence when unsure. No prose, JSON only."
)


def _clamp(v) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.5


def _extract_json(text: str) -> Optional[dict]:
    """Parse a JSON object from the model reply, tolerating ```json fences and
    surrounding prose. Returns None if no object is found."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        # strip a leading ```json / ``` fence and the trailing ```
        t = t.split("```", 2)
        t = t[1] if len(t) >= 2 else text
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    t = t.strip().strip("`").strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        # last resort: grab the outermost {...}
        i, j = t.find("{"), t.rfind("}")
        if 0 <= i < j:
            try:
                obj = json.loads(t[i:j + 1])
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
        return None


def _validate(payload: dict) -> tuple:
    """Return (nodes, edges) keeping only well-formed entries. Dangling edges
    (endpoints not among the kept nodes) are dropped."""
    raw_nodes = payload.get("nodes") if isinstance(payload, dict) else None
    if not isinstance(raw_nodes, list):
        return [], []
    nodes, ids = [], set()
    for n in raw_nodes:
        if not isinstance(n, dict):
            continue
        nid, name, kind = n.get("id"), n.get("name"), n.get("kind")
        if not (isinstance(nid, str) and isinstance(name, str) and kind in _VALID_KINDS):
            continue
        if nid in ids:
            continue
        ids.add(nid)
        nodes.append({
            "id": nid, "name": name, "kind": kind,
            "qualname": n.get("qualname") if isinstance(n.get("qualname"), str) else None,
            "confidence": _clamp(n.get("confidence", 0.5)),
        })
    edges = []
    raw_edges = payload.get("edges") if isinstance(payload.get("edges"), list) else []
    for e in raw_edges:
        if not isinstance(e, dict):
            continue
        src, dst = e.get("src"), e.get("dst")
        if src in ids and dst in ids and src != dst:
            edges.append({"src": src, "dst": dst, "confidence": _clamp(e.get("confidence", 0.5))})
    return nodes, edges


def infer_structure(
    graph: NVGraph,
    source: str,
    filename: str,
    *,
    provider: Optional[Provider] = None,
    _transport: Optional[Transport] = None,
) -> NVGraph:
    """Augment `graph` in place with PROVISIONAL inferred nodes/edges from the LLM.
    Returns the same graph. No-op (original unchanged) if no provider, an empty/
    non-JSON reply, or nothing validates. Gated on an LLM key."""
    provider = provider or Provider.from_env()
    if provider is None:
        return graph

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content":
            f"Infer the module graph of this model.\n\n--- {filename} ---\n{source}"},
    ]
    try:
        reply = provider.complete(messages, _transport=_transport, max_tokens=1200)
    except Exception:
        return graph

    payload = _extract_json(reply)
    if not payload:
        return graph
    nodes, edges = _validate(payload)

    # prefix inferred ids so they never collide with real node ids.
    pfx = "inferred:"
    for n in nodes:
        nid = pfx + n["id"]
        if graph.has_node(nid):
            continue
        meta = {}
        if n.get("qualname"):
            meta["qualname"] = n["qualname"]
        graph.add_node(
            nid, kind=n["kind"], name=n["name"], source="inferred",
            loc={"file": filename} if filename else None,
            meta=meta, attrs={"inferred": True, "confidence": n["confidence"]},
        )
    for e in edges:
        s, d = pfx + e["src"], pfx + e["dst"]
        if graph.has_node(s) and graph.has_node(d):
            graph.add_edge(s, d, kind="dataflow", source="inferred")
    return graph
