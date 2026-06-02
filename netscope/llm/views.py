"""Generated views — a prompt becomes a declarative VIEW SPEC the renderer applies.

"group by attention vs MLP", "highlight params > 1M", "color by dtype" turn into a
small, validated JSON spec of SAFE operations — never arbitrary code. The LLM only
ever produces the spec (gated on a key); `apply_view_spec` is a pure function that
stamps view flags (vhi / vdim / vcolor) onto cytoscape node data, which the
template styles. No eval, no injection — an unknown op is simply dropped.

Ops:
  {"op": "highlight", "where": <predicate>}   -> data.vhi   on matches
  {"op": "filter",    "where": <predicate>}   -> data.vdim  on NON-matches
  {"op": "colorBy",   "field": "kind|dtype|device|name"}  -> data.vcolor by value

Predicate keys (all optional, ANDed): kind, name_contains, params_gt, params_lt,
dtype, device.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from netscope.llm.provider import Provider, Transport

_OPS = {"highlight", "filter", "colorBy"}
_COLOR_FIELDS = {"kind", "dtype", "device", "name"}

VIEW_SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "ops": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": sorted(_OPS)},
                    "where": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string"},
                            "name_contains": {"type": "string"},
                            "params_gt": {"type": "number"},
                            "params_lt": {"type": "number"},
                            "dtype": {"type": "string"},
                            "device": {"type": "string"},
                        },
                    },
                    "field": {"type": "string", "enum": sorted(_COLOR_FIELDS)},
                },
                "required": ["op"],
            },
        },
    },
    "required": ["ops"],
}

# a small fixed palette for colorBy (stable per distinct value).
_PALETTE = ["#2ee6ff", "#34f5a8", "#ffc233", "#cf8bff", "#ff8d6b",
            "#6ad1ff", "#8affc2", "#ffd98a", "#e0a3ff", "#ff5a5f"]

_SYSTEM = (
    "You turn a user's request into a netscope VIEW SPEC: a JSON object that "
    "re-styles a graph of model layers. Respond with ONLY JSON matching:\n"
    + json.dumps(VIEW_SPEC_SCHEMA) +
    "\nOps: highlight (emphasize matching nodes), filter (dim non-matching), "
    "colorBy (recolor by a field). Predicate keys: kind, name_contains, "
    "params_gt, params_lt, dtype, device. Use the node names/fields you are given. "
    "JSON only, no prose."
)


# --- parsing / validation ----------------------------------------------------
def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        t = parts[1] if len(parts) >= 2 else t
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    t = t.strip().strip("`").strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if 0 <= i < j:
            try:
                obj = json.loads(t[i:j + 1])
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
        return None


def _valid_op(o: Any) -> Optional[dict]:
    if not isinstance(o, dict) or o.get("op") not in _OPS:
        return None
    op = {"op": o["op"]}
    if op["op"] == "colorBy":
        if o.get("field") not in _COLOR_FIELDS:
            return None
        op["field"] = o["field"]
    else:
        where = o.get("where")
        op["where"] = where if isinstance(where, dict) else {}
    return op


def parse_view_spec(text: str) -> Dict[str, Any]:
    """Parse + validate an LLM reply into a view spec. Unknown/malformed ops are
    dropped; a non-JSON reply yields an empty spec (no-op)."""
    payload = _extract_json(text)
    if not isinstance(payload, dict):
        return {"ops": []}
    raw = payload.get("ops")
    ops = [v for v in (_valid_op(o) for o in raw) if v] if isinstance(raw, list) else []
    return {"ops": ops}


# --- applying ----------------------------------------------------------------
def _node_field(data: dict, field: str):
    if field == "kind":
        return data.get("kind")
    if field == "name":
        return data.get("name")
    meta = data.get("meta") or {}
    return meta.get(field)


def _matches(data: dict, where: dict) -> bool:
    meta = data.get("meta") or {}
    if "kind" in where and data.get("kind") != where["kind"]:
        return False
    if "name_contains" in where:
        nc = where["name_contains"]
        if not isinstance(nc, str) or nc.lower() not in str(data.get("name", "")).lower():
            return False
    p = meta.get("params")
    if "params_gt" in where and not (isinstance(p, (int, float)) and p > where["params_gt"]):
        return False
    if "params_lt" in where and not (isinstance(p, (int, float)) and p < where["params_lt"]):
        return False
    if "dtype" in where and meta.get("dtype") != where["dtype"]:
        return False
    if "device" in where and meta.get("device") != where["device"]:
        return False
    # an empty/bogus predicate (no recognized key matched) -> no match
    return any(k in where for k in
               ("kind", "name_contains", "params_gt", "params_lt", "dtype", "device"))


def apply_view_spec(elements: dict, spec: dict) -> dict:
    """Stamp view flags onto a COPY of the cytoscape elements per the spec. Pure:
    returns new node/edge dicts, never mutates the input. The template reads
    data.vhi (highlight), data.vdim (dimmed), data.vcolor (recolor)."""
    nodes = [{"data": dict(n["data"])} for n in elements.get("nodes", [])]
    edges = [{"data": dict(e["data"])} for e in elements.get("edges", [])]
    color_map: Dict[Any, str] = {}

    for op in (spec or {}).get("ops", []):
        kind = op.get("op")
        if kind == "highlight":
            for n in nodes:
                if _matches(n["data"], op.get("where") or {}):
                    n["data"]["vhi"] = True
        elif kind == "filter":
            for n in nodes:
                if not _matches(n["data"], op.get("where") or {}):
                    n["data"]["vdim"] = True
        elif kind == "colorBy":
            field = op.get("field")
            for n in nodes:
                val = _node_field(n["data"], field)
                if val is None:
                    continue
                if val not in color_map:
                    color_map[val] = _PALETTE[len(color_map) % len(_PALETTE)]
                n["data"]["vcolor"] = color_map[val]
    return {"nodes": nodes, "edges": edges}


# --- generation (gated) ------------------------------------------------------
def generate_view_spec(
    prompt: str,
    elements: dict,
    *,
    provider: Optional[Provider] = None,
    _transport: Optional[Transport] = None,
) -> Dict[str, Any]:
    """Ask the LLM for a view spec for `prompt`, grounded in the graph's available
    node names + fields. Returns a validated spec (empty if no key / bad reply)."""
    provider = provider or Provider.from_env()
    if provider is None:
        return {"ops": []}
    names = sorted({n["data"].get("name") for n in elements.get("nodes", [])
                    if n["data"].get("name")})[:60]
    fields = sorted({k for n in elements.get("nodes", [])
                     for k in (n["data"].get("meta") or {}).keys()})
    ctx = (f"available node names: {', '.join(names)}\n"
           f"available meta fields: {', '.join(fields)}")
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Request: {prompt}\n\n{ctx}"},
    ]
    try:
        reply = provider.complete(messages, _transport=_transport, max_tokens=500)
    except Exception:
        return {"ops": []}
    return parse_view_spec(reply)


def _main(argv=None) -> int:
    """CLI: `python -m netscope.llm.views <graph.json> "<prompt>"` -> prints the
    validated view spec JSON (the extension shells out to this, then applies the
    spec to the live webview)."""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print('usage: python -m netscope.llm.views <graph.json> "<prompt>"', file=sys.stderr)
        return 2
    try:
        data = json.load(open(argv[0], encoding="utf-8"))
    except Exception as e:
        print(f"could not load graph: {e}", file=sys.stderr)
        return 1
    # accept either a cytoscape {nodes:[{data}]} OR an IR {nodes:[{...}]} dump.
    if data.get("nodes") and "data" not in (data["nodes"][0] or {}):
        elements = {"nodes": [{"data": {"id": n["id"], "name": n.get("name"),
                                        "kind": n.get("kind"), "meta": n.get("meta") or {}}}
                              for n in data["nodes"]], "edges": []}
    else:
        elements = data
    from netscope.llm import available
    if not available():
        print("no LLM key configured — set NETSCOPE_LLM_API_KEY (or "
              "OPENROUTER_API_KEY / OPENAI_API_KEY).", file=sys.stderr)
        return 3
    spec = generate_view_spec(argv[1], elements)
    sys.stdout.write(json.dumps(spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
