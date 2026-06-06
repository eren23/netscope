"""Grounded prompt construction: the LLM only ever sees real netscope data.

For a given node we assemble (a) its IR slice — kind, qualname, declared/observed
shapes, params; (b) its immediate neighbours in the dataflow; (c) any mismatch
warning the detector raised that touches it; and (d) the actual source lines at
the node's `loc`. The model is told to ground every claim in that context and to
cite file:line. This is what keeps answers honest — it explains/annotates the
graph, it doesn't invent architecture.
"""
from __future__ import annotations

import os
from typing import List, Optional

_SYSTEM = (
    "You are netscope's assistant, embedded in a neural-network tracer. You are "
    "given REAL data captured from a PyTorch/Hugging Face model: a node from the "
    "model's graph, its tensor shapes, its neighbours, any shape-mismatch warning, "
    "and the actual source lines. Answer ONLY from this context. Be concise and "
    "concrete. Cite file:line when you reference code. If a shape mismatch is "
    "present, explain the clash in plain terms and suggest the minimal fix "
    "(e.g. change a layer dim, add a flatten/reshape). Never invent layers or "
    "shapes that aren't in the context."
)

_QUESTION_PROMPTS = {
    "explain": "Explain what this part of the network does and how the tensor flows through it.",
    "why_warn": "Why is this node flagged with a warning? Explain the mismatch.",
    "suggest_fix": "Suggest the minimal code change to fix the issue on this node.",
}


def _fmt_shape(meta: dict, key: str) -> Optional[str]:
    s = meta.get(key)
    if isinstance(s, list) and s:
        return "[" + ", ".join(str(x) for x in s) + "]"
    return None


def _node_block(node: dict) -> str:
    meta = node.get("meta") or {}
    lines = [f"node: {node['name']} (kind={node['kind']})"]
    q = meta.get("qualname")
    if q:
        lines.append(f"qualname: {q}")
    for label, key in (("input shape", "in_shape"), ("output shape", "out_shape")):
        v = _fmt_shape(meta, key)
        if v:
            lines.append(f"{label}: {v}")
    if isinstance(meta.get("params"), int) and meta["params"]:
        lines.append(f"params: {meta['params']:,}")
    loc = node.get("loc")
    if loc:
        lines.append(f"source: {loc.get('file')}:{loc.get('line')}")
    return "\n".join(lines)


def _neighbours_block(graph, node_id: str) -> str:
    by_id = {n["id"]: n for n in graph.nodes()}
    ins, outs = [], []
    for e in graph.edges():
        if e["kind"] != "dataflow":
            continue
        if e["dst"] == node_id and e["src"] in by_id:
            ins.append(by_id[e["src"]])
        if e["src"] == node_id and e["dst"] in by_id:
            outs.append(by_id[e["dst"]])

    def names(ns, cap=8):
        # cap the list so a hub node with hundreds of neighbours doesn't blow the
        # context budget; note how many were elided.
        labels = [(n.get("meta") or {}).get("qualname") or n["name"] for n in ns]
        if not labels:
            return "(none)"
        if len(labels) > cap:
            return ", ".join(labels[:cap]) + f", …and {len(labels) - cap} more"
        return ", ".join(labels)

    return f"upstream (feeds in): {names(ins)}\ndownstream (consumes): {names(outs)}"


def _warning_block(graph, node_id: str) -> Optional[str]:
    warns = graph.to_dict().get("warnings", [])
    hits = [w for w in warns if w.get("src") == node_id or w.get("dst") == node_id]
    if not hits:
        return None
    return "warnings:\n" + "\n".join(f"  - {w['detail']}" for w in hits)


def _source_block(node: dict, context: int = 2, root: Optional[str] = None) -> Optional[str]:
    loc = node.get("loc")
    if not loc or not loc.get("file") or not loc.get("line"):
        return None
    path = loc["file"]
    # `root` (set on the MCP path, where the graph came from an UNTRUSTED saved
    # trace) restricts source reads to that tree, so a crafted `loc.file=/etc/...`
    # can't be slurped into the prompt. Unset (the in-process library path) reads
    # freely — there the graph is the user's own live trace.
    if root is not None:
        try:
            real, base = os.path.realpath(path), os.path.realpath(root)
            if real != base and os.path.commonpath([real, base]) != base:
                return None
            path = real
        except (OSError, ValueError):
            return None
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    line = loc["line"]
    lo = max(1, line - context)
    hi = min(len(src), line + context)
    out = []
    for i in range(lo, hi + 1):
        marker = ">" if i == line else " "
        text = src[i - 1]
        if len(text) > 200:            # clip a pathological long line (minified/data)
            text = text[:200] + "…"
        out.append(f"{marker} {i}: {text}")
    return "source near " + loc["file"].split("/")[-1] + ":\n" + "\n".join(out)


def build_messages(graph, node_id: str, *, question: str = "explain",
                   source_root: Optional[str] = None) -> List[dict]:
    """Build grounded [system, user] messages for a node + question kind.

    source_root: if set, source lines are only read from files under this dir —
    pass it when the graph came from an untrusted saved trace (the MCP path)."""
    by_id = {n["id"]: n for n in graph.nodes()}
    node = by_id.get(node_id)
    if node is None:
        raise KeyError(f"no node {node_id!r} in graph")

    ask = _QUESTION_PROMPTS.get(question, _QUESTION_PROMPTS["explain"])
    parts = [_node_block(node), _neighbours_block(graph, node_id)]
    warn = _warning_block(graph, node_id)
    if warn:
        parts.append(warn)
    source = _source_block(node, root=source_root)
    if source:
        parts.append(source)
    context = "\n\n".join(parts)

    user = f"{ask}\n\n--- context (real netscope data) ---\n{context}"
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]
