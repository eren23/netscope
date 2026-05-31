"""Mermaid sink — a cheap secondary export for docs / paper figures / quick
terminal-pasteable diagrams. `contains` edges are skipped (hierarchy is shown
via labels rather than Mermaid subgraphs to keep it robust)."""
from __future__ import annotations

import re


def _safe(node_id: str) -> str:
    return re.sub(r"\W", "_", node_id)


def _label(node: dict) -> str:
    name = node["name"]
    out = (node.get("meta") or {}).get("out_shape")
    if out:
        name += " " + "x".join(str(x) for x in out)
    return name.replace('"', "'")


def to_mermaid(g) -> str:
    lines = ["flowchart TD"]
    for n in g.nodes():
        lines.append(f'  {_safe(n["id"])}["{_label(n)}"]')
    for e in g.edges():
        if e["kind"] == "contains":
            continue
        lines.append(f'  {_safe(e["src"])} --> {_safe(e["dst"])}')
    return "\n".join(lines)
