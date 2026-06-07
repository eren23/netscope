"""Mermaid sink — a cheap secondary export for docs / paper figures / quick
terminal-pasteable diagrams. `contains` edges are skipped (hierarchy is shown
via labels rather than Mermaid subgraphs to keep it robust)."""

def _label(node: dict) -> str:
    name = node["name"]
    out = (node.get("meta") or {}).get("out_shape")
    if out:
        name += " " + "x".join(str(x) for x in out)
    return name.replace('"', "'")


def to_mermaid(g) -> str:
    lines = ["flowchart TD"]
    # injective id mapping: distinct node ids -> distinct n0/n1/… so that ids which
    # differ only in non-word chars (`a.b` vs `a/b`) don't collide into one node.
    ids: dict = {}

    def sid(node_id: str) -> str:
        return ids.setdefault(node_id, f"n{len(ids)}")

    for n in g.nodes():
        lines.append(f'  {sid(n["id"])}["{_label(n)}"]')
    for e in g.edges():
        if e["kind"] == "contains":
            continue
        lines.append(f'  {sid(e["src"])} --> {sid(e["dst"])}')
    return "\n".join(lines)
