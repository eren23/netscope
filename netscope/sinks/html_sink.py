"""Interactive Cytoscape HTML sink.

Maps the IR to Cytoscape elements (the `parent` field -> compound nodes, so the
hierarchy renders as nested boxes and `contains` edges are dropped) and injects
them into the shared `web/template.html`. That same template is reused verbatim
by the M4 VSCode webview, so the renderer is written once.
"""
from __future__ import annotations

import html as _html
import json
import os
import tempfile

_WEB = os.path.join(os.path.dirname(__file__), os.pardir, "web")
_TEMPLATE = os.path.join(_WEB, "template.html")
_VENDOR = os.path.join(_WEB, "vendor")

# Order matters: cytoscape core first, then dagre + its adapter, then
# expand-collapse. Inlined into the HTML so the graph is fully self-contained
# (no CDN) — works offline and inside locked-down VSCode webviews.
_VENDOR_LIBS = [
    "cytoscape.min.js",
    "dagre.min.js",
    "cytoscape-dagre.min.js",
    "cytoscape-expand-collapse.min.js",
]


def _vendor_scripts() -> str:
    out = []
    for name in _VENDOR_LIBS:
        path = os.path.join(_VENDOR, name)
        with open(path, encoding="utf-8") as f:
            src = f.read()
        # guard against a future broken vendor fetch landing an HTML 404 page
        if src.lstrip()[:5].lower() == "<html":
            raise RuntimeError(f"vendor lib {name} looks like an HTML error page, not JS")
        out.append(f"<!-- {name} -->\n<script>\n{src}\n</script>")
    return "\n".join(out)


def _node_label(node: dict) -> str:
    name = node["name"]
    out = (node.get("meta") or {}).get("out_shape")
    if out:
        return f"{name}\n{list(out)}"
    return name


def to_cytoscape(g) -> dict:
    from netscope.core.checks import detect_mismatches

    warnings = detect_mismatches(g)
    # nodes/edges touched by a warning, so the renderer can paint them red.
    warn_nodes = set()
    warn_pairs = set()
    for w in warnings:
        warn_nodes.add(w["src"])
        warn_nodes.add(w["dst"])
        warn_pairs.add((w["src"], w["dst"]))

    nodes = []
    for n in g.nodes():
        data = {
            "id": n["id"],
            "name": n["name"],
            "label": _node_label(n),
            "kind": n["kind"],
            "meta": n.get("meta") or {},
            "loc": n.get("loc"),
            "prov": n.get("source"),          # runtime | static | fused
            "attrs": n.get("attrs") or {},    # branch / reduce / ...
        }
        if n["id"] in warn_nodes:
            data["warn"] = True
        if (n.get("attrs") or {}).get("inferred"):
            data["inferred"] = True       # LLM-inferred -> rendered dashed/dim
        if n.get("parent"):
            data["parent"] = n["parent"]
        nodes.append({"data": data})

    edges = []
    for i, e in enumerate(g.edges()):
        if e["kind"] == "contains":  # implied by compound nesting
            continue
        # NB: cytoscape reserves data.source/target for endpoints; the producer
        # of the edge (runtime vs hint) goes in `flow`.
        data = {
            "id": f"e{i}", "source": e["src"], "target": e["dst"],
            "kind": e["kind"], "flow": e.get("source"),
        }
        if (e["src"], e["dst"]) in warn_pairs:
            data["warn"] = True
        tm = e.get("tensor_meta")
        if tm and tm.get("shape"):
            data["label"] = "x".join(str(x) for x in tm["shape"])
        edges.append({"data": data})

    return {"nodes": nodes, "edges": edges, "warnings": warnings}


def to_html(g, title: str = None) -> str:
    title = title or g.name or "netscope"
    with open(_TEMPLATE, encoding="utf-8") as f:
        tpl = f.read()
    elements = json.dumps(to_cytoscape(g))
    return (
        tpl.replace("__NETSCOPE_VENDOR__", _vendor_scripts())
        .replace("__NETSCOPE_ELEMENTS__", elements)
        .replace("__NETSCOPE_TITLE__", _html.escape(title))
    )


def show(g, path: str = None, open_browser: bool = True) -> str:
    if path is None:
        safe = (g.name or "graph").replace(os.sep, "_") or "graph"
        path = os.path.join(tempfile.gettempdir(), f"{safe}.netscope.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_html(g))
    if open_browser:
        import webbrowser

        webbrowser.open("file://" + os.path.abspath(path))
    return path
