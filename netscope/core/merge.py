"""Fuse a runtime graph with a static graph by source location.

This is the one integration point between producers. A runtime node and a static
node that share the same ``loc`` describe the same code, seen two ways: the
runtime node knows real shapes/dtypes (the executed path); the static node knows
declared structure (names, branch counts, votes — including code that never ran).
The fused node keeps the runtime ``meta`` and gains the static ``attrs``.

Nodes present on only one side are carried through unchanged, so static-only
structure (e.g. a vote stage the tracer can't see) still appears.
"""
from __future__ import annotations

from typing import Optional

from netscope.core.ir import NVGraph


def _loc_key(node: dict) -> Optional[tuple]:
    loc = node.get("loc")
    if not loc:
        return None
    return (loc.get("file"), loc.get("line"))


def merge(runtime: NVGraph, static: NVGraph) -> NVGraph:
    fused = NVGraph(name=runtime.name or static.name)

    static_by_loc = {}
    for n in static.nodes():
        key = _loc_key(n)
        if key is not None:
            static_by_loc[key] = n

    matched_static_ids = set()

    # 1) runtime nodes, fused with any static node sharing their loc. A static
    #    node fuses into AT MOST ONE runtime node (first match wins) — two runtime
    #    nodes can legitimately share a loc (a submodule called twice, a loop
    #    body), and duplicating the static node's attrs across both is wrong.
    for rt in runtime.nodes():
        key = _loc_key(rt)
        st = static_by_loc.get(key) if key is not None else None
        if st is not None and st["id"] in matched_static_ids:
            st = None   # already fused into an earlier runtime node at this loc
        attrs = dict(rt.get("attrs") or {})
        source = rt["source"]
        if st is not None:
            attrs.update(st.get("attrs") or {})
            source = "fused"
            matched_static_ids.add(st["id"])
        fused.add_node(
            rt["id"], kind=rt["kind"], name=rt["name"], parent=rt.get("parent"),
            source=source, loc=rt.get("loc"), meta=rt.get("meta"), attrs=attrs,
        )

    # 2) runtime edges
    for e in runtime.edges():
        fused.add_edge(e["src"], e["dst"], kind=e["kind"],
                       tensor_meta=e.get("tensor_meta"), source=e["source"],
                       condition=e.get("condition"))

    # 3) static-only nodes (structure the runtime never saw — e.g. a branch fan
    #    or a vote stage). BUT drop unmatched declared-dim nodes: those exist only
    #    for the static pre-check and are redundant with runtime module nodes, so
    #    an unmatched one is a layer that didn't run (e.g. an unused fallback
    #    class) — keeping it would float a stray node in the real trace.
    for st in static.nodes():
        if st["id"] in matched_static_ids:
            continue
        if (st.get("attrs") or {}).get("declared_dim"):
            continue
        fused.add_node(
            st["id"], kind=st["kind"], name=st["name"], parent=st.get("parent"),
            source="static", loc=st.get("loc"), meta=st.get("meta"),
            attrs=st.get("attrs"),
        )

    return fused
