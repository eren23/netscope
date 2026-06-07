"""The netscope intermediate representation (IR).

A small, typed, hierarchical graph stored on a ``networkx.DiGraph`` (so we get
serialization + graph algorithms for free). This is the single contract every
producer (runtime trace, static AST) and every sink (HTML/JSON/websocket)
speaks.

Every node carries ``loc`` (source location) and ``source`` (which producer
emitted it). Those two fields are the backbone of the later static<->runtime
fusion: nodes from different producers are merged by matching ``loc``.

Node ``kind``:  pipeline | stage | model | module | op
Edge ``kind``:  dataflow | control | contains
``source``:     runtime | static | fused
"""
from typing import Any, Optional

import networkx as nx

SCHEMA_VERSION = "1"


class NVGraph:
    """A hierarchical dataflow graph of an ML pipeline."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._g = nx.DiGraph()

    # -- nodes ----------------------------------------------------------------
    def add_node(
        self,
        id: str,
        *,
        kind: str,
        name: str,
        parent: Optional[str] = None,
        source: str = "runtime",
        loc: Optional[dict] = None,
        meta: Optional[dict] = None,
        attrs: Optional[dict] = None,
    ) -> str:
        self._g.add_node(
            id,
            kind=kind,
            name=name,
            parent=parent,
            source=source,
            loc=loc,
            meta=dict(meta) if meta else {},
            attrs=dict(attrs) if attrs else {},
        )
        return id

    def get_node(self, id: str) -> dict:
        return {"id": id, **self._g.nodes[id]}

    def has_node(self, id: str) -> bool:
        return self._g.has_node(id)

    def nodes(self) -> list[dict]:
        return [{"id": n, **d} for n, d in self._g.nodes(data=True)]

    def children(self, parent_id: str) -> list[str]:
        return [n for n, d in self._g.nodes(data=True) if d.get("parent") == parent_id]

    def update_meta(self, id: str, mapping: dict) -> None:
        """Merge ``mapping`` into a node's ``meta`` (used to fill output shapes)."""
        self._g.nodes[id]["meta"].update(mapping)

    # -- edges ----------------------------------------------------------------
    def add_edge(
        self,
        src: str,
        dst: str,
        *,
        kind: str,
        tensor_meta: Optional[dict] = None,
        source: str = "runtime",
        condition: Optional[str] = None,
    ) -> None:
        self._g.add_edge(
            src, dst, kind=kind, tensor_meta=tensor_meta, source=source, condition=condition
        )

    def get_edge(self, src: str, dst: str) -> dict:
        return {"src": src, "dst": dst, **self._g.edges[src, dst]}

    def edges(self) -> list[dict]:
        return [{"src": u, "dst": v, **d} for u, v, d in self._g.edges(data=True)]

    # -- serialization --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        from netscope.core.checks import detect_mismatches

        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "nodes": self.nodes(),
            "edges": self.edges(),
            "warnings": detect_mismatches(self),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NVGraph":
        """Rebuild a graph from ``to_dict()`` output (or a saved JSON trace).

        Inverse of ``to_dict`` for nodes + edges; ``warnings`` aren't stored back
        (they're recomputed on demand). Used to diff two persisted traces."""
        g = cls(name=data.get("name", ""))
        for n in data.get("nodes", []):
            g.add_node(
                n["id"], kind=n.get("kind", "module"), name=n.get("name", n["id"]),
                parent=n.get("parent"), source=n.get("source", "runtime"),
                loc=n.get("loc"), meta=n.get("meta"), attrs=n.get("attrs"),
            )
        for e in data.get("edges", []):
            # skip dangling edges — nx.add_edge would otherwise auto-create a junk
            # endpoint node (kind/name None) from a truncated or hand-edited trace.
            if not (g.has_node(e.get("src")) and g.has_node(e.get("dst"))):
                continue
            g.add_edge(
                e["src"], e["dst"], kind=e.get("kind", "dataflow"),
                tensor_meta=e.get("tensor_meta"), source=e.get("source", "runtime"),
                condition=e.get("condition"),
            )
        return g

    # -- sinks (lazy imports keep core decoupled from rendering) ---------------
    def to_json(self, indent: int = 2) -> str:
        from netscope.sinks.json_sink import to_json

        return to_json(self, indent=indent)

    def to_mermaid(self) -> str:
        from netscope.sinks.mermaid_sink import to_mermaid

        return to_mermaid(self)

    def to_html(self, title: Optional[str] = None) -> str:
        from netscope.sinks.html_sink import to_html

        return to_html(self, title=title)

    def show(self, path: Optional[str] = None, open_browser: bool = True) -> str:
        """Write an interactive standalone HTML graph and (optionally) open it."""
        from netscope.sinks.html_sink import show

        return show(self, path=path, open_browser=open_browser)
