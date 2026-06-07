"""Criterion #4 guard: netscope's internal import graph must stay acyclic.

Builds the module dependency graph by AST-scanning every netscope/*.py for
`import netscope...` / `from netscope... import ...` — module-level AND in-function
lazy imports both count — then asserts it is a DAG. The library leans on lazy
imports (e.g. core/ir.py imports the sinks inside methods) precisely to keep this
graph acyclic; this fails loudly if that ever regresses. networkx is already a
runtime dependency, so this needs nothing extra.
"""
import ast
from pathlib import Path

import networkx as nx

PKG = Path(__file__).resolve().parents[1] / "netscope"
FILES = sorted(PKG.rglob("*.py"))


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(PKG.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


MODULES = {_module_name(f) for f in FILES}


def _targets(path: Path) -> set[str]:
    """Resolved netscope module names this file imports (to the finest known module)."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("netscope"):
                    out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and node.module.startswith("netscope"):
                for alias in node.names:
                    sub = f"{node.module}.{alias.name}"
                    out.add(sub if sub in MODULES else node.module)
    # collapse each target onto the nearest module/package node we actually have
    resolved = set()
    for t in out:
        while t and t not in MODULES:
            t = t.rsplit(".", 1)[0] if "." in t else ""
        if t:
            resolved.add(t)
    return resolved


def test_netscope_import_graph_is_acyclic():
    g = nx.DiGraph()
    g.add_nodes_from(MODULES)
    for f in FILES:
        src = _module_name(f)
        for tgt in _targets(f):
            if tgt != src:
                g.add_edge(src, tgt)
    cycles = list(nx.simple_cycles(g))
    assert not cycles, f"circular imports in netscope: {cycles}"
