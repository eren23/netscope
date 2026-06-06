"""netscope — featherweight ML-pipeline graph tracer + visualizer.

Public API:
    netscope.graph(name)        open a capture session (context manager -> NVGraph)
    netscope.active_capture()   the live Capture inside a session, else None
    netscope.is_capturing()     True inside a session

Importing netscope auto-installs framework instrumentation via wrapt post-import
hooks (import-order independent): the torch forward-hook tracer activates while a
capture session is open. No decorators or model edits required.
"""
from __future__ import annotations

from netscope.core.capture import Capture, graph
from netscope.core.context import active_capture, is_capturing
from netscope.core.diff import annotate_diff as _annotate_diff
from netscope.core.diff import diff_graphs as _diff_graphs
from netscope.core.ir import SCHEMA_VERSION, NVGraph
from netscope.hints.api import branch, reduce, stage

_installed = False


def install() -> None:
    """Register framework instrumentation (idempotent)."""
    global _installed
    if _installed:
        return
    _installed = True
    try:
        import wrapt
    except Exception:
        return
    wrapt.register_post_import_hook(lambda *_: _install_torch(), "torch")
    wrapt.register_post_import_hook(lambda *_: _install_transformers(), "transformers")


def _install_torch() -> None:
    try:
        from netscope.instrument import torch_nn

        torch_nn.register()
    except Exception:
        pass


def _install_transformers() -> None:
    try:
        from netscope.instrument import transformers_hf

        transformers_hf.register()
    except Exception:
        pass


install()


def diff(before: "NVGraph", after: "NVGraph") -> dict:
    """Structured diff between two traces (before/after an edit, or two variants):
    nodes added/removed and shape/param deltas on the ones that stayed. Keyed by a
    stable identity (qualname > loc > name), so it survives the id shifts that come
    from inserting a layer."""
    return _diff_graphs(before, after)


def diff_view(before: "NVGraph", after: "NVGraph") -> "NVGraph":
    """A single graph with every node tagged `attrs.diff` (added/removed/changed/
    same) — call `.show()` on it to render the diff in color."""
    return _annotate_diff(before, after)


def roles(graph: "NVGraph") -> dict:
    """Architectural role breakdown of a traced model — `{attention: n, mlp: n,
    norm: n, ...}`. The transformer lens; the graph's `role:` overlay colors nodes
    by the same classification (attention vs MLP vs norm at a glance)."""
    from netscope.enrich.roles import role_counts
    return role_counts(graph)


__all__ = [
    "graph",
    "active_capture",
    "is_capturing",
    "install",
    "stage",
    "branch",
    "reduce",
    "diff",
    "diff_view",
    "roles",
    "NVGraph",
    "Capture",
    "SCHEMA_VERSION",
]
