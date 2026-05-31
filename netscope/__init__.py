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

__all__ = [
    "graph",
    "active_capture",
    "is_capturing",
    "install",
    "stage",
    "branch",
    "reduce",
    "NVGraph",
    "Capture",
    "SCHEMA_VERSION",
]
