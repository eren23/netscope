"""Registry of session-scoped instrumentors.

Some instrumentors are cheap to leave installed at import time (e.g. a wrapt
wrapper on ``transformers...generate``, gated by ``is_capturing``). Others —
notably torch's global per-module forward hooks — are only worth paying for
*while a session is open*. Those register here; ``capture.graph()`` calls
``enter_session()`` on entry and ``exit_session()`` on exit so the hooks exist
only during capture (zero hooks => zero overhead otherwise).

This is also the framework-extensibility seam: new frameworks register an
instrumentor without touching core.
"""
from __future__ import annotations

_session_instrumentors: list = []


def register_session_instrumentor(inst) -> None:
    """Register once per instrumentor type (idempotent)."""
    if any(type(i) is type(inst) for i in _session_instrumentors):
        return
    _session_instrumentors.append(inst)


def clear() -> None:
    _session_instrumentors.clear()


def enter_session() -> list:
    """Activate every instrumentor; return (inst, handle) pairs for teardown."""
    handles = []
    for inst in _session_instrumentors:
        try:
            handles.append((inst, inst.on_enter()))
        except Exception:
            pass
    return handles


def exit_session(handles) -> None:
    for inst, handle in handles:
        try:
            inst.on_exit(handle)
        except Exception:
            pass
