"""Registry of session-scoped instrumentors.

Some instrumentors are cheap to leave installed at import time (e.g. a wrapt
wrapper on ``transformers...generate``, gated by ``is_capturing``). Others —
notably torch's global per-module forward hooks — are only worth paying for
*while a session is open*. Those register here; ``capture.graph()`` calls
``enter_session()`` on entry and ``exit_session()`` on exit so the hooks exist
only during capture (zero hooks => zero overhead otherwise).

This is also the framework-extensibility seam: a new framework implements the
:class:`Instrumentor` contract and registers an instance via
:func:`register_session_instrumentor` — no core file changes required. See
``docs/extending-frameworks.md`` for a walkthrough using the torch adapter.
"""
import contextlib
from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class Instrumentor(Protocol):
    """What a session-scoped framework adapter must provide.

    ``on_enter`` installs the framework's tracing hooks when a capture session
    opens and returns an opaque *handle* (anything — a tuple of hook handles, a
    list, …); that same handle is passed back to ``on_exit`` to tear them down.
    Both run inside ``try/except`` (see :func:`enter_session` /
    :func:`exit_session`), so a misbehaving adapter degrades to "no trace from
    this framework" rather than breaking the user's program.

    Optional — not part of the required Protocol so frameworks that need no guard
    simply omit it: an adapter may also define ``inference_context(self)`` returning
    a context manager (e.g. ``torch.no_grad()``). The isolation re-run wraps the
    re-executed module in the combined guard collected by :func:`inference_context`.
    """

    def on_enter(self) -> object: ...

    def on_exit(self, handle: object, /) -> None: ...


_session_instrumentors: "list[Instrumentor]" = []


def register_session_instrumentor(inst: Instrumentor) -> None:
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


@contextlib.contextmanager
def inference_context() -> Iterator[None]:
    """Combined "inference mode" guard contributed by the registered instrumentors
    (e.g. torch's ``no_grad``), used for the isolation re-run so the focused
    re-execution doesn't build autograd graphs. A no-op when no adapter provides
    one — and for the no-framework case. Best-effort: never raises."""
    with contextlib.ExitStack() as stack:
        for inst in _session_instrumentors:
            make_guard = getattr(inst, "inference_context", None)
            if make_guard is None:
                continue
            try:
                cm = make_guard()
            except Exception:
                cm = None
            if cm is not None:
                try:
                    stack.enter_context(cm)
                except Exception:
                    pass
        yield
