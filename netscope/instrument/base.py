"""Generic span wrapper for callable-based auto-instrumentation.

This is the foundation the HF (and future provider) instrumentors build on,
following the proven `wrapt` + `safe_patch` pattern:

* ``span_wrapper`` returns a wrapt-style ``(wrapped, instance, args, kwargs)``
  wrapper that opens a span while capturing, derives node name/meta from the
  call, and is a pure pass-through otherwise.
* Tracing NEVER breaks the wrapped call: if name/meta derivation or span
  bookkeeping throws, the original call still runs and returns (MLflow's
  ``safe_patch`` rule).
"""
from __future__ import annotations

from typing import Callable, Optional

import wrapt

from netscope.core import context as ctx

NameFn = Callable[[object, tuple, dict], str]
MetaFn = Callable[[object, tuple, dict], Optional[dict]]


def span_wrapper(*, name_fn: NameFn, kind: str, meta_fn: Optional[MetaFn] = None):
    def wrapper(wrapped, instance, args, kwargs):
        cap = ctx.active_capture()
        if cap is None:
            return wrapped(*args, **kwargs)  # zero-overhead gate
        handle = None
        try:
            name = name_fn(instance, args, kwargs)
            meta = meta_fn(instance, args, kwargs) if meta_fn else None
            handle = cap.open_span(name, kind=kind, meta=meta)
        except Exception:
            handle = None  # never block the real call on a tracing bug
        try:
            return wrapped(*args, **kwargs)
        finally:
            if handle is not None:
                try:
                    cap.close_span(handle)
                except Exception:
                    pass

    return wrapper


def wrap_callable(
    fn: Callable, *, name_fn: NameFn, kind: str, meta_fn: Optional[MetaFn] = None
) -> Callable:
    """Wrap a single callable (used in tests + ad-hoc instrumentation)."""
    return wrapt.FunctionWrapper(fn, span_wrapper(name_fn=name_fn, kind=kind, meta_fn=meta_fn))


def safe_patch(
    module: str, name: str, *, name_fn: NameFn, kind: str, meta_fn: Optional[MetaFn] = None
) -> bool:
    """Patch ``module.name`` in place. Returns True on success, never raises."""
    try:
        wrapt.wrap_function_wrapper(
            module, name, span_wrapper(name_fn=name_fn, kind=kind, meta_fn=meta_fn)
        )
        return True
    except Exception:
        return False
