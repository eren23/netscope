"""Optional semantic hints.

Auto-tracing sees *calls*, not *intent*. These markers let a user name the
semantic regions that pure runtime capture can't infer — the `plan` stage, the
5 `branch`es, the `vote` reduce. Each marker is both a decorator and a context
manager, reentrancy-safe, records source loc for free, and is a pass-through
when no capture session is open (zero overhead in production).
"""
from __future__ import annotations

import contextlib
import functools
import inspect
import time
from typing import Optional

from netscope.core import context as ctx


def _now_if_profiling(cap):
    """A start timestamp when the session is profiling, else None (zero overhead)."""
    return time.perf_counter() if getattr(cap, "profile", False) else None


def _stamp_time(cap, node_id, t0):
    if t0 is not None:
        cap.graph.update_meta(node_id, {"time_ms": round((time.perf_counter() - t0) * 1000, 4)})


class _Marker:
    def __init__(self, name: str, *, kind: str, attrs: dict, loc: Optional[dict] = None):
        self._name = name
        self._kind = kind
        self._attrs = attrs
        self._loc = loc
        self._handle = None
        self._t0 = None

    # context-manager use: `with nv.stage("vote", reduce=True): ...`
    def __enter__(self):
        cap = ctx.active_capture()
        if cap is not None:
            self._handle = cap.open_span(
                self._name, kind=self._kind, attrs=self._attrs, loc=self._loc
            )
            self._t0 = _now_if_profiling(cap)   # per-region wall-time under profile
        return self

    def __exit__(self, *exc):
        cap = ctx.active_capture()
        if cap is not None and self._handle is not None:
            _stamp_time(cap, self._handle.node_id, self._t0)
            cap.close_span(self._handle)
        self._handle = None
        self._t0 = None
        return False

    # decorator use: `@nv.stage("plan")` — captures the def's loc once.
    def __call__(self, fn):
        loc = self._loc or _fn_loc(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            cap = ctx.active_capture()
            if cap is None:
                return fn(*args, **kwargs)
            handle = cap.open_span(self._name, kind=self._kind, attrs=self._attrs, loc=loc)
            t0 = _now_if_profiling(cap)
            try:
                return fn(*args, **kwargs)
            finally:
                _stamp_time(cap, handle.node_id, t0)
                cap.close_span(handle)

        return wrapper


def _fn_loc(fn) -> Optional[dict]:
    try:
        file = inspect.getsourcefile(fn) or inspect.getfile(fn)
        _, line = inspect.getsourcelines(fn)
        return {"file": file, "line": line}
    except Exception:
        return None


def stage(name: str, *, reduce: bool = False, **attrs) -> _Marker:
    if reduce:
        attrs["reduce"] = True
    return _Marker(name, kind="stage", attrs=attrs)


def branch(name: str, **attrs) -> _Marker:
    attrs["branch"] = True
    return _Marker(name, kind="stage", attrs=attrs)


def reduce(name: str, **attrs) -> _Marker:
    attrs["reduce"] = True
    return _Marker(name, kind="stage", attrs=attrs)


@contextlib.contextmanager
def step(label: Optional[str] = None):
    """Mark one generation / decode step — wrap each iteration of an autoregressive
    loop (`with netscope.step(): logits = model(ids)`). Auto-numbered (step 0, 1,
    …) and timed under ``profile=True``; together the steps form the generation
    timeline (see ``netscope.timeline``). A no-op outside a capture session.
    """
    cap = ctx.active_capture()
    if cap is None:
        yield None
        return
    idx = sum(1 for n in cap.graph.nodes() if "step" in (n.get("attrs") or {}))
    handle = cap.open_span(label or f"step {idx}", kind="stage", attrs={"step": idx})
    t0 = _now_if_profiling(cap)
    try:
        yield handle.node_id
    finally:
        _stamp_time(cap, handle.node_id, t0)
        cap.close_span(handle)
