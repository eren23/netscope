"""Optional semantic hints.

Auto-tracing sees *calls*, not *intent*. These markers let a user name the
semantic regions that pure runtime capture can't infer — the `plan` stage, the
5 `branch`es, the `vote` reduce. Each marker is both a decorator and a context
manager, reentrancy-safe, records source loc for free, and is a pass-through
when no capture session is open (zero overhead in production).
"""
from __future__ import annotations

import functools
import inspect
from typing import Optional

from netscope.core import context as ctx


class _Marker:
    def __init__(self, name: str, *, kind: str, attrs: dict, loc: Optional[dict] = None):
        self._name = name
        self._kind = kind
        self._attrs = attrs
        self._loc = loc
        self._handle = None

    # context-manager use: `with nv.stage("vote", reduce=True): ...`
    def __enter__(self):
        cap = ctx.active_capture()
        if cap is not None:
            self._handle = cap.open_span(
                self._name, kind=self._kind, attrs=self._attrs, loc=self._loc
            )
        return self

    def __exit__(self, *exc):
        cap = ctx.active_capture()
        if cap is not None and self._handle is not None:
            cap.close_span(self._handle)
        self._handle = None
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
            try:
                return fn(*args, **kwargs)
            finally:
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
