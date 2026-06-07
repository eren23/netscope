"""The capture session.

`graph(name)` opens a `Capture`, marks it active for the `with` block, activates
session-scoped instrumentors (torch forward hooks, etc.), and yields the
underlying `NVGraph`. Producers emit via `cap.span(...)` (context manager) or the
lower-level `open_span` / `close_span` pair (used by the torch pre/post forward
hooks, which are two separate callbacks and so cannot use a `with` block).
"""
import contextlib
import itertools
import os
from typing import Iterator, Optional

from netscope.core import context as ctx
from netscope.core import registry
from netscope.core.ir import NVGraph


class SpanHandle:
    __slots__ = ("node_id", "parent_token")

    def __init__(self, node_id: str, parent_token) -> None:
        self.node_id = node_id
        self.parent_token = parent_token


class Capture:
    def __init__(self, name: str = "", profile: bool = False) -> None:
        self.graph = NVGraph(name=name)
        self._counter = itertools.count()
        # opt-in wall-time measurement. Read by the torch hook; off by default so
        # the steady-state trace stays metadata-only and ~zero overhead.
        self.profile = profile

    def _new_id(self, name: str) -> str:
        return f"{name}#{next(self._counter)}"

    def open_span(
        self,
        name: str,
        *,
        kind: str,
        loc: Optional[dict] = None,
        meta: Optional[dict] = None,
        attrs: Optional[dict] = None,
    ) -> SpanHandle:
        node_id = self._new_id(name)
        parent = ctx.current_parent()
        self.graph.add_node(
            node_id, kind=kind, name=name, parent=parent,
            source="runtime", loc=loc, meta=meta, attrs=attrs,
        )
        if parent is not None:
            self.graph.add_edge(parent, node_id, kind="contains", source="runtime")
        token = ctx.push_parent(node_id)
        return SpanHandle(node_id, token)

    def close_span(self, handle: SpanHandle, *, meta_update: Optional[dict] = None) -> None:
        if meta_update:
            self.graph.update_meta(handle.node_id, meta_update)
        ctx.pop_parent(handle.parent_token)

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str,
        loc: Optional[dict] = None,
        meta: Optional[dict] = None,
        attrs: Optional[dict] = None,
    ) -> Iterator[str]:
        handle = self.open_span(name, kind=kind, loc=loc, meta=meta, attrs=attrs)
        try:
            yield handle.node_id
        finally:
            self.close_span(handle)


@contextlib.contextmanager
def graph(name: str = "", *, profile: bool = False) -> Iterator[NVGraph]:
    """Open a capture session. Yields the live NVGraph.

    profile=True additionally measures per-module wall-time (`meta.time_ms`); it's
    opt-in because timing has real overhead, whereas the default trace is
    metadata-only. Activation/param byte counts ride on every trace regardless
    (they're free — derived from shapes already captured).

    Sessions do not nest: a second `graph()` opened inside an active one would
    double-install the global torch hooks, so every module would be captured
    twice into the inner graph. We raise instead of silently corrupting — use one
    session, or close the outer first. (The internal isolation re-run opens a
    nested session deliberately; it clears the active capture first, so it is not
    affected by this guard.)
    """
    if ctx.active_capture() is not None:
        raise RuntimeError(
            "netscope is already capturing — graph() sessions cannot be nested. "
            "Close the outer `with netscope.graph(...)` first, or use a single "
            "session."
        )
    # NETSCOPE_PROFILE=1 forces profiling on without editing the user's graph()
    # call — this is how the extension's "Run & Trace (profiled)" turns it on.
    if not profile and os.environ.get("NETSCOPE_PROFILE"):
        profile = True
    cap = Capture(name, profile=profile)
    token = ctx.set_capture(cap)
    stack_token = ctx.push_clean_parent_scope()   # fresh stack; restored on exit
    handles = registry.enter_session()
    try:
        yield cap.graph
    finally:
        registry.exit_session(handles)
        from netscope.core.stage_flow import infer_stage_flow
        from netscope.sinks.file_sink import maybe_dump

        infer_stage_flow(cap.graph)
        maybe_dump(cap.graph)
        ctx.reset_capture(token)
        ctx.restore_parent_scope(stack_token)     # never leak a dangling parent
        _maybe_run_isolated(cap)


def _maybe_run_isolated(cap: "Capture") -> None:
    """If the run captured an isolation target (NETSCOPE_ISOLATE matched a
    submodule), re-run JUST that module on its real frozen input in a fresh
    session and dump the focused sub-trace to NETSCOPE_ISOLATE_OUT.

    Best-effort: never raises into the user's program. The nested session runs
    with NETSCOPE_OUT / NETSCOPE_ISOLATE cleared so it neither clobbers the main
    trace nor recurses.
    """
    stash = getattr(cap, "_isolate_stash", None)
    if not stash:
        return
    target, args, kwargs, name = stash
    iso_out = os.environ.get("NETSCOPE_ISOLATE_OUT")
    saved_out = os.environ.pop("NETSCOPE_OUT", None)
    saved_iso = os.environ.pop("NETSCOPE_ISOLATE", None)
    try:
        with graph(f"isolate:{name}") as ig:
            try:
                import torch

                with torch.no_grad():
                    target(*args, **kwargs)
            except Exception:
                pass  # a kwarg-heavy / stateful module may not re-run cleanly
        if iso_out:
            try:
                with open(iso_out, "w", encoding="utf-8") as f:
                    f.write(ig.to_json())
            except Exception:
                pass
    finally:
        if saved_out is not None:
            os.environ["NETSCOPE_OUT"] = saved_out
        if saved_iso is not None:
            os.environ["NETSCOPE_ISOLATE"] = saved_iso
