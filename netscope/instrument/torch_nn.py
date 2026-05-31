"""torch auto-instrumentation via global module forward hooks.

A session-scoped instrumentor: on session enter it registers torch's *global*
`register_module_forward_pre_hook` + `register_module_forward_hook` (one pair
fires for every nn.Module), and removes them on exit.

* pre-hook  -> opens a span (nesting by the contextvars parent stack -> module
  hierarchy), records input shape + params, and draws `dataflow` edges from any
  module that produced one of this module's input tensors.
* post-hook -> records the output shape, closes the span, and registers this
  module as the producer of its output tensor(s).

Producer bookkeeping is keyed by `id(tensor)` but each entry holds a *weakref*
to the producing tensor. A match counts only if the weakref still resolves to
the very same live object (`ref() is t`). This defeats CPython id() reuse: once
an intermediate tensor is freed its id may be handed to an unrelated tensor, but
the stale weakref is then dead, so no false edge is drawn. We keep only tensor
metadata (shape) + a weakref, never a strong reference, so nothing is retained.
"""
from __future__ import annotations

import os
import weakref
from typing import Iterator, Optional

from netscope.core import context as ctx
from netscope.core import registry
from netscope.enrich.params import own_params


def _is_tensor(x) -> bool:
    try:
        import torch

        return isinstance(x, torch.Tensor)
    except Exception:
        return False


def _shape(x) -> Optional[list]:
    return list(x.shape) if _is_tensor(x) else None


def _iter_tensors(obj) -> Iterator:
    """Yield tensors found directly in obj or one level inside a tuple/list."""
    if _is_tensor(obj):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            if _is_tensor(item):
                yield item


def _freeze(obj):
    """Detach+clone every tensor in obj (recursing into tuple/list/dict) so a
    snapshot of a module's real call survives to a later isolated re-run, immune
    to in-place mutation. Non-tensors pass through unchanged."""
    if _is_tensor(obj):
        try:
            return obj.detach().clone()
        except Exception:
            return obj
    if isinstance(obj, tuple):
        return tuple(_freeze(x) for x in obj)
    if isinstance(obj, list):
        return [_freeze(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _freeze(v) for k, v in obj.items()}
    return obj


def id2name_items(root):
    """(qualname, module) pairs for the root subtree; empty on any failure."""
    try:
        return list(root.named_modules())
    except Exception:
        return []


def _attach_isolation_capture(cap, module, qualname):
    """Register a per-module forward pre-hook on the isolation target that stashes
    its real positional AND keyword inputs (frozen) the first time it runs. Per-
    module hooks support ``with_kwargs`` across torch versions (the global hook
    does not), so this is how kwargs-taking blocks become re-runnable in isolation.
    Returns the hook handle (removed on session exit)."""

    def grab(mod, args, kwargs):
        if getattr(cap, "_isolate_stash", None) is None:
            try:
                cap._isolate_stash = (
                    mod, _freeze(tuple(args)), _freeze(dict(kwargs or {})), qualname,
                )
            except Exception:
                pass

    try:
        return module.register_forward_pre_hook(grab, with_kwargs=True)
    except TypeError:
        # very old torch without with_kwargs: positional-only (kwargs lost, but
        # modules whose kwargs have defaults still re-run fine).
        def grab_pos(mod, args):
            if getattr(cap, "_isolate_stash", None) is None:
                try:
                    cap._isolate_stash = (mod, _freeze(tuple(args)), {}, qualname)
                except Exception:
                    pass

        return module.register_forward_pre_hook(grab_pos)


class TorchForwardInstrumentor:
    """Adds torch global forward hooks while a capture session is open."""

    def on_enter(self):
        from torch.nn.modules.module import (
            register_module_forward_hook,
            register_module_forward_pre_hook,
        )

        pending: list = []      # stack of SpanHandle (sync, nested execution)
        producers: dict = {}    # id(tensor) -> (node_id, weakref(tensor))
        id2name: dict = {}      # id(submodule) -> qualified name, per top-level fwd
        extra: list = []        # per-module hooks attached for isolation (removed on exit)
        isolate_target = os.environ.get("NETSCOPE_ISOLATE") or None

        def pre(module, args):
            cap = ctx.active_capture()
            if cap is None:
                return
            if not pending:
                # this module is the root of the current top-level forward; map
                # every submodule's id -> its qualified name (e.g. model.layers.2)
                # so each node is addressable for click-to-source + isolation.
                id2name.clear()
                try:
                    for nm, m in module.named_modules():
                        id2name[id(m)] = nm
                except Exception:
                    pass
                # isolation: attach a per-module pre-hook to the target module so
                # we capture its REAL positional AND keyword inputs (the global
                # hook above is positional-only). Targeted -> zero cost otherwise.
                if isolate_target is not None:
                    target = next(
                        (m for nm, m in id2name_items(module) if nm == isolate_target),
                        None,
                    )
                    if target is not None:
                        extra.append(_attach_isolation_capture(cap, target, isolate_target))
            qualname = id2name.get(id(module))
            meta = {"params": own_params(module)}
            in_shape = _shape(args[0]) if args else None
            if in_shape is not None:
                meta["in_shape"] = in_shape
            if qualname:
                meta["qualname"] = qualname
            handle = cap.open_span(type(module).__name__, kind="module", meta=meta)
            pending.append(handle)
            # dataflow: link the producer of each input tensor -> this module,
            # but only if the recorded weakref still points to the SAME tensor
            # (guards against id() reuse after an intermediate tensor is freed).
            for t in _iter_tensors(args):
                rec = producers.get(id(t))
                if rec is None:
                    continue
                prod_node, wref = rec
                if wref() is t and prod_node != handle.node_id:
                    cap.graph.add_edge(
                        prod_node, handle.node_id, kind="dataflow",
                        tensor_meta={"shape": list(t.shape)}, source="runtime",
                    )

        def post(module, args, output):
            cap = ctx.active_capture()
            if cap is None or not pending:
                return
            handle = pending.pop()
            out_shape = _shape(output)
            cap.close_span(
                handle,
                meta_update={"out_shape": out_shape} if out_shape is not None else None,
            )
            for t in _iter_tensors(output):
                try:
                    producers[id(t)] = (handle.node_id, weakref.ref(t))
                except TypeError:
                    pass  # some tensor subclasses don't support weakref

        pre_handle = register_module_forward_pre_hook(pre)
        post_handle = register_module_forward_hook(post)
        # `extra` is a live reference: per-module isolation hooks appended during
        # the run land here and get removed on exit alongside the global pair.
        return (pre_handle, post_handle, extra)

    def on_exit(self, handles) -> None:
        flat = []
        for h in handles:
            (flat.extend if isinstance(h, list) else flat.append)(h)
        for h in flat:
            try:
                h.remove()
            except Exception:
                pass


def register() -> None:
    registry.register_session_instrumentor(TorchForwardInstrumentor())
