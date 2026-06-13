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
import inspect
import os
import time
import weakref
from typing import Iterator, Optional

from netscope.core import context as ctx
from netscope.core import registry
from netscope.enrich.params import own_param_bytes, own_params


def _root_qualname_locs(root) -> dict:
    """Build {qualname: {file, line}} for `root`'s submodules by statically
    scanning the source file that defines root's class. Best-effort: returns {}
    on any failure (no source, C-defined, parse error) so the tracer falls back
    to loc=None. Reused by click-to-source, inline shape hints, and squiggles.

    The root module itself has qualname "" (it's no submodule of anything), so it
    has no `self.x = ...` line; we map "" to the root's CLASS DEFINITION line, so a
    directly-called custom module (Encoder(), ClassifierHead()) still gets a loc.
    """
    try:
        from netscope.static.module_loc import qualname_locs_for_file

        src_file = inspect.getsourcefile(type(root))
        if not src_file:
            return {}
        locs = qualname_locs_for_file(src_file)
        if "" not in locs:
            try:
                _, line = inspect.getsourcelines(type(root))
                locs[""] = {"file": src_file, "line": line}
            except (OSError, TypeError):
                pass
        return locs
    except Exception:
        return {}


def _is_tensor(x) -> bool:
    try:
        import torch

        return isinstance(x, torch.Tensor)
    except Exception:
        return False


def _shape(x) -> Optional[list]:
    return list(x.shape) if _is_tensor(x) else None


def _dtype(x) -> Optional[str]:
    """A short dtype name ('float32', 'float16', ...) or None for non-tensors."""
    if not _is_tensor(x):
        return None
    try:
        return str(x.dtype).split(".")[-1]   # 'torch.float32' -> 'float32'
    except Exception:
        return None


def _device(x) -> Optional[str]:
    """The device a tensor lives on ('cpu', 'cuda:0', ...) or None."""
    if not _is_tensor(x):
        return None
    try:
        return str(x.device)
    except Exception:
        return None


def _act_bytes(x) -> Optional[int]:
    """Activation memory of a tensor = elements × bytes-per-element. Free: the
    output tensor is already in hand in the post-hook, so this is exact and adds
    no overhead (no extra allocation, just two cheap property reads)."""
    if not _is_tensor(x):
        return None
    try:
        return int(x.numel() * x.element_size())
    except Exception:
        return None


def _first_tensor(obj):
    """The first tensor reachable in obj (descending dict/tuple/list), or None.
    Used to read dtype/device off a module's representative input/output."""
    for t in _iter_tensors(obj):
        return t
    return None


def _iter_tensors(obj, _depth: int = 0) -> Iterator:
    """Yield every tensor reachable in obj, descending tuple/list/dict (bounded
    depth). HuggingFace modules return dicts / ModelOutput (a Mapping) and nested
    tuples, so a one-level scan dropped their producer edges — walk the structure.
    """
    if _depth > 6:
        return
    if _is_tensor(obj):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _iter_tensors(item, _depth + 1)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _iter_tensors(item, _depth + 1)
    else:
        # ModelOutput / dataclass-like: expose tensors via .values() or vars().
        # Guard tightly so we never iterate something huge or stateful.
        values = getattr(obj, "values", None)
        if callable(values) and not _is_tensor(obj):
            try:
                items = list(values())
            except Exception:
                items = None
            if items is not None:
                for item in items:
                    yield from _iter_tensors(item, _depth + 1)


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

    return module.register_forward_pre_hook(grab, with_kwargs=True)


class TorchForwardInstrumentor:
    """Adds torch global forward hooks while a capture session is open."""

    def on_enter(self):
        from torch.nn.modules.module import (
            register_module_forward_hook,
            register_module_forward_pre_hook,
        )

        pending: list = []      # stack of SpanHandle (sync, nested execution)
        starts: list = []       # parallel stack of perf_counter() starts (profile)
        producers: dict = {}    # id(tensor) -> (node_id, weakref(tensor))
        id2name: dict = {}      # id(submodule) -> qualified name, per top-level fwd
        qual2loc: dict = {}     # qualname -> {file, line}, per top-level fwd
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
                # ...and map each qualname -> the source line it's constructed on,
                # so every node gets a `loc` (click-to-source + inline hints +
                # mismatch squiggles all key off this). Best-effort, static-only.
                qual2loc.clear()
                qual2loc.update(_root_qualname_locs(module))
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
            meta: dict[str, object] = {"params": own_params(module)}
            pbytes = own_param_bytes(module)
            if pbytes:
                meta["param_bytes"] = pbytes   # free: count × dtype size, for the memory overlay
            in_t = _first_tensor(args)
            in_shape = _shape(in_t)
            if in_shape is not None:
                meta["in_shape"] = in_shape
            # dtype/device read off the representative input tensor — lets the user
            # see float16 vs float32 and cpu vs cuda placement (a real debugging
            # need on mixed-precision / multi-GPU models).
            dt, dev = _dtype(in_t), _device(in_t)
            if dt is not None:
                meta["dtype"] = dt
            if dev is not None:
                meta["device"] = dev
            loc = None
            # NB: a ROOT module's qualname is "" (it's nobody's submodule) — that's
            # a valid name, not "missing", so test `is not None` rather than truth.
            # Don't write an empty qualname into meta (it carries no info), but DO
            # resolve its loc (the root's class-def line) so it's still clickable.
            if qualname is not None:
                if qualname:
                    meta["qualname"] = qualname
                loc = qual2loc.get(qualname)
            handle = cap.open_span(type(module).__name__, kind="module", meta=meta, loc=loc)
            pending.append(handle)
            # timing (opt-in): push a start NOW, last thing before forward runs, so
            # the paired post-hook measures (this module + its children) wall-time.
            # Kept in lockstep with `pending` (pushed/popped on the same paths).
            starts.append(time.perf_counter() if getattr(cap, "profile", False) else None)
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
            elapsed_end = time.perf_counter()   # grab first, before our own bookkeeping
            cap = ctx.active_capture()
            if cap is None or not pending:
                return
            handle = pending.pop()
            t0 = starts.pop() if starts else None
            # out_shape from the output if it's a bare tensor; for a tuple/dict
            # output (MultiheadAttention, HF ModelOutput) fall back to the first
            # reachable tensor so the node still shows a representative shape.
            out_t = output if _is_tensor(output) else _first_tensor(output)
            out_shape = _shape(out_t)
            update: dict[str, object] = {}
            if out_shape is not None:
                update["out_shape"] = out_shape
            ab = _act_bytes(out_t)
            if ab is not None:
                update["act_bytes"] = ab           # free: output activation memory
            if t0 is not None:
                update["time_ms"] = round((elapsed_end - t0) * 1000, 4)
            # fill dtype/device from the output if the input didn't provide them.
            n = cap.graph.get_node(handle.node_id)
            cur_meta = n.get("meta") or {}
            if "dtype" not in cur_meta and _dtype(out_t) is not None:
                update["dtype"] = _dtype(out_t)
            if "device" not in cur_meta and _device(out_t) is not None:
                update["device"] = _device(out_t)
            cap.close_span(handle, meta_update=update or None)
            for t in _iter_tensors(output):
                try:
                    producers[id(t)] = (handle.node_id, weakref.ref(t))
                except TypeError:
                    pass  # some tensor subclasses don't support weakref

        pre_handle = register_module_forward_pre_hook(pre)
        # always_call=True so the post-hook fires even when a forward RAISES — then
        # `pending`/`starts`/the parent stack unwind cleanly instead of leaking a
        # half-open span that would mis-parent and mislabel the next forward.
        post_handle = register_module_forward_hook(post, always_call=True)
        # `extra` is a live reference: per-module isolation hooks appended during
        # the run land here and get removed on exit alongside the global pair.
        return (pre_handle, post_handle, extra)

    def on_exit(self, handles) -> None:
        flat: list = []
        for h in handles:
            (flat.extend if isinstance(h, list) else flat.append)(h)
        for h in flat:
            try:
                h.remove()
            except Exception as e:
                # a hook that won't remove would fire on the NEXT session with a
                # dead capture -> warn (don't silently leak it). Keep going so one
                # bad handle doesn't strand the rest.
                import warnings

                warnings.warn(f"netscope: failed to remove a torch hook: {e}",
                              RuntimeWarning, stacklevel=2)


def register() -> None:
    registry.register_session_instrumentor(TorchForwardInstrumentor())
