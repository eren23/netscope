"""torch.fx fallback producer — real structure from a model INSTANCE, no forward.

The source-AST producer gives ~1 node for models built via `from_config` /
factory functions (no literal layers in user source). When we hold the model
object, `torch.fx.symbolic_trace` recovers the real module graph + dataflow
*without executing a forward* — for the models fx can trace.

fx fails on dynamic control flow (data-dependent branches/loops — many LLMs,
`nn.TransformerEncoderLayer`). This is therefore BEST-EFFORT: `trace_model`
returns an `NVGraph` on success and `None` on any failure, so the caller falls
back to runtime tracing (which handles everything).
"""
import inspect
from typing import Optional

from netscope.core.ir import NVGraph


def _qual_locs(model) -> dict:
    """qualname -> {file, line} for the model's submodules, reusing the same
    static scan the runtime tracer uses for click-to-source. Best-effort."""
    try:
        from netscope.static.module_loc import qualname_locs_for_file

        src = inspect.getsourcefile(type(model))
        return qualname_locs_for_file(src) if src else {}
    except Exception:
        return {}


def trace_model(model) -> Optional[NVGraph]:
    """Symbolically trace `model` with torch.fx and build an IR graph of its
    modules + dataflow. Returns None if fx can't trace it (caller falls back)."""
    try:
        import torch.fx as fx
    except Exception:
        return None
    try:
        gm = fx.symbolic_trace(model)
    except Exception:
        return None   # dynamic control flow etc. -> let the caller use runtime

    try:
        return _build(gm, model)
    except Exception:
        return None


def _build(gm, model) -> NVGraph:
    g = NVGraph(name=type(model).__name__)
    locs = _qual_locs(model)
    submodules = dict(model.named_modules())

    # fx node -> our node id, for the module nodes we keep (call_module only).
    fxnode_to_id = {}
    counter = 0
    for fxn in gm.graph.nodes:
        if fxn.op != "call_module":
            continue
        target = fxn.target               # the submodule qualname, e.g. "layer1.0.conv1"
        sub = submodules.get(target)
        cls = type(sub).__name__ if sub is not None else str(target)
        counter += 1
        nid = f"fx#{counter}"
        fxnode_to_id[fxn] = nid
        meta = {"qualname": target}
        try:
            from netscope.enrich.params import own_params

            if sub is not None:
                p = own_params(sub)
                if p:
                    meta["params"] = p
        except Exception:
            pass
        g.add_node(nid, kind="module", name=cls, source="static",
                   loc=locs.get(target), meta=meta, attrs={"fx": True})

    # dataflow: an edge prod->cons when cons consumes prod's output. Walk each
    # call_module's input fx-nodes back to the nearest producing call_module.
    for fxn, cons_id in fxnode_to_id.items():
        for inp in fxn.all_input_nodes:
            prod_id = _nearest_module_producer(inp, fxnode_to_id)
            if prod_id is not None and prod_id != cons_id:
                g.add_edge(prod_id, cons_id, kind="dataflow", source="static")
    return g


def _nearest_module_producer(fxn, fxnode_to_id, _seen=None) -> Optional[str]:
    """Walk back through non-module fx ops (functional calls, getattr, etc.) to
    the nearest call_module that produced this value."""
    if _seen is None:
        _seen = set()
    if id(fxn) in _seen:
        return None
    _seen.add(id(fxn))
    if fxn in fxnode_to_id:
        return fxnode_to_id[fxn]
    for inp in getattr(fxn, "all_input_nodes", []):
        r = _nearest_module_producer(inp, fxnode_to_id, _seen)
        if r is not None:
            return r
    return None
