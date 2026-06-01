"""Declared-dim static pre-check — catch a wiring clash before any run.

Reads literal layer dims from `__init__` (`nn.Linear(in, out)`,
`nn.Conv2d(cin, cout, ...)`) and the `forward` call order, emits module nodes
carrying the DECLARED shapes (`[1, feat]`, batch sentinel so the existing
mismatch detector — which ignores axis 0 — can compare feature dims) plus
`dataflow` edges between consecutively-wired layers. `detect_mismatches` then
flags an obvious clash for free, through the same `warnings` channel the editor
squiggles already read.

Conservative by design: a layer is only checkable when its relevant dim is a
literal int and the wiring is a direct `self.b(self.a(x))` / sequential chain.
Anything ambiguous is simply not wired, so it is never falsely flagged.
"""
from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

# constructor -> ((in positional index, in kwarg name), (out index, out kwarg)).
# Linear(in_features, out_features); Conv2d(in_channels, out_channels, kernel, ...).
_DIMMED = {
    "Linear": ((0, "in_features"), (1, "out_features")),
    "Conv1d": ((0, "in_channels"), (1, "out_channels")),
    "Conv2d": ((0, "in_channels"), (1, "out_channels")),
    "Conv3d": ((0, "in_channels"), (1, "out_channels")),
}


def _arg_dim(call: ast.Call, pos: int, kw: str) -> Optional[int]:
    """A literal int from positional index `pos` or keyword `kw`, else None.
    Covers both `nn.Linear(64, 256)` and `nn.Linear(in_features=64, ...)`."""
    if len(call.args) > pos:
        v = _literal_int(call.args[pos])
        if v is not None:
            return v
    for kwarg in call.keywords:
        if kwarg.arg == kw:
            return _literal_int(kwarg.value)
    return None


def _callee(call: ast.Call) -> Optional[str]:
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def _literal_int(arg: ast.expr) -> Optional[int]:
    if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
        return arg.value
    return None


class _Layer:
    __slots__ = ("name", "line", "in_dim", "out_dim", "kind")

    def __init__(self, name, line, in_dim, out_dim, kind):
        self.name = name
        self.line = line
        self.in_dim = in_dim       # int or None
        self.out_dim = out_dim     # int or None
        self.kind = kind           # constructor name, e.g. "Linear"


def _find_init(cls: ast.ClassDef) -> Optional[ast.FunctionDef]:
    for item in cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            return item
    return None


def _find_forward(cls: ast.ClassDef) -> Optional[ast.FunctionDef]:
    for item in cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == "forward":
            return item
    return None


_CONTAINERS = {"Sequential", "ModuleList"}


def _layer_from_call(name: str, call: ast.Call, line: int) -> Optional["_Layer"]:
    """A _Layer for a dim-carrying constructor call, or None."""
    ctor = _callee(call)
    if ctor not in _DIMMED:
        return None
    (in_i, in_kw), (out_i, out_kw) = _DIMMED[ctor]
    return _Layer(name, line, _arg_dim(call, in_i, in_kw),
                  _arg_dim(call, out_i, out_kw), ctor)


def _collect_layers(init: ast.FunctionDef) -> Dict[str, _Layer]:
    """Map attr-name -> _Layer for each `self.<name> = nn.Linear/Conv*(...)`, and
    each dim-carrying positional child of `self.<name> = nn.Sequential(a, b, ...)`
    / `nn.ModuleList([...])` named by index (`name.0`, `name.1`) to mirror how
    named_modules() names them."""
    layers: Dict[str, _Layer] = {}
    for stmt in ast.walk(init):
        if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
            continue
        if len(stmt.targets) != 1:
            continue
        tgt = stmt.targets[0]
        if (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "self"
                and _callee(stmt.value) in _CONTAINERS):
            # index-named children: Sequential(child0, child1, ...) or
            # ModuleList([child0, ...]). Only dim-carrying children become layers.
            call = stmt.value
            children = list(call.args)
            if len(children) == 1 and isinstance(children[0], (ast.List, ast.Tuple)):
                children = list(children[0].elts)
            for i, child in enumerate(children):
                if isinstance(child, ast.Call):
                    lyr = _layer_from_call(f"{tgt.attr}.{i}", child, child.lineno)
                    if lyr is not None:
                        layers[f"{tgt.attr}.{i}"] = lyr
            continue
        if not (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "self"):
            continue
        call = stmt.value
        ctor = _callee(call)
        if ctor not in _DIMMED:
            continue
        (in_i, in_kw), (out_i, out_kw) = _DIMMED[ctor]
        in_dim = _arg_dim(call, in_i, in_kw)        # positional OR keyword
        out_dim = _arg_dim(call, out_i, out_kw)
        layers[tgt.attr] = _Layer(tgt.attr, stmt.lineno, in_dim, out_dim, ctor)
    return layers


def _self_attr_of_call(node: ast.Call, layers: Dict[str, _Layer]) -> Optional[str]:
    """If `node` is `self.<attr>(...)` for a known layer attr, return the attr."""
    f = node.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self":
        return f.attr if f.attr in layers else None
    return None


def _producer_of_arg(arg, layers, var2layer) -> Optional[str]:
    """Which layer produced `arg`? Either a direct `self.a(...)` call (nesting) or
    a variable previously bound to a layer's output (intermediate-var wiring)."""
    if isinstance(arg, ast.Call):
        return _self_attr_of_call(arg, layers)
    if isinstance(arg, ast.Name):
        return var2layer.get(arg.id)
    return None


def _wiring_pairs(forward: ast.FunctionDef, layers: Dict[str, _Layer]) -> List[Tuple[str, str]]:
    """Producer->consumer attr-name pairs from the forward body. Handles both
    direct nesting `self.b(self.a(x))` AND the common intermediate-variable style
    `h = self.a(x); self.b(h)` (and reassignment `x = self.a(x); x = self.b(x)`),
    by tracking which layer's output each local variable last held.

    Conservative + linear: walks top-level statements in order (good enough for
    typical straight-line forward()s); branches/loops are not flow-analyzed."""
    pairs: List[Tuple[str, str]] = []
    var2layer: Dict[str, str] = {}   # local var name -> the layer attr that produced it

    def record_consumers(call: ast.Call) -> None:
        consumer = _self_attr_of_call(call, layers)
        if consumer is not None and call.args:
            prod = _producer_of_arg(call.args[0], layers, var2layer)
            if prod is not None and prod != consumer:
                pairs.append((prod, consumer))

    for stmt in forward.body:
        # find every self.layer(...) call inside this statement (for the edge)
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                record_consumers(node)
        # then update var bindings: `name = <expr producing a layer output>`
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            var = stmt.targets[0].id
            produced = _outermost_layer(stmt.value, layers, var2layer)
            if produced is not None:
                var2layer[var] = produced
            else:
                var2layer.pop(var, None)   # rebound to a non-layer value
    return pairs


def _outermost_layer(expr, layers, var2layer) -> Optional[str]:
    """The layer attr whose output this expression evaluates to: `self.a(x)` ->
    'a'; `self.a(self.b(x))` -> 'a' (outermost); a bare var -> its bound layer."""
    if isinstance(expr, ast.Call):
        direct = _self_attr_of_call(expr, layers)
        if direct is not None:
            return direct
    if isinstance(expr, ast.Name):
        return var2layer.get(expr.id)
    return None


def add_declared_dims(graph, source: str, filename: str) -> None:
    """Augment `graph` (an NVGraph) with declared-dim module nodes + wiring edges
    parsed from `source`, so detect_mismatches can flag a static clash. No-op on
    parse failure (best-effort, never raises into the caller)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return

    counter = 0
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        init = _find_init(cls)
        forward = _find_forward(cls)
        if init is None or forward is None:
            continue
        layers = _collect_layers(init)
        if not layers:
            continue
        # one node per layer, declared shapes as [1, feat] (batch sentinel)
        ids: Dict[str, str] = {}
        for name, lyr in layers.items():
            counter += 1
            nid = f"dim#{counter}"
            ids[name] = nid
            meta: dict = {"qualname": name}
            if lyr.in_dim is not None:
                meta["in_shape"] = [1, lyr.in_dim]
            if lyr.out_dim is not None:
                meta["out_shape"] = [1, lyr.out_dim]
            graph.add_node(
                nid, kind="module", name=lyr.kind, source="static",
                loc={"file": filename, "line": lyr.line}, meta=meta,
                # mark these as declared-dim nodes: they exist for the static
                # pre-check only and are REDUNDANT with runtime module nodes.
                # merge() drops any that don't loc-match a runtime node, so they
                # never float free in a real trace (e.g. a layer in an unused
                # fallback class that never executed).
                attrs={"declared_dim": True},
            )
        # wire producer->consumer edges from the forward call order
        for producer, consumer in _wiring_pairs(forward, layers):
            if producer in ids and consumer in ids:
                graph.add_edge(ids[producer], ids[consumer],
                               kind="dataflow", source="static")
        # Sequential children chain implicitly (a.0 -> a.1 -> a.2 ...) — wire
        # consecutive index-named siblings so a clash inside a Sequential is caught.
        by_container: Dict[str, list] = {}
        for name in layers:
            if "." in name:
                container, _, idx = name.rpartition(".")
                if idx.isdigit():
                    by_container.setdefault(container, []).append((int(idx), name))
        for siblings in by_container.values():
            siblings.sort()
            for (_, prev), (_, nxt) in zip(siblings, siblings[1:]):
                if prev in ids and nxt in ids:
                    graph.add_edge(ids[prev], ids[nxt], kind="dataflow", source="static")
