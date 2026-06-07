"""Static AST producer — graph structure WITHOUT running the code.

Parses a source file and recovers semantics that runtime tracing is blind to:

* ``for _ in range(N):``        -> a stage node with ``attrs.repeat = N`` (the
                                   branch fan-out, e.g. sfumato's 5 branches)
* ``Counter(...).most_common``  -> a ``reduce`` stage node (the majority vote)
* ``@nv.stage("name")`` / ``@stage("name")`` on a def -> a named stage node

Every node carries ``source="static"`` and its ``loc`` so ``core.merge`` can fuse
it onto the runtime graph by source location. Output is a plain ``NVGraph``.
"""
import ast
from typing import Optional

from netscope.core.ir import NVGraph


def _is_range_loop(node: ast.For) -> bool:
    call = node.iter
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "range"
        and bool(call.args)
    )


def _range_count(node: ast.For) -> Optional[int]:
    """The branch count IF it is a literal int, else None (e.g. range(n_branches))."""
    if not _is_range_loop(node) or not isinstance(node.iter, ast.Call):
        return None
    last = node.iter.args[-1]
    if isinstance(last, ast.Constant) and isinstance(last.value, int):
        return last.value
    return None


def _counter_var_names(tree: ast.AST) -> set:
    """Names bound from `x = Counter(...)` anywhere in the tree, so a later
    `x.most_common(...)` is recognised as a vote (the real sfumato form)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            f = node.value.func
            fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if fname == "Counter":
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        names.add(tgt.id)
    return names


def _is_most_common(call: ast.Call, counter_vars: set) -> bool:
    # match  <expr>.most_common(...)  where <expr> is an inline Counter(...) chain
    # OR a variable previously bound to a Counter(...).
    if not (isinstance(call.func, ast.Attribute) and call.func.attr == "most_common"):
        return False
    recv = call.func.value
    if isinstance(recv, ast.Name) and recv.id in counter_vars:
        return True
    return "Counter" in ast.dump(recv)


def _stage_name_from_decorator(dec: ast.expr) -> Optional[str]:
    # match  stage("x")  or  nv.stage("x")  (also branch/reduce)
    if isinstance(dec, ast.Call):
        f = dec.func
        fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if fname in ("stage", "branch", "reduce") and dec.args:
            a0 = dec.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                return a0.value
    return None


class _Visitor(ast.NodeVisitor):
    def __init__(self, g: NVGraph, filename: str, counter_vars: set):
        self.g = g
        self.filename = filename
        self._counter_vars = counter_vars
        self._n = 0

    def _add(self, name: str, *, kind: str, line: int, attrs: dict) -> None:
        self._n += 1
        self.g.add_node(
            f"static#{self._n}", kind=kind, name=name, source="static",
            loc={"file": self.filename, "line": line}, attrs=attrs,
        )

    def visit_For(self, node: ast.For):
        if _is_range_loop(node):
            count = _range_count(node)              # None when range(variable)
            attrs: dict[str, object] = {"branch": True}
            label = "branch loop"
            if count is not None:
                attrs["repeat"] = count
                label = f"loop x{count}"
            self._add(label, kind="stage", line=node.lineno, attrs=attrs)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if _is_most_common(node, self._counter_vars):
            self._add("vote", kind="stage", line=node.lineno, attrs={"reduce": True})
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for dec in node.decorator_list:
            sname = _stage_name_from_decorator(dec)
            if sname is not None:
                self._add(sname, kind="stage", line=node.lineno, attrs={"declared": True})
                break
        self.generic_visit(node)


def analyze_source(source: str, filename: str = "<unknown>") -> NVGraph:
    g = NVGraph(name=filename)
    tree = ast.parse(source)
    counter_vars = _counter_var_names(tree)
    _Visitor(g, filename, counter_vars).visit(tree)
    # declared-dim pre-check: emit module nodes carrying literal layer dims +
    # wiring edges, so detect_mismatches flags a clash BEFORE any run.
    from netscope.static.dims import add_declared_dims

    add_declared_dims(g, source, filename)
    return g


def analyze_file(path: str) -> NVGraph:
    with open(path, encoding="utf-8") as f:
        return analyze_source(f.read(), filename=path)
