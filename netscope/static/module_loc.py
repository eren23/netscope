"""Map a module's qualified name -> the source line where it was constructed.

The torch tracer tags every node with `meta.qualname` (the `named_modules()`
name, e.g. "encoder" or "body.1"). This recovers a `{file, line}` for that name
by statically scanning the model's defining file for the `self.<name> = <Call>`
assignments in `__init__` — and, for container constructors like
`nn.Sequential(child, child, ...)`, the positional children that `named_modules()`
names by index ("body.0", "body.1", ...).

Pure AST, no execution. Best-effort: anything it can't resolve simply isn't in
the map, and the tracer falls back to `loc=None` (today's behavior).
"""
import ast
from typing import Dict, Optional

# container constructors whose positional Module args are named by index
_INDEXED_CONTAINERS = {"Sequential", "ModuleList"}


def _callee_name(call: ast.Call) -> Optional[str]:
    """The bare constructor name: nn.Linear(...) -> 'Linear', Foo(...) -> 'Foo'."""
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def _find_init(classdef: ast.ClassDef) -> Optional[ast.FunctionDef]:
    for item in classdef.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            return item
    return None


def _self_attr_target(node: ast.Assign) -> Optional[str]:
    """For `self.<name> = ...`, return '<name>' (single target only)."""
    if len(node.targets) != 1:
        return None
    tgt = node.targets[0]
    if (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
            and tgt.value.id == "self"):
        return tgt.attr
    return None


def qualname_locs(source: str, filename: str = "<unknown>") -> Dict[str, dict]:
    """Return {qualname: {"file": filename, "line": lineno}} for every
    `self.<name> = <ModuleConstructor>(...)` found in the file's classes,
    including index-named children of Sequential/ModuleList."""
    out: Dict[str, dict] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out

    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        init = _find_init(cls)
        if init is None:
            continue
        for stmt in ast.walk(init):
            if not isinstance(stmt, ast.Assign):
                continue
            name = _self_attr_target(stmt)
            if name is None or not isinstance(stmt.value, ast.Call):
                continue
            call = stmt.value
            out[name] = {"file": filename, "line": stmt.lineno}
            # container children named by position: Sequential(a, b, c) -> name.0..
            if _callee_name(call) in _INDEXED_CONTAINERS:
                for i, arg in enumerate(call.args):
                    if isinstance(arg, ast.Call):
                        out[f"{name}.{i}"] = {"file": filename, "line": arg.lineno}
    return out


def qualname_locs_for_file(path: str) -> Dict[str, dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return qualname_locs(f.read(), filename=path)
    except (OSError, UnicodeDecodeError):
        return {}
