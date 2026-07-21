"""Self-healing shapes — deterministically fix a declared-dim shape clash.

A static `shape_mismatch` (the declared-dim pre-check in `static/dims.py`) has a
mechanical fix: the consumer declares an in-dim (`nn.Linear(128, …)`) but its
producer emits a different one (256). Reconcile by changing the consumer's in-dim
literal to what actually arrives, at the consumer's own source line — located
precisely via the AST, so we edit the in-features arg and nothing else.

Deterministic and offline (no LLM): the fix for a clean dim clash is unambiguous.
`propose_fixes` is read-only (dry-run); `apply_fixes` writes. Ambiguous or
non-literal cases are simply not proposed — never a wrong edit.

    from netscope.autofix import propose_fixes, apply_fixes
    from netscope.static.ast_producer import analyze_file
    fixes = propose_fixes(analyze_file("model.py"))   # [{file, line, old, new, ...}]
    apply_fixes(fixes)                                 # write them, then re-analyze
"""
from __future__ import annotations

import ast
from typing import List, Optional

from netscope.core.checks import detect_mismatches
from netscope.static.dims import _DIMMED, _callee


def _feature_dims(consumer_meta: dict, producer_meta: dict) -> Optional[tuple]:
    """(consumer's declared in-dim, producer's emitted out-dim) for a feature clash,
    or None. Both are the last-axis (feature) dim of the static [1, feat] shapes."""
    ci = (consumer_meta.get("in_shape") or [None])[-1]
    po = (producer_meta.get("out_shape") or [None])[-1]
    if ci is None or po is None or ci == po:
        return None
    return ci, po


def _in_arg_node(call: ast.Call, pos: int, kw: str):
    if len(call.args) > pos:
        return call.args[pos]
    for k in call.keywords:
        if k.arg == kw:
            return k.value
    return None


def _in_dim_patch(file: str, line: int, cur: int, need: int) -> Optional[dict]:
    """Rewrite the in-features literal `cur` -> `need` in the dim-carrying ctor at
    `line`. Uses AST column offsets so only that one argument changes. None if the
    call/arg can't be pinned unambiguously (then we simply propose nothing)."""
    try:
        with open(file, encoding="utf-8") as f:
            src_lines = f.read().splitlines()
    except OSError:
        return None
    try:
        tree = ast.parse("\n".join(src_lines))
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or node.lineno != line:
            continue
        ctor = _callee(node)
        if ctor not in _DIMMED:
            continue
        (in_i, in_kw), _ = _DIMMED[ctor]
        arg = _in_arg_node(node, in_i, in_kw)
        if not (isinstance(arg, ast.Constant) and arg.value == cur):
            continue
        ln = arg.lineno
        if not (1 <= ln <= len(src_lines)):
            return None
        old = src_lines[ln - 1]
        new = old[: arg.col_offset] + str(need) + old[arg.end_col_offset:]
        if new == old:
            return None
        return {"file": file, "line": ln, "old": old, "new": new}
    return None


def propose_fixes(graph) -> List[dict]:
    """Dry-run: structured edits that reconcile each static shape_mismatch by
    changing the consumer's declared in-dim to the producer's out-dim. Read-only —
    returns `[{file, line, old, new, qualname, detail}]`."""
    nodes = {n["id"]: n for n in graph.nodes()}
    fixes: List[dict] = []
    for w in detect_mismatches(graph):
        if w.get("kind") != "shape_mismatch":
            continue      # rank_mismatch ("add a flatten()") isn't a clean literal swap
        consumer = nodes.get(w["dst"])
        producer = nodes.get(w["src"])
        if not consumer or not producer:
            continue
        loc = consumer.get("loc") or {}
        meta = consumer.get("meta") or {}
        dims = _feature_dims(meta, producer.get("meta") or {})
        if not (loc.get("file") and loc.get("line") and dims):
            continue
        patch = _in_dim_patch(loc["file"], loc["line"], dims[0], dims[1])
        if patch:
            patch.update(qualname=meta.get("qualname"), detail=w.get("detail"))
            fixes.append(patch)
    return fixes


def apply_fixes(fixes: List[dict]) -> int:
    """Write the proposed edits back to disk (one line replaced per fix, original
    line endings preserved). Returns the number applied."""
    by_file: dict = {}
    for fx in fixes:
        by_file.setdefault(fx["file"], {})[fx["line"]] = fx["new"]
    applied = 0
    for file, edits in by_file.items():
        with open(file, encoding="utf-8") as f:
            lines = f.readlines()
        for ln, new in edits.items():
            if 1 <= ln <= len(lines):
                eol = "\n" if lines[ln - 1].endswith("\n") else ""
                lines[ln - 1] = new + eol
                applied += 1
        with open(file, "w", encoding="utf-8") as f:
            f.writelines(lines)
    return applied


def _main(argv=None) -> int:
    """`netscope fix <file.py> [--apply]` — show (or, with --apply, write) the
    dim-clash fixes. Dry-run by default; it edits your source, so you opt in."""
    import sys

    from netscope.static.ast_producer import analyze_file

    argv = list(sys.argv[1:] if argv is None else argv)
    do_apply = "--apply" in argv
    paths = [a for a in argv if not a.startswith("-")]
    if not paths:
        print("usage: netscope fix <file.py> [--apply]", file=sys.stderr)
        return 2
    path = paths[0]
    try:
        fixes = propose_fixes(analyze_file(path))
    except Exception as e:                       # parse/IO — surface cleanly
        print(f"netscope fix error: {e}", file=sys.stderr)
        return 1
    if not fixes:
        print("no fixable shape mismatches found.")
        return 0
    for fx in fixes:
        print(f"{fx['file']}:{fx['line']}  ({fx.get('qualname')})")
        print(f"  - {fx['old'].strip()}")
        print(f"  + {fx['new'].strip()}")
    if not do_apply:
        print(f"\n{len(fixes)} fix(es) proposed — re-run with --apply to write them.")
        return 0
    n = apply_fixes(fixes)
    remaining = detect_mismatches(analyze_file(path))
    print(f"\napplied {n} fix(es); {len(remaining)} mismatch(es) remain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
