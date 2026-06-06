"""Trace diffing — structural + shape diff between two traces.

A consumer of two IR graphs (not a new producer): given a `before` and `after`
NVGraph — two versions of a model, or the same model before/after an edit — report
what changed. The hard part is *identity*: node ids are counter-based (`Linear#3`)
and shift the instant you insert a layer, so two traces of "the same" model would
look entirely different by id. We key on a STABLE identity instead:

    qualname  (the attribute path, e.g. `layers.2.attn`)   — best
    loc       (file, line)                                  — for ops without one
    (name, parent)                                          — last resort

`diff_graphs` returns a structured diff. `annotate_diff` returns a single graph
(the `after` state, plus ghost `removed` nodes) with every node tagged
`attrs.diff ∈ {added, removed, changed, same}` so the existing renderer paints it
with no special-casing — diffing is just another set of node attrs.
"""
from __future__ import annotations

from typing import Optional

from netscope.core.ir import NVGraph

# meta fields a diff compares — structure + the metadata that actually matters
# when you tweak a model. dtype/device catch a precision/placement change even
# when shapes are identical.
_COMPARED = ("out_shape", "in_shape", "params", "dtype", "device")


def _key(node: dict) -> tuple:
    """A stable cross-graph identity for a node (see module docstring)."""
    meta = node.get("meta") or {}
    q = meta.get("qualname")
    if q:
        return ("q", q)
    loc = node.get("loc")
    if loc and loc.get("file"):
        return ("l", loc.get("file"), loc.get("line"))
    return ("n", node.get("name"), node.get("parent"))


def _occ_keyed(graph: NVGraph) -> list:
    """[(occ_key, node)] — `occ_key` disambiguates siblings that share a `_key`
    (e.g. two ops with the same name+parent and no qualname/loc). The first
    occurrence keeps the bare key; later ones get a `(key, n)` suffix. Without this
    such siblings collapse in the index and a genuine add/remove is hidden."""
    seen: dict = {}
    out = []
    for n in graph.nodes():
        k = _key(n)
        c = seen.get(k, 0)
        seen[k] = c + 1
        out.append((k if c == 0 else (k, c), n))
    return out


def _index(graph: NVGraph) -> dict:
    """occ_key -> node (see _occ_keyed)."""
    return {ok: n for ok, n in _occ_keyed(graph)}


def _summary(node: dict, key: tuple) -> dict:
    meta = node.get("meta") or {}
    return {
        "key": key,
        "qualname": meta.get("qualname"),
        "name": node.get("name"),
        "kind": node.get("kind"),
        "out_shape": meta.get("out_shape"),
        "params": meta.get("params"),
    }


def _snap(node: dict) -> dict:
    meta = node.get("meta") or {}
    return {f: meta.get(f) for f in _COMPARED}


def _changed_fields(a: dict, b: dict) -> list:
    """Which compared meta fields differ between a (before) and b (after)."""
    am, bm = a.get("meta") or {}, b.get("meta") or {}
    out = []
    for f in _COMPARED:
        if am.get(f) != bm.get(f):
            # ignore the case where both are absent/empty (None vs missing)
            if am.get(f) or bm.get(f):
                out.append(f)
    return out


def diff_graphs(before: NVGraph, after: NVGraph) -> dict:
    """Structured diff. Returns {added, removed, changed, same, summary}.

    - added/removed: nodes present on only one side (each a `_summary`).
    - changed: matched nodes whose compared meta differs, with `fields` (what
      changed), `before`/`after` snapshots.
    - same: count of matched, unchanged nodes.
    """
    bidx, aidx = _index(before), _index(after)
    bkeys, akeys = set(bidx), set(aidx)

    added = [_summary(aidx[k], k) for k in akeys - bkeys]
    removed = [_summary(bidx[k], k) for k in bkeys - akeys]
    changed, same = [], 0
    for k in akeys & bkeys:
        fields = _changed_fields(bidx[k], aidx[k])
        if fields:
            changed.append({**_summary(aidx[k], k), "fields": fields,
                            "before": _snap(bidx[k]), "after": _snap(aidx[k])})
        else:
            same += 1

    _sort = lambda lst: sorted(lst, key=lambda x: str(x["key"]))
    added, removed, changed = _sort(added), _sort(removed), _sort(changed)
    return {
        "added": added, "removed": removed, "changed": changed, "same": same,
        "summary": {"added": len(added), "removed": len(removed),
                    "changed": len(changed), "same": same},
    }


def _detail_str(changed_entry: dict) -> str:
    """Human one-liner for a changed node: `out_shape [2,16]→[2,32]; params 100→200`."""
    parts = []
    b, a = changed_entry["before"], changed_entry["after"]
    for f in changed_entry["fields"]:
        parts.append(f"{f} {b.get(f)}→{a.get(f)}")
    return "; ".join(parts)


def annotate_diff(before: NVGraph, after: NVGraph) -> NVGraph:
    """A single graph for rendering the diff: the `after` graph with each node
    tagged `attrs.diff` (added|changed|same), plus the `before`-only nodes carried
    in as `removed` ghosts (id-prefixed to avoid collisions, parent remapped to a
    surviving node where possible)."""
    d = diff_graphs(before, after)
    changed_by_key = {c["key"]: c for c in d["changed"]}
    after_occ = _occ_keyed(after)
    bidx = _index(before)
    aidx = {ok: n for ok, n in after_occ}
    after_id_by_key = {ok: n["id"] for ok, n in after_occ}

    out = NVGraph(name=after.name or before.name)

    for ok, n in after_occ:
        if ok in bidx:
            tag = "changed" if ok in changed_by_key else "same"
        else:
            tag = "added"
        attrs = dict(n.get("attrs") or {})
        attrs["diff"] = tag
        if tag == "changed":
            attrs["diff_detail"] = _detail_str(changed_by_key[ok])
        out.add_node(n["id"], kind=n["kind"], name=n["name"], parent=n.get("parent"),
                     source=n.get("source", "runtime"), loc=n.get("loc"),
                     meta=n.get("meta"), attrs=attrs)

    for e in after.edges():
        out.add_edge(e["src"], e["dst"], kind=e["kind"],
                     tensor_meta=e.get("tensor_meta"),
                     source=e.get("source", "runtime"), condition=e.get("condition"))

    # ghost the removed nodes so you can see what disappeared.
    before_key_by_id = {n["id"]: ok for ok, n in _occ_keyed(before)}
    for ok, n in _occ_keyed(before):
        if ok in aidx:
            continue
        p: Optional[str] = n.get("parent")
        new_parent = None
        if p is not None:
            # parent still in `after` -> attach there; else if the parent was ALSO
            # removed, attach to ITS ghost (don't flatten removed subtrees to root).
            new_parent = after_id_by_key.get(before_key_by_id.get(p))
            if new_parent is None and p in before_key_by_id:
                new_parent = f"removed::{p}"
        attrs = dict(n.get("attrs") or {})
        attrs["diff"] = "removed"
        out.add_node(f"removed::{n['id']}", kind=n["kind"], name=n["name"],
                     parent=new_parent, source="removed", loc=n.get("loc"),
                     meta=n.get("meta"), attrs=attrs)

    return out


def _main(argv=None) -> int:
    """CLI: `python -m netscope.core.diff before.json after.json [--html out] [--json]`.

    Diffs two saved traces (the path the extension takes); prints a summary, and
    optionally writes a colored, annotated diff graph."""
    import argparse
    import json

    ap = argparse.ArgumentParser(prog="netscope.diff",
                                 description="Diff two netscope traces (JSON dumps).")
    ap.add_argument("before"); ap.add_argument("after")
    ap.add_argument("--html", help="write the annotated, colored diff graph here")
    ap.add_argument("--graph-json", help="write the annotated diff graph as IR JSON "
                    "(diff-tagged nodes) — what the extension loads into its webview")
    ap.add_argument("--json", action="store_true", help="emit the structured diff as JSON")
    a = ap.parse_args(argv)

    with open(a.before, encoding="utf-8") as f:
        before = NVGraph.from_dict(json.load(f))
    with open(a.after, encoding="utf-8") as f:
        after = NVGraph.from_dict(json.load(f))

    d = diff_graphs(before, after)
    if a.json:
        print(json.dumps(d, indent=2, default=str))
    else:
        s = d["summary"]
        print(f"diff: +{s['added']} added  -{s['removed']} removed  "
              f"~{s['changed']} changed  ={s['same']} same")
        for x in d["added"]:
            print(f"  + {x['qualname'] or x['name']}  out={x.get('out_shape')}")
        for x in d["removed"]:
            print(f"  - {x['qualname'] or x['name']}  out={x.get('out_shape')}")
        for c in d["changed"]:
            print(f"  ~ {c['qualname'] or c['name']}  {_detail_str(c)}")
    if a.graph_json:
        with open(a.graph_json, "w", encoding="utf-8") as f:
            f.write(annotate_diff(before, after).to_json())
        print(f"diff graph json -> {a.graph_json}")
    if a.html:
        annotate_diff(before, after).show(path=a.html, open_browser=False)
        print(f"diff graph -> {a.html}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
