"""Mismatch detection over the IR — the "show errors" layer.

Scans dataflow edges for shape incompatibilities between a producer's output and
a consumer's expected input. Returns structured warnings; the renderer paints
the offending edges/nodes and lists the warnings. Conservative by design: if a
shape is missing or ambiguous, we say nothing (no false alarms).
"""
from __future__ import annotations

from typing import Optional


def _shape(meta_shape) -> Optional[list]:
    """A usable shape (list of ints), or None."""
    if not meta_shape:
        return None
    try:
        return [int(x) for x in meta_shape]
    except (TypeError, ValueError):
        return None


def _label(node) -> str:
    """How to name a node in a warning. Prefer the qualified attribute name
    (`backbone`, `layers.2.attn`) when present — it disambiguates two layers of
    the same class (two `Linear`s) — else the class name."""
    q = (node.get("meta") or {}).get("qualname")
    return q or node["name"]


def _check_edge(a, b, src, dst) -> Optional[dict]:
    a_shape = _shape((a.get("meta") or {}).get("out_shape"))
    b_shape = _shape((b.get("meta") or {}).get("in_shape"))
    if a_shape is None or b_shape is None:
        return None

    a_name, b_name = _label(a), _label(b)
    base = {"src": src, "dst": dst, "severity": "error"}

    # 1) rank mismatch first — e.g. Conv2d (N,C,H,W) into a Linear (N,F). The
    #    feature-dim check would give a confusing number; the real fix is a
    #    flatten/reshape, so say that.
    if len(a_shape) != len(b_shape):
        return {
            **base, "kind": "rank_mismatch",
            "detail": (
                f"{a_name} emits a {len(a_shape)}-D tensor but "
                f"{b_name} expects {len(b_shape)}-D — missing a flatten()/reshape()?"
            ),
        }

    # 2) same rank, differing NON-BATCH dims (everything after axis 0). This
    #    catches a Linear in!=out (1-D feature) AND a Conv channel clash (the
    #    differing axis isn't always the last one). Batch (axis 0) is ignored.
    a_feat, b_feat = a_shape[1:], b_shape[1:]
    if a_feat != b_feat:
        # report the single differing dim if there's exactly one, else the shapes
        diffs = [i for i in range(len(a_feat)) if a_feat[i] != b_feat[i]]
        if len(diffs) == 1:
            i = diffs[0]
            detail = (f"{a_name} emits dim {a_feat[i]} but "
                      f"{b_name} expects {b_feat[i]} (axis {i + 1})")
        else:
            detail = (f"{a_name} emits non-batch shape {a_feat} but "
                      f"{b_name} expects {b_feat}")
        return {**base, "kind": "shape_mismatch", "detail": detail}
    return None


def detect_mismatches(graph) -> list:
    warnings = []
    nodes = {n["id"]: n for n in graph.nodes()}
    for e in graph.edges():
        if e["kind"] != "dataflow":
            continue
        a = nodes.get(e["src"])
        b = nodes.get(e["dst"])
        if not a or not b:
            continue
        w = _check_edge(a, b, e["src"], e["dst"])
        if w is not None:
            warnings.append(w)
    return warnings
