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


def _feature_axis(rank: int) -> int:
    """The axis carrying the channel/feature dim, by PyTorch layout convention.

    4-D NCHW → channels live at axis 1 (H/W are spatial, change freely under
    stride/pool). Everything else (2-D NC, 3-D NSC) → the LAST axis is the
    feature/embedding dim; axis 1 of a 3-D tensor is the SEQUENCE length, which
    legitimately differs across a cross-attention edge. Returning the right axis
    is what keeps us from flagging seq-length / spatial changes as wiring bugs.
    """
    if rank == 4:
        return 1
    return rank - 1


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

    # 2) same rank: compare ONLY the feature/channel axis (see _feature_axis).
    #    A clash there — Linear in!=out, a transformer embed-dim mismatch, a Conv
    #    channel clash — is a real wiring bug. Differences confined to non-feature
    #    axes (sequence length on a 3-D tensor, spatial H/W on a 4-D one) are
    #    intentional reshaping, NOT bugs, so we stay silent (the "no false alarms"
    #    rule; flagging encoder src_len vs decoder tgt_len was the dogfood bug).
    if len(a_shape) < 2:
        return None   # rank-1: no batch/feature split to reason about
    fa = _feature_axis(len(a_shape))
    if a_shape[fa] != b_shape[fa]:
        detail = (f"{a_name} emits dim {a_shape[fa]} but "
                  f"{b_name} expects {b_shape[fa]} (axis {fa})")
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
