"""Mismatch detection over the IR — the "show errors" layer.

Scans dataflow edges for shape incompatibilities between a producer's output and
a consumer's expected input. Returns structured warnings; the renderer paints
the offending edges/nodes and lists the warnings. Conservative by design: if a
shape is missing or ambiguous, we say nothing (no false alarms).
"""
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


def _is_conv(node) -> bool:
    """Does this node look like a conv/pooling op (channels-first layout)?"""
    s = f"{node.get('name', '')} {(node.get('meta') or {}).get('qualname', '')}".lower()
    return "conv" in s or "pool" in s


def _feature_axis(rank: int, conv: bool = False) -> int:
    """The axis carrying the channel/feature dim, by PyTorch layout convention.

    4-D NCHW (and conv 3-D NCL) → channels at axis 1 (spatial H/W/L change freely
    under stride/pool). Non-conv 2-D NC / 3-D NSC → the LAST axis is the feature/
    embedding dim; axis 1 of a 3-D NSC tensor is the SEQUENCE length, which differs
    legitimately across a cross-attention edge. Getting this right keeps us from
    flagging seq-length / spatial changes — while still catching a Conv1d channel
    clash (which the old rank-3=last-axis assumption silently missed).
    """
    if rank == 4:
        return 1
    if rank == 3:
        return 1 if conv else 2     # NCL (conv) vs NSC (transformer)
    return rank - 1


def _check_edge(a, b, src, dst, edge_shape=None) -> Optional[dict]:
    # prefer the ACTUAL tensor that flowed on THIS edge (edge.tensor_meta) over the
    # producer node's single representative out_shape. A multi-output module — an
    # FPN / multi-scale backbone returning several feature maps ([512,28,28],
    # [1024,14,14], …) — has one out_shape but feeds different-dim tensors down
    # different edges; comparing the node shape there false-flags the consumers
    # wired to the other scales (the RT-DETR / detection-model dogfood bug).
    a_shape = _shape(edge_shape) or _shape((a.get("meta") or {}).get("out_shape"))
    b_shape = _shape((b.get("meta") or {}).get("in_shape"))
    if a_shape is None or b_shape is None:
        return None

    a_name, b_name = _label(a), _label(b)
    base = {"src": src, "dst": dst, "severity": "error"}

    # 1) rank mismatch first — e.g. Conv2d (N,C,H,W) into a Linear (N,F). The
    #    feature-dim check would give a confusing number; the real fix is a
    #    flatten/reshape, so say that. But only in the HIGHER->LOWER direction: a
    #    genuine "forgot flatten()" is a 4-D feature map flowing into a 2-D consumer.
    #    The REVERSE (a lower-rank producer into a higher-rank consumer) is almost
    #    always an auxiliary/broadcast input — a rotary / sine position table, a
    #    mask, a scalar — not a wiring bug, so stay silent (conservative; flagging it
    #    false-alarms on every transformer's rotary_emb -> attention, the SAM3 dogfood).
    if len(a_shape) != len(b_shape):
        if len(a_shape) > len(b_shape):
            return {
                **base, "kind": "rank_mismatch",
                "detail": (
                    f"{a_name} emits a {len(a_shape)}-D tensor but "
                    f"{b_name} expects {len(b_shape)}-D — missing a flatten()/reshape()?"
                ),
            }
        return None

    # 2) same rank: compare ONLY the feature/channel axis (see _feature_axis).
    #    A clash there — Linear in!=out, a transformer embed-dim mismatch, a Conv
    #    channel clash — is a real wiring bug. Differences confined to non-feature
    #    axes (sequence length on a 3-D tensor, spatial H/W on a 4-D one) are
    #    intentional reshaping, NOT bugs, so we stay silent (the "no false alarms"
    #    rule; flagging encoder src_len vs decoder tgt_len was the dogfood bug).
    if len(a_shape) < 2:
        return None   # rank-1: no batch/feature split to reason about
    fa = _feature_axis(len(a_shape), conv=_is_conv(a) or _is_conv(b))
    if a_shape[fa] != b_shape[fa]:
        detail = (f"{a_name} emits dim {a_shape[fa]} but "
                  f"{b_name} expects {b_shape[fa]} (axis {fa})")
        return {**base, "kind": "shape_mismatch", "detail": detail}
    return None


def detect_mismatches(graph) -> list:
    warnings = []
    nodes = {n["id"]: n for n in graph.nodes()}
    # dataflow in-degree per consumer: a node fed by >1 producer merges them
    # (concat in an FPN/PAN/U-Net neck, a residual add). Its in_shape is the
    # COMBINED tensor, so comparing it against any single producer's edge is a false
    # alarm — only a clean 1->1 wiring can be soundly checked.
    indeg: dict = {}
    for e in graph.edges():
        if e["kind"] == "dataflow":
            indeg[e["dst"]] = indeg.get(e["dst"], 0) + 1
    for e in graph.edges():
        if e["kind"] != "dataflow":
            continue
        if indeg.get(e["dst"], 0) > 1:
            continue   # consumer merges several inputs — per-edge check is invalid
        a = nodes.get(e["src"])
        b = nodes.get(e["dst"])
        if not a or not b:
            continue
        w = _check_edge(a, b, e["src"], e["dst"], (e.get("tensor_meta") or {}).get("shape"))
        if w is not None:
            warnings.append(w)
    return warnings
