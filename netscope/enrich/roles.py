"""Architectural role of a node — the transformer-aware lens.

A coarse, name-based classifier: given a node, what *kind of component* is it —
attention, MLP, norm, embedding, …? It reads the module's class name + qualified
attribute path (`layers.0.self_attn.q_proj`), so a leaf inside an attention block
(`q_proj`) still classifies as `attention` via its parent path. Heuristic by
design (standard HF / torch naming); never raises, falls back to `other`.

This powers the "group by attention vs MLP" view — color a transformer by role and
its alternating attention/MLP structure pops out at a glance.
"""
# checked in order — block-level roles (attention/mlp) BEFORE the generic `linear`,
# so an attention block's `q_proj` (which also matches "proj") lands in `attention`.
_ROLE_KEYS = [
    ("attention", ("attention", "attn", "mha", "self_attn", "selfattention", "crossattention")),
    ("mlp", ("mlp", "feedforward", "feed_forward", "ffn", "swiglu", "geglu", "moe", "experts")),
    ("norm", ("layernorm", "rmsnorm", "batchnorm", "groupnorm", "norm", "ln_f", "ln_1", "ln_2")),
    ("embedding", ("embedding", "embed", "wte", "wpe", "tok_emb", "pos_emb", "rotary")),
    ("activation", ("relu", "gelu", "silu", "swish", "sigmoid", "tanh", "softmax", "act_fn", "activation")),
    ("conv", ("conv", "pool")),
    ("linear", ("linear", "proj", "dense", "lm_head", "out_proj", "fc")),
]


def node_role(node: dict) -> str:
    """Coarse role: attention | mlp | norm | embedding | activation | conv | linear | other."""
    name = node.get("name") or ""
    qual = (node.get("meta") or {}).get("qualname") or ""
    s = f"{name} {qual}".lower()
    for role, keys in _ROLE_KEYS:
        if any(k in s for k in keys):
            return role
    return "other"


def role_counts(graph) -> dict:
    """How many nodes of each role — a one-line architectural summary of a model.
    Counts leaf-ish nodes (module/op/model), not the pipeline/stage scaffolding."""
    counts: dict = {}
    for n in graph.nodes():
        if n.get("kind") in ("module", "op", "model"):
            r = node_role(n)
            counts[r] = counts.get(r, 0) + 1
    return counts
