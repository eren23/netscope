"""Memory / OOM predictor — "will it fit?".

Extrapolate the cost data netscope already captured (per-node `act_bytes`,
`param_bytes`, `out_shape`, `dtype`, `kv_cache`) to a *target* batch/seq and
estimate peak GPU memory, flagging OOM against a VRAM budget.

Honesty model:
- **params** — exact (sum of leaf `param_bytes`; `own_param_bytes` already uses
  recurse=False, so containers contribute 0 and nothing is double-counted).
- **kv_cache** — exact formula, scaled to the target seq (the term that dominates
  an LLM at long context, and the usual real OOM cause).
- **activations** — a *linear* extrapolation of the captured `act_bytes` along the
  batch axis (axis 0) and any axis whose traced size matches the traced seq. Nodes
  whose batch axis can't be identified are flagged `uncertain` rather than scaled.

Real GPU peak is never the paper sum (CUDA context, allocator slack,
fragmentation), so an `overhead`/`reserve` calibration knob is exposed — tune it
to your setup. Inference (no-grad) footprint; training multiplies by grad +
optimizer state (deferred).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_GiB = 1024 ** 3
_UNIT_BYTES = {"": _GiB, "g": _GiB, "gb": _GiB, "gib": _GiB, "t": _GiB * 1024,
               "tb": _GiB * 1024, "tib": _GiB * 1024, "m": 1024 ** 2, "mb": 1024 ** 2,
               "mib": 1024 ** 2, "b": 1}


def _parse_vram(vram, vram_gb) -> "int | None":
    """VRAM budget in bytes. `vram_gb=24` or `vram="24GB"`; a bare number/G/GB/GiB
    is GiB (binary — what nvidia-smi reports and what OOM is actually measured in)."""
    if vram_gb is not None:
        return int(float(vram_gb) * _GiB)
    if vram is None:
        return None
    if isinstance(vram, (int, float)):
        return int(vram * _GiB)
    m = re.match(r"\s*([\d.]+)\s*([a-zA-Z]*)", str(vram))
    if not m:
        return None
    return int(float(m.group(1)) * _UNIT_BYTES.get(m.group(2).lower(), _GiB))


def _crossover_seq(peak_at, vram_bytes, hi: int = 1 << 20) -> "int | None":
    """Largest seq that still fits the budget (peak is monotonic in seq). None if
    it fits across the whole range (no crossover) or never fits (OOM even at 1)."""
    if peak_at(1) > vram_bytes or peak_at(hi) <= vram_bytes:
        return None
    lo = 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if peak_at(mid) <= vram_bytes:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _fmt(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024 or unit == "GB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{int(nbytes)} B"
        nbytes /= 1024
    return f"{nbytes:.1f} GB"


@dataclass
class MemoryReport:
    components: dict            # {"params", "kv_cache", "activations_peak"} bytes
    by_layer: list             # [{qualname, act_at_target, uncertain}], biggest first
    peak_bytes: int
    target: dict               # {"batch", "seq"}
    vram_bytes: "int | None" = None
    oom: bool = False
    crossover_seq: "int | None" = None

    def to_text(self) -> str:
        c = self.components
        lines = [
            f"netscope.memory — batch={self.target['batch']}, seq={self.target['seq']}",
            f"  params        {_fmt(c['params']):>10}",
            f"  kv_cache      {_fmt(c['kv_cache']):>10}",
            f"  activations   {_fmt(c['activations_peak']):>10}   (peak of top layers)",
            f"  {'─' * 24}",
            f"  estimated peak {_fmt(self.peak_bytes):>9}   (incl. overhead + reserve)",
        ]
        if self.vram_bytes is not None:
            verdict = "⚠ OOM" if self.oom else "✓ fits"
            line = f"  budget         {_fmt(self.vram_bytes):>9} → {verdict}"
            if self.crossover_seq is not None:
                line += f"   (fits up to seq {self.crossover_seq})"
            lines.append(line)
        return "\n".join(lines)


_DTYPE_BYTES = {
    "float64": 8, "double": 8, "float32": 4, "float": 4, "int64": 8, "long": 8,
    "int32": 4, "int": 4, "bfloat16": 2, "float16": 2, "half": 2, "int16": 2,
    "int8": 1, "uint8": 1, "bool": 1,
}


def _dtype_bytes(dt: "str | None", default: int = 2) -> int:
    # default 2: KV caches are almost always fp16/bf16, so it's the safe guess when
    # a node didn't record its dtype. ponytail: bump the default if fp32 KV matters.
    return _DTYPE_BYTES.get(dt or "", default)


def _sum_param_bytes(nodes) -> int:
    return int(sum((n.get("meta") or {}).get("param_bytes", 0) or 0 for n in nodes))


def _kv_bytes(nodes, batch, seq) -> int:
    """Exact KV-cache bytes at the target (batch, seq), from every node carrying a
    `kv_cache` = {layers, shape:[b,heads,seq,head_dim], seq}. heads, head_dim and
    layers are structural (fixed); only batch and seq scale.

    A generation trace fires the model's forward once per decode step, so the SAME
    growing cache shows up on many nodes — summing them would overcount ×steps.
    Group by cache signature (layers, heads, head_dim): max within a group (it's
    one cache), sum across distinct groups (genuinely separate caches, e.g. two
    models in a pipeline)."""
    groups: dict = {}
    for n in nodes:
        meta = n.get("meta") or {}
        kv = meta.get("kv_cache")
        shape = (kv or {}).get("shape") or []
        layers = (kv or {}).get("layers")
        if not kv or len(shape) < 4 or not layers:
            continue
        heads, head_dim = shape[1], shape[3]
        b = batch or shape[0]
        s = seq or kv.get("seq") or shape[2]
        # the cache's own dtype (fp16/bf16) — NOT the module's input dtype, which is
        # int64 for a token-id input and would overcount 4×. Fall back to the node's.
        elsize = _dtype_bytes(kv.get("dtype") or meta.get("dtype"))
        nbytes = layers * b * heads * s * head_dim * elsize
        key = (layers, heads, head_dim)
        groups[key] = max(groups.get(key, 0), nbytes)
    return int(sum(groups.values()))


def _infer_traced_batch(nodes) -> "int | None":
    """The batch dim the trace ran at — the most common axis-0 size across nodes
    that carry an out_shape (batch is reliably axis 0 in torch layouts)."""
    c: Counter = Counter()
    for n in nodes:
        sh = (n.get("meta") or {}).get("out_shape")
        if sh:
            c[sh[0]] += 1
    return c.most_common(1)[0][0] if c else None


def _scaled_activations(nodes, b0, batch, s0, seq) -> list:
    """Per-node activation bytes scaled to the target (batch, seq). Linear in the
    batch axis (0); also linear in any axis whose traced size == the traced seq.
    A node whose axis-0 size != b0 (attention [b,heads,q,k], a transpose, a scalar
    reduction) can't be confidently scaled → flagged `uncertain`, left unscaled."""
    out = []
    for n in nodes:
        meta = n.get("meta") or {}
        act, sh = meta.get("act_bytes"), meta.get("out_shape")
        if not act or not sh:
            continue
        uncertain = not (b0 and batch and sh[0] == b0)
        factor = 1.0
        if not uncertain:
            factor = batch / b0
            if seq and s0:
                for ax in sh[1:]:
                    if ax == s0:
                        factor *= seq / s0
        out.append({
            "id": n.get("id"),
            "qualname": meta.get("qualname") or n.get("name", ""),
            "act_at_target": int(round(act * factor)),
            "uncertain": uncertain,
        })
    out.sort(key=lambda r: r["act_at_target"], reverse=True)
    return out


def memory(graph, *, batch: "int | None" = None, seq: "int | None" = None,
           vram=None, vram_gb=None, top_k: int = 2, overhead: float = 1.1,
           reserve_gb: float = 1.0, annotate: bool = False,
           traced_batch: "int | None" = None,
           traced_seq: "int | None" = None) -> MemoryReport:
    """Estimate peak GPU memory for a traced graph at a target batch/seq, and flag
    OOM against a VRAM budget — "will it fit?".

    ``overhead`` (multiplier) and ``reserve_gb`` (flat floor) are the calibration
    knobs: real GPU peak carries CUDA-context + allocator slack + fragmentation the
    paper sum can't see (~1–2 GB + ~10–20%). Defaults are a sane inference starting
    point — tune them to your setup. Pass ``vram="24GB"`` (or ``vram_gb=24``) to get
    an OOM verdict + the crossover seq (the longest context that still fits).

    ``annotate=True`` writes each node's predicted activation bytes into its meta
    (``meta.pred_bytes``) so the graph's ``cost: predicted mem`` overlay heatmaps
    it — the layer that dominates at the target scale glows red in ``g.show()``.
    """
    nodes = graph.nodes()
    params = _sum_param_bytes(nodes)
    b0 = traced_batch or _infer_traced_batch(nodes)
    target_batch = batch or b0
    s0 = traced_seq
    reserve_bytes = reserve_gb * _GiB

    def peak_at(s):
        by = _scaled_activations(nodes, b0, target_batch, s0, s)
        ap = int(sum(r["act_at_target"] for r in by[:top_k]))
        kvb = _kv_bytes(nodes, target_batch, s)
        return int((params + kvb + ap) * overhead + reserve_bytes), by, ap, kvb

    peak, by_layer, act_peak, kv = peak_at(seq)
    if annotate:
        for row in by_layer:
            if row.get("id") is not None:
                graph.update_meta(row["id"], {"pred_bytes": row["act_at_target"]})
    components = {"params": params, "kv_cache": kv, "activations_peak": act_peak}
    vram_bytes = _parse_vram(vram, vram_gb)
    oom = vram_bytes is not None and peak > vram_bytes
    crossover = (_crossover_seq(lambda s: peak_at(s)[0], vram_bytes)
                 if vram_bytes is not None else None)
    return MemoryReport(components=components, by_layer=by_layer, peak_bytes=peak,
                        target={"batch": target_batch, "seq": seq},
                        vram_bytes=vram_bytes, oom=oom, crossover_seq=crossover)
