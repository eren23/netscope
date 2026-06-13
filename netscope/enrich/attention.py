"""Per-head attention reductions — the value-side LLM lens.

Attention weight tensors are huge ([batch, heads, q, k]); we never retain them.
Given a weight tensor (already in hand inside a forward hook), reduce it to a few
scalars per head and throw the tensor away:

- entropy : mean attention entropy per query (focused≈0, diffuse≈log k)
- dist    : mean attention distance |query_pos - key_pos| (how far back it looks)
- last    : fraction of attention mass on the final key position

Pure + framework-light: works on any torch tensor shaped [heads, q, k] or
[batch, heads, q, k] (batch is averaged out). Returns one dict per head.
"""
from typing import List


def head_stats(weights) -> List[dict]:
    import torch

    w = weights.detach().to(torch.float32)
    if w.dim() == 4:            # [batch, heads, q, k] -> mean over batch
        w = w.mean(dim=0)
    if w.dim() != 3:
        return []
    heads, q, k = w.shape
    # entropy per (head, query), averaged over queries
    p = w.clamp_min(0)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    ent = -(p * (p.clamp_min(1e-9)).log()).sum(dim=-1).mean(dim=-1)   # [heads]
    # attention distance: sum_k p * |q_idx - k_idx|, averaged over queries
    qi = torch.arange(q, device=w.device).view(q, 1)
    ki = torch.arange(k, device=w.device).view(1, k)
    dist_mat = (qi - ki).abs().to(torch.float32)                     # [q, k]
    dist = (p * dist_mat).sum(dim=-1).mean(dim=-1)                   # [heads]
    last = p[:, :, -1].mean(dim=-1)                                  # [heads]
    return [
        {"entropy": round(float(ent[h]), 4),
         "dist": round(float(dist[h]), 4),
         "last": round(float(last[h]), 4)}
        for h in range(heads)
    ]
