"""Hero demo — a real, KNOWN LLM architecture (Qwen3), no weight download.

We fetch only the tiny config.json from the Hub and instantiate the architecture
with random init (`from_config`) — the *real* Qwen3 module graph, no gigabytes of
weights, runs on CPU in a second. We trim to a few decoder blocks so each block's
internals (attention q/k/v/o projections, the gated MLP, the RMS norms) stay
legible; bump LAYERS to see the whole stack.

    python examples/real_model_demo.py            # Qwen3, 4 blocks
    LAYERS=28 python examples/real_model_demo.py  # the full model

Falls back to Qwen2.5, then a hand-built decoder, if the Hub is unreachable.
"""
from __future__ import annotations

import os

import torch

import netscope

LAYERS = int(os.environ.get("LAYERS", "4"))
CANDIDATES = ["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B-Instruct"]


def build_real(name: str):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(name)
    if hasattr(cfg, "num_hidden_layers"):
        cfg.num_hidden_layers = min(LAYERS, cfg.num_hidden_layers)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    vocab = getattr(cfg, "vocab_size", 32000)
    return model, torch.randint(0, vocab, (1, 8)), type(model).__name__


def build_fallback():
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, d=256, h=4, ff=1024):
            super().__init__()
            self.norm1 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, h, batch_first=True)
            self.norm2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

        def forward(self, x):
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
            return x + self.mlp(self.norm2(x))

    class TinyLM(nn.Module):
        def __init__(self, vocab=8000, d=256, n=LAYERS):
            super().__init__()
            self.embed = nn.Embedding(vocab, d)
            self.blocks = nn.ModuleList([Block(d) for _ in range(n)])
            self.norm = nn.LayerNorm(d)
            self.lm_head = nn.Linear(d, vocab, bias=False)

        def forward(self, ids):
            x = self.embed(ids)
            for b in self.blocks:
                x = b(x)
            return self.lm_head(self.norm(x))

    return TinyLM().eval(), torch.randint(0, 8000, (1, 8)), "TinyDecoderLM (fallback)"


def main() -> None:
    model = ids = label = None
    for name in CANDIDATES:
        try:
            model, ids, label = build_real(name)
            print(f"loaded REAL architecture: {name} ({label}), {LAYERS} decoder block(s)")
            break
        except Exception as e:
            print(f"  {name} unavailable ({type(e).__name__}: {str(e)[:80]})")
    if model is None:
        model, ids, label = build_fallback()
        print(f"using fallback: {label}")

    with netscope.graph(label) as g, torch.no_grad():
        model(ids)

    nodes = g.nodes()
    total = sum(n["meta"].get("params", 0) for n in nodes)
    print(f"captured {len(nodes)} module nodes, {len(g.edges())} edges, {total:,} params")
    out = g.show(path="/tmp/real_model.netscope.html", open_browser=False)
    print(f"interactive graph -> {out}")


if __name__ == "__main__":
    main()
