"""Hero demo 2 — a Transformer encoder block, zero instrumentation.

    python examples/transformer_demo.py

Shows the attention + feed-forward substructure of a single
`nn.TransformerEncoderLayer` with the (batch, seq, d_model) tensor flowing
through it — captured automatically from one forward pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import netscope


def main() -> None:
    layer = nn.TransformerEncoderLayer(
        d_model=64, nhead=8, dim_feedforward=256, batch_first=True
    )
    x = torch.randn(2, 16, 64)  # (batch, seq_len, d_model)

    with netscope.graph("transformer-encoder-layer") as g:
        layer(x)

    nodes = g.nodes()
    total = sum(n["meta"].get("params", 0) for n in nodes)
    print(f"encoder layer: {len(nodes)} module nodes, {total:,} params")
    print(g.to_mermaid())

    out = g.show(path="/tmp/transformer.netscope.html", open_browser=False)
    print(f"interactive graph -> {out}")


if __name__ == "__main__":
    main()
