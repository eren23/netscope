"""Demo — the "show errors" feature: a shape mismatch you'd hit while wiring.

The story: you built an encoder and a classifier head separately, then connected
encoder -> head. But the encoder emits a 256-dim feature and the head expects
128. netscope captures the REAL shapes from a forward pass, sees the dataflow edge
you declared, and flags the clash — a red edge + a ⚠ warnings list — instead of
you discovering it only when the tensors finally collide at runtime.

    python examples/mismatch_demo.py
"""
from __future__ import annotations

import torch
import torch.nn as nn

import netscope


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 256)        # emits a 256-d feature

    def forward(self, x):
        return torch.relu(self.proj(x))


class ClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 10)          # but expects a 128-d feature!

    def forward(self, x):
        return self.fc(x)


def _node_id(g, name):
    return next(n["id"] for n in g.nodes() if n["name"] == name)


def main() -> None:
    enc = Encoder().eval()
    head = ClassifierHead().eval()

    with netscope.graph("mismatch-demo") as g, torch.no_grad():
        enc(torch.randn(1, 64))               # Encoder node: real out_shape [1, 256]
        head(torch.randn(1, 128))             # ClassifierHead node: real in_shape [1, 128]

        # the connection you *meant* to make: encoder feature -> head input.
        # Declared INSIDE the trace so it's part of the captured graph (and so the
        # NETSCOPE_OUT dump the editor reads includes it -> red squiggle in-editor).
        g.add_edge(_node_id(g, "Encoder"), _node_id(g, "ClassifierHead"),
                   kind="dataflow", source="hint")

    from netscope.core.checks import detect_mismatches
    warns = detect_mismatches(g)
    print(f"captured {len(g.nodes())} nodes; {len(warns)} mismatch(es) detected:")
    for w in warns:
        print(f"  WARN  {w['detail']}")

    out = g.show(path="/tmp/mismatch_demo.netscope.html", open_browser=False)
    print(f"interactive graph -> {out}")


if __name__ == "__main__":
    main()
