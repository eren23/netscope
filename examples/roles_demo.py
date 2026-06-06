"""Demo — the transformer role lens: color a model by architectural component.

netscope classifies every node by role (attention / MLP / norm / embedding / …)
from its module naming. In the graph, hit the **⊕ role** button to recolor the
model by role — a transformer's alternating attention / norm / feed-forward
structure pops out at a glance.

    python examples/roles_demo.py
"""
from __future__ import annotations

import torch
import torch.nn as nn

import netscope


def main() -> None:
    # NB: train mode so the submodule hooks fire — in eval, torch fuses
    # TransformerEncoderLayer into one fast-path call and the structure collapses.
    enc = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(64, 8, batch_first=True), num_layers=3)
    with netscope.graph("encoder") as g, torch.no_grad():
        enc(torch.randn(2, 10, 64))

    print("architectural roles:")
    for role, n in sorted(netscope.roles(g).items(), key=lambda kv: -kv[1]):
        print(f"  {role:12s} {n}")

    out = g.show(path="/tmp/roles_demo.netscope.html", open_browser=False)
    print(f"graph -> {out}  (hit the '⊕ role' button to color by attention / MLP / norm)")


if __name__ == "__main__":
    main()
