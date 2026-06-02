"""Demo — catch a wiring bug BEFORE you run anything (static pre-check).

netscope reads the declared layer dims straight from the source — no forward
pass, no tensors — and flags an obvious clash. In the editor this is the
"netscope: Show Graph" command (or just opening the file): a red squiggle appears
on the offending line without executing a thing.

    python examples/static_dim_check_demo.py
"""
from __future__ import annotations

import torch.nn as nn

from netscope.static.ast_producer import analyze_file


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(784, 512)     # emits a 512-d feature
        self.classifier = nn.Linear(256, 10)    # BUG: expects 256, not 512

    def forward(self, x):
        return self.classifier(self.backbone(x))


def main() -> None:
    # analyze THIS file statically — nothing is executed, no model is built
    g = analyze_file(__file__)
    warnings = g.to_dict()["warnings"]
    print(f"static analysis of {__file__.split('/')[-1]} — no run, no tensors:")
    if not warnings:
        print("  no wiring clashes found")
    for w in warnings:
        line = next((n["loc"]["line"] for n in g.nodes() if n["id"] == w["dst"]), "?")
        print(f"  L{line}  {w['detail']}")


if __name__ == "__main__":
    main()
