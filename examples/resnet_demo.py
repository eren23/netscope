"""Hero demo 1 — a popular known model, zero instrumentation.

    python examples/resnet_demo.py

Just `import netscope`, wrap one forward in `netscope.graph(...)`, and the entire
ResNet-18 hierarchy is captured with real per-layer tensor shapes and parameter
counts. No decorators, no model edits. Writes an interactive standalone graph.
"""
from __future__ import annotations

import torch
from torchvision.models import resnet18

import netscope


def main() -> None:
    model = resnet18(weights=None).eval()
    x = torch.randn(1, 3, 224, 224)

    with netscope.graph("resnet18") as g, torch.no_grad():
        model(x)

    nodes = g.nodes()
    total = sum(n["meta"].get("params", 0) for n in nodes)
    convs = sum("Conv2d" in n["name"] for n in nodes)
    print(f"resnet18: {len(nodes)} module nodes, {convs} Conv2d, {total:,} params")

    out = g.show(path="/tmp/resnet18.netscope.html", open_browser=False)
    print(f"interactive graph -> {out}")


if __name__ == "__main__":
    main()
