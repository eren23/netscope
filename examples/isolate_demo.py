"""Demo - trace ONE part of a network in isolation.

A full trace shows the whole forward pass. But sometimes you want to study just
one block - re-run only `model.layer2`, or one transformer decoder layer, on the
*real* tensor that flowed into it, without the rest of the pipeline.

In the editor: Run & Trace, click a node, hit "isolate this part".

From code (what the extension does under the hood): set NETSCOPE_ISOLATE to the
module's qualified name and NETSCOPE_ISOLATE_OUT to a path. After the run,
netscope re-runs just that submodule on its captured real input and writes the
focused sub-trace there.

    python examples/isolate_demo.py
"""
from __future__ import annotations

import json
import os
import tempfile

import torch
from torchvision.models import resnet18

import netscope
from netscope.core.ir import NVGraph

TARGET = "layer2"  # a ResNet stage (Sequential of BasicBlocks)


def main() -> None:
    model = resnet18(weights=None).eval()

    iso_out = os.path.join(tempfile.gettempdir(), "isolate_layer2.json")
    os.environ["NETSCOPE_ISOLATE"] = TARGET
    os.environ["NETSCOPE_ISOLATE_OUT"] = iso_out

    with netscope.graph("resnet18") as g, torch.no_grad():
        model(torch.randn(1, 3, 64, 64))

    print(f"full trace: {len(g.nodes())} nodes")

    sub = json.loads(open(iso_out).read())
    root = next(n for n in sub["nodes"] if n["parent"] is None)
    print(f"isolated '{TARGET}': {len(sub['nodes'])} nodes, "
          f"root={root['name']}, ran on real input {root['meta'].get('in_shape')}")

    # render the focused sub-trace as its own standalone graph
    ig = NVGraph(name=f"isolate:{TARGET}")
    for n in sub["nodes"]:
        ig.add_node(n["id"], kind=n["kind"], name=n["name"], parent=n["parent"],
                    source=n["source"], loc=n.get("loc"), meta=n.get("meta"),
                    attrs=n.get("attrs"))
    for e in sub["edges"]:
        ig.add_edge(e["src"], e["dst"], kind=e["kind"],
                    tensor_meta=e.get("tensor_meta"), source=e.get("source", "runtime"))
    out = ig.show(path="/tmp/isolate_demo.netscope.html", open_browser=False)
    print(f"interactive isolated graph -> {out}")


if __name__ == "__main__":
    main()
