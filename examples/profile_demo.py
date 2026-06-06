"""Demo — the profiler overlay: per-layer cost, colored hot.

Activation + param MEMORY ride on every trace for free (derived from the shapes
netscope already captures). Pass profile=True to also measure per-layer wall-TIME
(opt-in, since timing has overhead — the default trace stays metadata-only).

Open the graph and use the HUD "cost:" selector to recolor nodes by time / memory
/ params — the fat or slow layer glows red.

    python examples/profile_demo.py
"""
from __future__ import annotations

import torch
import torch.nn as nn

import netscope


def main() -> None:
    # one fat middle layer so the heatmap has a clear hot spot
    model = nn.Sequential(
        nn.Linear(64, 1024), nn.ReLU(),
        nn.Linear(1024, 1024), nn.ReLU(),    # the param / compute hog
        nn.Linear(1024, 10),
    )
    with netscope.graph("profile-demo", profile=True) as g, torch.no_grad():
        model.train(False)(torch.randn(8, 64))

    print("per-layer cost (params · activation · time):")
    for n in g.nodes():
        m = n.get("meta") or {}
        if m.get("param_bytes"):
            print(f"  {n['name']:8s} {m['param_bytes']:>9,} B params · "
                  f"{m.get('act_bytes', 0):>7,} B act · {m.get('time_ms', '-')} ms")

    out = g.show(path="/tmp/profile_demo.netscope.html", open_browser=False)
    print(f"graph -> {out}  (HUD 'cost:' selector recolors by time / memory / params)")


if __name__ == "__main__":
    main()
