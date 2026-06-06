"""Demo — trace diffing: see exactly what changed between two model versions.

The iteration story: you widen a layer and insert another, re-trace, and netscope
shows precisely what's new, what shifted shape/params, and what's gone — instead
of eyeballing two graphs side by side. Matched by a STABLE identity (the module's
qualified name), so it survives the node-id shuffle an inserted layer causes.

    python examples/diff_demo.py
"""
from __future__ import annotations

import torch
import torch.nn as nn

import netscope


class Net(nn.Module):
    """One model, two versions: `wide=True` widens `enc` 128->256 and inserts a
    `mid` block. Same class, so the root matches and only the real edits show."""

    def __init__(self, wide: bool = False):
        super().__init__()
        self.enc = nn.Linear(64, 256 if wide else 128)
        if wide:
            self.mid = nn.Linear(256, 128)
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        h = torch.relu(self.enc(x))
        if hasattr(self, "mid"):
            h = torch.relu(self.mid(h))
        return self.head(h)


def main() -> None:
    with netscope.graph("before") as before, torch.no_grad():
        Net(wide=False).train(False)(torch.randn(1, 64))
    with netscope.graph("after") as after, torch.no_grad():
        Net(wide=True).train(False)(torch.randn(1, 64))

    d = netscope.diff(before, after)
    s = d["summary"]
    print(f"diff: +{s['added']} added  ~{s['changed']} changed  "
          f"-{s['removed']} removed  ={s['same']} same")
    _nm = lambda x: x["qualname"] or x["name"]
    for x in d["added"]:
        print(f"  + {_nm(x):8s} {x['out_shape']}")
    for c in d["changed"]:
        b, a = c["before"], c["after"]
        print(f"  ~ {_nm(c):8s} out {b['out_shape']} -> {a['out_shape']}")

    out = netscope.diff_view(before, after).show(
        path="/tmp/diff_demo.netscope.html", open_browser=False)
    print(f"colored diff graph -> {out}  (added=green, changed=amber, removed=ghost)")


if __name__ == "__main__":
    main()
