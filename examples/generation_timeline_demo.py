"""Demo — the generation timeline: trace an autoregressive loop, step by step.

Wrap each decode step in `with netscope.step():`. netscope records the steps in
order, times each (profile=True), and you watch the sequence length grow. The naive
loop re-processes the whole sequence each step (no KV cache), so the steps also get
slower — exactly the cost a KV cache removes.

`netscope.timeline(g)` returns the per-step summary; the graph shows the steps as a
left-to-right sequence — flip on the `cost: time` overlay to see prefill vs decode.

    python examples/generation_timeline_demo.py
"""
from __future__ import annotations

import torch
import torch.nn as nn

import netscope


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 48, d: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.block = nn.TransformerEncoderLayer(d, 4, batch_first=True)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        return self.head(self.block(self.embed(ids)))


def main() -> None:
    model = TinyLM()
    ids = torch.randint(0, 48, (1, 4))          # a 4-token prompt

    with netscope.graph("generate", profile=True) as g, torch.no_grad():
        for _ in range(6):                       # 6 decode steps
            with netscope.step():
                logits = model(ids)
                nxt = logits[:, -1:].argmax(-1)
                ids = torch.cat([ids, nxt], dim=1)

    print("generation timeline:")
    print(f"  {'step':>4} {'seq':>4} {'modules':>8} {'time_ms':>9}")
    for s in netscope.timeline(g):
        seq = s["out_shape"][1] if s["out_shape"] else "?"
        print(f"  {s['step']:>4} {seq!s:>4} {s['modules']:>8} {s['time_ms']!s:>9}")

    out = g.show(path="/tmp/generation_timeline.netscope.html", open_browser=False)
    print(f"graph -> {out}  (steps left-to-right; profiled — try the 'cost: time' overlay)")


if __name__ == "__main__":
    main()
