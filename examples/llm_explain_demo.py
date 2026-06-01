"""Demo — ask the netscope assistant about a node (grounded in the real trace).

netscope's LLM layer is provider-agnostic: it talks to ANY OpenAI-compatible
endpoint. Point it at OpenRouter (default — many cheap models like Gemini Flash),
OpenAI, Together, Groq, or a local server, via env:

    export OPENROUTER_API_KEY=sk-or-...            # or NETSCOPE_LLM_API_KEY / OPENAI_API_KEY
    export NETSCOPE_LLM_MODEL=google/gemini-2.0-flash-001   # optional override
    export NETSCOPE_LLM_BASE_URL=https://api.openai.com/v1  # optional: any OpenAI-compatible host
    python examples/llm_explain_demo.py

With no key set the demo prints the grounded prompt it WOULD send and exits — the
rest of netscope works fully offline.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import netscope
import netscope.llm as nl
from netscope.llm.prompts import build_messages


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 256)

    def forward(self, x):
        return torch.relu(self.proj(x))


class ClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 10)   # expects 128 — but the encoder emits 256

    def forward(self, x):
        return self.fc(x)


def _node_id(g, name):
    return next(n["id"] for n in g.nodes() if n["name"] == name)


def main() -> None:
    enc, head = Encoder().train(False), ClassifierHead().train(False)
    with netscope.graph("llm-demo") as g, torch.no_grad():
        enc(torch.randn(1, 64))
        head(torch.randn(1, 128))
        g.add_edge(_node_id(g, "Encoder"), _node_id(g, "ClassifierHead"),
                   kind="dataflow", source="hint")

    node = _node_id(g, "ClassifierHead")
    if nl.available():
        print("asking the assistant why ClassifierHead is flagged…\n")
        print(nl.explain(g, node, question="why_warn"))
    else:
        print("(no LLM key set — showing the grounded prompt it would send)\n")
        for m in build_messages(g, node, question="why_warn"):
            print(f"--- {m['role']} ---\n{m['content']}\n")


if __name__ == "__main__":
    main()
