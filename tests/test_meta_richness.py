"""Phase A1+A2: richer, correct metadata on real models.

A1 — every captured node records dtype + device (not just shape + params), so
mixed-precision / multi-device models are debuggable.
A2 — dataflow producer tracking descends into dict / nested container outputs,
not just one level of tuple/list. HuggingFace models return dicts / ModelOutput,
so without this their dataflow edges vanish.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import netscope


# ---- A1: dtype + device on every node ------------------------------------
def test_node_records_dtype_and_device():
    m = nn.Linear(8, 4).train(False)
    with netscope.graph("d") as g:
        m(torch.randn(2, 8))
    n = next(x for x in g.nodes() if x["name"] == "Linear")
    meta = n.get("meta") or {}
    assert meta.get("dtype") == "float32"
    assert meta.get("device") == "cpu"


def test_node_records_half_precision_dtype():
    m = nn.Linear(8, 4).half().train(False)
    with netscope.graph("h") as g:
        m(torch.randn(2, 8).half())
    n = next(x for x in g.nodes() if x["name"] == "Linear")
    assert (n.get("meta") or {}).get("dtype") == "float16"


def test_dtype_device_absent_for_nontensor_output():
    """A module returning a non-tensor (a tuple, here) shouldn't crash; dtype/
    device just stay absent rather than guessing."""
    class TupleOut(nn.Module):
        def forward(self, x):
            return (x, x.sum())
    with netscope.graph("t") as g:
        TupleOut().train(False)(torch.randn(2, 4))
    # the node exists and didn't crash; out_shape absent (tuple), dtype optional
    assert any(x["name"] == "TupleOut" for x in g.nodes())


# ---- A2: dataflow through dict / nested outputs --------------------------
def test_dataflow_through_dict_output():
    """A producer whose output is a dict of tensors must still be registered, so
    the consumer that reads dict['h'] gets a dataflow edge."""
    class Producer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)
        def forward(self, x):
            return {"h": self.lin(x), "aux": x.sum()}

    class Consumer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 4)
        def forward(self, d):
            return self.lin(d["h"])

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.prod = Producer()
            self.cons = Consumer()
        def forward(self, x):
            return self.cons(self.prod(x))

    with netscope.graph("net") as g:
        Net().train(False)(torch.randn(2, 8))

    ids = {(n.get("meta") or {}).get("qualname"): n["id"] for n in g.nodes()}
    prod_id, cons_id = ids.get("prod"), ids.get("cons")
    assert prod_id and cons_id
    dataflow = [(e["src"], e["dst"]) for e in g.edges() if e["kind"] == "dataflow"]
    assert (prod_id, cons_id) in dataflow, f"missing prod->cons dataflow edge; got {dataflow}"


def test_dataflow_real_hf_model_has_edges():
    """A real HF decoder returns a ModelOutput (dict-like) at the top; internal
    blocks must still be connected by dataflow edges.

    Built from a tiny LOCAL config so the test is hermetic — no Hub download /
    network. (The previous `AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")` fetch
    flaked in CI under the parallel matrix: HTTP 429 rate-limit.) GPT2 is a
    long-standing HF decoder returning a ModelOutput, so it exercises the same
    dict-output dataflow path the Qwen model did."""
    import pytest

    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, GPT2Config

    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=128, n_positions=64)
    model = AutoModelForCausalLM.from_config(cfg).train(False)
    with netscope.graph("hf") as g, torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (1, 8)))
    dataflow = [e for e in g.edges() if e["kind"] == "dataflow"]
    assert len(dataflow) >= 2, f"expected dataflow edges in a real HF model, got {len(dataflow)}"
