from __future__ import annotations

import torch

from netscope.instrument.torch_nn import _kv_cache_shape


def test_legacy_tuple_of_kv():
    # legacy HF: past_key_values = ((k, v), (k, v), ...) per layer; k=[b,heads,seq,hd]
    k = torch.zeros(1, 8, 5, 64)
    v = torch.zeros(1, 8, 5, 64)
    out = {"past_key_values": ((k, v), (k, v))}
    info = _kv_cache_shape(out)
    assert info == {"layers": 2, "shape": [1, 8, 5, 64], "seq": 5, "dtype": "float32"}


def test_cache_object_with_key_cache():
    class _Cache:                       # mimics an early v5 DynamicCache
        key_cache = [torch.zeros(1, 8, 7, 64)]
    out = {"past_key_values": _Cache()}
    info = _kv_cache_shape(out)
    assert info["seq"] == 7 and info["shape"][-1] == 64


def test_cache_object_with_layers_keys():
    # transformers v5.12+ DynamicCache: .layers, each with .keys / .values tensors
    class _Layer:
        keys = torch.zeros(1, 8, 9, 64)
        values = torch.zeros(1, 8, 9, 64)

    class _Cache:
        layers = [_Layer(), _Layer()]

    info = _kv_cache_shape({"past_key_values": _Cache()})
    assert info == {"layers": 2, "shape": [1, 8, 9, 64], "seq": 9, "dtype": "float32"}


def test_no_kv_returns_none():
    assert _kv_cache_shape(torch.zeros(2, 3)) is None
    assert _kv_cache_shape({"logits": torch.zeros(1, 5, 10)}) is None


import netscope


class _KVModel(torch.nn.Module):
    def forward(self, x):
        k = torch.zeros(1, 8, x.shape[1], 64)
        return {"logits": x, "past_key_values": ((k, k),)}


def test_kv_cache_recorded_only_when_opted_in():
    m, x = _KVModel(), torch.zeros(1, 5, 16)
    with netscope.graph("on", capture={"kv_cache"}) as g:
        m(x)
    assert any((n.get("meta") or {}).get("kv_cache", {}).get("seq") == 5 for n in g.nodes())

    with netscope.graph("off") as g2:        # default: nothing recorded
        m(x)
    assert all("kv_cache" not in (n.get("meta") or {}) for n in g2.nodes())
