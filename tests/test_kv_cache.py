from __future__ import annotations

import torch

from netscope.instrument.torch_nn import _kv_cache_shape


def test_legacy_tuple_of_kv():
    # legacy HF: past_key_values = ((k, v), (k, v), ...) per layer; k=[b,heads,seq,hd]
    k = torch.zeros(1, 8, 5, 64)
    v = torch.zeros(1, 8, 5, 64)
    out = {"past_key_values": ((k, v), (k, v))}
    info = _kv_cache_shape(out)
    assert info == {"layers": 2, "shape": [1, 8, 5, 64], "seq": 5}


def test_cache_object_with_key_cache():
    class _Cache:                       # mimics a v5 DynamicCache
        key_cache = [torch.zeros(1, 8, 7, 64)]
    out = {"past_key_values": _Cache()}
    info = _kv_cache_shape(out)
    assert info["seq"] == 7 and info["shape"][-1] == 64


def test_no_kv_returns_none():
    assert _kv_cache_shape(torch.zeros(2, 3)) is None
    assert _kv_cache_shape({"logits": torch.zeros(1, 5, 10)}) is None
