"""Memory / OOM predictor — extrapolate the captured cost data to a target
batch/seq and estimate peak GPU memory.

Params and KV-cache are exact; activations are a linear extrapolation of the
captured `act_bytes` (flagged where the batch axis can't be identified).
"""
from __future__ import annotations

import torch
import torch.nn as nn

import netscope


def test_params_are_exact():
    # Linear(4,8): 40 params; Linear(8,2): 18 params; ReLU: 0  -> 58 params × 4 bytes
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    with netscope.graph("m") as g:
        model(torch.randn(2, 4))
    r = netscope.memory(g)
    assert r.components["params"] == 58 * 4


def test_activations_scale_linearly_with_batch():
    model = nn.Linear(4, 16)
    with netscope.graph("m") as g:
        model(torch.randn(2, 4))          # traced at batch 2
    base = netscope.memory(g, batch=2)
    big = netscope.memory(g, batch=8)     # 4× the batch
    # the Linear's activation ([2,16] -> [8,16]) scales exactly ×4
    assert base.components["activations_peak"] == 2 * 16 * 4        # act_bytes at batch 2
    assert big.components["activations_peak"] == base.components["activations_peak"] * 4


def test_kv_cache_is_exact_and_scales_with_seq():
    from netscope.core.ir import NVGraph
    g = NVGraph("m")
    g.add_node("dec", kind="module", name="Decoder",
               meta={"kv_cache": {"layers": 4, "shape": [1, 8, 16, 64], "seq": 16},
                     "dtype": "float16"})
    r16 = netscope.memory(g, batch=1, seq=16)
    # layers × batch × heads × seq × head_dim × 2 bytes (fp16) — the exact formula
    assert r16.components["kv_cache"] == 4 * 1 * 8 * 16 * 64 * 2
    r32 = netscope.memory(g, batch=1, seq=32)
    assert r32.components["kv_cache"] == r16.components["kv_cache"] * 2   # linear in seq


def test_overhead_and_reserve_knobs_raise_the_estimate():
    model = nn.Linear(4, 16)
    with netscope.graph("m") as g:
        model(torch.randn(2, 4))
    lo = netscope.memory(g, overhead=1.0, reserve_gb=0)      # raw sum, no calibration
    hi = netscope.memory(g, overhead=2.0, reserve_gb=0)      # 2× multiplier
    assert hi.peak_bytes > lo.peak_bytes
    withres = netscope.memory(g, overhead=1.0, reserve_gb=1)  # flat +1 GiB floor
    assert withres.peak_bytes == lo.peak_bytes + 1024 ** 3


def test_oom_verdict_and_crossover_seq():
    from netscope.core.ir import NVGraph
    g = NVGraph("m")
    # chunky KV: 32 layers × 32 heads × 128 head_dim, fp16
    # kv(batch=1, seq=S) = 32*1*32*S*128*2 = 262144 * S bytes
    g.add_node("dec", kind="module", name="Decoder",
               meta={"kv_cache": {"layers": 32, "shape": [1, 32, 128, 128], "seq": 128},
                     "dtype": "float16"})
    fits = netscope.memory(g, batch=1, seq=1024, vram_gb=1, overhead=1.0, reserve_gb=0)
    assert fits.oom is False                        # 0.25 GiB < 1 GiB
    oom = netscope.memory(g, batch=1, seq=8192, vram_gb=1, overhead=1.0, reserve_gb=0)
    assert oom.oom is True                          # 2 GiB > 1 GiB
    # peak(S) = 262144·S crosses 1 GiB at S = 4096 (last seq that still fits)
    assert oom.crossover_seq == 4096


def test_reduction_node_is_flagged_uncertain_not_scaled():
    from netscope.core.ir import NVGraph
    g = NVGraph("m")
    g.add_node("lin1", kind="module", name="Linear",
               meta={"out_shape": [2, 16], "act_bytes": 2 * 16 * 4, "qualname": "lin1"})
    g.add_node("lin2", kind="module", name="Linear",
               meta={"out_shape": [2, 8], "act_bytes": 2 * 8 * 4, "qualname": "lin2"})
    # a batch-reduction (e.g. a mean/loss) -> [16]: axis-0 isn't the batch (b0=2)
    g.add_node("pool", kind="module", name="Mean",
               meta={"out_shape": [16], "act_bytes": 16 * 4, "qualname": "pool"})
    by = {l["qualname"]: l for l in netscope.memory(g, batch=8).by_layer}
    assert by["lin1"]["uncertain"] is False and by["pool"]["uncertain"] is True
    assert by["lin1"]["act_at_target"] == 2 * 16 * 4 * 4        # scaled ×4 (batch 2→8)
    assert by["pool"]["act_at_target"] == 16 * 4               # uncertain → left unscaled


def test_kv_dominates_a_real_llm_at_long_context():
    # end-to-end on a real model: KV cache is captured (v5 DynamicCache), counted
    # once across the two forwards that report it, scales exactly with seq, and uses
    # the cache's own fp32 element size — dominating the fixed params at long context.
    import pytest
    torch_ = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(n_layer=4, n_head=8, n_embd=128, vocab_size=64, n_positions=64)
    model = GPT2LMHeadModel(cfg)
    model.train(False)
    with netscope.graph("fwd", capture={"kv_cache"}) as g:
        with torch_.no_grad():
            model(torch_.zeros(1, 6, dtype=torch_.long), use_cache=True)

    r1k = netscope.memory(g, batch=1, seq=1024)
    r8k = netscope.memory(g, batch=1, seq=8192)
    # 4 layers × 1 × 8 heads × 8192 × 16 head_dim × 4 bytes (fp32 cache)
    assert r8k.components["kv_cache"] == 4 * 1 * 8 * 8192 * 16 * 4
    assert r8k.components["kv_cache"] == r1k.components["kv_cache"] * 8   # exact, linear in seq
    assert r8k.components["kv_cache"] > r8k.components["params"]          # KV dominates at long ctx


def test_to_text_summarizes_the_verdict():
    from netscope.core.ir import NVGraph
    g = NVGraph("m")
    g.add_node("dec", kind="module", name="Decoder",
               meta={"kv_cache": {"layers": 8, "shape": [1, 8, 128, 64], "seq": 128,
                                  "dtype": "float16"}})
    txt = netscope.memory(g, batch=1, seq=8192, vram_gb=1).to_text()
    assert isinstance(txt, str) and txt
    assert "peak" in txt.lower() and "kv" in txt.lower()
    assert "OOM" in txt                     # this config exceeds a 1 GiB budget


def test_annotate_stamps_pred_bytes_for_the_overlay():
    model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 4))
    with netscope.graph("m") as g:
        model(torch.randn(2, 4))            # traced at batch 2
    netscope.memory(g, batch=16, annotate=True)   # 8× batch -> stamp meta.pred_bytes
    pred = [p for p in ((n.get("meta") or {}).get("pred_bytes") for n in g.nodes()) if p]
    # the widest layer's predicted activation is its traced act_bytes × 8
    assert pred and max(pred) == 2 * 64 * 4 * 8
