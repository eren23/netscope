"""Will it fit? — `netscope.memory()` predicts peak GPU memory at a target
batch/seq and flags OOM, extrapolating the cost data from a single trace.

    python examples/memory_demo.py

Params and KV-cache are exact; activations are a linear estimate (flagged where
the batch axis can't be identified). `overhead`/`reserve_gb` are calibration knobs
— real GPU peak carries CUDA-context + allocator slack the paper sum can't see.
"""
import torch
import torch.nn as nn

import netscope


def mlp_footprint() -> None:
    model = nn.Sequential(nn.Linear(512, 2048), nn.ReLU(), nn.Linear(2048, 512))
    with netscope.graph("mlp") as g:
        model(torch.randn(8, 512))          # trace once at batch 8
    print("== an MLP, extrapolated to batch=1024 on a 24 GB card ==")
    print(netscope.memory(g, batch=1024, vram_gb=24).to_text())
    print()


def llm_kv_blowup() -> None:
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except Exception:
        print("(install `transformers` to see the KV-cache blow-up demo)")
        return
    cfg = GPT2Config(n_layer=12, n_head=12, n_embd=768, vocab_size=1000, n_positions=64)
    model = GPT2LMHeadModel(cfg)
    model.train(False)
    with netscope.graph("gpt2", capture={"kv_cache"}) as g:   # trace once at seq 8
        with torch.no_grad():
            model(torch.zeros(1, 8, dtype=torch.long), use_cache=True)

    print("== a 12-layer GPT-2, traced at seq=8, asked 'will it fit at long context?' ==")
    for seq in (2048, 8192, 32768):
        r = netscope.memory(g, batch=1, seq=seq, vram_gb=4)
        verdict = "OOM" if r.oom else "fits"
        print(f"  seq={seq:>6}:  kv={r.components['kv_cache'] / 1e6:7.1f} MB   "
              f"peak≈{r.peak_bytes / 1e9:.2f} GB  → {verdict}")
    print()
    print(netscope.memory(g, batch=1, seq=32768, vram_gb=4).to_text())


if __name__ == "__main__":
    mlp_footprint()
    llm_kv_blowup()
