"""The netscope showcase — every headline feature on real models, one command.

    python examples/showcase.py                # writes /tmp/netscope-gallery/

Traces a roster of real architectures (all built from config — no weight
downloads, CPU-fast), runs the memory/OOM predictor, the generation-timeline
strip, and the `netscope fix` loop, and writes a browsable, self-contained
gallery: an index page linking one interactive graph per model, with the memory
reports and the autofix transcript inlined.

Models whose extra dependency is missing (torchvision, ultralytics) are skipped
with a note — the always-available subset (transformers + torch) still runs.
"""
from __future__ import annotations

import html
import os
import shutil
import sys
import tempfile
import textwrap

import torch

import netscope

OUT = os.environ.get("NETSCOPE_GALLERY", "/tmp/netscope-gallery")


# --- the roster --------------------------------------------------------------

def gpt2_generation():
    """GPT-2 from config, a real decode loop: netscope.step() + attention/KV
    capture -> the generation-timeline strip; memory() -> the KV blow-up story;
    annotate=True -> the 'cost: predicted mem' overlay in the graph."""
    from transformers import GPT2Config, GPT2LMHeadModel

    # attn_implementation="eager": transformers v5 defaults to SDPA, which cannot
    # materialize attention weights — output_attentions comes back EMPTY without
    # this. Required for capture={"attention"} on any HF v5 model.
    cfg = GPT2Config(n_layer=6, n_head=12, n_embd=384, vocab_size=1000, n_positions=64,
                     attn_implementation="eager")
    model = GPT2LMHeadModel(cfg)
    model.train(False)
    ids = torch.randint(0, 1000, (1, 4))
    with netscope.graph("gpt2 · generate", profile=True,
                        capture={"attention", "kv_cache"}) as g, torch.no_grad():
        for _ in range(6):
            with netscope.step():
                # output_attentions: HF returns attention weights only when asked
                # (netscope injects this automatically for generate(); a manual
                # decode loop asks explicitly). netscope reduces them to per-head
                # scalars and drops the tensors.
                out = model(ids, use_cache=True, output_attentions=True)
                ids = torch.cat([ids, out.logits[:, -1:].argmax(-1)], dim=1)

    reports = []
    for seq in (2048, 8192, 32768):
        r = netscope.memory(g, batch=1, seq=seq, vram_gb=4)
        reports.append(r.to_text())
    netscope.memory(g, batch=64, annotate=True)      # stamp pred_bytes for the overlay
    notes = ("A real autoregressive decode, step by step. The strip under the graph "
             "colors each step by attention focus; pick <i>cost: predicted mem</i> in "
             "the HUD to see which layer dominates at batch 64.")
    return g, {"memory": "\n\n".join(reports), "notes": notes}


def qwen3():
    """A real, known LLM architecture — only the config.json comes from the Hub."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    cfg.num_hidden_layers = 4
    model = AutoModelForCausalLM.from_config(cfg)
    model.train(False)
    with netscope.graph("Qwen3 · 4 blocks") as g, torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (1, 8)))
    return g, {"notes": "The real Qwen3 module graph (4 decoder blocks kept), from its "
                        "Hub config — try the <i>⊕ role</i> lens: attention / MLP / norm."}


def sam3():
    """Meta's SAM 3 (detect + segment + track), shrunk from config: the full
    vision-ViT + CLIP-text + DETR + mask-decoder pipeline, folded readable."""
    from transformers import (
        CLIPTextConfig, Sam3Config, Sam3DETRDecoderConfig, Sam3DETREncoderConfig,
        Sam3GeometryEncoderConfig, Sam3Model, Sam3ViTConfig, Sam3VisionConfig,
    )
    cfg = Sam3Config(
        vision_config=Sam3VisionConfig(backbone_config=Sam3ViTConfig(num_hidden_layers=2)),
        text_config=CLIPTextConfig(num_hidden_layers=2),
        geometry_encoder_config=Sam3GeometryEncoderConfig(num_layers=1),
        detr_encoder_config=Sam3DETREncoderConfig(num_layers=1),
        detr_decoder_config=Sam3DETRDecoderConfig(num_layers=1, num_queries=20),
    )
    model = Sam3Model(cfg)
    model.train(False)
    img = cfg.vision_config.backbone_config.image_size
    ids = torch.tensor([[49406, 320, 2368, 49407]])          # bos "a cat" eos
    with netscope.graph("SAM 3") as g, torch.no_grad():
        model(pixel_values=torch.randn(1, 3, img, img), input_ids=ids,
              attention_mask=torch.ones_like(ids))
    return g, {"notes": "SAM 3's real architecture (848M in the wild), compact from "
                        "config: vision ViT + CLIP text + DETR detector + mask decoder."}


def rtdetr():
    from transformers import RTDetrConfig, RTDetrModel

    model = RTDetrModel(RTDetrConfig(num_queries=20))
    model.train(False)
    with netscope.graph("RT-DETR") as g, torch.no_grad():
        model(torch.randn(1, 3, 224, 224))
    return g, {"notes": "639 layers auto-folded to a readable pipeline — and zero "
                        "false mismatch warnings on its multi-scale backbone."}


def resnet18():
    from torchvision.models import resnet18 as _resnet18

    model = _resnet18(weights=None)
    model.train(False)
    with netscope.graph("resnet18", profile=True) as g, torch.no_grad():
        model(torch.randn(1, 3, 224, 224))
    r = netscope.memory(g, batch=256, vram_gb=8)
    return g, {"memory": r.to_text(),
               "notes": "The classic — 11.7M params, repeated BasicBlocks folded. "
                        "Profiled: try the cost heatmap. Will it fit at batch 256?"}


def yolov8n():
    from ultralytics import YOLO

    model = YOLO("yolov8n.yaml").model
    model.train(False)
    with netscope.graph("YOLOv8n") as g, torch.no_grad():
        model(torch.randn(1, 3, 640, 640))
    return g, {"notes": "272 layers from the YAML alone (no weights), folded; concat "
                        "necks and multi-scale heads wired without false alarms."}


ROSTER = [
    ("gpt2-generate", "GPT-2 · watch it think", gpt2_generation),
    ("qwen3", "Qwen3 · real LLM from config", qwen3),
    ("sam3", "SAM 3 · detect + segment + track", sam3),
    ("rtdetr", "RT-DETR · 639 layers, folded", rtdetr),
    ("resnet18", "ResNet-18 · the classic", resnet18),
    ("yolov8n", "YOLOv8n · from YAML", yolov8n),
]

_FIX_SRC = textwrap.dedent(
    """
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(4, 256)
            self.head = nn.Linear(128, 10)   # expects 128, but enc emits 256

        def forward(self, x):
            return self.head(self.enc(x))
    """
)


def autofix_transcript() -> str:
    """Run the real propose -> apply -> re-analyze loop on a temp clash file and
    return the transcript, exactly as `netscope fix` prints it."""
    from netscope.autofix import apply_fixes, propose_fixes
    from netscope.core.checks import detect_mismatches
    from netscope.static.ast_producer import analyze_file

    d = tempfile.mkdtemp()
    path = os.path.join(d, "model.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_FIX_SRC)
    lines = [f"$ netscope fix model.py"]
    fixes = propose_fixes(analyze_file(path))
    for fx in fixes:
        lines += [f"model.py:{fx['line']}  ({fx.get('qualname')})",
                  f"  - {fx['old'].strip()}", f"  + {fx['new'].strip()}"]
    lines += ["", f"$ netscope fix model.py --apply"]
    n = apply_fixes(fixes)
    remaining = detect_mismatches(analyze_file(path))
    lines += [f"applied {n} fix(es); {len(remaining)} mismatch(es) remain."]
    shutil.rmtree(d, ignore_errors=True)
    return "\n".join(lines)


# --- the gallery -------------------------------------------------------------

_CARD = """<div class="card">
  <h2><a href="{page}">{title}</a></h2>
  <p>{notes}</p>{extra}
</div>"""

_INDEX = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>netscope · showcase</title><style>
 body {{ background:#06080c; color:#e8eef2; font-family:ui-monospace,Menlo,monospace;
        max-width:900px; margin:40px auto; padding:0 20px; }}
 h1 {{ letter-spacing:.3em; text-transform:uppercase; font-size:18px; }}
 h1 b {{ color:#22d3ee; }} h2 {{ font-size:15px; margin:0 0 6px; }}
 a {{ color:#22d3ee; text-decoration:none; }} a:hover {{ text-decoration:underline; }}
 .card {{ border:1px solid #1e2c40; border-radius:12px; padding:16px 18px; margin:14px 0;
         background:#0a0e15; }}
 .card p {{ color:#8fa3bb; font-size:13px; line-height:1.55; margin:6px 0; }}
 pre {{ background:#04121e; border:1px solid #1e2c40; border-radius:8px; padding:12px;
       font-size:12px; line-height:1.5; overflow-x:auto; color:#c9e3f5; }}
 .skip {{ color:#6b7888; font-size:12px; }}
</style></head><body>
<h1><b>net</b>scope · showcase</h1>
<p>Real architectures, traced from config (no weight downloads). Every graph is a
self-contained HTML — open one and explore: fold blocks, hover the wiring, switch
the cost/role/attention lenses.</p>
{cards}
<div class="card"><h2>netscope fix · self-healing shapes</h2>
<p>Detect a declared-dim clash and apply the exact edit — deterministic, offline:</p>
<pre>{fix}</pre></div>
{skips}
</body></html>"""


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    cards, skips = [], []
    for slug, title, build in ROSTER:
        try:
            g, extras = build()
            page = f"{slug}.html"
            g.show(path=os.path.join(OUT, page), open_browser=False)
            extra = ""
            if extras.get("memory"):
                extra = f"\n  <pre>{html.escape(extras['memory'])}</pre>"
            cards.append(_CARD.format(page=page, title=html.escape(title),
                                      notes=extras.get("notes", ""), extra=extra))
            print(f"  ✓ {title}  ->  {page}")
        except Exception as e:
            reason = f"{type(e).__name__}: {str(e)[:90]}"
            skips.append(f"<p class='skip'>· {html.escape(title)} skipped — "
                         f"{html.escape(reason)}</p>")
            print(f"  · {title} skipped ({reason})")
    fix = autofix_transcript()
    print("  ✓ netscope fix transcript")
    index = os.path.join(OUT, "index.html")
    with open(index, "w", encoding="utf-8") as f:
        f.write(_INDEX.format(cards="\n".join(cards), fix=html.escape(fix),
                              skips="\n".join(skips)))
    print(f"\ngallery -> {index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
