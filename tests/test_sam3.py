"""SAM3 (Segment Anything Model 3, transformers v5): a real CLIP-text-conditioned
DETR detector + mask decoder. netscope must trace it end-to-end — producer
dataflow edges threaded through its nested blocks — with NO false mismatch
warnings (SAM3's rotary_emb emits a 2-D table into 4-D attention; that lower->
higher rank edge is an auxiliary input, not a missing flatten(), and is now
ignored by core.checks).

Hermetic: built from a tiny LOCAL config (shrunk layer counts) — no 848M Hub
download, no network. Skips cleanly where SAM3 is unavailable (transformers < 5,
i.e. Python 3.9).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Sam3Model"):
    pytest.skip("SAM3 requires transformers v5", allow_module_level=True)

import netscope
from netscope.core.checks import detect_mismatches


def _tiny_sam3():
    """A shrunk SAM3 (layer counts cut: ~81M vs the 848M default) so a forward is
    fast + memory-light, while exercising the real SAM3 module graph."""
    from transformers import (
        Sam3Config, Sam3Model, Sam3VisionConfig, Sam3ViTConfig, CLIPTextConfig,
        Sam3GeometryEncoderConfig, Sam3DETREncoderConfig, Sam3DETRDecoderConfig,
    )
    cfg = Sam3Config(
        vision_config=Sam3VisionConfig(backbone_config=Sam3ViTConfig(num_hidden_layers=2)),
        text_config=CLIPTextConfig(num_hidden_layers=2),
        geometry_encoder_config=Sam3GeometryEncoderConfig(num_layers=1),
        detr_encoder_config=Sam3DETREncoderConfig(num_layers=1),
        detr_decoder_config=Sam3DETRDecoderConfig(num_layers=1, num_queries=20),
    )
    return Sam3Model(cfg).train(False), cfg


def test_sam3_traces_with_dataflow_and_no_false_warnings():
    model, cfg = _tiny_sam3()
    img = cfg.vision_config.backbone_config.image_size
    pixel_values = torch.randn(1, 3, img, img)
    input_ids = torch.tensor([[49406, 320, 2368, 49407]])  # bos "a cat" eos
    attention_mask = torch.ones_like(input_ids)

    with netscope.graph("sam3") as g, torch.no_grad():
        model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

    names = [n["name"] for n in g.nodes()]
    # the real SAM3 blocks are captured (vision ViT, attention, FPN, DETR, ...)
    assert any("Sam3" in nm for nm in names), f"expected Sam3* module nodes, got {set(names)}"
    assert any("Attention" in nm for nm in names)

    # dataflow producer edges threaded through the nested blocks
    dataflow = [e for e in g.edges() if e["kind"] == "dataflow"]
    assert len(dataflow) >= 20, f"expected dataflow edges in SAM3, got {len(dataflow)}"

    # and NO false mismatch warnings on a clean model — in particular the
    # rotary_emb (2-D) -> attention (4-D) edge must not trip the rank check.
    warns = detect_mismatches(g)
    assert warns == [], f"unexpected mismatch warnings on a clean SAM3: {warns}"
