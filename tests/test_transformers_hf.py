"""HF generate-wrapper helpers.

The torch hooks capture HF model *forwards*; this adapter additionally wraps
``GenerationMixin.generate`` so a whole ``model.generate(...)`` shows up as one
`model` node (the LLM-call boundary in pipelines like sfumato's Qwen planner).
These are the pure helpers that produce that node's name + meta.
"""
from __future__ import annotations

from netscope.instrument import transformers_hf


class _FakeModel:
    pass


def test_gen_name_is_classname_dot_generate():
    assert transformers_hf._gen_name(_FakeModel(), (), {}) == "_FakeModel.generate"


def test_gen_meta_extracts_max_new_tokens():
    assert transformers_hf._gen_meta(None, (), {"max_new_tokens": 64}) == {"max_new_tokens": 64}


def test_gen_meta_is_none_without_max_new_tokens():
    assert transformers_hf._gen_meta(None, (), {}) is None


def test_register_is_idempotent():
    # transformers is importable in the test env, so the post-import hook already
    # ran register() at import time; calling again hits the `_installed` guard.
    transformers_hf.register()
    assert transformers_hf._installed is True


import netscope
from netscope.instrument.transformers_hf import _maybe_request_attentions


def test_injects_output_attentions_when_capturing_attention():
    with netscope.graph("g", capture={"attention"}):
        kwargs = _maybe_request_attentions({})
        assert kwargs.get("output_attentions") is True


def test_does_not_inject_by_default():
    with netscope.graph("g"):
        assert "output_attentions" not in _maybe_request_attentions({})


def test_respects_user_explicit_value():
    with netscope.graph("g", capture={"attention"}):
        kwargs = _maybe_request_attentions({"output_attentions": False})
        assert kwargs["output_attentions"] is False   # never override the user


def test_generate_becomes_a_model_node_under_capture():
    """A real model.generate() routes through the wrapped GenerationMixin.generate
    and shows up as one `model` node — the LLM-call boundary (sfumato's Qwen
    planner). Hermetic: a tiny GPT-2 from config, random weights, no download
    (same approach as the SAM3 dogfood), so it runs in a plain unit test."""
    import pytest
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=16, n_positions=32,
                     bos_token_id=0, eos_token_id=1, pad_token_id=1)
    model = GPT2LMHeadModel(cfg)
    model.train(False)                       # inference mode (== .eval())
    ids = torch.zeros(1, 2, dtype=torch.long)
    mask = torch.ones(1, 2, dtype=torch.long)
    with netscope.graph("gen") as g:
        with torch.no_grad():
            model.generate(ids, attention_mask=mask, max_new_tokens=2, do_sample=False)

    # Contract: a generate() call is captured as a `model` node carrying the right
    # name + meta. We don't pin the *count*: under pytest's cross-file import order
    # the post-import hook can stack the wrapper more than once (harmless — a real
    # single `import netscope` wraps generate exactly once), which would produce a
    # node per wrap. Assert the properties, not the multiplicity.
    gen = [n for n in g.nodes() if str(n.get("name", "")).endswith(".generate")]
    assert gen, "model.generate() should be captured as a model node"
    assert all(n["kind"] == "model" for n in gen)
    assert all(n["name"] == "GPT2LMHeadModel.generate" for n in gen)
    assert all((n.get("meta") or {}).get("max_new_tokens") == 2 for n in gen)
