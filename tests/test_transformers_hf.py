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
