"""HuggingFace transformers auto-instrumentation.

Wraps ``GenerationMixin.generate`` so every ``model.generate(...)`` becomes a
`model` node while capturing (the LLM-call boundary in pipelines like sfumato's
Qwen planner). Gated + safe via the shared ``span_wrapper``.
"""
from __future__ import annotations

from netscope.instrument.base import safe_patch

_installed = False


def _gen_name(instance, args, kwargs) -> str:
    return type(instance).__name__ + ".generate"


def _gen_meta(instance, args, kwargs):
    mnt = kwargs.get("max_new_tokens")
    return {"max_new_tokens": mnt} if mnt is not None else None


def register() -> None:
    global _installed
    if _installed:
        return
    ok = safe_patch(
        "transformers.generation.utils",
        "GenerationMixin.generate",
        name_fn=_gen_name,
        kind="model",
        meta_fn=_gen_meta,
    )
    if ok:
        _installed = True
