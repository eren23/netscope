"""HuggingFace transformers auto-instrumentation.

Wraps ``GenerationMixin.generate`` so every ``model.generate(...)`` becomes a
`model` node while capturing (the LLM-call boundary in pipelines like sfumato's
Qwen planner). Uses a bespoke wrapt patch (instead of the shared ``safe_patch``)
so the wrapper can inject ``output_attentions=True`` when attention capture is on.
"""
from netscope.core import context as ctx

_installed = False


def _gen_name(instance, args, kwargs) -> str:
    return type(instance).__name__ + ".generate"


def _gen_meta(instance, args, kwargs):
    mnt = kwargs.get("max_new_tokens")
    return {"max_new_tokens": mnt} if mnt is not None else None


def _maybe_request_attentions(kwargs: dict) -> dict:
    """When attention capture is on, ask HF to return attention weights (it won't
    by default). Uses setdefault so an explicit user value always wins."""
    cap = ctx.active_capture()
    if cap is not None and cap.wants("attention"):
        kwargs.setdefault("output_attentions", True)
    return kwargs


def register() -> None:
    global _installed
    if _installed:
        return
    try:
        import wrapt
    except Exception:
        return

    @wrapt.patch_function_wrapper("transformers.generation.utils", "GenerationMixin.generate")
    def _wrapped(wrapped, instance, args, kwargs):
        cap = ctx.active_capture()
        if cap is None:
            return wrapped(*args, **kwargs)        # zero-overhead gate, no copy
        # tracing must NEVER break the wrapped call (the MLflow safe_patch rule):
        # if injection or the span machinery throws, generate() still runs.
        handle = None
        try:
            kwargs = _maybe_request_attentions(dict(kwargs))
            handle = cap.open_span(_gen_name(instance, args, kwargs), kind="model",
                                   meta=_gen_meta(instance, args, kwargs))
        except Exception:
            handle = None
        try:
            return wrapped(*args, **kwargs)
        finally:
            if handle is not None:
                try:
                    cap.close_span(handle)
                except Exception:
                    pass
    _installed = True
