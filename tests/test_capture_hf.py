"""M1: HuggingFace transformers auto-instrumentation.

Importing netscope registers a wrapt post-import hook for `transformers`; once
transformers is imported, `GenerationMixin.generate` is auto-wrapped (so every
`model.generate(...)` becomes a `model` node while capturing). We can't load an
8B model in a unit test, so we verify the wrap is applied structurally here; the
behavior of the wrapper itself is covered in test_instrument_base.
"""
from __future__ import annotations

import netscope  # noqa: F401  -- registers the transformers post-import hook
import transformers  # noqa: F401  -- triggers the hook -> wraps generate


def test_generate_is_auto_wrapped():
    from transformers.generation.utils import GenerationMixin

    # wrapt proxies expose the original via __wrapped__
    assert hasattr(GenerationMixin.generate, "__wrapped__")
