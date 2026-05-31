"""M1: zero overhead outside a capture session.

The expensive per-module forward hooks exist ONLY while a session is open. This
is a deterministic check (no timing): torch keeps global hooks in module-level
OrderedDicts; outside a session our hook count must be exactly the baseline, and
inside it must be baseline + 1 (one pre + one post).
"""
from __future__ import annotations

import torch  # noqa: F401  -- ensures the torch post-import hook has fired

import netscope


def test_no_global_forward_hooks_outside_session():
    import torch.nn.modules.module as M

    base_pre = len(M._global_forward_pre_hooks)
    base_post = len(M._global_forward_hooks)

    with netscope.graph("g"):
        assert len(M._global_forward_pre_hooks) == base_pre + 1
        assert len(M._global_forward_hooks) == base_post + 1

    assert len(M._global_forward_pre_hooks) == base_pre
    assert len(M._global_forward_hooks) == base_post
