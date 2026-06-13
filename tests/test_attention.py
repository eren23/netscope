from __future__ import annotations

import math

import torch

from netscope.enrich.attention import head_stats


def test_focused_head_low_entropy_high_last():
    # one head, q=k=4, every query attends ONLY to the last key.
    w = torch.zeros(1, 4, 4)
    w[0, :, -1] = 1.0
    (h,) = head_stats(w)
    assert h["entropy"] == 0.0          # all mass on one key
    assert h["last"] == 1.0
    assert math.isclose(h["dist"], 1.5)   # queries 0..3 -> |q-3| mean = 1.5


def test_uniform_head_max_entropy():
    w = torch.full((1, 4, 4), 0.25)      # uniform over 4 keys
    (h,) = head_stats(w)
    assert math.isclose(h["entropy"], math.log(4), rel_tol=1e-5)
    assert math.isclose(h["last"], 0.25, rel_tol=1e-5)


def test_4d_input_is_meaned_over_batch_and_lists_each_head():
    w = torch.rand(2, 3, 5, 5)           # [batch, heads, q, k]
    stats = head_stats(w)
    assert len(stats) == 3               # one dict per head
    assert set(stats[0]) == {"entropy", "dist", "last"}


def test_non_3d_or_4d_input_returns_empty():
    assert head_stats(torch.rand(4, 4)) == []        # 2D
    assert head_stats(torch.rand(5)) == []            # 1D
