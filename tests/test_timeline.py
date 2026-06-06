"""Generation-step timeline — netscope.step() markers + netscope.timeline().

A real autoregressive loop: each step re-runs a tiny LM on a growing sequence, so
the timeline must come back ordered, with the sequence length growing step to step,
and per-step wall-time present under profile.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import netscope


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 32, d: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        return self.head(self.embed(ids))


def _generate(steps: int, profile: bool = False):
    model = TinyLM()
    ids = torch.randint(0, 32, (1, 2))
    with netscope.graph("gen", profile=profile) as g, torch.no_grad():
        for _ in range(steps):
            with netscope.step():
                logits = model(ids)
                ids = torch.cat([ids, logits[:, -1:].argmax(-1)], dim=1)
    return g


def test_steps_are_recorded_in_order():
    tl = netscope.timeline(_generate(4))
    assert [s["step"] for s in tl] == [0, 1, 2, 3]


def test_sequence_length_grows_per_step():
    seqs = [s["out_shape"][1] for s in netscope.timeline(_generate(4))]
    assert seqs == sorted(seqs) and seqs[-1] > seqs[0]   # the autoregressive signature


def test_profile_records_per_step_wall_time():
    tl = netscope.timeline(_generate(3, profile=True))
    assert tl and all(isinstance(s["time_ms"], (int, float)) for s in tl)


def test_default_trace_has_no_step_timing():
    tl = netscope.timeline(_generate(3, profile=False))
    assert all(s["time_ms"] is None for s in tl)


def test_step_outside_a_session_is_a_noop():
    with netscope.step():        # no active capture -> must not raise
        pass


def test_custom_step_label():
    model = TinyLM()
    with netscope.graph("g") as g, torch.no_grad():
        with netscope.step("prefill"):
            model(torch.randint(0, 32, (1, 3)))
    tl = netscope.timeline(g)
    assert tl[0]["label"] == "prefill" and tl[0]["step"] == 0
