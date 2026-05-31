"""Hero demo — sfumato's cmajc pipeline (AR-plan -> diffuse-refine x5 -> vote).

Runs on CPU in milliseconds, no 8B model required, and exercises ALL of netscope:

  * runtime auto-capture of real (tiny) torch modules -> live tensor shapes
  * optional nv.stage / nv.branch / nv.reduce hints -> the semantic boxes that
    pure tracing can't infer (which calls are branches, which is the vote)
  * the STATIC AST producer run on sfumato's ACTUAL e4/runner.py -> recovers the
    branch fan-out + majority vote straight from their source, no execution

    python examples/sfumato_cmajc.py

Mirrors the real pipeline in github.com/eren23/sfumato (cmajc, 82.5% on GSM8K):
a Qwen AR planner, 5 LLaDA diffusion branches at temperature>0, then a majority
vote over the extracted answers.
"""
from __future__ import annotations

import os
from collections import Counter

import torch
import torch.nn as nn

import netscope

D = 32  # tiny hidden-size stand-in for the demo


class ARPlanner(nn.Module):
    """Stand-in for the Qwen AR planner (generate_plan)."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(D, D)

    def forward(self, problem):
        return torch.tanh(self.proj(problem))


class DiffusionRefiner(nn.Module):
    """Stand-in for one LLaDA denoise branch (denoise_block)."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Linear(D, D)

    def forward(self, prefix):
        return torch.relu(self.block(prefix))


def extract_answer(vec) -> int:
    # argmax stand-in for grade.extract_answer (the "numeric answer" a branch lands on)
    return int(vec.argmax().item())


def run_cmajc(problem, planner, refiner, n_branches=5):
    """Faithful mini-cmajc, annotated with netscope hints."""
    with netscope.stage("plan"):                       # Qwen AR planner
        plan = planner(problem)

    branch_answers = []
    for b in range(n_branches):                      # 5 diffusion branches
        with netscope.branch(f"diffuse[{b}]"):         # LLaDA denoise, temp>0
            torch.manual_seed(b)
            refined = refiner(plan + 0.1 * torch.randn(D))
        branch_answers.append(extract_answer(refined))

    with netscope.reduce("vote"):                      # majority vote / consensus
        winner, _ = Counter(branch_answers).most_common(1)[0]
    return winner


def static_view_of_real_sfumato():
    """Run the static AST producer on sfumato's ACTUAL runner.py, if present."""
    path = os.path.expanduser("~/Documents/AI/sfumato/e4/runner.py")
    if not os.path.exists(path):
        print("(real sfumato runner.py not found; skipping static view)")
        return
    from netscope.static.ast_producer import analyze_file

    sg = analyze_file(path)
    branches = [n for n in sg.nodes() if n["attrs"].get("branch")]
    votes = [n for n in sg.nodes() if n["attrs"].get("reduce")]
    print(f"static analysis of real sfumato {os.path.basename(path)}:")
    print(f"  {len(branches)} branch region(s), {len(votes)} vote/reduce region(s) "
          f"recovered from source — NO execution")
    for n in branches + votes:
        print(f"    L{n['loc']['line']:<4} {n['name']:<14} {n['attrs']}")


def main() -> None:
    planner = ARPlanner().eval()
    refiner = DiffusionRefiner().eval()
    problem = torch.randn(D)

    with netscope.graph("sfumato-cmajc") as g, torch.no_grad():
        winner = run_cmajc(problem, planner, refiner)

    nodes = g.nodes()
    stages = [n for n in nodes if n["kind"] == "stage"]
    n_branch = sum(1 for n in stages if n["attrs"].get("branch"))
    n_reduce = sum(1 for n in stages if n["attrs"].get("reduce"))
    print(f"runtime graph: {len(nodes)} nodes "
          f"({n_branch} branches, {n_reduce} reduce), winner answer = {winner}")
    print()
    static_view_of_real_sfumato()

    out = g.show(path="/tmp/sfumato_cmajc.netscope.html", open_browser=False)
    print(f"\ninteractive graph -> {out}")


if __name__ == "__main__":
    main()
