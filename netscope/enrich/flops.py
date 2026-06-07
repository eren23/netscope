"""FLOPs / MACs counting — opt-in, best-effort.

FLOPs need a real forward pass and a per-op cost model, so unlike params they
are NOT computed during capture (that would fight the capture-once design).
Instead this is an explicit call the user (or a demo) opts into. We delegate to
`thop` if installed; anything goes wrong -> ``None`` (never raises), since this
is enrichment, not core behavior.
"""
from typing import Optional


def flops_available() -> bool:
    try:
        import thop  # noqa: F401

        return True
    except Exception:
        return False


def count_flops(model, sample_input) -> Optional[int]:
    """Total multiply-accumulates for one forward of ``model(sample_input)``.

    Returns None if thop is unavailable or profiling fails. FLOPs ~= 2 * MACs.
    """
    try:
        import contextlib
        import io

        from thop import profile

        # thop prints per-op registration noise; silence it.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            macs, _params = profile(model, inputs=(sample_input,), verbose=False)
        return int(macs)
    except Exception:
        return None
