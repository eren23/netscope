"""The VSCode extension ships a copy of the web renderer (the Cytoscape template +
vendored JS) under extension/media/, because a packaged VSIX has no sibling
netscope/web/ to read at runtime. The two trees must stay byte-identical — if they
drift, the editor webview renders differently from the standalone g.show() HTML,
the exact bug the single-source design exists to prevent. This guards that.

If this fails after an intentional edit to netscope/web/, re-copy the changed files
into extension/media/ (same relative paths) so the two trees match again.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "netscope" / "web"
COPY = ROOT / "extension" / "media"

WEB_FILES = sorted(p.relative_to(SOURCE) for p in SOURCE.rglob("*") if p.is_file())


@pytest.mark.parametrize("rel", WEB_FILES, ids=str)
def test_extension_media_matches_netscope_web(rel):
    copy = COPY / rel
    assert copy.exists(), f"extension/media/{rel} is missing — copy it from netscope/web/{rel}"
    assert copy.read_bytes() == (SOURCE / rel).read_bytes(), (
        f"extension/media/{rel} has drifted from netscope/web/{rel} — re-sync the copy"
    )
