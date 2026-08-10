"""The character corpus the dashboard can render — one source of truth for the fonts.

``scripts/02_build_fonts.py`` subsets the shipped faces to it; ``tests/test_font_coverage.py``
asserts every character in it resolves to a shipped face.  The pair closes a loop a CSS stack
cannot close alone: an unresolved family renders its characters in whatever the host happens
to install, silently — which is how the Japanese half of this bilingual UI spent its life in
VL Gothic while the CSS named IBM Plex.

Scope = tracked text that reaches the browser: the schema YAML (every ``ja``/``en`` UI string
and level label), the dashboard's own Python literals, and the committed model JSON whose
labels are read back into figures.  Tracked-only, so the corpus is a property of the commit
and the font build stays replayable from a clean checkout.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SURFACES = ("src/rehab_sci/", "schema/", "models/")
TEXT_SUFFIXES = (".py", ".css", ".yaml", ".yml", ".json", ".html")
# The faces themselves are tracked binaries under a scanned surface.
EXCLUDE = ("src/rehab_sci/dashboard/assets/fonts/",)


def corpus_files() -> list[Path]:
    """Tracked files whose text can reach the DOM."""
    listing = subprocess.run(
        ["git", "ls-files", "-z", *SURFACES],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    return [
        ROOT / rel
        for rel in listing
        if rel.endswith(TEXT_SUFFIXES) and not rel.startswith(EXCLUDE)
    ]


def dashboard_charset() -> frozenset[str]:
    """Every character the corpus can put on screen, printable ASCII included.

    ASCII is unconditional rather than observed: numbers, units and IDs are formatted at
    runtime, so the digits and punctuation a face must carry are not all findable in source.
    """
    chars = {chr(cp) for cp in range(0x20, 0x7F)}
    for path in corpus_files():
        chars.update(path.read_text(encoding="utf-8"))
    return frozenset(chars - set("\t\n\r"))
