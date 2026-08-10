"""Typography gate: every character the UI can render resolves to a shipped face.

Unlike the rest of the harness these tests need no data — the corpus is the committed
schema, dashboard source and model JSON, so they run on a bare checkout.

The failure they exist to catch is silent.  A CSS stack that names a family the host
lacks does not error; the browser substitutes, and the substitute can be a serif in a
sans UI or VL Gothic in a Plex one.  Nothing in a render test notices.  So the promise
is checked at its source: corpus ⊆ ⋃ cmaps of the woff2 files actually committed, and
every ``font-family`` the CSS declares ends in a family those files define.
"""

from __future__ import annotations

import re
import unicodedata

import pytest
from font_charset import ROOT, dashboard_charset
from fontTools.ttLib import TTFont

FONTS = ROOT / "src" / "rehab_sci" / "dashboard" / "assets" / "fonts"
CSS = ROOT / "src" / "rehab_sci" / "dashboard" / "assets" / "style.css"

# Stack tails the browser resolves without a download.  ``sans-serif`` / ``monospace``
# are the generic last resorts every stack is allowed to end on; anything else named
# after the shipped families would be an unshipped dependency on the host.
GENERIC = {"sans-serif", "monospace", "serif", "system-ui", "ui-monospace"}


def shipped_faces() -> dict[str, frozenset[int]]:
    """Committed woff2 filename → the codepoints it draws."""
    faces = {}
    for path in sorted(FONTS.glob("*.woff2")):
        font = TTFont(path, lazy=True)
        cmap: set[int] = set()
        for table in font["cmap"].tables:
            cmap |= set(table.cmap)
        faces[path.name] = frozenset(cmap)
    return faces


@pytest.fixture(scope="module")
def faces() -> dict[str, frozenset[int]]:
    found = shipped_faces()
    assert found, f"no woff2 committed under {FONTS} — run scripts/02_build_fonts.py"
    return found


def test_every_character_has_a_shipped_glyph(faces):
    """No character in the corpus may depend on a font the user happens to have."""
    covered: set[int] = set().union(*faces.values())
    missing = sorted({ch for ch in dashboard_charset() if ord(ch) not in covered}, key=ord)
    assert not missing, "characters with no shipped glyph (they render in a host font):\n" + "\n".join(
        f"  U+{ord(ch):04X} {ch} {unicodedata.name(ch, '?')}" for ch in missing
    )


def test_stacks_cover_their_own_scripts(faces):
    """Each stack must carry CJK itself, not lean on the next stack's face.

    ``--font-mono`` is the one that bites: the JA headline metric is ``58% 対 21%``, so a
    mono stack of Latin faces alone drops a kanji into a host font mid-number.
    """
    css = CSS.read_text(encoding="utf-8")
    families = {
        name: frozenset().union(
            *(cmap for face, cmap in faces.items() if face.startswith(f"ibm-plex-{name}-"))
        )
        for name in ("sans", "mono", "sans-jp")
    }
    declared = {
        "sans": families["sans"] | families["sans-jp"],
        "mono": families["mono"] | families["sans-jp"],
    }
    for var, covered in declared.items():
        assert f"--font-{var}:" in css, f"style.css no longer declares --font-{var}"
        missing = sorted({ch for ch in dashboard_charset() if ord(ch) not in covered}, key=ord)
        assert not missing, (
            f"--font-{var} cannot render {len(missing)} character(s): "
            + "".join(missing[:20])
        )


def test_no_stack_names_an_unshipped_family():
    """A stack may only name families we ship, plus the CSS generics."""
    css = CSS.read_text(encoding="utf-8")
    defined = set(re.findall(r'@font-face\s*\{[^}]*?font-family:\s*"([^"]+)"', css, re.S))
    offenders: dict[str, list[str]] = {}
    for var, value in re.findall(r"--font-(sans|mono):([^;]+);", css):
        named = [f.strip().strip('"').strip() for f in value.split(",")]
        unknown = [f for f in named if f and f not in defined and f not in GENERIC]
        if unknown:
            offenders[var] = unknown
    assert not offenders, f"stacks name families we do not ship: {offenders}"


def test_faces_declare_matching_line_metrics(faces):
    """The JP face must lay out on the Latin face's line box.

    Plex Sans JP ships hhea 1060/-440 against Plex Sans' 1025/-275; mixed inline it would
    stretch every line holding a kanji.  The CSS corrects that with ``ascent-override`` /
    ``descent-override``, so the overrides have to match the Latin face's real metrics.
    """
    latin = TTFont(FONTS / "ibm-plex-sans-400.woff2", lazy=True)
    upm = latin["head"].unitsPerEm
    expected = {
        "ascent-override": round(100 * latin["hhea"].ascent / upm, 1),
        "descent-override": round(100 * -latin["hhea"].descent / upm, 1),
        "line-gap-override": round(100 * latin["hhea"].lineGap / upm, 1),
    }
    css = CSS.read_text(encoding="utf-8")
    jp_blocks = [
        block
        for block in re.findall(r"@font-face\s*\{(.*?)\}", css, re.S)
        if "ibm-plex-sans-jp-" in block
    ]
    assert len(jp_blocks) == 4, f"expected 4 JP @font-face blocks, found {len(jp_blocks)}"
    for block in jp_blocks:
        for prop, want in expected.items():
            found = re.search(rf"{prop}:\s*([\d.]+)%", block)
            assert found, f"JP @font-face missing {prop}"
            assert float(found.group(1)) == want, (
                f"{prop} is {found.group(1)}%, Plex Sans needs {want}%"
            )
