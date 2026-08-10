"""Build the dashboard's self-hosted webfonts: pinned IBM Plex → subset woff2.

Run with ``uv run python scripts/02_build_fonts.py`` (needs the ``dev`` group's
``fonttools[woff]``).  Idempotent: same pins + same fontTools produce byte-identical
output, so the committed ``assets/fonts/`` tree is replayable from a clean base.

Why subset locally instead of vendoring an off-the-shelf webfont build: the Google /
Fontsource ``latin`` subsets drop U+2192 ``→``, U+2248 ``≈``, U+2265 ``≥`` and Greek —
characters the headline metrics and Methods prose use ("43% → 82%", "±23 → ±13", "ΔUEMS"),
which would then render from a fallback face mid-string.  ``UNICODES`` below keeps them.

Uncovered by IBM Plex at any subset: ⇔ ∈ ≧ ≫ ⊇ ▲ (fall back to a system face).
"""

from __future__ import annotations

import io
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "rehab_sci" / "dashboard" / "assets" / "fonts"

REGISTRY = "https://registry.npmjs.org"
# Pinned upstream packages: IBM's own distribution, OFL-1.1.
PACKAGES = {
    "sans": ("@ibm/plex-sans", "1.1.0"),
    "mono": ("@ibm/plex-mono", "2.5.0"),
}
# family → {css weight: upstream style name}.  Ship exactly the weights the CSS asks for.
FACES = {
    "sans": {400: "Regular", 500: "Medium", 600: "SemiBold", 700: "Bold"},
    "mono": {400: "Regular", 600: "SemiBold"},
}
UPSTREAM_FILE = "package/fonts/complete/woff2/IBMPlex{family}-{style}.woff2"
UPSTREAM_LICENSE = "package/LICENSE.txt"

# Everything the bilingual UI renders in a latin face: ASCII, Latin-1/Ext-A, the
# punctuation and math the copy uses, arrows for the →/↔ metrics, Greek for Δ/η/μ.
UNICODES = [
    "0020-007E",  # basic latin
    "00A0-00FF",  # latin-1: § ± ² × · à ï
    "0100-017F",  # latin extended-A
    "0370-03FF",  # greek: Δ Σ α δ η κ μ ρ σ φ (sans only; mono has no greek)
    "2000-206F",  # general punctuation: thin spaces, dashes, quotes, ellipsis, ‰ ′ ″
    "2070-209F",  # super/subscripts
    "20AC",  # €
    "2122",  # ™
    "2190-2199",  # ← ↑ → ↓ ↔
    "2212",  # −
    "2248",  # ≈
    "2260-2265",  # ≠ ≤ ≥
]


def fetch_package(name: str, version: str) -> dict[str, bytes]:
    """Download one pinned npm tarball, returning its member paths → bytes."""
    base = name.split("/")[1]
    url = f"{REGISTRY}/{name}/-/{base}-{version}.tgz"
    print(f"fetch {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        blob = response.read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        return {
            member.name: tar.extractfile(member).read()  # type: ignore[union-attr]
            for member in tar.getmembers()
            if member.isfile()
        }


def subset_face(raw: bytes, out_path: Path) -> None:
    """Cut one complete face down to ``UNICODES``, keeping every layout feature.

    ``tnum`` is load-bearing: the metric readouts and tables ask for tabular figures.
    """
    font = TTFont(io.BytesIO(raw))
    # Keep upstream's head.modified: recalculating stamps build time into the bytes and
    # every rerun would then produce a different file.
    font.recalcTimestamp = False
    options = subset.Options()
    options.layout_features = ["*"]
    options.name_IDs = ["*"]
    options.notdef_outline = True
    options.flavor = "woff2"
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(unicodes=subset.parse_unicodes(",".join(UNICODES)))
    subsetter.subset(font)
    font.flavor = "woff2"
    font.save(out_path)


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    for family, (package, version) in PACKAGES.items():
        members = fetch_package(package, version)
        # One OFL for both families — IBM ships byte-identical license text.
        (OUT / "OFL.txt").write_bytes(members[UPSTREAM_LICENSE])
        for weight, style in FACES[family].items():
            key = UPSTREAM_FILE.format(family=family.capitalize(), style=style)
            out_path = OUT / f"ibm-plex-{family}-{weight}.woff2"
            subset_face(members[key], out_path)
            print(f"  {out_path.name:28s} {out_path.stat().st_size / 1024:6.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
