"""Build the dashboard's self-hosted webfonts: pinned IBM Plex → subset woff2.

Run with ``uv run python scripts/02_build_fonts.py`` (needs the ``dev`` group's
``fonttools[woff]``).  Idempotent: same pins + same fontTools produce byte-identical
output, so the committed ``assets/fonts/`` tree is replayable from a clean base.

Three families, because the UI is bilingual and numeric:

* **Plex Sans** — Latin, Greek, punctuation, arrows, the U+2212 minus.
* **Plex Mono** — the headline metrics and tabular readouts.
* **Plex Sans JP** — kana, kanji and the JIS math symbols (⇔ ∈ ≧ ≫ ⊇ ▲) that a Latin
  face has no reason to carry.  It is the tail of *both* stacks: a JA metric string
  reads ``58% 対 21%``, so even the mono elements need CJK behind them.

Why subset locally instead of vendoring off-the-shelf webfont builds: the Google /
Fontsource ``latin`` subsets drop U+2192 ``→``, U+2248 ``≈``, U+2265 ``≥`` and Greek —
characters the headline metrics and Methods prose use ("43% → 82%", "±23 → ±13", "ΔUEMS")
— and a stock Plex Sans JP is 2.3 MB per weight.  Subsetting to ``RANGES`` ∪ the corpus
in ``font_charset`` keeps every character the repo can render and nothing else;
``tests/test_font_coverage.py`` fails if that promise ever breaks.
"""

from __future__ import annotations

import io
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import NamedTuple

from font_charset import dashboard_charset
from fontTools import subset
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "rehab_sci" / "dashboard" / "assets" / "fonts"

REGISTRY = "https://registry.npmjs.org"


class Family(NamedTuple):
    package: str  # pinned npm package, IBM's own distribution, OFL-1.1
    version: str
    member: str  # tarball path of one complete face, ``{style}``-templated
    weights: dict[int, str]  # css weight → upstream style name


# Ship exactly the weights the CSS asks for: a weight with no face gets synthesised or
# rounded up to the next one that exists.
FAMILIES = {
    "sans": Family(
        "@ibm/plex-sans",
        "1.1.0",
        "package/fonts/complete/woff2/IBMPlexSans-{style}.woff2",
        {400: "Regular", 500: "Medium", 600: "SemiBold", 700: "Bold"},
    ),
    "mono": Family(
        "@ibm/plex-mono",
        "2.5.0",
        "package/fonts/complete/woff2/IBMPlexMono-{style}.woff2",
        {400: "Regular", 600: "SemiBold"},
    ),
    "sans-jp": Family(
        "@ibm/plex-sans-jp",
        "3.0.0",
        "package/fonts/complete/woff2/hinted/IBMPlexSansJP-{style}.woff2",
        {400: "Regular", 500: "Medium", 600: "SemiBold", 700: "Bold"},
    ),
}
# Both packages carry the same OFL text; take it from one so reruns are deterministic.
LICENSE_FROM = "sans"
UPSTREAM_LICENSE = "package/LICENSE.txt"

# Latin-side ranges kept whole rather than corpus-clipped: they cost little, and prose
# picking up a new dash, prime or Greek letter should not have to wait for a font rebuild.
RANGES = [
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
    with urllib.request.urlopen(url, timeout=300) as response:
        blob = response.read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        return {
            member.name: tar.extractfile(member).read()  # type: ignore[union-attr]
            for member in tar.getmembers()
            if member.isfile()
        }


def subset_face(raw: bytes, out_path: Path, unicodes: list[int]) -> None:
    """Cut one complete face down to ``unicodes``, keeping every layout feature.

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
    subsetter.populate(unicodes=unicodes)
    subsetter.subset(font)
    font.flavor = "woff2"
    font.save(out_path)


def main() -> int:
    # A face keeps a character when the corpus needs it or a Latin range covers it; the
    # intersection with what upstream actually draws is fontTools' job.
    wanted = sorted(
        {ord(ch) for ch in dashboard_charset()} | set(subset.parse_unicodes(",".join(RANGES)))
    )
    print(f"corpus + ranges: {len(wanted)} codepoints")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    total = 0
    for name, family in FAMILIES.items():
        members = fetch_package(family.package, family.version)
        if name == LICENSE_FROM:
            (OUT / "OFL.txt").write_bytes(members[UPSTREAM_LICENSE])
        for weight, style in family.weights.items():
            out_path = OUT / f"ibm-plex-{name}-{weight}.woff2"
            subset_face(members[family.member.format(style=style)], out_path, wanted)
            size = out_path.stat().st_size
            total += size
            print(f"  {out_path.name:28s} {size / 1024:7.1f} KB")
    print(f"  {'total':28s} {total / 1024:7.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
