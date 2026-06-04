"""Shared domain constants — single source of truth (imports nothing from the project)."""

from __future__ import annotations

from typing import Final

# AIS grade ↔ ordinal severity code (A=1 … E=5).  data/loader.py builds the
# AIS_ord column from the forward map; models + dashboard invert it to label
# predictions.  Both directions derive from one literal to prevent drift.
AIS_LETTER_TO_ORD: Final[dict[str, int]] = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
AIS_ORD_TO_LETTER: Final[dict[int, str]] = {v: k for k, v in AIS_LETTER_TO_ORD.items()}
