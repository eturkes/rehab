"""Profile ALL_SCIDATA.csv to discover schema, dtypes, missingness, and factor levels.

Writes ONLY descriptive metadata (no patient rows) to ``schema/raw_profile.json``
— a git-ignored, locally-regenerated artifact.  Identifier / bookkeeping columns
(``IDENTIFIER_COLS``) are redacted to existence + missingness only: their value
stats would otherwise expose real medical-record numbers (IDNumber min/max/median)
and record-entry timestamps (STAMP level samples), which must never be committed.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "ALL_SCIDATA.csv"
OUT = ROOT / "schema" / "raw_profile.json"

# Identifier / bookkeeping columns: emit existence + missingness ONLY, never their
# values.  Numeric stats would leak real medical-record numbers (IDNumber) and
# per-row indices; categorical level samples would leak real record-entry
# timestamps (STAMP).  Redacting them keeps this profile genuinely PHI-free.
IDENTIFIER_COLS = {
    "IDNumber",
    "STAMP",
    "KeyRecordNumber",
    "TIMESRecordNumber",
    "AnualCaseNumber",
}


def main() -> int:
    if not RAW.exists():
        print(f"Missing {RAW}", file=sys.stderr)
        return 1

    # encoding detected as cp932 (Shift-JIS superset used in Japan)
    df = pd.read_csv(
        RAW,
        encoding="cp932",
        low_memory=False,
        keep_default_na=False,
        na_values=["", "_"],  # the file uses '_' for missing
    )

    n_rows, n_cols = df.shape
    n_patients = df["IDNumber"].nunique() if "IDNumber" in df.columns else None

    columns: list[dict] = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        # tentative numeric coercion to detect numeric-but-stored-as-object
        coerced = pd.to_numeric(non_null, errors="coerce")
        is_numeric_like = coerced.notna().mean() > 0.95 and len(non_null) > 0

        info: dict = {
            "name": col,
            "dtype": str(s.dtype),
            "n_non_null": int(non_null.size),
            "n_unique": int(non_null.nunique()),
            "missing_pct": round(float(s.isna().mean()) * 100, 2),
            "numeric_like": bool(is_numeric_like),
        }
        if col in IDENTIFIER_COLS:
            info["redacted"] = True  # identifier column — value stats suppressed
        elif is_numeric_like:
            info["numeric"] = {
                "min": (None if coerced.empty else float(coerced.min())),
                "max": (None if coerced.empty else float(coerced.max())),
                "mean": (None if coerced.empty else round(float(coerced.mean()), 3)),
                "median": (None if coerced.empty else float(coerced.median())),
                "n_zero": int((coerced == 0).sum()),
            }
        else:
            # categorical — capture up to 30 levels with frequencies (descriptive only)
            counts: Counter[str] = Counter(non_null.astype(str).tolist())
            top = counts.most_common(30)
            info["categorical"] = {
                "levels_total": len(counts),
                "levels_sample": [{"value": v, "count": c} for v, c in top],
            }
        columns.append(info)

    profile = {
        "source_file": RAW.name,
        "encoding": "cp932",
        "missing_sentinel": "_",
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "n_unique_patients": int(n_patients) if n_patients is not None else None,
        "columns": columns,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}  rows={n_rows} cols={n_cols} patients={n_patients}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
