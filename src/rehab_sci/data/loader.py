"""ALL_SCIDATA.csv loader + cleaner. Patient data is held in-memory only — NEVER persisted to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from rehab_sci.schema import Schema, load_schema

RAW_PATH_DEFAULT: Final[Path] = Path(__file__).resolve().parents[3] / "ALL_SCIDATA.csv"

# Cord level → ordinal index (smaller = more rostral / more severe lesion when used as NLI).
_CORD_ORDER: Final[list[str]] = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
    "L1", "L2", "L3", "L4", "L5",
    "S1", "S2", "S3", "S45",
]
_CORD_INDEX: Final[dict[str, int]] = {lvl: i for i, lvl in enumerate(_CORD_ORDER)}
_AIS_NUMERIC: Final[dict[str, int]] = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}


def cord_level_to_int(level: str | None) -> float:
    if level is None or (isinstance(level, float) and np.isnan(level)):
        return np.nan
    return float(_CORD_INDEX.get(str(level).strip(), np.nan))


def ais_to_int(grade: str | None) -> float:
    if grade is None or (isinstance(grade, float) and np.isnan(grade)):
        return np.nan
    return float(_AIS_NUMERIC.get(str(grade).strip(), np.nan))


def _coerce_numeric(s: pd.Series, allow_bool: bool = False) -> pd.Series:
    """Coerce a column to numeric. If ``allow_bool``, FALSE/TRUE/NT are mapped to 0/1/NaN."""
    if allow_bool:
        m = {
            "0": 0.0, "1": 0.0 if False else 1.0,
            "FALSE": 0.0, "False": 0.0, "false": 0.0,
            "TRUE": 1.0, "True": 1.0, "true": 1.0,
            "NT": np.nan, "_": np.nan, "": np.nan, np.nan: np.nan,
        }
        return s.map(lambda v: m.get(str(v) if v is not np.nan else v, np.nan)).astype(float)
    return pd.to_numeric(s, errors="coerce")


def _split_mfrankel(val: str | float) -> tuple[float, float]:
    """Split 'X/Y' modified-Frankel / Frankel pair into ordinal codes.

    mFrankel order: A,B1,B2,B3,C1,C2,D0,D1,D2,D3,E   →   1..11
    Frankel order:  A,B,C,D,E                         →   1..5
    """
    mfrankel_order = ["A", "B1", "B2", "B3", "C1", "C2", "D0", "D1", "D2", "D3", "E"]
    frankel_order = ["A", "B", "C", "D", "E"]
    m_idx = {k: i + 1 for i, k in enumerate(mfrankel_order)}
    f_idx = {k: i + 1 for i, k in enumerate(frankel_order)}
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return (np.nan, np.nan)
    s = str(val).strip()
    if s in ("_", "_/_", ""):
        return (np.nan, np.nan)
    parts = s.split("/")
    if len(parts) == 1:
        parts = [parts[0], parts[0]]
    a, b = parts[0].strip(), parts[1].strip()
    return (float(m_idx.get(a, np.nan)), float(f_idx.get(b, np.nan)))


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Load raw CSV with cp932 encoding. Patient data stays in-memory only."""
    p = Path(path) if path else RAW_PATH_DEFAULT
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Please place ALL_SCIDATA.csv in the project root."
        )
    return pd.read_csv(
        p,
        encoding="cp932",
        low_memory=False,
        keep_default_na=False,
        na_values=["", "_", "NA"],
    )


def normalize(df: pd.DataFrame, schema: Schema | None = None) -> pd.DataFrame:
    """Apply schema-driven cleaning: dtypes, level normalization, derived columns."""
    schema = schema or load_schema()
    df = df.copy()

    # ---- per-column normalization ----
    for col in df.columns:
        spec = schema.by_raw(col)
        if spec is None:
            continue

        if spec.dtype == "numeric" or spec.dtype == "ordinal":
            # NonKeyMuscle was stored as 0/FALSE/TRUE — treat as boolean
            allow_bool = spec.family == "non_key_muscle"
            df[col] = _coerce_numeric(df[col], allow_bool=allow_bool)
            if spec.range is not None:
                lo, hi = spec.range
                df[col] = df[col].mask((df[col] < lo) | (df[col] > hi))
        elif spec.dtype == "categorical" and spec.levels:
            df[col] = (
                df[col]
                .astype("string")
                .map(lambda v: schema.normalize_level(spec.levels, v) if pd.notna(v) else pd.NA)
            )
        elif spec.dtype == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ---- derived columns, batched to avoid fragmentation ----
    new_cols: dict[str, pd.Series] = {}

    if "mFrankel_Frankel" in df.columns:
        pairs = df["mFrankel_Frankel"].map(_split_mfrankel)
        new_cols["mFrankel_ord"] = pairs.map(lambda x: x[0])
        new_cols["Frankel_ord"] = pairs.map(lambda x: x[1])

    for src, tgt in [
        ("NLI", "NLI_ord"),
        ("NLI_manual", "NLI_manual_ord"),
        ("RightSensoryLevel", "RightSensoryLevel_ord"),
        ("LeftSensoryLevel", "LeftSensoryLevel_ord"),
        ("RightMotorLevel", "RightMotorLevel_ord"),
        ("LeftMotorLevel", "LeftMotorLevel_ord"),
    ]:
        if src in df.columns:
            new_cols[tgt] = df[src].map(cord_level_to_int)

    if "AIS" in df.columns:
        new_cols["AIS_ord"] = df["AIS"].map(ais_to_int)
    if "AIS_manual" in df.columns:
        new_cols["AIS_manual_ord"] = df["AIS_manual"].map(ais_to_int)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def add_isncsci_summaries(df: pd.DataFrame, schema: Schema | None = None) -> pd.DataFrame:
    """Compute UEMS / LEMS / total motor / per-modality sensory totals per row.

    UEMS = sum of bilateral key-muscle scores at C5..T1 (max 50).
    LEMS = sum of bilateral key-muscle scores at L2..S1 (max 50).
    Total motor = UEMS + LEMS (max 100).
    Light-touch total = sum over 28 dermatomes × 2 sides (max 112). Same for pin-prick.
    """
    schema = schema or load_schema()

    upper = ["C5", "C6", "C7", "C8", "T1"]
    lower = ["L2", "L3", "L4", "L5", "S1"]

    def _sum_existing(cols: list[str]) -> pd.Series:
        existing = [c for c in cols if c in df.columns]
        if not existing:
            return pd.Series(np.nan, index=df.index)
        # row is valid if at least half of the cells are present
        mat = df[existing]
        return mat.sum(axis=1, min_count=max(1, len(existing) // 2))

    upper_cols = [f"{side}KeyMuscle{lvl}" for side in ("Right", "Left") for lvl in upper]
    lower_cols = [f"{side}KeyMuscle{lvl}" for side in ("Right", "Left") for lvl in lower]
    uems = _sum_existing(upper_cols)
    lems = _sum_existing(lower_cols)
    total_motor = uems.fillna(0) + lems.fillna(0)
    total_motor = total_motor.where(uems.notna() | lems.notna())

    derms = schema.isncsci["sensory"]["dermatomes"]
    lt_cols = [f"{side}LightTouch{lvl}" for side in ("Right", "Left") for lvl in derms]
    pp_cols = [f"{side}PinPrick{lvl}" for side in ("Right", "Left") for lvl in derms]
    lt_total = _sum_existing(lt_cols)
    pp_total = _sum_existing(pp_cols)

    return pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "UEMS": uems,
                    "LEMS": lems,
                    "TotalMotor": total_motor,
                    "LightTouchTotal": lt_total,
                    "PinPrickTotal": pp_total,
                },
                index=df.index,
            ),
        ],
        axis=1,
    )


def add_scim_subscales(df: pd.DataFrame, schema: Schema | None = None) -> pd.DataFrame:
    """Compute SCIM-III sub-scale and total scores."""
    schema = schema or load_schema()
    subs: dict[str, pd.Series] = {}
    for key, sub in schema.scim["subscales"].items():
        items = [c for c in sub["items"] if c in df.columns]
        if not items:
            subs[f"SCIM_{key}"] = pd.Series(np.nan, index=df.index)
            continue
        # require at least half of items present per row
        subs[f"SCIM_{key}"] = df[items].sum(axis=1, min_count=max(1, len(items) // 2))
    new_df = pd.DataFrame(subs, index=df.index)
    new_df["SCIM_total"] = new_df.sum(axis=1, min_count=1)
    return pd.concat([df, new_df], axis=1)


def load_clean(path: Path | str | None = None, schema: Schema | None = None) -> pd.DataFrame:
    """Public entrypoint: load → normalize → add ISNCSCI summaries → add SCIM subscales."""
    schema = schema or load_schema()
    df = load_raw(path)
    df = normalize(df, schema=schema)
    df = add_isncsci_summaries(df, schema=schema)
    df = add_scim_subscales(df, schema=schema)
    return df
