"""Construct the analysis-ready frame: one row per patient-episode.

The unit of analysis is ``KeyRecordNumber``.  The raw CSV ships a perfect
1200 × 26 grid (1200 episodes × 26 timepoint slots = 31,200 rows), but 301 of
those episodes are pure placeholder rows: ``IDNumber`` is null AND every
admission feature is null AND every outcome is null.  They contribute nothing
to modelling or cohort statistics, so :func:`build_analysis_dataset` filters
them out at load time.  The post-filter universe is 899 episodes (the trained
model already excluded them via ``dropna(IDNumber, outcome)``; this filter
just makes the cohort counts honest too).

Admission features are taken from the ``0day`` timepoint, with backfill from later
acute-phase timepoints (72h, 2w, 4w) so that missing baseline rows do not silently
drop patients.  The outcome ``y_discharge_scim`` is the ``discharge`` timepoint
SCIM-III total.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rehab_sci.data.loader import load_clean
from rehab_sci.schema import Schema, load_schema

# Order in which we look for an admission-time value.  First non-null wins.
ADMISSION_FALLBACK: list[str] = ["0day", "72h", "2w", "4w"]

# Features the model is allowed to see.  Strictly admission-only — never leak future info.
ADMISSION_FEATURES: list[str] = [
    # demographics
    "年齢",
    "性別",
    # injury context
    "外傷性_非外傷性",
    "対麻痺_四肢麻痺",
    "糖尿病",
    "OPLL",
    "DISH",
    "ALLEN分類",
    "損傷部位",
    "受傷機転",
    "保険",
    # ISNCSCI / AIS
    "AIS_ord",
    "mFrankel_ord",
    "Frankel_ord",
    "NLI_ord",
    "RightSensoryLevel_ord",
    "LeftSensoryLevel_ord",
    "RightMotorLevel_ord",
    "LeftMotorLevel_ord",
    "VAC",
    "DAP",
    "UEMS",
    "LEMS",
    "TotalMotor",
    "LightTouchTotal",
    "PinPrickTotal",
    # admission ADL baseline (some patients have early SCIM)
    "SCIM_self_care",
    "SCIM_respiration_sphincter",
    "SCIM_mobility",
    "SCIM_total",
]

# Numeric & categorical splits — needed for model encoders.
NUMERIC_FEATURES: list[str] = [
    "年齢",
    "AIS_ord",
    "mFrankel_ord",
    "Frankel_ord",
    "NLI_ord",
    "RightSensoryLevel_ord",
    "LeftSensoryLevel_ord",
    "RightMotorLevel_ord",
    "LeftMotorLevel_ord",
    "UEMS",
    "LEMS",
    "TotalMotor",
    "LightTouchTotal",
    "PinPrickTotal",
    "SCIM_self_care",
    "SCIM_respiration_sphincter",
    "SCIM_mobility",
    "SCIM_total",
]
CATEGORICAL_FEATURES: list[str] = [
    "性別",
    "外傷性_非外傷性",
    "対麻痺_四肢麻痺",
    "糖尿病",
    "OPLL",
    "DISH",
    "ALLEN分類",
    "損傷部位",
    "受傷機転",
    "保険",
    "VAC",
    "DAP",
]


@dataclass
class AnalysisFrame:
    df: pd.DataFrame
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    outcome_col: str
    longitudinal: pd.DataFrame
    schema: Schema


def _first_non_null(group: pd.DataFrame, col: str, order: list[str]) -> object:
    timepoint_to_row = {row["TIME_Name"]: row for _, row in group.iterrows()}
    for tp in order:
        row = timepoint_to_row.get(tp)
        if row is not None and pd.notna(row.get(col)):
            return row[col]
    return pd.NA


def build_episode_frame(longitudinal: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long longitudinal frame to one row per episode (KeyRecordNumber)."""
    grouped = longitudinal.groupby("KeyRecordNumber", sort=True)

    # vectorized first-non-null over the admission fallback timepoints for each feature
    parts: list[pd.DataFrame] = []
    for tp in ADMISSION_FALLBACK:
        sl = longitudinal[longitudinal["TIME_Name"] == tp].set_index("KeyRecordNumber")
        parts.append(sl)

    # union of episode IDs across the early timepoints
    episodes = sorted(set().union(*[set(p.index) for p in parts]))
    episode_idx = pd.Index(episodes, name="KeyRecordNumber")

    feat = pd.DataFrame(index=episode_idx)
    for col in ADMISSION_FEATURES:
        for p in parts:
            if col not in p.columns:
                continue
            s = p[col].reindex(episode_idx)
            if col not in feat.columns:
                feat[col] = s
            else:
                feat[col] = feat[col].where(feat[col].notna(), s)

    # passport: patient ID + a couple of stable demographics for joining/reporting
    pid = grouped["IDNumber"].first().reindex(episode_idx)
    feat["IDNumber"] = pid

    # business year of the episode — meta only (temporal-validation split key);
    # never a model feature (absent from ADMISSION_FEATURES → never in feature_cols).
    feat["BusinessYear"] = grouped["BusinessYear"].first().reindex(episode_idx)

    # raw letter / level columns kept for visualization (NOT used as model features)
    for col in ("AIS", "NLI"):
        for p in parts:
            if col not in p.columns:
                continue
            s = p[col].reindex(episode_idx)
            if col not in feat.columns:
                feat[col] = s
            else:
                feat[col] = feat[col].where(feat[col].notna(), s)

    # outcomes at discharge: SCIM total + 3 subscales + AIS ordinal + (sparse) WISCI.
    discharge = longitudinal[longitudinal["TIME_Name"] == "discharge"].set_index("KeyRecordNumber")
    feat["y_discharge_scim"] = discharge["SCIM_total"].reindex(episode_idx)
    feat["y_discharge_scim_self_care"] = discharge["SCIM_self_care"].reindex(episode_idx)
    feat["y_discharge_scim_resp_sphincter"] = discharge["SCIM_respiration_sphincter"].reindex(episode_idx)
    feat["y_discharge_scim_mobility"] = discharge["SCIM_mobility"].reindex(episode_idx)
    feat["y_discharge_ais"] = discharge["AIS_ord"].reindex(episode_idx)
    feat["y_discharge_wisci"] = discharge["WalkingIndex"].reindex(episode_idx)
    max_scim = grouped["SCIM_total"].max().reindex(episode_idx)
    feat["y_max_scim"] = max_scim
    last_scim_idx = (
        longitudinal.dropna(subset=["SCIM_total"]).sort_values("TIMES").groupby("KeyRecordNumber").tail(1)
    )
    feat["y_last_scim"] = last_scim_idx.set_index("KeyRecordNumber")["SCIM_total"].reindex(episode_idx)
    feat["y_last_timepoint"] = last_scim_idx.set_index("KeyRecordNumber")["TIME_Name"].reindex(episode_idx)

    # baseline (0day) — for the simulator default values
    baseline = longitudinal[longitudinal["TIME_Name"] == "0day"].set_index("KeyRecordNumber")
    feat["baseline_scim"] = baseline["SCIM_total"].reindex(episode_idx)

    # length of stay (already on every row but constant per patient)
    feat["LOS_days"] = grouped["入院期間"].first().reindex(episode_idx)

    return feat.reset_index()


def _identify_ghost_episodes(ep: pd.DataFrame, admission_features: list[str]) -> set[int]:
    """Return KeyRecordNumbers of pure placeholder episodes.

    A ghost episode is one where ``IDNumber`` is null AND every admission
    feature is null.  Empirically, the 301 episodes matching this rule also
    have null outcomes (SCIM / AIS / WISCI / LOS) across every timepoint, so
    they contribute nothing modellable or visualizable.  The trained model
    already excludes them via ``dropna(IDNumber, outcome)``; filtering them
    here additionally fixes the cohort-level episode counts.
    """
    cols = [c for c in admission_features if c in ep.columns]
    if not cols:
        return set()
    is_ghost = ep["IDNumber"].isna() & ep[cols].isna().all(axis=1)
    return set(ep.loc[is_ghost, "KeyRecordNumber"].tolist())


def build_analysis_dataset() -> AnalysisFrame:
    schema = load_schema()
    long_df = load_clean(schema=schema)
    ep = build_episode_frame(long_df)
    ghost_krs = _identify_ghost_episodes(ep, ADMISSION_FEATURES)
    if ghost_krs:
        ep = ep[~ep["KeyRecordNumber"].isin(ghost_krs)].reset_index(drop=True)
        long_df = long_df[~long_df["KeyRecordNumber"].isin(ghost_krs)].reset_index(drop=True)
    feature_cols = ADMISSION_FEATURES.copy()
    # ensure no leakage: outcome columns are not in feature_cols
    feature_cols = [c for c in feature_cols if c in ep.columns]
    return AnalysisFrame(
        df=ep,
        feature_cols=feature_cols,
        numeric_cols=[c for c in NUMERIC_FEATURES if c in ep.columns],
        categorical_cols=[c for c in CATEGORICAL_FEATURES if c in ep.columns],
        outcome_col="y_discharge_scim",
        longitudinal=long_df,
        schema=schema,
    )


def _replace_nan_to_none(o):  # for JSON dumps
    if isinstance(o, float) and np.isnan(o):
        return None
    return o
