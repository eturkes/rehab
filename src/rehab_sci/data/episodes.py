"""Per-episode views used by the dashboard's Patient explorer tab.

Pure data shaping; no Plotly or Dash references here.  Returns plain frames
and dicts.  All functions are stateless and idempotent given their inputs.

Key choices baked in here:

* The canonical longitudinal axis is :data:`PATIENT_TIMELINE` — the same
  subset used by ``fig_recovery_curves`` (admission → 6m → discharge).
  Beyond 6m the raw frame is almost entirely empty (see AGENT_NOTES §1),
  so showing them would be visual noise.
* Cohort percentile bands are stratified by **admission** attributes
  (``対麻痺_四肢麻痺``, ``AIS``) from the episode frame, not by each
  longitudinal row's current state.  Clinically the right comparator for
  "where is my patient relative to others like them" is the cohort with
  the same *baseline* injury severity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PATIENT_TIMELINE: list[str] = [
    "0day", "72h", "2w", "4w", "6w",
    "2m", "3m", "4m", "5m", "6m",
    "discharge",
]

# Columns we surface in the per-timepoint patient view.
PATIENT_VIEW_COLS: list[str] = [
    "SCIM_total",
    "SCIM_self_care",
    "SCIM_respiration_sphincter",
    "SCIM_mobility",
    "AIS",
    "AIS_ord",
    "NLI",
    "UEMS",
    "LEMS",
    "TotalMotor",
    "LightTouchTotal",
    "PinPrickTotal",
]


@dataclass(frozen=True)
class PatientOption:
    """One row of the patient picker."""

    id_number: int
    n_episodes: int
    age: float | None
    sex: str | None
    paralysis: str | None
    ais_admit: str | None
    key_records: tuple[int, ...]


def list_patient_options(ep: pd.DataFrame) -> list[PatientOption]:
    """Return one PatientOption per ``IDNumber``, sorted by IDNumber."""
    e = ep.dropna(subset=["IDNumber"]).copy()
    e["IDNumber"] = e["IDNumber"].astype("int64")
    grouped = e.sort_values("KeyRecordNumber").groupby("IDNumber", sort=True)
    out: list[PatientOption] = []
    for pid, g in grouped:
        first = g.iloc[0]
        age_val = first.get("年齢")
        try:
            age = float(age_val) if pd.notna(age_val) else None
        except (TypeError, ValueError):
            age = None
        out.append(
            PatientOption(
                id_number=int(pid),
                n_episodes=int(g.shape[0]),
                age=age,
                sex=None if pd.isna(first.get("性別")) else str(first.get("性別")),
                paralysis=(
                    None if pd.isna(first.get("対麻痺_四肢麻痺"))
                    else str(first.get("対麻痺_四肢麻痺"))
                ),
                ais_admit=None if pd.isna(first.get("AIS")) else str(first.get("AIS")),
                key_records=tuple(int(x) for x in g["KeyRecordNumber"].tolist()),
            )
        )
    return out


def episode_admission_features(ep: pd.DataFrame, key_record: int, feature_cols: list[str]) -> dict:
    """Return a dict of admission features for one episode, defaulting NaN to None.

    Used to seed the model's input vector when computing the per-patient
    prediction + SHAP.  Caller is responsible for filling remaining NaNs
    with simulator defaults if desired.
    """
    row = ep.loc[ep["KeyRecordNumber"] == key_record]
    if row.empty:
        return {}
    rec = row.iloc[0].to_dict()
    return {c: (None if pd.isna(rec.get(c)) else rec.get(c)) for c in feature_cols}


def patient_timeline(long_df: pd.DataFrame, key_record: int) -> pd.DataFrame:
    """Return one row per timepoint in :data:`PATIENT_TIMELINE` for the episode.

    Missing timepoints are still represented (all-NaN row) so the consumer
    can render gaps in the timeline rather than silently re-indexing.
    """
    sub = long_df[long_df["KeyRecordNumber"] == key_record]
    keep_cols = [c for c in PATIENT_VIEW_COLS if c in sub.columns]
    sub = sub[["TIME_Name", *keep_cols]].copy()
    sub["TIME_Name"] = pd.Categorical(sub["TIME_Name"], categories=PATIENT_TIMELINE, ordered=True)
    sub = sub.dropna(subset=["TIME_Name"])
    sub = sub.sort_values("TIME_Name").set_index("TIME_Name")
    # Re-index so every canonical timepoint has a row, even if empty.
    return sub.reindex(PATIENT_TIMELINE)


def patient_meta(ep: pd.DataFrame, key_record: int) -> dict:
    """Demographics + admission injury summary for the meta strip."""
    row = ep.loc[ep["KeyRecordNumber"] == key_record]
    if row.empty:
        return {}
    r = row.iloc[0]
    pid = r.get("IDNumber")
    return {
        "key_record": int(key_record),
        "id_number": None if pd.isna(pid) else int(pid),
        "age": None if pd.isna(r.get("年齢")) else float(r.get("年齢")),
        "sex": None if pd.isna(r.get("性別")) else str(r.get("性別")),
        "paralysis": (
            None if pd.isna(r.get("対麻痺_四肢麻痺"))
            else str(r.get("対麻痺_四肢麻痺"))
        ),
        "ais_admit": None if pd.isna(r.get("AIS")) else str(r.get("AIS")),
        "nli_admit": None if pd.isna(r.get("NLI")) else str(r.get("NLI")),
        "y_discharge_scim": (
            None if pd.isna(r.get("y_discharge_scim"))
            else float(r.get("y_discharge_scim"))
        ),
        "y_discharge_ais": (
            None if pd.isna(r.get("y_discharge_ais"))
            else float(r.get("y_discharge_ais"))
        ),
        "los_days": (
            None if pd.isna(r.get("LOS_days"))
            else float(r.get("LOS_days"))
        ),
    }


def cohort_percentile_bands(
    long_df: pd.DataFrame,
    ep: pd.DataFrame,
    value_col: str,
    group_keys: list[str],
    min_n: int = 5,
    timeline: list[str] | None = None,
) -> pd.DataFrame:
    """Per-(timepoint × group) percentile bands for ``value_col``.

    Stratification keys come from the **episode frame** (admission attrs),
    not the long frame's current-state attrs.  The long frame is left-joined
    onto a small (KeyRecordNumber → group_keys) table before grouping.

    Returns columns: ``[TIME_Name, *group_keys, p10, p25, p50, p75, p90, n]``
    with rows where ``n < min_n`` dropped (too few patients to draw a band).
    """
    if not group_keys:
        raise ValueError("group_keys must contain at least one episode-level key.")
    timeline = timeline or PATIENT_TIMELINE
    # Rename the stratum keys before merging — the long frame carries its own
    # per-row copies of demographics / injury fields, and a naïve merge would
    # produce ``_x`` / ``_y`` suffixed columns.
    band_keys = [f"_band_{k}" for k in group_keys]
    key_table = ep[["KeyRecordNumber", *group_keys]].copy()
    key_table = key_table.rename(columns=dict(zip(group_keys, band_keys, strict=True)))
    key_table = key_table.dropna(subset=band_keys)
    merged = long_df[["KeyRecordNumber", "TIME_Name", value_col]].merge(
        key_table, on="KeyRecordNumber", how="inner"
    )
    merged = merged[merged["TIME_Name"].isin(timeline)]
    merged = merged.dropna(subset=[value_col])
    if merged.empty:
        return pd.DataFrame(columns=["TIME_Name", *group_keys, "p10", "p25", "p50", "p75", "p90", "n"])
    g = merged.groupby(["TIME_Name", *band_keys], observed=True)[value_col]
    out = g.agg(
        p10=lambda s: float(np.nanpercentile(s, 10)),
        p25=lambda s: float(np.nanpercentile(s, 25)),
        p50=lambda s: float(np.nanpercentile(s, 50)),
        p75=lambda s: float(np.nanpercentile(s, 75)),
        p90=lambda s: float(np.nanpercentile(s, 90)),
        n="count",
    ).reset_index()
    out = out[out["n"] >= min_n].copy()
    out = out.rename(columns=dict(zip(band_keys, group_keys, strict=True)))
    out["TIME_Name"] = pd.Categorical(out["TIME_Name"], categories=timeline, ordered=True)
    return out.sort_values(["TIME_Name", *group_keys]).reset_index(drop=True)
