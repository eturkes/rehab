"""Outcome registry — the source of truth for what `train.py` predicts.

Each :class:`OutcomeSpec` declares one prediction target.  The training
pipeline iterates over :data:`OUTCOMES` and persists one bundle per spec
under ``models/{spec.key}/``.  The dashboard simulator reads the same
registry and renders an outcome-selector.

Six outcomes are currently registered:

* Four regression heads — SCIM-III total + the three subscales — share
  the same admission-feature matrix and use split-conformal 80 % PIs.
* One ordinal-classification head — AIS A→E at discharge — uses
  LightGBM multiclass with classes sorted by severity (A=1 … E=5) so
  that ordinality is at least preserved in the column ordering of
  ``predict_proba``.  Ordinal-aware metrics (quadratic-weighted κ,
  MAE-on-ordinal-code) are reported alongside accuracy.
* One log-scale regression head — length-of-stay in days — uses a
  ``log1p`` transform so the conformal residuals are symmetric on the
  natural scale of the variable (LOS is right-skewed: 1–788 d, median
  ≈ 140 d).  Predictions and PI bounds are back-transformed to days
  before clipping.

WISCI is intentionally absent (only 50 episodes have a discharge value
post-ghost-filter — below regression power; see AGENT_NOTES §1).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OutcomeSpec:
    key: str  # artifact prefix + dashboard selector value
    target_col: str  # column name on the episode frame
    task: str  # "regression" | "multiclass"
    display_key: str  # ui_strings.yaml key (rendered per-lang)
    unit_key: str | None = None  # ui_strings.yaml key for units (regression)
    clip_min: float | None = None  # applied to the back-transformed prediction
    clip_max: float | None = None
    transform: str | None = None  # None | "log1p"
    class_codes: tuple[int, ...] = field(default_factory=tuple)  # multiclass only
    class_labels: tuple[str, ...] = field(default_factory=tuple)  # display letters


OUTCOMES: tuple[OutcomeSpec, ...] = (
    OutcomeSpec(
        key="scim_total",
        target_col="y_discharge_scim",
        task="regression",
        display_key="outcome_scim_total",
        unit_key="unit_score",
        clip_min=0.0,
        clip_max=100.0,
    ),
    OutcomeSpec(
        key="scim_self_care",
        target_col="y_discharge_scim_self_care",
        task="regression",
        display_key="outcome_scim_self_care",
        unit_key="unit_score",
        clip_min=0.0,
        clip_max=20.0,
    ),
    OutcomeSpec(
        key="scim_resp_sphincter",
        target_col="y_discharge_scim_resp_sphincter",
        task="regression",
        display_key="outcome_scim_resp_sphincter",
        unit_key="unit_score",
        clip_min=0.0,
        clip_max=40.0,
    ),
    OutcomeSpec(
        key="scim_mobility",
        target_col="y_discharge_scim_mobility",
        task="regression",
        display_key="outcome_scim_mobility",
        unit_key="unit_score",
        clip_min=0.0,
        clip_max=40.0,
    ),
    OutcomeSpec(
        key="ais_discharge",
        target_col="y_discharge_ais",
        task="multiclass",
        display_key="outcome_ais_discharge",
        unit_key=None,
        class_codes=(1, 2, 3, 4, 5),
        class_labels=("A", "B", "C", "D", "E"),
    ),
    OutcomeSpec(
        key="los_days",
        target_col="LOS_days",
        task="regression",
        display_key="outcome_los_days",
        unit_key="unit_days",
        clip_min=0.0,
        clip_max=None,
        transform="log1p",
    ),
)


def get(key: str) -> OutcomeSpec:
    for spec in OUTCOMES:
        if spec.key == key:
            return spec
    raise KeyError(f"unknown outcome key: {key!r}")
