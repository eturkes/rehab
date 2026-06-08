"""Startup data loading and global state for the dashboard.

Every other dashboard module imports its globals from here. This module
has no dependencies on other dashboard modules (except theme for template
registration), keeping the import graph acyclic.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from rehab_sci.dashboard.theme import apply_template
from rehab_sci.data.dataset import build_analysis_dataset
from rehab_sci.data.episodes import list_patient_options
from rehab_sci.models.outcomes import OUTCOMES, OutcomeSpec
from rehab_sci.schema import load_schema

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models"

# ---------- one-time startup ----------
apply_template()
SCHEMA = load_schema()
AF = build_analysis_dataset()
EP = AF.df
LONG = AF.longitudinal

with (MODELS_DIR / "training_metrics.json").open(encoding="utf-8") as _f:
    METRICS = json.load(_f)
with (MODELS_DIR / "simulator_defaults.json").open(encoding="utf-8") as _f:
    SIM_DEFAULTS = json.load(_f)
FEATURE_SPEC = joblib.load(MODELS_DIR / "feature_spec.joblib")

DEFAULT_OUTCOME = "scim_total"


def _load_outcome_bundle(spec: OutcomeSpec) -> dict:
    od = MODELS_DIR / spec.key
    bundle: dict = {
        "key": spec.key,
        "spec": spec,
        "task": spec.task,
        "feature_spec": joblib.load(od / "feature_spec.joblib"),
        "shap": joblib.load(od / "shap_test.joblib"),
        "metrics": METRICS["outcomes"][spec.key],
    }
    if spec.task == "regression":
        bundle["median"] = joblib.load(od / "lgbm_median.joblib")
        bundle["p10"] = joblib.load(od / "lgbm_p10.joblib")
        bundle["p90"] = joblib.load(od / "lgbm_p90.joblib")
    elif spec.task == "multiclass":
        bundle["clf"] = joblib.load(od / "lgbm_multiclass.joblib")
    return bundle


OUTCOME_BUNDLES: dict[str, dict] = {s.key: _load_outcome_bundle(s) for s in OUTCOMES}
SCIM_TOTAL_BUNDLE = OUTCOME_BUNDLES[DEFAULT_OUTCOME]

with (MODELS_DIR / "subgroups.json").open(encoding="utf-8") as _f:
    _raw_subgroups = json.load(_f)
if "results" in _raw_subgroups:
    SUBGROUPS: dict[str, dict] = {DEFAULT_OUTCOME: _raw_subgroups}
else:
    SUBGROUPS = _raw_subgroups

_traj_path = MODELS_DIR / "trajectory" / "bundle.joblib"
TRAJECTORY_BUNDLE: dict | None = joblib.load(_traj_path) if _traj_path.exists() else None

_arch_path = MODELS_DIR / "archetypes" / "archetypes.joblib"
ARCHETYPE_DATA: dict | None = joblib.load(_arch_path) if _arch_path.exists() else None

# F23 data-quality scorecard (aggregate, identifier-free). Absent until the report
# is generated (`python -m rehab_sci.data.quality`); the Methods panel degrades gracefully.
_dq_path = MODELS_DIR / "dataquality_summary.json"
if _dq_path.exists():
    with _dq_path.open(encoding="utf-8") as _f:
        DATAQUALITY: dict | None = json.load(_f)
else:
    DATAQUALITY = None

# F24 temporal (out-of-time) validation scorecard. Absent until the backtest is
# generated (`python -m rehab_sci.models.temporal`); the Methods panel degrades gracefully.
_temporal_path = MODELS_DIR / "temporal_metrics.json"
if _temporal_path.exists():
    with _temporal_path.open(encoding="utf-8") as _f:
        TEMPORAL: dict | None = json.load(_f)
else:
    TEMPORAL = None

# G1 landmark (dynamic) prediction. `landmark_metrics.json` (tracked) drives the Methods
# value-of-observation curve; `landmark/bundle.joblib` (gitignored) holds the per-landmark
# models for dynamic inference. Both absent until `python -m rehab_sci.models.landmark`.
_landmark_path = MODELS_DIR / "landmark_metrics.json"
if _landmark_path.exists():
    with _landmark_path.open(encoding="utf-8") as _f:
        LANDMARK: dict | None = json.load(_f)
else:
    LANDMARK = None

_landmark_bundle_path = MODELS_DIR / "landmark" / "bundle.joblib"
LANDMARK_BUNDLE: dict | None = (
    joblib.load(_landmark_bundle_path) if _landmark_bundle_path.exists() else None
)

# G3 observed-trajectory phenotyping (growth mixture model). `phenotype_metrics.json`
# (tracked, identifier-free) records the fit; `phenotypes/phenotypes.joblib` (gitignored)
# carries class-mean curves + per-episode assignments driving the Overview phenotype surface.
# Both absent until `python -m rehab_sci.models.phenotypes`; the panel degrades gracefully.
_pheno_path = MODELS_DIR / "phenotypes" / "phenotypes.joblib"
PHENOTYPE_DATA: dict | None = joblib.load(_pheno_path) if _pheno_path.exists() else None

# G4 AIS-grade conversion modeling. `conversion_metrics.json` (tracked, identifier-free) drives
# the Methods landscape/reliability/driver panels; `conversion/bundle.joblib` (gitignored) holds
# the calibrated endpoint + ordinal-magnitude heads for per-row inference (compute.predict_conversion).
# Both absent until `python -m rehab_sci.models.conversion`; every conversion surface degrades gracefully.
_conversion_path = MODELS_DIR / "conversion_metrics.json"
if _conversion_path.exists():
    with _conversion_path.open(encoding="utf-8") as _f:
        CONVERSION: dict | None = json.load(_f)
else:
    CONVERSION = None

_conversion_bundle_path = MODELS_DIR / "conversion" / "bundle.joblib"
CONVERSION_BUNDLE: dict | None = (
    joblib.load(_conversion_bundle_path) if _conversion_bundle_path.exists() else None
)

# G6 AIS multi-state recovery. `multistate_metrics.json` (tracked, identifier-free) drives the
# Methods cohort-dynamics panels (occupancy / first-passage conversion / transition / sojourn +
# the improve-head calibration & drivers); `multistate/bundle.joblib` (gitignored) holds the
# per-admission-grade Markov curves + the calibrated improve-by-6m head for per-row inference
# (compute.predict_multistate).  Both absent until `python -m rehab_sci.models.multistate`; every
# multi-state surface degrades gracefully.
_multistate_path = MODELS_DIR / "multistate_metrics.json"
if _multistate_path.exists():
    with _multistate_path.open(encoding="utf-8") as _f:
        MULTISTATE: dict | None = json.load(_f)
else:
    MULTISTATE = None

_multistate_bundle_path = MODELS_DIR / "multistate" / "bundle.joblib"
MULTISTATE_BUNDLE: dict | None = (
    joblib.load(_multistate_bundle_path) if _multistate_bundle_path.exists() else None
)

# G7 functional-independence profile. `independence_metrics.json` (tracked, identifier-free) drives
# the Methods scorecard / calibration overlay / SHAP-driver heatmap / per-AIS base-rate landscape +
# the per-item drilldown; `independence/bundle.joblib` (gitignored) holds the 18 calibrated per-item
# heads for per-row inference (compute.predict_independence).  Both absent until
# `python -m rehab_sci.models.independence`; every independence surface degrades gracefully.
_independence_path = MODELS_DIR / "independence_metrics.json"
if _independence_path.exists():
    with _independence_path.open(encoding="utf-8") as _f:
        INDEPENDENCE: dict | None = json.load(_f)
else:
    INDEPENDENCE = None

_independence_bundle_path = MODELS_DIR / "independence" / "bundle.joblib"
INDEPENDENCE_BUNDLE: dict | None = (
    joblib.load(_independence_bundle_path) if _independence_bundle_path.exists() else None
)

# G8 recovery topography map. `topography_metrics.json` (tracked, identifier-free) drives the
# Methods cohort atlas / per-modality calibration / per-segment scorecard + drivers + the
# per-segment drilldown; `topography/bundle.joblib` (gitignored) holds the 132 calibrated
# per-ISNCSCI-segment heads for per-row inference (compute.predict_topography).  Both absent until
# `python -m rehab_sci.models.topography`; every topography surface degrades gracefully.
_topography_path = MODELS_DIR / "topography_metrics.json"
if _topography_path.exists():
    with _topography_path.open(encoding="utf-8") as _f:
        TOPOGRAPHY: dict | None = json.load(_f)
else:
    TOPOGRAPHY = None

_topography_bundle_path = MODELS_DIR / "topography" / "bundle.joblib"
TOPOGRAPHY_BUNDLE: dict | None = (
    joblib.load(_topography_bundle_path) if _topography_bundle_path.exists() else None
)

# G10 neurological-level descent. `level_descent_metrics.json` (tracked, identifier-free) drives
# the Methods scorecard / landscape / per-level reliability+driver+confusion drilldown;
# `level_descent/bundle.joblib` (gitignored) holds the per-level calibrated descent + ordinal
# magnitude heads for per-row inference (compute.predict_level_descent).  Both absent until
# `python -m rehab_sci.models.level_descent`; every level-descent surface degrades gracefully.
_level_descent_path = MODELS_DIR / "level_descent_metrics.json"
if _level_descent_path.exists():
    with _level_descent_path.open(encoding="utf-8") as _f:
        LEVEL_DESCENT: dict | None = json.load(_f)
else:
    LEVEL_DESCENT = None

_level_descent_bundle_path = MODELS_DIR / "level_descent" / "bundle.joblib"
LEVEL_DESCENT_BUNDLE: dict | None = (
    joblib.load(_level_descent_bundle_path) if _level_descent_bundle_path.exists() else None
)

PATIENT_OPTIONS = list_patient_options(EP)
PATIENT_OPTIONS_BY_ID = {p.id_number: p for p in PATIENT_OPTIONS}
