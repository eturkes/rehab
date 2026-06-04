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

PATIENT_OPTIONS = list_patient_options(EP)
PATIENT_OPTIONS_BY_ID = {p.id_number: p for p in PATIENT_OPTIONS}
