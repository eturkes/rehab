"""Pure computation helpers for model inference, conformal PI, and SHAP.

Shared by simulator, patient, and report callbacks. No Dash or Plotly deps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rehab_sci.constants import AIS_ORD_TO_LETTER
from rehab_sci.dashboard.state import (
    EP,
    FEATURE_SPEC,
    OUTCOME_BUNDLES,
    SIM_DEFAULTS,
    TRAJECTORY_BUNDLE,
)
from rehab_sci.data.episodes import episode_admission_features
from rehab_sci.models.outcomes import OUTCOMES, OutcomeSpec


# ---------- conformal q resolution ----------
def resolve_group_q(q_by_group: dict | None, marginal: float, X: pd.DataFrame) -> float:
    """Resolve Mondrian q for a single-row input.

    Priority: AIS group -> paralysis group -> marginal.
    """
    if q_by_group is None or len(X) == 0:
        return marginal
    row = X.iloc[0]
    ais_qs = q_by_group.get("ais", {})
    if ais_qs and "AIS_ord" in X.columns:
        ais_val = row["AIS_ord"]
        if pd.notna(ais_val):
            letter = AIS_ORD_TO_LETTER.get(int(ais_val))
            if letter and letter in ais_qs:
                return float(ais_qs[letter])
    para_qs = q_by_group.get("paralysis", {})
    if para_qs and "対麻痺_四肢麻痺" in X.columns:
        para_val = row["対麻痺_四肢麻痺"]
        if pd.notna(para_val) and str(para_val) in para_qs:
            return float(para_qs[str(para_val)])
    return marginal


def resolve_conformal_q(fspec: dict, X: pd.DataFrame) -> float:
    return resolve_group_q(
        fspec.get("conformal_q_by_group"),
        float(fspec.get("conformal_q_transformed", 0.0)),
        X,
    )


def resolve_aps_q(fspec: dict, X: pd.DataFrame) -> float:
    return resolve_group_q(
        fspec.get("aps_q_by_group"),
        float(fspec.get("aps_q_hat", 1.0)),
        X,
    )


# ---------- trajectory ----------
def predict_trajectory(X: pd.DataFrame) -> dict | None:
    """Predict SCIM-total at each trajectory timepoint for a single-row input."""
    if TRAJECTORY_BUNDLE is None:
        return None
    tps = TRAJECTORY_BUNDLE["timepoints"]
    models = TRAJECTORY_BUNDLE["models"]
    conf = TRAJECTORY_BUNDLE["conformal"]
    clip_min = TRAJECTORY_BUNDLE.get("clip_min", 0.0)
    clip_max = TRAJECTORY_BUNDLE.get("clip_max", 100.0)

    out_tps: list[str] = []
    out_pred: list[float] = []
    out_lo: list[float] = []
    out_hi: list[float] = []

    for tp in tps:
        tp_models = models[tp]
        tp_conf = conf[tp]
        pred_t = float(tp_models["median"].predict(X)[0])
        pred = max(clip_min, min(clip_max, pred_t))
        q_t = resolve_group_q(
            tp_conf["q_by_group"], float(tp_conf["q_transformed"]), X,
        )
        lo_conf = max(clip_min, min(clip_max, pred_t - q_t))
        hi_conf = max(clip_min, min(clip_max, pred_t + q_t))
        lo_q = max(clip_min, min(clip_max, float(tp_models["p10"].predict(X)[0])))
        hi_q = max(clip_min, min(clip_max, float(tp_models["p90"].predict(X)[0])))

        out_tps.append(tp)
        out_pred.append(pred)
        out_lo.append(min(lo_conf, lo_q))
        out_hi.append(max(hi_conf, hi_q))

    return {"timepoints": out_tps, "pred": out_pred, "lo": out_lo, "hi": out_hi}


# ---------- APS ----------
def aps_prediction_set(proba_row: np.ndarray, q_hat: float) -> list[int]:
    order = np.argsort(-proba_row)
    cumsum = 0.0
    pred_set: list[int] = []
    for j in order:
        pred_set.append(int(j))
        cumsum += proba_row[j]
        if cumsum >= q_hat:
            break
    return sorted(pred_set)


# ---------- scalar transforms ----------
def inv_transform_scalar(x: float, transform: str | None) -> float:
    if transform == "log1p":
        return float(np.expm1(x))
    return float(x)


def clip_scalar(x: float, lo: float | None, hi: float | None) -> float:
    if lo is not None and x < lo:
        x = lo
    if hi is not None and x > hi:
        x = hi
    return x


def format_value(col: str, value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    if isinstance(value, float):
        return f"{value:.0f}"
    return str(value)


# ---------- reference predictions (What-if, PDF report) ----------
def compute_ref_predictions(X: pd.DataFrame) -> dict:
    """Compute predictions for all outcomes on a single-row X."""
    outcomes: dict[str, dict] = {}
    for spec in OUTCOMES:
        bundle = OUTCOME_BUNDLES[spec.key]
        fspec = bundle["feature_spec"]
        if spec.task == "regression":
            transform = fspec.get("transform")
            cmin = fspec.get("clip_min")
            cmax = fspec.get("clip_max")
            q_t = resolve_conformal_q(fspec, X)
            pred_t = float(bundle["median"].predict(X)[0])
            pred_p10_t = float(bundle["p10"].predict(X)[0])
            pred_p90_t = float(bundle["p90"].predict(X)[0])
            pred = clip_scalar(inv_transform_scalar(pred_t, transform), cmin, cmax)
            lo_conf = clip_scalar(inv_transform_scalar(pred_t - q_t, transform), cmin, cmax)
            hi_conf = clip_scalar(inv_transform_scalar(pred_t + q_t, transform), cmin, cmax)
            lo_q = clip_scalar(inv_transform_scalar(pred_p10_t, transform), cmin, cmax)
            hi_q = clip_scalar(inv_transform_scalar(pred_p90_t, transform), cmin, cmax)
            outcomes[spec.key] = {
                "task": "regression",
                "pred": pred,
                "lo": min(lo_conf, lo_q),
                "hi": max(hi_conf, hi_q),
            }
        else:
            class_labels = list(fspec.get("class_labels", spec.class_labels))
            proba = np.asarray(bundle["clf"].predict_proba(X)[0], dtype=float)
            pred_idx = int(np.argmax(proba))
            outcomes[spec.key] = {
                "task": "multiclass",
                "pred_class": class_labels[pred_idx],
                "pred_prob": float(proba[pred_idx]),
                "proba": proba.tolist(),
            }
    return outcomes


# ---------- simulator input collection ----------
def collect_sim_inputs(num_vals, num_ids, cat_vals, cat_ids) -> pd.DataFrame:
    row: dict[str, object] = {}
    for ident, v in zip(num_ids, num_vals, strict=False):
        row[ident["col"]] = v
    for ident, v in zip(cat_ids, cat_vals, strict=False):
        row[ident["col"]] = v
    for c in FEATURE_SPEC["feature_cols"]:
        if c not in row or row[c] is None:
            row[c] = SIM_DEFAULTS.get(c)
    X = pd.DataFrame([{c: row.get(c) for c in FEATURE_SPEC["feature_cols"]}])
    for c in FEATURE_SPEC["categorical_cols"]:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for c in FEATURE_SPEC["numeric_cols"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


# ---------- SHAP for single-row inference ----------
def shap_for_row_regression(X: pd.DataFrame, model) -> tuple[np.ndarray, float]:
    import shap

    expl = shap.TreeExplainer(model)
    values = expl.shap_values(X)
    base = (
        float(expl.expected_value)
        if np.isscalar(expl.expected_value)
        else float(expl.expected_value[0])
    )
    return values[0], base


def shap_for_row_class(
    X: pd.DataFrame, clf, class_idx: int, n_classes: int,
) -> tuple[np.ndarray, float]:
    import shap

    expl = shap.TreeExplainer(clf)
    raw = expl.shap_values(X)
    if isinstance(raw, list):
        per_class = np.asarray(raw[class_idx])[0]
        base_arr = expl.expected_value
        base = (
            float(base_arr)
            if np.isscalar(base_arr)
            else float(np.asarray(base_arr).ravel()[class_idx])
        )
    else:
        arr = np.asarray(raw)
        if arr.ndim == 3 and arr.shape[0] == n_classes and arr.shape[-1] != n_classes:
            arr = np.transpose(arr, (1, 2, 0))
        per_class = arr[0, :, class_idx]
        base_arr = expl.expected_value
        if np.isscalar(base_arr):
            base = float(base_arr)
        else:
            base = float(np.asarray(base_arr).ravel()[class_idx])
    return per_class, base


# ---------- episode helpers ----------
def episode_row_for_model(key_record: int) -> pd.DataFrame:
    """Build a one-row model input from an episode's admission features."""
    feat = episode_admission_features(EP, key_record, FEATURE_SPEC["feature_cols"])
    for c in FEATURE_SPEC["feature_cols"]:
        if feat.get(c) is None:
            feat[c] = SIM_DEFAULTS.get(c)
    X = pd.DataFrame([{c: feat.get(c) for c in FEATURE_SPEC["feature_cols"]}])
    for c in FEATURE_SPEC["categorical_cols"]:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for c in FEATURE_SPEC["numeric_cols"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def episode_has_admission(key_record: int) -> bool:
    feat = episode_admission_features(EP, key_record, FEATURE_SPEC["feature_cols"])
    return any(v is not None for v in feat.values())


def get_observed_for_outcome(key_record: int, spec: OutcomeSpec) -> float | None:
    row = EP.loc[EP["KeyRecordNumber"] == key_record]
    if row.empty or spec.target_col not in row.columns:
        return None
    v = row.iloc[0][spec.target_col]
    return None if pd.isna(v) else float(v)
