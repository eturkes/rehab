"""Train one model per outcome spec + split-conformal PI + SHAP cache.

Iterates over :data:`rehab_sci.models.outcomes.OUTCOMES` and writes one bundle
per outcome under ``models/{spec.key}/``::

    models/
      scim_total/      lgbm_median.joblib  lgbm_p10.joblib  lgbm_p90.joblib
                       feature_spec.joblib  shap_test.joblib
      scim_self_care/  …                                      (same shape)
      scim_resp_sphincter/                                    (same shape)
      scim_mobility/                                          (same shape)
      ais_discharge/   lgbm_multiclass.joblib  feature_spec.joblib  shap_test.joblib
      los_days/        lgbm_median.joblib  …  (transform=log1p)
    models/training_metrics.json   — dict keyed by spec.key
    models/simulator_defaults.json — outcome-independent feature defaults

Design choices (see also AGENT_NOTES §3)
----------------------------------------
* **Group split by patient (IDNumber)** prevents leakage when a patient has
  multiple episodes.  Used for every outcome.
* **Regression heads** (SCIM total + 3 subscales + LOS) — LightGBM median
  + 10% / 90% quantile heads.  Split-conformal PI on a calibration fold
  gives a marginal 80 % coverage guarantee.  The conformal q is computed
  on the *transformed* scale (log1p for LOS, identity otherwise) so the
  interval is symmetric on the natural modelling scale; bounds are
  back-transformed and clipped to the outcome's range before reporting.
* **Multiclass head** (AIS A→E at discharge) — LightGBM multiclass with
  ``class_weight="balanced"`` (AIS-D dominates ~59 %).  Classes are
  encoded by severity (A=0 … E=4) so ``predict_proba`` columns and the
  cached SHAP last axis are both ordinally sorted.  Ordinal-aware
  metrics (quadratic-weighted κ, MAE-on-ordinal-code) are reported
  alongside accuracy.  APS conformal classification sets provide 80 %
  coverage guarantee with Mondrian per-group calibration.
* **TreeSHAP** is cached on the held-out test set only.  Regression
  outcomes store a 2D ``(n, p)`` matrix; multiclass stores a 3D
  ``(n, p, K)`` tensor with K=5 (the AIS axis).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from rehab_sci.data.dataset import AnalysisFrame, build_analysis_dataset
from rehab_sci.models.outcomes import OUTCOMES, OutcomeSpec

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 20260518

AIS_ORD_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
AIS_ORD_COL = "AIS_ord"
PARALYSIS_COL = "対麻痺_四肢麻痺"
MONDRIAN_MIN_N = 8

TRAJECTORY_TIMEPOINTS = ("72h", "2w", "4w", "6w", "2m", "3m", "4m", "5m", "6m")


# ----------------------------- prep helpers ------------------------------

def _prep(
    ep: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    used = ep.dropna(subset=[target_col, "IDNumber"]).copy()
    X = used[feature_cols].copy()
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for c in numeric_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    y = used[target_col]
    groups = used["IDNumber"].astype("float64").astype("int64")
    cat_cols_in_X = [c for c in categorical_cols if c in X.columns]
    return X, y, groups, cat_cols_in_X


def _apply_transform(y: pd.Series | np.ndarray, transform: str | None) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    if transform == "log1p":
        return np.log1p(arr)
    return arr


def _inverse_transform(y: np.ndarray, transform: str | None) -> np.ndarray:
    if transform == "log1p":
        return np.expm1(y)
    return y


def _clip(arr: np.ndarray, lo: float | None, hi: float | None) -> np.ndarray:
    out = arr
    if lo is not None:
        out = np.maximum(out, lo)
    if hi is not None:
        out = np.minimum(out, hi)
    return out


# ----------------------------- LightGBM params ---------------------------

def _params_lgbm_reg() -> dict:
    return dict(
        objective="regression",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=-1,
        min_data_in_leaf=15,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=2,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )


def _params_quantile(alpha: float) -> dict:
    p = _params_lgbm_reg()
    p["objective"] = "quantile"
    p["alpha"] = alpha
    return p


def _params_lgbm_clf(n_classes: int) -> dict:
    return dict(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=-1,
        min_data_in_leaf=10,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=2,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbosity=-1,
    )


def _fit_reg(params, X_tr, y_tr, X_val, y_val, cat_cols):
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def _fit_clf(params, X_tr, y_tr, X_val, y_val, cat_cols):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def _grouped_holdout(
    X: pd.DataFrame, y: np.ndarray, groups: pd.Series, test_size: float
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return train_idx, test_idx


# ----------------------------- CV scorers --------------------------------

def _cv_score_reg(
    X: pd.DataFrame,
    y_raw: pd.Series,
    groups: pd.Series,
    cat_cols: list[str],
    transform: str | None,
    clip_min: float | None,
    clip_max: float | None,
    n_splits: int = 5,
) -> dict:
    """CV metrics reported on the *raw* (back-transformed, clipped) scale."""
    gkf = GroupKFold(n_splits=n_splits)
    y_t = _apply_transform(y_raw, transform)
    y_raw_arr = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
    r2s, rmses, maes = [], [], []
    for tr, te in gkf.split(X, y_t, groups=groups):
        inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
        inner_tr, inner_val = next(inner.split(X.iloc[tr], y_t[tr], groups=groups.iloc[tr]))
        model = _fit_reg(
            _params_lgbm_reg(),
            X.iloc[tr].iloc[inner_tr],
            y_t[tr][inner_tr],
            X.iloc[tr].iloc[inner_val],
            y_t[tr][inner_val],
            cat_cols,
        )
        pred_t = model.predict(X.iloc[te])
        pred = _clip(_inverse_transform(pred_t, transform), clip_min, clip_max)
        r2s.append(r2_score(y_raw_arr[te], pred))
        rmses.append(float(np.sqrt(mean_squared_error(y_raw_arr[te], pred))))
        maes.append(mean_absolute_error(y_raw_arr[te], pred))
    return {
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "folds": n_splits,
    }


def _cv_score_multiclass(
    X: pd.DataFrame,
    y_codes: np.ndarray,
    groups: pd.Series,
    cat_cols: list[str],
    class_codes: tuple[int, ...],
    n_splits: int = 5,
) -> dict:
    gkf = GroupKFold(n_splits=n_splits)
    accs, kappas, ord_maes = [], [], []
    code_arr = np.array(class_codes)
    n_classes = len(class_codes)
    for tr, te in gkf.split(X, y_codes, groups=groups):
        inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
        inner_tr, inner_val = next(inner.split(X.iloc[tr], y_codes[tr], groups=groups.iloc[tr]))
        model = _fit_clf(
            _params_lgbm_clf(n_classes),
            X.iloc[tr].iloc[inner_tr],
            y_codes[tr][inner_tr],
            X.iloc[tr].iloc[inner_val],
            y_codes[tr][inner_val],
            cat_cols,
        )
        pred_idx = np.asarray(model.predict(X.iloc[te]), dtype=int)
        accs.append(accuracy_score(y_codes[te], pred_idx))
        kappas.append(cohen_kappa_score(y_codes[te], pred_idx, weights="quadratic"))
        ord_maes.append(mean_absolute_error(code_arr[y_codes[te]], code_arr[pred_idx]))
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "kappa_quadratic_mean": float(np.mean(kappas)),
        "kappa_quadratic_std": float(np.std(kappas)),
        "ordinal_mae_mean": float(np.mean(ord_maes)),
        "ordinal_mae_std": float(np.std(ord_maes)),
        "folds": n_splits,
    }


# ----------------------------- Mondrian conformal --------------------------


def _conformal_q(residuals: np.ndarray, alpha: float) -> float:
    q_idx = int(np.ceil((len(residuals) + 1) * (1 - alpha))) - 1
    q_idx = max(0, min(q_idx, len(residuals) - 1))
    return float(np.sort(residuals)[q_idx])


def _compute_mondrian_q(
    residuals_t: np.ndarray,
    X_cal: pd.DataFrame,
    alpha: float,
) -> dict:
    """Per-AIS-grade and per-paralysis-class conformal quantiles.

    Groups with fewer than ``MONDRIAN_MIN_N`` calibration samples are omitted;
    inference falls back to the next coarser group or to the marginal.
    """
    result: dict = {}
    if AIS_ORD_COL in X_cal.columns:
        ais_vals = pd.to_numeric(X_cal[AIS_ORD_COL], errors="coerce").to_numpy()
        ais_qs: dict[str, float] = {}
        for code, letter in AIS_ORD_TO_LETTER.items():
            mask = ais_vals == code
            if int(mask.sum()) >= MONDRIAN_MIN_N:
                ais_qs[letter] = _conformal_q(residuals_t[mask], alpha)
        result["ais"] = ais_qs
    if PARALYSIS_COL in X_cal.columns:
        para_vals = X_cal[PARALYSIS_COL].astype(str).to_numpy()
        para_qs: dict[str, float] = {}
        for label in ("TETRA", "PARA", "NONE"):
            mask = para_vals == label
            if int(mask.sum()) >= MONDRIAN_MIN_N:
                para_qs[label] = _conformal_q(residuals_t[mask], alpha)
        result["paralysis"] = para_qs
    return result


def _resolve_mondrian_q_array(
    marginal_q: float,
    q_by_group: dict,
    X: pd.DataFrame,
) -> np.ndarray:
    """Per-row conformal q: AIS group -> paralysis group -> marginal."""
    n = len(X)
    q_arr = np.full(n, marginal_q)
    resolved = np.zeros(n, dtype=bool)
    ais_qs = q_by_group.get("ais", {})
    if ais_qs and AIS_ORD_COL in X.columns:
        ais_vals = pd.to_numeric(X[AIS_ORD_COL], errors="coerce").to_numpy()
        for code, letter in AIS_ORD_TO_LETTER.items():
            if letter in ais_qs:
                mask = ais_vals == code
                q_arr[mask] = ais_qs[letter]
                resolved |= mask
    para_qs = q_by_group.get("paralysis", {})
    if para_qs and PARALYSIS_COL in X.columns:
        para_vals = X[PARALYSIS_COL].astype(str).to_numpy()
        for label, q in para_qs.items():
            mask = (~resolved) & (para_vals == label)
            q_arr[mask] = q
    return q_arr


def _mondrian_test_coverage(
    y_raw: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    X: pd.DataFrame,
) -> dict:
    """Per-group coverage on the test set using Mondrian PI bounds."""
    covered = (y_raw >= lo) & (y_raw <= hi)
    result: dict = {}
    if AIS_ORD_COL in X.columns:
        ais_vals = pd.to_numeric(X[AIS_ORD_COL], errors="coerce").to_numpy()
        ais_cov: dict = {}
        for code, letter in AIS_ORD_TO_LETTER.items():
            mask = ais_vals == code
            n = int(mask.sum())
            if n > 0:
                ais_cov[letter] = {"n": n, "coverage": round(float(covered[mask].mean()), 4)}
        result["ais"] = ais_cov
    if PARALYSIS_COL in X.columns:
        para_vals = X[PARALYSIS_COL].astype(str).to_numpy()
        para_cov: dict = {}
        for label in ("TETRA", "PARA", "NONE"):
            mask = para_vals == label
            n = int(mask.sum())
            if n > 0:
                para_cov[label] = {"n": n, "coverage": round(float(covered[mask].mean()), 4)}
        result["paralysis"] = para_cov
    return result


# ----------------------------- APS conformal classification -----------------


def _aps_scores(proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """APS nonconformity scores for conformal classification sets.

    For each sample: sort class probabilities descending, accumulate until the
    true class is included.  The score is the total accumulated mass.
    """
    n = len(y_true)
    scores = np.zeros(n)
    for i in range(n):
        order = np.argsort(-proba[i])
        cumsum = 0.0
        for j in order:
            cumsum += proba[i, j]
            if j == y_true[i]:
                scores[i] = cumsum
                break
    return scores


def _aps_prediction_set(proba_row: np.ndarray, q_hat: float) -> list[int]:
    """Class indices in the APS prediction set for one sample."""
    order = np.argsort(-proba_row)
    cumsum = 0.0
    pred_set: list[int] = []
    for j in order:
        pred_set.append(int(j))
        cumsum += proba_row[j]
        if cumsum >= q_hat:
            break
    return sorted(pred_set)


def _aps_test_metrics(
    proba: np.ndarray,
    y_true: np.ndarray,
    q_arr: np.ndarray,
    X: pd.DataFrame,
) -> dict:
    """Coverage and avg set size on test set using per-row Mondrian APS q."""
    n = len(y_true)
    covered = np.zeros(n, dtype=bool)
    sizes = np.zeros(n, dtype=int)
    for i in range(n):
        pset = _aps_prediction_set(proba[i], q_arr[i])
        covered[i] = y_true[i] in pset
        sizes[i] = len(pset)
    result: dict = {
        "coverage": round(float(covered.mean()), 4),
        "avg_set_size": round(float(sizes.mean()), 3),
        "n": n,
    }
    if AIS_ORD_COL in X.columns:
        ais_vals = pd.to_numeric(X[AIS_ORD_COL], errors="coerce").to_numpy()
        ais_cov: dict = {}
        for code, letter in AIS_ORD_TO_LETTER.items():
            mask = ais_vals == code
            ng = int(mask.sum())
            if ng > 0:
                ais_cov[letter] = {
                    "n": ng,
                    "coverage": round(float(covered[mask].mean()), 4),
                    "avg_set_size": round(float(sizes[mask].mean()), 3),
                }
        result["ais"] = ais_cov
    if PARALYSIS_COL in X.columns:
        para_vals = X[PARALYSIS_COL].astype(str).to_numpy()
        para_cov: dict = {}
        for label in ("TETRA", "PARA", "NONE"):
            mask = para_vals == label
            ng = int(mask.sum())
            if ng > 0:
                para_cov[label] = {
                    "n": ng,
                    "coverage": round(float(covered[mask].mean()), 4),
                    "avg_set_size": round(float(sizes[mask].mean()), 3),
                }
        result["paralysis"] = para_cov
    return result


# ----------------------------- per-outcome trainers ----------------------

def _train_regression(
    spec: OutcomeSpec,
    af: AnalysisFrame,
    out_root: Path,
) -> dict:
    target = spec.target_col
    transform = spec.transform
    X, y_raw, groups, cat_cols = _prep(
        af.df, af.feature_cols, af.numeric_cols, af.categorical_cols, target
    )
    y_t = _apply_transform(y_raw, transform)
    y_raw_arr = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
    print(
        f"[{spec.key}]  rows={len(X)}  features={X.shape[1]}  "
        f"patients={groups.nunique()}  task=regression  transform={transform or 'none'}"
    )

    cv = _cv_score_reg(X, y_raw, groups, cat_cols, transform, spec.clip_min, spec.clip_max)
    print(
        f"   CV  R²={cv['r2_mean']:.3f}±{cv['r2_std']:.3f}  "
        f"RMSE={cv['rmse_mean']:.3f}  MAE={cv['mae_mean']:.3f}"
    )

    # holdout + calibration
    train_idx, test_idx = _grouped_holdout(X, y_t, groups, test_size=0.2)
    X_dev, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_dev_t, y_te_t = y_t[train_idx], y_t[test_idx]
    y_te_raw = y_raw_arr[test_idx]
    g_dev = groups.iloc[train_idx]

    train_idx2, calib_idx = _grouped_holdout(X_dev, y_dev_t, g_dev, test_size=0.20)
    X_tr, X_cal = X_dev.iloc[train_idx2], X_dev.iloc[calib_idx]
    y_tr_t, y_cal_t = y_dev_t[train_idx2], y_dev_t[calib_idx]

    median_dev = _fit_reg(_params_lgbm_reg(), X_tr, y_tr_t, X_cal, y_cal_t, cat_cols)
    p10_dev = _fit_reg(_params_quantile(0.1), X_tr, y_tr_t, X_cal, y_cal_t, cat_cols)
    p90_dev = _fit_reg(_params_quantile(0.9), X_tr, y_tr_t, X_cal, y_cal_t, cat_cols)

    # conformal half-width on the TRANSFORMED scale (so bounds remain symmetric on log-LOS)
    pred_cal_t = median_dev.predict(X_cal)
    residuals_t = np.abs(y_cal_t - pred_cal_t)
    alpha = 0.2
    conformal_q_t = _conformal_q(residuals_t, alpha)

    # Mondrian: per-group conformal q on the calibration fold
    q_by_group = _compute_mondrian_q(residuals_t, X_cal, alpha)
    q_by_group["marginal"] = conformal_q_t
    q_by_group["min_n"] = MONDRIAN_MIN_N

    pred_t = median_dev.predict(X_te)
    pred = _clip(_inverse_transform(pred_t, transform), spec.clip_min, spec.clip_max)
    lo = _clip(_inverse_transform(pred_t - conformal_q_t, transform), spec.clip_min, spec.clip_max)
    hi = _clip(_inverse_transform(pred_t + conformal_q_t, transform), spec.clip_min, spec.clip_max)
    covered = float(((y_te_raw >= lo) & (y_te_raw <= hi)).mean())

    # Mondrian test-set coverage (each test point uses its group-specific q)
    q_te_mond = _resolve_mondrian_q_array(conformal_q_t, q_by_group, X_te)
    lo_mond = _clip(_inverse_transform(pred_t - q_te_mond, transform), spec.clip_min, spec.clip_max)
    hi_mond = _clip(_inverse_transform(pred_t + q_te_mond, transform), spec.clip_min, spec.clip_max)
    covered_mond = float(((y_te_raw >= lo_mond) & (y_te_raw <= hi_mond)).mean())
    mondrian_coverage = _mondrian_test_coverage(y_te_raw, lo_mond, hi_mond, X_te)

    pred10 = _clip(_inverse_transform(p10_dev.predict(X_te), transform), spec.clip_min, spec.clip_max)
    pred90 = _clip(_inverse_transform(p90_dev.predict(X_te), transform), spec.clip_min, spec.clip_max)
    qcov = float(((y_te_raw >= pred10) & (y_te_raw <= pred90)).mean())

    test_metrics = {
        "r2": float(r2_score(y_te_raw, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_te_raw, pred))),
        "mae": float(mean_absolute_error(y_te_raw, pred)),
        "conformal_q_half_width_transformed": conformal_q_t,
        "conformal_coverage_80": covered,
        "conformal_coverage_80_mondrian": covered_mond,
        "mondrian_coverage": mondrian_coverage,
        "mondrian_q_by_group": {
            k: v for k, v in q_by_group.items() if k != "min_n"
        },
        "raw_quantile_coverage_80": qcov,
        "n_train": int(len(X_tr)),
        "n_calib": int(len(X_cal)),
        "n_test": int(len(X_te)),
        "n_dev_patients": int(g_dev.nunique()),
        "n_test_patients": int(groups.iloc[test_idx].nunique()),
    }
    print(
        f"   TEST  R²={test_metrics['r2']:.3f}  RMSE={test_metrics['rmse']:.3f}  "
        f"MAE={test_metrics['mae']:.3f}  conformal80={covered:.2%}  q={conformal_q_t:.3f}"
    )
    ais_qs = q_by_group.get("ais", {})
    para_qs = q_by_group.get("paralysis", {})
    ais_str = "  ".join(f"{g}={q:.1f}" for g, q in sorted(ais_qs.items())) if ais_qs else "–"
    para_str = "  ".join(f"{g}={q:.1f}" for g, q in sorted(para_qs.items())) if para_qs else "–"
    print(f"   MONDRIAN  coverage={covered_mond:.2%}  AIS:[{ais_str}]  Para:[{para_str}]")

    # refit on ALL data
    full_params = _params_lgbm_reg()
    full_params["n_estimators"] = int(median_dev.best_iteration_ or 400)
    final_median = lgb.LGBMRegressor(**full_params)
    final_median.fit(X, y_t, categorical_feature=cat_cols)

    fp10 = _params_quantile(0.1)
    fp10["n_estimators"] = int(p10_dev.best_iteration_ or 400)
    final_p10 = lgb.LGBMRegressor(**fp10)
    final_p10.fit(X, y_t, categorical_feature=cat_cols)

    fp90 = _params_quantile(0.9)
    fp90["n_estimators"] = int(p90_dev.best_iteration_ or 400)
    final_p90 = lgb.LGBMRegressor(**fp90)
    final_p90.fit(X, y_t, categorical_feature=cat_cols)

    # SHAP on test (transformed-scale contributions; back-transform happens
    # only for the headline number, not for the explanation).
    explainer = shap.TreeExplainer(final_median)
    shap_values = explainer.shap_values(X_te)
    base_value = (
        float(explainer.expected_value)
        if np.isscalar(explainer.expected_value)
        else float(explainer.expected_value[0])
    )

    abs_mean = np.abs(shap_values).mean(axis=0)
    importance = sorted(
        zip(X.columns.tolist(), abs_mean.tolist(), strict=False),
        key=lambda x: -x[1],
    )

    # persist
    od = out_root / spec.key
    od.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_median, od / "lgbm_median.joblib")
    joblib.dump(final_p10, od / "lgbm_p10.joblib")
    joblib.dump(final_p90, od / "lgbm_p90.joblib")
    joblib.dump(
        {
            "X_test_index": list(X_te.index),
            "shap_values": shap_values,
            "X_test": X_te,
            "base_value": base_value,
            "y_test": y_te_raw,
            "y_pred": pred,
        },
        od / "shap_test.joblib",
    )
    feature_spec = {
        "feature_cols": list(X.columns),
        "numeric_cols": list(af.numeric_cols),
        "categorical_cols": list(af.categorical_cols),
        "task": "regression",
        "transform": transform,
        "clip_min": spec.clip_min,
        "clip_max": spec.clip_max,
        "conformal_q_transformed": conformal_q_t,
        "conformal_q_by_group": q_by_group,
        "base_value": base_value,
    }
    joblib.dump(feature_spec, od / "feature_spec.joblib")

    return {
        "task": "regression",
        "target_col": target,
        "transform": transform,
        "clip_min": spec.clip_min,
        "clip_max": spec.clip_max,
        "cv": cv,
        "test": test_metrics,
        "global_importance_top25": [
            {"feature": f, "abs_mean_shap": v} for f, v in importance[:25]
        ],
        "n_features": int(X.shape[1]),
        "n_episodes_with_outcome": int(len(X)),
        "n_patients_with_outcome": int(groups.nunique()),
    }


def _train_multiclass(
    spec: OutcomeSpec,
    af: AnalysisFrame,
    out_root: Path,
) -> dict:
    target = spec.target_col
    X, y_raw, groups, cat_cols = _prep(
        af.df, af.feature_cols, af.numeric_cols, af.categorical_cols, target
    )
    class_codes = spec.class_codes
    n_classes = len(class_codes)
    code_to_idx = {int(c): i for i, c in enumerate(class_codes)}
    y_raw_int = pd.to_numeric(y_raw, errors="coerce").astype(int).to_numpy()
    y_codes = np.array([code_to_idx[int(v)] for v in y_raw_int])
    code_arr = np.array(class_codes)

    print(
        f"[{spec.key}]  rows={len(X)}  features={X.shape[1]}  "
        f"patients={groups.nunique()}  task=multiclass  K={n_classes}  "
        f"class_dist={dict(zip(spec.class_labels, np.bincount(y_codes, minlength=n_classes).tolist(), strict=False))}"
    )

    cv = _cv_score_multiclass(X, y_codes, groups, cat_cols, class_codes)
    print(
        f"   CV  acc={cv['accuracy_mean']:.3f}±{cv['accuracy_std']:.3f}  "
        f"κ_quad={cv['kappa_quadratic_mean']:.3f}±{cv['kappa_quadratic_std']:.3f}  "
        f"ordMAE={cv['ordinal_mae_mean']:.3f}"
    )

    # holdout + calibration fold (for APS conformal classification sets)
    train_idx, test_idx = _grouped_holdout(X, y_codes, groups, test_size=0.2)
    X_dev, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_dev, y_te = y_codes[train_idx], y_codes[test_idx]
    g_dev = groups.iloc[train_idx]

    train_idx2, calib_idx = _grouped_holdout(X_dev, y_dev, g_dev, test_size=0.20)
    X_tr_full, X_cal = X_dev.iloc[train_idx2], X_dev.iloc[calib_idx]
    y_tr_full, y_cal = y_dev[train_idx2], y_dev[calib_idx]
    g_tr_full = g_dev.iloc[train_idx2]

    inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
    inner_tr, inner_val = next(inner.split(X_tr_full, y_tr_full, groups=g_tr_full))
    X_tr, X_val = X_tr_full.iloc[inner_tr], X_tr_full.iloc[inner_val]
    y_tr, y_val = y_tr_full[inner_tr], y_tr_full[inner_val]

    clf_dev = _fit_clf(_params_lgbm_clf(n_classes), X_tr, y_tr, X_val, y_val, cat_cols)
    pred_idx = np.asarray(clf_dev.predict(X_te), dtype=int)

    # APS conformal classification sets on calibration fold
    alpha = 0.2
    proba_cal = np.asarray(clf_dev.predict_proba(X_cal), dtype=float)
    aps_scores = _aps_scores(proba_cal, y_cal)
    aps_q_hat = _conformal_q(aps_scores, alpha)
    aps_q_by_group = _compute_mondrian_q(aps_scores, X_cal, alpha)
    aps_q_by_group["marginal"] = aps_q_hat
    aps_q_by_group["min_n"] = MONDRIAN_MIN_N

    proba_te = np.asarray(clf_dev.predict_proba(X_te), dtype=float)
    aps_q_te = _resolve_mondrian_q_array(aps_q_hat, aps_q_by_group, X_te)
    aps_test = _aps_test_metrics(proba_te, y_te, aps_q_te, X_te)

    test_metrics = {
        "accuracy": float(accuracy_score(y_te, pred_idx)),
        "kappa_quadratic": float(cohen_kappa_score(y_te, pred_idx, weights="quadratic")),
        "ordinal_mae": float(mean_absolute_error(code_arr[y_te], code_arr[pred_idx])),
        "n_train": int(len(X_tr)),
        "n_calib": int(len(X_cal)),
        "n_test": int(len(X_te)),
        "n_dev_patients": int(g_dev.nunique()),
        "n_test_patients": int(groups.iloc[test_idx].nunique()),
        "class_codes": [int(c) for c in class_codes],
        "class_labels": list(spec.class_labels),
        "aps_q_hat": aps_q_hat,
        "aps_coverage_80": aps_test["coverage"],
        "aps_avg_set_size": aps_test["avg_set_size"],
        "aps_mondrian_coverage": aps_test,
        "aps_q_by_group": {
            k: v for k, v in aps_q_by_group.items() if k != "min_n"
        },
    }
    print(
        f"   TEST  acc={test_metrics['accuracy']:.3f}  "
        f"κ_quad={test_metrics['kappa_quadratic']:.3f}  "
        f"ordMAE={test_metrics['ordinal_mae']:.3f}"
    )
    print(
        f"   APS  coverage={aps_test['coverage']:.2%}  "
        f"avg_set_size={aps_test['avg_set_size']:.2f}  "
        f"q_hat={aps_q_hat:.4f}"
    )
    ais_aps_qs = aps_q_by_group.get("ais", {})
    para_aps_qs = aps_q_by_group.get("paralysis", {})
    ais_str = "  ".join(f"{g}={q:.3f}" for g, q in sorted(ais_aps_qs.items())) if ais_aps_qs else "–"
    para_str = "  ".join(f"{g}={q:.3f}" for g, q in sorted(para_aps_qs.items())) if para_aps_qs else "–"
    print(f"   APS-MONDRIAN  AIS:[{ais_str}]  Para:[{para_str}]")

    full_params = _params_lgbm_clf(n_classes)
    full_params["n_estimators"] = int(clf_dev.best_iteration_ or 400)
    final_clf = lgb.LGBMClassifier(**full_params)
    final_clf.fit(X, y_codes, categorical_feature=cat_cols)

    # multiclass SHAP — newer shap returns (n, p, K); older versions a list of K (n, p).
    explainer = shap.TreeExplainer(final_clf)
    raw_shap = explainer.shap_values(X_te)
    if isinstance(raw_shap, list):
        shap_arr = np.stack(raw_shap, axis=-1)  # (n, p, K)
    else:
        shap_arr = np.asarray(raw_shap)
        if shap_arr.ndim == 3 and shap_arr.shape[0] == n_classes and shap_arr.shape[-1] != n_classes:
            shap_arr = np.transpose(shap_arr, (1, 2, 0))
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_value = [float(v) for v in np.asarray(explainer.expected_value).ravel().tolist()]
    else:
        base_value = [float(explainer.expected_value)] * n_classes
    # global importance: mean |SHAP| across samples and classes
    abs_mean = np.abs(shap_arr).mean(axis=(0, 2))
    importance = sorted(
        zip(X.columns.tolist(), abs_mean.tolist(), strict=False),
        key=lambda x: -x[1],
    )

    od = out_root / spec.key
    od.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_clf, od / "lgbm_multiclass.joblib")
    joblib.dump(
        {
            "X_test_index": list(X_te.index),
            "shap_values": shap_arr,
            "X_test": X_te,
            "base_value": base_value,
            "class_codes": [int(c) for c in class_codes],
            "class_labels": list(spec.class_labels),
            "y_test": y_te,
            "y_pred": pred_idx,
            "y_pred_proba": proba_te,
        },
        od / "shap_test.joblib",
    )
    feature_spec = {
        "feature_cols": list(X.columns),
        "numeric_cols": list(af.numeric_cols),
        "categorical_cols": list(af.categorical_cols),
        "task": "multiclass",
        "class_codes": [int(c) for c in class_codes],
        "class_labels": list(spec.class_labels),
        "base_value": base_value,
        "aps_q_hat": aps_q_hat,
        "aps_q_by_group": aps_q_by_group,
    }
    joblib.dump(feature_spec, od / "feature_spec.joblib")

    return {
        "task": "multiclass",
        "target_col": target,
        "class_codes": [int(c) for c in class_codes],
        "class_labels": list(spec.class_labels),
        "cv": cv,
        "test": test_metrics,
        "global_importance_top25": [
            {"feature": f, "abs_mean_shap": v} for f, v in importance[:25]
        ],
        "n_features": int(X.shape[1]),
        "n_episodes_with_outcome": int(len(X)),
        "n_patients_with_outcome": int(groups.nunique()),
    }


# ----------------------------- trajectory training -----------------------


def _train_trajectory(af: AnalysisFrame, out_root: Path) -> dict:
    """Train per-timepoint SCIM-total models for recovery trajectory forecasting.

    One LightGBM regression head per timepoint in ``TRAJECTORY_TIMEPOINTS``,
    using the same 32 admission features as the discharge models.  Artifacts
    are bundled into ``models/trajectory/bundle.joblib``.
    """
    long = af.longitudinal
    ep = af.df.copy()

    for tp in TRAJECTORY_TIMEPOINTS:
        tp_data = long.loc[long["TIME_Name"] == tp, ["KeyRecordNumber", "SCIM_total"]]
        tp_data = tp_data.set_index("KeyRecordNumber")["SCIM_total"]
        ep[f"_traj_{tp}"] = tp_data.reindex(ep["KeyRecordNumber"]).values

    all_models: dict[str, dict] = {}
    all_conformal: dict[str, dict] = {}
    all_metrics: dict[str, dict] = {}
    print("=" * 60)
    print("TRAJECTORY  training (SCIM-total at intermediate timepoints)")
    print("=" * 60)

    for tp in TRAJECTORY_TIMEPOINTS:
        target_col = f"_traj_{tp}"
        X, y_raw, groups, cat_cols = _prep(
            ep, af.feature_cols, af.numeric_cols, af.categorical_cols, target_col
        )
        y_arr = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
        print(
            f"[trajectory/{tp}]  n={len(X)}  patients={groups.nunique()}"
        )

        train_idx, test_idx = _grouped_holdout(X, y_arr, groups, test_size=0.2)
        X_dev, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_dev, y_te = y_arr[train_idx], y_arr[test_idx]
        g_dev = groups.iloc[train_idx]

        train_idx2, calib_idx = _grouped_holdout(X_dev, y_dev, g_dev, test_size=0.20)
        X_tr, X_cal = X_dev.iloc[train_idx2], X_dev.iloc[calib_idx]
        y_tr, y_cal = y_dev[train_idx2], y_dev[calib_idx]

        median_dev = _fit_reg(_params_lgbm_reg(), X_tr, y_tr, X_cal, y_cal, cat_cols)
        p10_dev = _fit_reg(_params_quantile(0.1), X_tr, y_tr, X_cal, y_cal, cat_cols)
        p90_dev = _fit_reg(_params_quantile(0.9), X_tr, y_tr, X_cal, y_cal, cat_cols)

        pred_cal = median_dev.predict(X_cal)
        residuals = np.abs(y_cal - pred_cal)
        alpha = 0.2
        conformal_q = _conformal_q(residuals, alpha)
        q_by_group = _compute_mondrian_q(residuals, X_cal, alpha)
        q_by_group["marginal"] = conformal_q
        q_by_group["min_n"] = MONDRIAN_MIN_N

        pred_te = np.clip(median_dev.predict(X_te), 0.0, 100.0)
        q_te = _resolve_mondrian_q_array(conformal_q, q_by_group, X_te)
        lo_te = np.clip(pred_te - q_te, 0.0, 100.0)
        hi_te = np.clip(pred_te + q_te, 0.0, 100.0)
        covered = float(((y_te >= lo_te) & (y_te <= hi_te)).mean())

        pred10_te = np.clip(p10_dev.predict(X_te), 0.0, 100.0)
        pred90_te = np.clip(p90_dev.predict(X_te), 0.0, 100.0)
        qcov = float(((y_te >= pred10_te) & (y_te <= pred90_te)).mean())

        test_m = {
            "r2": float(r2_score(y_te, pred_te)),
            "rmse": float(np.sqrt(mean_squared_error(y_te, pred_te))),
            "mae": float(mean_absolute_error(y_te, pred_te)),
            "conformal_q": conformal_q,
            "conformal_coverage_80": covered,
            "raw_quantile_coverage_80": qcov,
            "n_train": int(len(X_tr)),
            "n_calib": int(len(X_cal)),
            "n_test": int(len(X_te)),
            "n_total": int(len(X)),
        }
        print(
            f"   TEST  R²={test_m['r2']:.3f}  RMSE={test_m['rmse']:.3f}  "
            f"MAE={test_m['mae']:.3f}  conf80={covered:.2%}  q={conformal_q:.1f}"
        )

        # Refit on all data
        full_params = _params_lgbm_reg()
        full_params["n_estimators"] = int(median_dev.best_iteration_ or 400)
        final_median = lgb.LGBMRegressor(**full_params)
        final_median.fit(X, y_arr, categorical_feature=cat_cols)

        fp10 = _params_quantile(0.1)
        fp10["n_estimators"] = int(p10_dev.best_iteration_ or 400)
        final_p10 = lgb.LGBMRegressor(**fp10)
        final_p10.fit(X, y_arr, categorical_feature=cat_cols)

        fp90 = _params_quantile(0.9)
        fp90["n_estimators"] = int(p90_dev.best_iteration_ or 400)
        final_p90 = lgb.LGBMRegressor(**fp90)
        final_p90.fit(X, y_arr, categorical_feature=cat_cols)

        all_models[tp] = {
            "median": final_median,
            "p10": final_p10,
            "p90": final_p90,
        }
        all_conformal[tp] = {
            "q_transformed": conformal_q,
            "q_by_group": q_by_group,
        }
        all_metrics[tp] = test_m

    traj_dir = out_root / "trajectory"
    traj_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "timepoints": list(TRAJECTORY_TIMEPOINTS),
        "models": all_models,
        "conformal": all_conformal,
        "clip_min": 0.0,
        "clip_max": 100.0,
    }
    joblib.dump(bundle, traj_dir / "bundle.joblib")
    print(f"\nTrajectory bundle → {traj_dir / 'bundle.joblib'}")
    return all_metrics


# ----------------------------- shared feature defaults -------------------

def _simulator_defaults(af: AnalysisFrame) -> tuple[dict, dict]:
    """Return (defaults, ranges_and_categories) over the full episode frame.

    These describe the *features*, not any specific outcome, so they are
    written once at the top level for the dashboard simulator to use.
    """
    ep = af.df
    X = ep[af.feature_cols].copy()
    for c in af.categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for c in af.numeric_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    defaults: dict[str, object] = {}
    ranges: dict[str, dict] = {}
    categories: dict[str, list[str]] = {}
    for c in af.numeric_cols:
        if c in X.columns:
            s = pd.to_numeric(X[c], errors="coerce").dropna()
            defaults[c] = None if s.empty else float(s.median())
            if not s.empty:
                ranges[c] = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "q05": float(s.quantile(0.05)),
                    "q95": float(s.quantile(0.95)),
                    "median": float(s.median()),
                }
    for c in af.categorical_cols:
        if c in X.columns:
            mode = X[c].mode(dropna=True)
            defaults[c] = None if mode.empty else str(mode.iloc[0])
            categories[c] = [
                str(v)
                for v in pd.Series(X[c].dropna().unique()).astype(str).tolist()
                if str(v) not in {"<NA>", "nan"}
            ]
    return defaults, {"ranges": ranges, "categories": categories}


# ----------------------------- entry point -------------------------------

def main() -> None:
    af = build_analysis_dataset()
    print(
        f"episode frame: {len(af.df)} episodes  "
        f"{af.df['IDNumber'].nunique(dropna=True)} patients  "
        f"{len(af.feature_cols)} features"
    )

    all_metrics: dict[str, dict] = {}
    for spec in OUTCOMES:
        if spec.task == "regression":
            m = _train_regression(spec, af, OUT)
        elif spec.task == "multiclass":
            m = _train_multiclass(spec, af, OUT)
        else:
            raise ValueError(f"unknown task: {spec.task!r}")
        all_metrics[spec.key] = m
        print()

    # trajectory models (SCIM-total at intermediate timepoints)
    traj_metrics = _train_trajectory(af, OUT)
    print()

    defaults, feature_universe = _simulator_defaults(af)
    # one shared feature_spec for the simulator's input-builder layer
    shared_feature_spec = {
        "feature_cols": list(af.feature_cols),
        "numeric_cols": list(af.numeric_cols),
        "categorical_cols": list(af.categorical_cols),
        "ranges": feature_universe["ranges"],
        "categories": feature_universe["categories"],
    }
    joblib.dump(shared_feature_spec, OUT / "feature_spec.joblib")

    # write metrics
    metrics_payload = {
        "outcomes": all_metrics,
        "outcome_keys": [s.key for s in OUTCOMES],
        "trajectory": traj_metrics,
        "trajectory_timepoints": list(TRAJECTORY_TIMEPOINTS),
        "random_state": RANDOM_STATE,
        "n_episodes_total": int(len(af.df)),
        "n_patients_total": int(af.df["IDNumber"].nunique(dropna=True)),
        "n_features": int(len(af.feature_cols)),
    }
    (OUT / "training_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )
    (OUT / "simulator_defaults.json").write_text(
        json.dumps(defaults, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote artifacts to {OUT}")


if __name__ == "__main__":
    main()
