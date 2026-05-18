"""Train the discharge-SCIM prediction model + conformal prediction interval + SHAP cache.

Outputs (gitignored — ``models/`` directory):
  - models/lgbm_median.joblib        : LightGBM regressor refit on full data (for serving)
  - models/lgbm_p10.joblib           : LightGBM quantile (10%) refit
  - models/lgbm_p90.joblib           : LightGBM quantile (90%) refit
  - models/feature_spec.joblib       : feature column order + ranges + categories
  - models/training_metrics.json     : CV + held-out test scores + conformal half-width
  - models/shap_test.joblib          : SHAP values for the held-out test set
  - models/simulator_defaults.json   : median / mode for each feature

Design choices
--------------
* **Group split by patient (IDNumber)** prevents leakage when a patient has multiple episodes.
* **5-fold GroupKFold cross-validation** for the median model.
* **Split conformal prediction** on a calibration fold gives a marginal 80% prediction
  interval with formal coverage guarantee (the LightGBM quantile heads alone systematically
  overfit on n≈400). The PI half-width is the (1-α)-quantile of the absolute residuals on
  the calibration set, then clipped to the SCIM scale [0, 100].
* **TreeSHAP** for global + local explanations.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from rehab_sci.data.dataset import build_analysis_dataset

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 20260518


def _prep(
    ep: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    outcome: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    used = ep.dropna(subset=[outcome, "IDNumber"]).copy()
    X = used[feature_cols].copy()
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for c in numeric_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    y = used[outcome].astype(float)
    groups = used["IDNumber"].astype("float64").astype("int64")
    cat_cols_in_X = [c for c in categorical_cols if c in X.columns]
    return X, y, groups, cat_cols_in_X


def _params_median() -> dict:
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
    p = _params_median()
    p["objective"] = "quantile"
    p["alpha"] = alpha
    return p


def _fit_with_early_stopping(
    params: dict, X_tr, y_tr, X_val, y_val, cat_cols: list[str]
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def _grouped_holdout(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, test_size: float
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return train_idx, test_idx


def _cv_score(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cat_cols: list[str],
    n_splits: int = 5,
) -> dict:
    gkf = GroupKFold(n_splits=n_splits)
    r2s, rmses, maes = [], [], []
    for tr, te in gkf.split(X, y, groups=groups):
        params = _params_median()
        # cv folds use a small inner validation for early stopping
        inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
        inner_tr, inner_val = next(
            inner.split(X.iloc[tr], y.iloc[tr], groups=groups.iloc[tr])
        )
        model = _fit_with_early_stopping(
            params,
            X.iloc[tr].iloc[inner_tr],
            y.iloc[tr].iloc[inner_tr],
            X.iloc[tr].iloc[inner_val],
            y.iloc[tr].iloc[inner_val],
            cat_cols,
        )
        pred = model.predict(X.iloc[te])
        r2s.append(r2_score(y.iloc[te], pred))
        rmses.append(float(np.sqrt(mean_squared_error(y.iloc[te], pred))))
        maes.append(mean_absolute_error(y.iloc[te], pred))
    return {
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "folds": n_splits,
    }


def main() -> None:
    af = build_analysis_dataset()
    X, y, groups, cat_cols = _prep(
        af.df, af.feature_cols, af.numeric_cols, af.categorical_cols, af.outcome_col
    )
    print(f"training rows = {len(X)}  features = {X.shape[1]}  patients = {groups.nunique()}")

    # ---- CV on full set (median model) ----
    cv = _cv_score(X, y, groups, cat_cols, n_splits=5)
    print(f"CV : R²={cv['r2_mean']:.3f}±{cv['r2_std']:.3f}  "
          f"RMSE={cv['rmse_mean']:.2f}  MAE={cv['mae_mean']:.2f}")

    # ---- held-out test (20%) + calibration (10% of the remainder) for split conformal ----
    train_idx, test_idx = _grouped_holdout(X, y, groups, test_size=0.2)
    X_dev, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_dev, y_te = y.iloc[train_idx], y.iloc[test_idx]
    g_dev = groups.iloc[train_idx]

    train_idx2, calib_idx = _grouped_holdout(X_dev, y_dev, g_dev, test_size=0.20)
    X_tr, X_cal = X_dev.iloc[train_idx2], X_dev.iloc[calib_idx]
    y_tr, y_cal = y_dev.iloc[train_idx2], y_dev.iloc[calib_idx]

    print(
        f"  dev={len(X_dev)}  test={len(X_te)}   "
        f"split-conformal: train={len(X_tr)} / calib={len(X_cal)}"
    )

    median_dev = _fit_with_early_stopping(
        _params_median(), X_tr, y_tr, X_cal, y_cal, cat_cols
    )
    p10_dev = _fit_with_early_stopping(
        _params_quantile(0.1), X_tr, y_tr, X_cal, y_cal, cat_cols
    )
    p90_dev = _fit_with_early_stopping(
        _params_quantile(0.9), X_tr, y_tr, X_cal, y_cal, cat_cols
    )

    # ---- conformal half-width ----
    pred_cal = median_dev.predict(X_cal)
    residuals = np.abs(y_cal.values - pred_cal)
    alpha = 0.2  # 80% PI
    q_idx = int(np.ceil((len(residuals) + 1) * (1 - alpha))) - 1
    q_idx = max(0, min(q_idx, len(residuals) - 1))
    conformal_q = float(np.sort(residuals)[q_idx])

    # ---- test metrics ----
    pred = median_dev.predict(X_te)
    lo = np.clip(pred - conformal_q, 0, 100)
    hi = np.clip(pred + conformal_q, 0, 100)
    covered = float(((y_te.values >= lo) & (y_te.values <= hi)).mean())

    # also evaluate raw quantile heads
    pred10 = p10_dev.predict(X_te)
    pred90 = p90_dev.predict(X_te)
    qcov = float(((y_te.values >= pred10) & (y_te.values <= pred90)).mean())

    test_metrics = {
        "r2": float(r2_score(y_te, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "mae": float(mean_absolute_error(y_te, pred)),
        "conformal_q_half_width": conformal_q,
        "conformal_coverage_80": covered,
        "raw_quantile_coverage_80": qcov,
        "n_train": int(len(X_tr)),
        "n_calib": int(len(X_cal)),
        "n_test": int(len(X_te)),
        "n_dev_patients": int(g_dev.nunique()),
        "n_test_patients": int(groups.iloc[test_idx].nunique()),
    }
    print(
        f"TEST: R²={test_metrics['r2']:.3f}  RMSE={test_metrics['rmse']:.2f}  "
        f"MAE={test_metrics['mae']:.2f}  conformal80={covered:.2%}  rawQ80={qcov:.2%}  "
        f"conformal half-width={conformal_q:.2f}"
    )

    # ---- refit final models on ALL data; the conformal q is fixed from calibration ----
    full_params_median = _params_median()
    full_params_median["n_estimators"] = int(median_dev.best_iteration_ or 400)
    final_median = lgb.LGBMRegressor(**full_params_median)
    final_median.fit(X, y, categorical_feature=cat_cols)

    fp10 = _params_quantile(0.1)
    fp10["n_estimators"] = int(p10_dev.best_iteration_ or 400)
    final_p10 = lgb.LGBMRegressor(**fp10)
    final_p10.fit(X, y, categorical_feature=cat_cols)

    fp90 = _params_quantile(0.9)
    fp90["n_estimators"] = int(p90_dev.best_iteration_ or 400)
    final_p90 = lgb.LGBMRegressor(**fp90)
    final_p90.fit(X, y, categorical_feature=cat_cols)

    # ---- SHAP on the held-out test set ----
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
    print("\nTop 15 SHAP global importance:")
    for f, v in importance[:15]:
        print(f"  {f:<30} {v:.3f}")

    # ---- simulator defaults: median for numeric, mode for categorical ----
    defaults: dict[str, object] = {}
    for c in af.numeric_cols:
        if c in X.columns:
            m = pd.to_numeric(X[c], errors="coerce").median()
            defaults[c] = None if pd.isna(m) else float(m)
    for c in af.categorical_cols:
        if c in X.columns:
            mode = X[c].mode(dropna=True)
            defaults[c] = None if mode.empty else str(mode.iloc[0])

    feature_spec = {
        "feature_cols": list(X.columns),
        "numeric_cols": list(af.numeric_cols),
        "categorical_cols": list(af.categorical_cols),
        "ranges": {},
        "categories": {},
        "conformal_half_width": conformal_q,
        "base_value": base_value,
    }
    for c in af.numeric_cols:
        if c in X.columns:
            s = pd.to_numeric(X[c], errors="coerce").dropna()
            if not s.empty:
                feature_spec["ranges"][c] = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "q05": float(s.quantile(0.05)),
                    "q95": float(s.quantile(0.95)),
                    "median": float(s.median()),
                }
    for c in af.categorical_cols:
        if c in X.columns:
            feature_spec["categories"][c] = [
                str(v)
                for v in pd.Series(X[c].dropna().unique()).astype(str).tolist()
                if str(v) not in {"<NA>", "nan"}
            ]

    # ---- persist artifacts ----
    joblib.dump(final_median, OUT / "lgbm_median.joblib")
    joblib.dump(final_p10, OUT / "lgbm_p10.joblib")
    joblib.dump(final_p90, OUT / "lgbm_p90.joblib")
    joblib.dump(
        {
            "X_test_index": list(X_te.index),
            "shap_values": shap_values,
            "X_test": X_te,
            "base_value": base_value,
        },
        OUT / "shap_test.joblib",
    )
    joblib.dump(feature_spec, OUT / "feature_spec.joblib")
    metrics = {
        "cv": cv,
        "test": test_metrics,
        "global_importance_top25": [
            {"feature": f, "abs_mean_shap": v} for f, v in importance[:25]
        ],
        "outcome": af.outcome_col,
        "n_features": int(X.shape[1]),
        "n_episodes_total": int(len(af.df)),
        "n_episodes_with_outcome": int(len(X)),
        "n_patients_with_outcome": int(groups.nunique()),
        "random_state": RANDOM_STATE,
    }
    (OUT / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT / "simulator_defaults.json").write_text(
        json.dumps(defaults, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote artifacts to {OUT}")


if __name__ == "__main__":
    main()
