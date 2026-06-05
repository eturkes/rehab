"""AIS-grade conversion modeling (G4) — predict the admission->discharge AIS *transition*.

Where the production ``y_discharge_ais`` head predicts the absolute discharge grade, this
module targets the clinically salient *conversion*: does the patient improve, by how much,
and — restricted to the at-risk cohort — do they cross a clinically meaningful threshold.
Conversion is strongly gated by admission grade (a ceiling at AIS D, no room at E), so each
endpoint is modelled on its eligible cohort with admission grade retained as a feature.

Three heads
-----------
* **Clinical endpoint panel** (binary, calibrated probabilities):
    - ``motor_incomplete``: motor-complete admission (AIS A/B) -> motor-incomplete discharge (>=C).
    - ``ambulatory``:       non-ambulatory admission (AIS A-C) -> ambulatory-capable discharge (>=D).
  Each on its at-risk cohort; LightGBM + Platt (sigmoid) calibration on out-of-fold predictions.
* **Ordinal magnitude** (multiclass + APS): improvement size {0, +1, >=+2} on the room-to-improve
  cohort (admission A-D).  Deterioration (rare, ~1.5%) folds into class 0.

Methodology (robust for small cohorts; few heads)
-------------------------------------------------
Grouped 5-fold CV by ``IDNumber`` yields out-of-fold (OOF) predictions for every episode; all
reported metrics, the Platt calibrator, and the APS quantile are computed on those OOF
predictions (the APS q is therefore a *cross-conformal* pool of the per-fold nonconformity
scores — a valid small-sample alternative to the single split-conformal fold ``train.py`` uses
on its larger cohorts).  Final heads are refit on the full cohort for the persisted bundle,
reusing the OOF calibrator / APS q (conservative, mirroring ``landmark.py``'s train-then-refit
conformal).  Global SHAP importances (the drivers panel) are *descriptive*, computed in-sample
on the final head — used to rank drivers, not to claim out-of-sample attribution.

Diagnostic + inference layer, like :mod:`rehab_sci.models.landmark` / :mod:`~.temporal`: writes
its own tracked ``models/conversion_metrics.json`` (identifier-free) + a git-ignored
``models/conversion/bundle.joblib`` and **never touches** ``train.py``'s production artifacts, so
the byte-repro of ``training_metrics.json`` is preserved.  It reuses ``train.py``'s LightGBM
params / fit / typing helpers and ``conformal.py``'s APS so the methodology matches production.

Persisted bundle shape (consumed by dashboard/compute.py::predict_conversion)
-----------------------------------------------------------------------------
    feature_cols, numeric_cols, categorical_cols, adm_col, dis_col
    endpoints[key] = {clf, calibrator, adm_grades, discharge_min, feature_cols, base_rate}
    magnitude      = {clf, aps_q_hat, class_codes, mag_cap, adm_grades, feature_cols}
The Platt ``calibrator`` is a 1-feature ``LogisticRegression`` over the LightGBM logit; apply it
at inference with the same logit transform (``_apply_platt`` is mirrored in compute.py to avoid
importing this module — and thus shap — into the dashboard process).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from rehab_sci.constants import AIS_ORD_TO_LETTER
from rehab_sci.data.dataset import AnalysisFrame, build_analysis_dataset
from rehab_sci.models.conformal import _aps_prediction_set, _aps_scores, _conformal_q
from rehab_sci.models.train import (
    RANDOM_STATE,
    _fit_clf,
    _params_lgbm_clf,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models"

ALPHA = 0.2  # 80% target coverage, matching production
N_SPLITS = 5  # grouped CV folds for OOF predictions
N_CAL_BINS = 5  # quantile bins for the reliability curve (small cohorts -> few bins)

ADM_COL = "AIS_ord"  # admission AIS as ordinal (1=A .. 5=E)
DIS_COL = "y_discharge_ais"  # discharge AIS as ordinal (same encoding)

# Clinical endpoint panel: each a binary conversion on its at-risk admission cohort.
# ``discharge_min`` is the minimum discharge ordinal that counts as a positive conversion.
ENDPOINTS: tuple[dict, ...] = (
    {"key": "motor_incomplete", "adm_grades": (1, 2), "discharge_min": 3},  # A/B -> C+ (motor-incomplete)
    {"key": "ambulatory", "adm_grades": (1, 2, 3), "discharge_min": 4},      # A-C -> D/E (ambulatory-capable)
)

# Ordinal magnitude: improvement size clipped to {0, +1, >=+2} on the room-to-improve cohort.
MAG_CAP = 2
MAG_ADM_GRADES: tuple[int, ...] = (1, 2, 3, 4)  # exclude E (5): no room to improve


# ----------------------------- feature construction ----------------------------

def _typed_X(used: pd.DataFrame, af: AnalysisFrame) -> pd.DataFrame:
    """Admission feature matrix with the schema's categorical / numeric dtypes applied."""
    X = used[af.feature_cols].copy()
    for c in af.categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for c in af.numeric_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _cohort(ep: pd.DataFrame, adm_grades: tuple[int, ...]) -> pd.DataFrame:
    """Episodes admitted at one of ``adm_grades`` with a discharge AIS and a real IDNumber."""
    adm = pd.to_numeric(ep[ADM_COL], errors="coerce")
    dis = pd.to_numeric(ep[DIS_COL], errors="coerce")
    return ep[adm.isin(adm_grades) & dis.notna() & ep["IDNumber"].notna()].copy()


# ----------------------------- LightGBM params / fit ----------------------------

def _params_binary() -> dict:
    """Binary conversion params — no ``class_weight`` (endpoints are near-balanced; weighting
    would distort the probabilities we then Platt-calibrate)."""
    return dict(
        objective="binary",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=-1,
        min_data_in_leaf=10,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=2,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )


def _fit_binary(params, X_tr, y_tr, X_val, y_val, cat_cols):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def _refit(params: dict, X: pd.DataFrame, y: np.ndarray, cat_cols: list[str], best_iter: int):
    """Refit a classifier on the full cohort at a fixed iteration count (no eval split)."""
    params = dict(params)
    params["n_estimators"] = int(best_iter or 400)
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y, categorical_feature=cat_cols)
    return model


# ----------------------------- Platt (sigmoid) calibration ----------------------------

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _fit_platt(prob: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Fit a 1-feature logistic recalibration over the LightGBM logit (Platt scaling).

    ``C`` is large so the fit is effectively unregularized — a clean 2-parameter sigmoid map.
    """
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(_logit(prob).reshape(-1, 1), y)
    return lr


def _apply_platt(cal: LogisticRegression, prob: np.ndarray) -> np.ndarray:
    return cal.predict_proba(_logit(prob).reshape(-1, 1))[:, 1]


# ----------------------------- out-of-fold prediction ----------------------------

def _oof_binary(X, y, groups, cat_cols) -> tuple[np.ndarray, int]:
    """Grouped-CV out-of-fold positive-class probabilities + median best-iteration."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.full(len(y), np.nan)
    iters: list[int] = []
    for tr, te in gkf.split(X, y, groups=groups):
        inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
        i_tr, i_val = next(inner.split(X.iloc[tr], y[tr], groups=groups.iloc[tr]))
        model = _fit_binary(
            _params_binary(),
            X.iloc[tr].iloc[i_tr], y[tr][i_tr],
            X.iloc[tr].iloc[i_val], y[tr][i_val], cat_cols,
        )
        oof[te] = np.asarray(model.predict_proba(X.iloc[te]), float)[:, 1]
        iters.append(int(model.best_iteration_ or 400))
    assert not np.isnan(oof).any(), "GroupKFold left an episode unpredicted"
    return oof, int(np.median(iters))


def _oof_multiclass(X, y_codes, groups, cat_cols, n_classes) -> tuple[np.ndarray, int]:
    """Grouped-CV out-of-fold class-probability matrix + median best-iteration."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros((len(y_codes), n_classes))
    seen = np.zeros(len(y_codes), dtype=bool)
    iters: list[int] = []
    for tr, te in gkf.split(X, y_codes, groups=groups):
        inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
        i_tr, i_val = next(inner.split(X.iloc[tr], y_codes[tr], groups=groups.iloc[tr]))
        model = _fit_clf(
            _params_lgbm_clf(n_classes),
            X.iloc[tr].iloc[i_tr], y_codes[tr][i_tr],
            X.iloc[tr].iloc[i_val], y_codes[tr][i_val], cat_cols,
        )
        oof[te] = np.asarray(model.predict_proba(X.iloc[te]), float)
        seen[te] = True
        iters.append(int(model.best_iteration_ or 400))
    assert seen.all(), "GroupKFold left an episode unpredicted"
    return oof, int(np.median(iters))


# ----------------------------- metrics helpers ----------------------------

def _calibration_curve(prob: np.ndarray, y: np.ndarray, n_bins: int) -> dict:
    """Reliability curve over quantile bins (avoids empty bins on small cohorts)."""
    edges = np.unique(np.quantile(prob, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, len(edges) - 2)
    out: dict[str, list] = {"pred_mean": [], "obs_freq": [], "count": []}
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        out["pred_mean"].append(float(prob[m].mean()))
        out["obs_freq"].append(float(y[m].mean()))
        out["count"].append(int(m.sum()))
    return out


def _shap_top(model, X: pd.DataFrame, top_n: int = 12) -> list[dict]:
    """Descriptive global driver ranking: top features by mean |SHAP| on the full cohort."""
    import shap  # lazy: keeps shap out of the dashboard import path

    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):  # older shap: [neg, pos]
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:  # newer shap on binary: (n, p, 2)
        sv = sv[:, :, -1]
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(-imp)[:top_n]
    return [{"feature": str(X.columns[i]), "mean_abs": float(imp[i])} for i in order]


# ----------------------------- per-head drivers ----------------------------

def _run_endpoint(spec: dict, ep: pd.DataFrame, af: AnalysisFrame) -> tuple[dict, dict]:
    """Fit + score one binary conversion endpoint; return (metrics, persisted-model)."""
    cohort = _cohort(ep, spec["adm_grades"])
    X = _typed_X(cohort, af)
    y = (pd.to_numeric(cohort[DIS_COL], errors="coerce") >= spec["discharge_min"]).astype(int).to_numpy()
    groups = cohort["IDNumber"].astype("float64").astype("int64")
    cat_cols = [c for c in af.categorical_cols if c in X.columns]

    oof, best_iter = _oof_binary(X, y, groups, cat_cols)
    cal = _fit_platt(oof, y)
    oof_cal = _apply_platt(cal, oof)

    base = float(y.mean())
    adm_int = pd.to_numeric(cohort[ADM_COL], errors="coerce").to_numpy().astype(int)
    rate_by_grade = {
        AIS_ORD_TO_LETTER[g]: {"rate": float(y[adm_int == g].mean()), "n": int((adm_int == g).sum())}
        for g in spec["adm_grades"] if (adm_int == g).any()
    }

    final = _refit(_params_binary(), X, y, cat_cols, best_iter)
    metrics = {
        "n": len(y),
        "n_pos": int(y.sum()),
        "base_rate": base,
        "auc": float(roc_auc_score(y, oof)),
        "brier": float(brier_score_loss(y, oof_cal)),
        "brier_raw": float(brier_score_loss(y, oof)),
        "brier_baseline": float(base * (1 - base)),
        "logloss": float(log_loss(y, np.clip(oof_cal, 1e-6, 1 - 1e-6))),
        "calibration_raw": _calibration_curve(oof, y, N_CAL_BINS),
        "calibration": _calibration_curve(oof_cal, y, N_CAL_BINS),
        "rate_by_admission_grade": rate_by_grade,
        "adm_grades": list(spec["adm_grades"]),
        "discharge_min": spec["discharge_min"],
        "shap_top": _shap_top(final, X),
    }
    model = {
        "clf": final,
        "calibrator": cal,
        "adm_grades": list(spec["adm_grades"]),
        "discharge_min": spec["discharge_min"],
        "feature_cols": list(X.columns),
        "base_rate": base,
    }
    return metrics, model


def _run_magnitude(ep: pd.DataFrame, af: AnalysisFrame) -> tuple[dict, dict]:
    """Fit + score the ordinal improvement-magnitude head; return (metrics, persisted-model)."""
    cohort = _cohort(ep, MAG_ADM_GRADES)
    X = _typed_X(cohort, af)
    a = pd.to_numeric(cohort[ADM_COL], errors="coerce").to_numpy()
    d = pd.to_numeric(cohort[DIS_COL], errors="coerce").to_numpy()
    mag = np.clip(d - a, 0, MAG_CAP).astype(int)  # {0, +1, >=+2}; deterioration folds into 0
    groups = cohort["IDNumber"].astype("float64").astype("int64")
    cat_cols = [c for c in af.categorical_cols if c in X.columns]
    n_classes = MAG_CAP + 1

    oof, best_iter = _oof_multiclass(X, mag, groups, cat_cols, n_classes)
    pred = oof.argmax(axis=1)
    q = float(_conformal_q(_aps_scores(oof, mag), ALPHA))
    sets = [_aps_prediction_set(oof[i], q) for i in range(len(mag))]

    adm_int = a.astype(int)
    class_dist_by_grade: dict[str, dict] = {}
    for g in MAG_ADM_GRADES:
        gm = adm_int == g
        if gm.any():
            vals, cnts = np.unique(mag[gm], return_counts=True)
            class_dist_by_grade[AIS_ORD_TO_LETTER[g]] = {
                "n": int(gm.sum()),
                "dist": {int(v): int(c) for v, c in zip(vals, cnts, strict=True)},
            }

    final = _refit(_params_lgbm_clf(n_classes), X, mag, cat_cols, best_iter)
    vals, cnts = np.unique(mag, return_counts=True)
    metrics = {
        "n": len(mag),
        "class_counts": {int(v): int(c) for v, c in zip(vals, cnts, strict=True)},
        "accuracy": float(accuracy_score(mag, pred)),
        "kappa_quadratic": float(cohen_kappa_score(mag, pred, weights="quadratic")),
        "ordinal_mae": float(mean_absolute_error(mag, pred)),
        "aps_q_hat": q,
        "aps_coverage_80": float(np.mean([mag[i] in sets[i] for i in range(len(mag))])),
        "aps_avg_set_size": float(np.mean([len(s) for s in sets])),
        "confusion": confusion_matrix(mag, pred, labels=list(range(n_classes))).tolist(),
        "class_dist_by_admission_grade": class_dist_by_grade,
        "adm_grades": list(MAG_ADM_GRADES),
        "mag_cap": MAG_CAP,
    }
    model = {
        "clf": final,
        "aps_q_hat": q,
        "class_codes": list(range(n_classes)),
        "mag_cap": MAG_CAP,
        "adm_grades": list(MAG_ADM_GRADES),
        "feature_cols": list(X.columns),
    }
    return metrics, model


def _landscape(ep: pd.DataFrame) -> dict:
    """Descriptive conversion landscape over every episode with both admission + discharge AIS."""
    adm = pd.to_numeric(ep[ADM_COL], errors="coerce")
    dis = pd.to_numeric(ep[DIS_COL], errors="coerce")
    m = adm.notna() & dis.notna()
    a = adm[m].astype(int).to_numpy()
    delta = dis[m].astype(int).to_numpy() - a
    vals, cnts = np.unique(delta, return_counts=True)
    improve_by_grade = {
        AIS_ORD_TO_LETTER[g]: {"improve_rate": float((delta[a == g] >= 1).mean()), "n": int((a == g).sum())}
        for g in sorted(np.unique(a))
    }
    return {
        "n_with_both_ais": int(m.sum()),
        "any_improve_rate": float((delta >= 1).mean()),
        "stable_rate": float((delta == 0).mean()),
        "deteriorate_rate": float((delta <= -1).mean()),
        "delta_distribution": {int(v): int(c) for v, c in zip(vals, cnts, strict=True)},
        "improve_rate_by_admission_grade": improve_by_grade,
    }


# ----------------------------- entry point ----------------------------

def main() -> None:
    af = build_analysis_dataset()
    ep = af.df
    print("=" * 64)
    print("AIS-GRADE CONVERSION (G4) — admission->discharge transition modeling")
    print("=" * 64)

    landscape = _landscape(ep)
    print(f"episodes with both admission + discharge AIS: {landscape['n_with_both_ais']}")
    print(f"any >=1-grade improvement: {landscape['any_improve_rate']:.1%}  "
          f"stable {landscape['stable_rate']:.1%}  deteriorate {landscape['deteriorate_rate']:.1%}")

    endpoints_metrics: dict[str, dict] = {}
    endpoints_models: dict[str, dict] = {}
    for spec in ENDPOINTS:
        print(f"\n[endpoint:{spec['key']}]  adm={[AIS_ORD_TO_LETTER[g] for g in spec['adm_grades']]} "
              f"-> discharge>={AIS_ORD_TO_LETTER[spec['discharge_min']]}")
        m, mod = _run_endpoint(spec, ep, af)
        endpoints_metrics[spec["key"]] = m
        endpoints_models[spec["key"]] = mod
        print(f"   n={m['n']} pos={m['n_pos']} base={m['base_rate']:.0%}  "
              f"AUC {m['auc']:.3f}  Brier {m['brier']:.3f} (base {m['brier_baseline']:.3f})")

    print("\n[magnitude]  ordinal improvement size {0,+1,>=+2}")
    mag_metrics, mag_model = _run_magnitude(ep, af)
    print(f"   n={mag_metrics['n']} classes={mag_metrics['class_counts']}  "
          f"κ {mag_metrics['kappa_quadratic']:.3f}  ordMAE {mag_metrics['ordinal_mae']:.3f}  "
          f"APS cov {mag_metrics['aps_coverage_80']:.0%} set {mag_metrics['aps_avg_set_size']:.2f}")

    out_dir = OUT / "conversion"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "feature_cols": list(af.feature_cols),
        "numeric_cols": list(af.numeric_cols),
        "categorical_cols": list(af.categorical_cols),
        "adm_col": ADM_COL,
        "dis_col": DIS_COL,
        "endpoints": endpoints_models,
        "magnitude": mag_model,
    }
    joblib.dump(bundle, out_dir / "bundle.joblib")

    payload = {
        "random_state": RANDOM_STATE,
        "alpha": ALPHA,
        "n_splits": N_SPLITS,
        "adm_col": ADM_COL,
        "dis_col": DIS_COL,
        "landscape": landscape,
        "endpoints": endpoints_metrics,
        "magnitude": mag_metrics,
    }
    (OUT / "conversion_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {out_dir / 'bundle.joblib'} and {OUT / 'conversion_metrics.json'}")


if __name__ == "__main__":
    main()
