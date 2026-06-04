"""Temporal (out-of-time) validation via rolling-origin expanding-window backtest.

The production models (``train.py``) are scored with a *random* GroupKFold split
by patient.  That split is blind to calendar time, so it cannot reveal drift:
case-mix, practice patterns, and coding shift across the 2014-2025 span.  This
module measures that drift.

For each origin / test year ``T`` in :data:`TEST_YEARS` it trains on every
episode with ``BusinessYear < T`` and tests on the episodes from year ``T``
(expanding window).  Per outcome and per origin it reports:

* point accuracy — R2/RMSE/MAE (regression) or accuracy / quadratic-weighted
  kappa / ordinal-MAE (AIS multiclass);
* uncertainty calibration out-of-time — marginal split-conformal 80% PI
  coverage + mean width (regression) or marginal APS 80% set coverage +
  average set size (AIS).

Methodology mirrors ``train.py`` exactly, with two deliberate differences:
the dev/test holdout is a *temporal* cut instead of a random group split, and
conformal/APS calibration is *marginal* (per-origin per-group folds are far too
small for the Mondrian variant).  The split is group-safe: a patient whose
episodes straddle the boundary is kept wholly in the past, so their test-year
episodes are dropped (and counted as ``n_dropped_overlap``).

This is a *diagnostic*: it writes ``models/temporal_metrics.json`` and touches
no artifact the dashboard loads for prediction.  The in-time baselines echoed
into that file are read back from ``models/training_metrics.json`` (random
GroupKFold CV for point metrics; the random holdout for coverage).

Run::

    uv run python -m rehab_sci.models.temporal
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit

from rehab_sci.data.dataset import AnalysisFrame, build_analysis_dataset
from rehab_sci.models.conformal import _aps_scores, _aps_test_metrics, _conformal_q
from rehab_sci.models.outcomes import OUTCOMES, OutcomeSpec
from rehab_sci.models.train import (
    RANDOM_STATE,
    _apply_transform,
    _clip,
    _fit_clf,
    _fit_reg,
    _inverse_transform,
    _params_lgbm_clf,
    _params_lgbm_reg,
    _prep,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models"

YEAR_COL = "BusinessYear"
ALPHA = 0.2  # 80% target coverage — matches train.py
# Expanding-window origins: train on BusinessYear < T, test on year T.
# Start at 2020 so the first training window spans 2014-2019 (6 yrs).
TEST_YEARS: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025)
MIN_DEV = 40  # skip an origin if the past holds fewer labelled dev episodes
MIN_TEST = 5  # skip an origin if the test year holds fewer labelled episodes


# ----------------------------- split helpers -----------------------------

def _prep_with_year(
    af: AnalysisFrame, spec: OutcomeSpec
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], pd.Series]:
    """``train._prep`` plus the per-row BusinessYear (aligned to X's index)."""
    X, y_raw, groups, cat_cols = _prep(
        af.df, af.feature_cols, af.numeric_cols, af.categorical_cols, spec.target_col
    )
    year = pd.to_numeric(af.df.loc[X.index, YEAR_COL], errors="coerce")
    return X, y_raw, groups, cat_cols, year


def _origin_masks(
    groups: pd.Series, year: pd.Series, test_year: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Boolean dev/test masks for one origin, group-safe by patient.

    dev = years < test_year; test = year == test_year minus any episode whose
    patient already appears in dev (kept wholly in the past).  Returns
    ``(dev_mask, test_mask, n_dropped_overlap)``.
    """
    dev = (year < test_year).to_numpy()
    test = (year == test_year).to_numpy()
    dev_ids = set(groups[dev].tolist())
    in_dev_patient = groups.isin(dev_ids).to_numpy()
    overlap = test & in_dev_patient
    return dev, test & ~in_dev_patient, int(overlap.sum())


# ----------------------------- per-origin evals --------------------------

def _eval_regression_origin(
    X: pd.DataFrame,
    y_raw: pd.Series,
    groups: pd.Series,
    cat_cols: list[str],
    year: pd.Series,
    spec: OutcomeSpec,
    test_year: int,
) -> dict | None:
    dev, test, n_overlap = _origin_masks(groups, year, test_year)
    if int(dev.sum()) < MIN_DEV or int(test.sum()) < MIN_TEST:
        return None
    y_t = _apply_transform(y_raw, spec.transform)
    y_raw_arr = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)

    X_dev, X_te = X.iloc[dev], X.iloc[test]
    y_dev_t = y_t[dev]
    g_dev = groups.iloc[dev]

    # calibration carve from the past (random within the training window);
    # X_cal doubles as the early-stopping eval set, exactly as train._train_regression.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr, cal = next(gss.split(X_dev, y_dev_t, groups=g_dev))
    X_tr, X_cal = X_dev.iloc[tr], X_dev.iloc[cal]
    y_tr_t, y_cal_t = y_dev_t[tr], y_dev_t[cal]

    median = _fit_reg(_params_lgbm_reg(), X_tr, y_tr_t, X_cal, y_cal_t, cat_cols)
    q = _conformal_q(np.abs(y_cal_t - median.predict(X_cal)), ALPHA)

    pred_t = median.predict(X_te)
    pred = _clip(_inverse_transform(pred_t, spec.transform), spec.clip_min, spec.clip_max)
    lo = _clip(_inverse_transform(pred_t - q, spec.transform), spec.clip_min, spec.clip_max)
    hi = _clip(_inverse_transform(pred_t + q, spec.transform), spec.clip_min, spec.clip_max)
    y_te = y_raw_arr[test]
    return {
        "test_year": int(test_year),
        "n_train": len(X_tr),
        "n_calib": len(X_cal),
        "n_test": len(X_te),
        "n_test_patients": int(groups.iloc[test].nunique()),
        "n_dropped_overlap": n_overlap,
        "r2": float(r2_score(y_te, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "mae": float(mean_absolute_error(y_te, pred)),
        "conformal_coverage_80": float(((y_te >= lo) & (y_te <= hi)).mean()),
        "pi_mean_width": float(np.mean(hi - lo)),
        "conformal_q_transformed": float(q),
    }


def _eval_multiclass_origin(
    X: pd.DataFrame,
    y_raw: pd.Series,
    groups: pd.Series,
    cat_cols: list[str],
    year: pd.Series,
    spec: OutcomeSpec,
    test_year: int,
) -> dict | None:
    dev, test, n_overlap = _origin_masks(groups, year, test_year)
    if int(dev.sum()) < MIN_DEV or int(test.sum()) < MIN_TEST:
        return None
    class_codes = spec.class_codes
    n_classes = len(class_codes)
    code_to_idx = {int(c): i for i, c in enumerate(class_codes)}
    y_int = pd.to_numeric(y_raw, errors="coerce").astype(int).to_numpy()
    y_codes = np.array([code_to_idx[int(v)] for v in y_int])
    code_arr = np.array(class_codes)

    X_dev, X_te = X.iloc[dev], X.iloc[test]
    y_dev, y_te = y_codes[dev], y_codes[test]
    g_dev = groups.iloc[dev]

    # cal carve from dev (APS calibration); inner val carve for early stopping —
    # mirrors train._train_multiclass.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_full, cal = next(gss.split(X_dev, y_dev, groups=g_dev))
    X_tr_full, X_cal = X_dev.iloc[tr_full], X_dev.iloc[cal]
    y_tr_full, y_cal = y_dev[tr_full], y_dev[cal]
    g_tr_full = g_dev.iloc[tr_full]

    inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
    itr, ival = next(inner.split(X_tr_full, y_tr_full, groups=g_tr_full))
    X_tr, X_val = X_tr_full.iloc[itr], X_tr_full.iloc[ival]
    y_tr, y_val = y_tr_full[itr], y_tr_full[ival]

    clf = _fit_clf(_params_lgbm_clf(n_classes), X_tr, y_tr, X_val, y_val, cat_cols)
    proba_cal = np.asarray(clf.predict_proba(X_cal), dtype=float)
    q_hat = _conformal_q(_aps_scores(proba_cal, y_cal), ALPHA)

    pred_idx = np.asarray(clf.predict(X_te), dtype=int)
    proba_te = np.asarray(clf.predict_proba(X_te), dtype=float)
    aps = _aps_test_metrics(proba_te, y_te, np.full(len(X_te), q_hat), X_te)
    labels = list(range(n_classes))
    return {
        "test_year": int(test_year),
        "n_train": len(X_tr),
        "n_calib": len(X_cal),
        "n_test": len(X_te),
        "n_test_patients": int(groups.iloc[test].nunique()),
        "n_dropped_overlap": n_overlap,
        "accuracy": float(accuracy_score(y_te, pred_idx)),
        "kappa_quadratic": float(
            cohen_kappa_score(y_te, pred_idx, labels=labels, weights="quadratic")
        ),
        "ordinal_mae": float(mean_absolute_error(code_arr[y_te], code_arr[pred_idx])),
        "aps_coverage_80": aps["coverage"],
        "aps_avg_set_size": aps["avg_set_size"],
        "aps_q_hat": float(q_hat),
        "class_dist_test": {
            spec.class_labels[i]: int((y_te == i).sum()) for i in range(n_classes)
        },
    }


# ----------------------------- baselines + summary -----------------------

def _load_baselines() -> dict:
    path = OUT / "training_metrics.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("outcomes", {})
    except FileNotFoundError:
        return {}


def _baseline_for(spec: OutcomeSpec, metrics: dict) -> dict:
    m = metrics.get(spec.key, {})
    cv, test = m.get("cv", {}), m.get("test", {})
    if spec.task == "regression":
        return {
            "r2": cv.get("r2_mean"),
            "rmse": cv.get("rmse_mean"),
            "mae": cv.get("mae_mean"),
            "conformal_coverage_80": test.get("conformal_coverage_80"),
            "source": "random GroupKFold CV (point) + random holdout (coverage)",
        }
    return {
        "accuracy": cv.get("accuracy_mean"),
        "kappa_quadratic": cv.get("kappa_quadratic_mean"),
        "ordinal_mae": cv.get("ordinal_mae_mean"),
        "aps_coverage_80": test.get("aps_coverage_80"),
        "aps_avg_set_size": test.get("aps_avg_set_size"),
        "source": "random GroupKFold CV (point) + random holdout (APS coverage)",
    }


def _summarize(origins: list[dict], task: str, baseline: dict) -> dict:
    if not origins:
        return {}

    def mean(k: str) -> float | None:
        vals = [o[k] for o in origins if o.get(k) is not None]
        return float(np.mean(vals)) if vals else None

    if task == "regression":
        s = {
            "r2_oot_mean": mean("r2"),
            "rmse_oot_mean": mean("rmse"),
            "mae_oot_mean": mean("mae"),
            "coverage_oot_mean": mean("conformal_coverage_80"),
            "pi_mean_width_oot_mean": mean("pi_mean_width"),
        }
        if baseline.get("r2") is not None and s["r2_oot_mean"] is not None:
            s["r2_delta_vs_baseline"] = s["r2_oot_mean"] - baseline["r2"]
        if baseline.get("conformal_coverage_80") is not None and s["coverage_oot_mean"] is not None:
            s["coverage_delta_vs_baseline"] = (
                s["coverage_oot_mean"] - baseline["conformal_coverage_80"]
            )
        return s
    s = {
        "accuracy_oot_mean": mean("accuracy"),
        "kappa_quadratic_oot_mean": mean("kappa_quadratic"),
        "ordinal_mae_oot_mean": mean("ordinal_mae"),
        "aps_coverage_oot_mean": mean("aps_coverage_80"),
        "aps_avg_set_size_oot_mean": mean("aps_avg_set_size"),
    }
    if baseline.get("accuracy") is not None and s["accuracy_oot_mean"] is not None:
        s["accuracy_delta_vs_baseline"] = s["accuracy_oot_mean"] - baseline["accuracy"]
    if baseline.get("aps_coverage_80") is not None and s["aps_coverage_oot_mean"] is not None:
        s["aps_coverage_delta_vs_baseline"] = (
            s["aps_coverage_oot_mean"] - baseline["aps_coverage_80"]
        )
    return s


# ----------------------------- driver ------------------------------------

def main() -> None:
    af = build_analysis_dataset()
    baselines = _load_baselines()
    out: dict = {
        "config": {
            "test_years": list(TEST_YEARS),
            "scheme": (
                "expanding-window rolling-origin: train on BusinessYear < test_year, "
                "test on test_year; group-safe by IDNumber; marginal conformal/APS; "
                f"alpha={ALPHA}"
            ),
            "random_state": RANDOM_STATE,
            "year_col": YEAR_COL,
            "min_dev": MIN_DEV,
            "min_test": MIN_TEST,
            "alpha": ALPHA,
        },
        "outcomes": {},
    }
    for spec in OUTCOMES:
        X, y_raw, groups, cat_cols, year = _prep_with_year(af, spec)
        baseline = _baseline_for(spec, baselines)
        origins: list[dict] = []
        print(
            f"\n[{spec.key}] task={spec.task}  labelled={len(X)}  "
            f"years={int(year.min())}-{int(year.max())}"
        )
        for ty in TEST_YEARS:
            r = (
                _eval_regression_origin(X, y_raw, groups, cat_cols, year, spec, ty)
                if spec.task == "regression"
                else _eval_multiclass_origin(X, y_raw, groups, cat_cols, year, spec, ty)
            )
            if r is None:
                print(f"   {ty}: skipped (insufficient dev/test)")
                continue
            origins.append(r)
            if spec.task == "regression":
                print(
                    f"   {ty}: n_te={r['n_test']:3d}  R2={r['r2']:+.3f}  "
                    f"MAE={r['mae']:6.2f}  cov80={r['conformal_coverage_80']:.0%}  "
                    f"w={r['pi_mean_width']:6.1f}  drop={r['n_dropped_overlap']}"
                )
            else:
                print(
                    f"   {ty}: n_te={r['n_test']:3d}  acc={r['accuracy']:.3f}  "
                    f"k={r['kappa_quadratic']:+.3f}  ordMAE={r['ordinal_mae']:.2f}  "
                    f"APScov={r['aps_coverage_80']:.0%}  set={r['aps_avg_set_size']:.2f}"
                )
        out["outcomes"][spec.key] = {
            "task": spec.task,
            "baseline": baseline,
            "origins": origins,
            "summary": _summarize(origins, spec.task, baseline),
        }
    path = OUT / "temporal_metrics.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path.relative_to(ROOT)}  ({len(out['outcomes'])} outcomes)")


if __name__ == "__main__":
    main()
