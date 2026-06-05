"""Landmark (dynamic) prediction — sharpen the discharge prognosis as early recovery is observed.

For each landmark time ``L`` in :data:`LANDMARKS`, fit one model per discharge outcome from
the admission feature set **plus** a block of last-observed-on-or-before-``L`` (LOCF) clinical
measures (SCIM total + subscales, AIS grade, motor / sensory totals).  As ``L`` advances and
more of the patient's *actual* early recovery is observed, point accuracy rises and the
conformal prediction interval narrows — the "value of observation" curve.

Beyond the full-block landmark head, each landmark also fits one **single-add head per measure**
(admission features + exactly that one ``L_<measure>``), each with its own marginal conformal q /
APS set.  These power the value-of-information surfaces (G2): because every measure has its own
calibrated interval over the *same* baseline + split, they can be ranked by the PI tightening each
single observation buys — prescribing *what to measure next* for a given patient, not just that
observation helps on average.

This is a **diagnostic + inference layer, like** :mod:`rehab_sci.models.temporal`: it writes
its own tracked ``models/landmark_metrics.json`` and a ``models/landmark/bundle.joblib`` and
**never touches** :mod:`rehab_sci.models.train`'s production artifacts, so the byte-repro of
``training_metrics.json`` is preserved.  It reuses ``train.py``'s prep / fit / transform
helpers and ``conformal.py``'s split-conformal / APS so the methodology matches production.

Landmark conditioning (still-admitted risk set)
-----------------------------------------------
An episode is eligible at landmark ``L`` only if it has a tracked observation at an
*intermediate* timepoint at or after ``L`` — evidence the patient was still admitted at ``L``.
This is standard landmarking: it avoids predicting a discharge outcome for a patient already
discharged before ``L`` (the immortal-time / leakage trap).  Because this risk set shrinks
(and skews sicker / longer-stay) as ``L`` grows, each landmark also fits a paired
**admission-only baseline** on the *same* eligible cohort and split; the landmark-minus-baseline
gap therefore isolates the value of the observed scores, controlling for the cohort shift.

Two deltas vs production training: conformal/APS ``q`` is **marginal** (per-landmark cohorts
are too small for stable Mondrian groups), and the persisted PI uses that marginal conformal
interval directly (no quantile-head union).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
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
from rehab_sci.models.conformal import _aps_prediction_set, _aps_scores, _conformal_q
from rehab_sci.models.outcomes import OUTCOMES, OutcomeSpec
from rehab_sci.models.train import (
    RANDOM_STATE,
    _apply_transform,
    _clip,
    _fit_clf,
    _fit_reg,
    _grouped_holdout,
    _inverse_transform,
    _params_lgbm_clf,
    _params_lgbm_reg,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models"

ALPHA = 0.2  # 80% target coverage, matching production

# Landmark times (intermediate-recovery checkpoints) and their nominal day offsets.
LANDMARKS: tuple[str, ...] = ("72h", "2w", "4w", "6w", "2m", "3m")
LANDMARK_DAYS: dict[str, int] = {"72h": 3, "2w": 14, "4w": 28, "6w": 42, "2m": 60, "3m": 90}

# Clinical measures carried forward (last-observed-on-or-before L) as the landmark block.
# All numeric.  These are exactly the recovery-tracking columns the loader computes per row.
LANDMARK_COLS: tuple[str, ...] = (
    "SCIM_total",
    "SCIM_self_care",
    "SCIM_respiration_sphincter",
    "SCIM_mobility",
    "AIS_ord",
    "UEMS",
    "LEMS",
    "TotalMotor",
    "LightTouchTotal",
    "PinPrickTotal",
)
LM_PREFIX = "L_"  # landmark-feature name prefix, e.g. SCIM_total -> L_SCIM_total

# Intermediate timepoint order (chronological), excluding the terminal "discharge" slot.
TIMEPOINT_ORDER: tuple[str, ...] = (
    "0day", "72h", "2w", "4w", "6w", "2m", "3m", "4m", "5m", "6m",
    "7m", "8m", "9m", "10m", "11m", "1y", "2y", "3y", "4y", "5y",
    "6y", "7y", "8y", "9y", "10y",
)
_OIDX: dict[str, int] = {tp: i for i, tp in enumerate(TIMEPOINT_ORDER)}

MIN_COHORT = 120  # skip an (outcome, landmark) cell below this many eligible episodes


# ----------------------------- landmark feature construction ----------------------------

def _latest_intermediate_oidx(long: pd.DataFrame) -> pd.Series:
    """Per-episode index of the latest intermediate timepoint carrying any tracked observation.

    Drives the still-admitted-at-L risk set: an episode is eligible at landmark L iff this
    value is >= the order-index of L.  The terminal ``discharge`` slot is excluded from
    ``TIMEPOINT_ORDER`` so a discharge-only measurement never counts as "still admitted".
    """
    cols = list(LANDMARK_COLS)
    sub = long[["KeyRecordNumber", "TIME_Name", *cols]].copy()
    sub["_oi"] = sub["TIME_Name"].map(_OIDX)
    has_obs = sub[cols].notna().any(axis=1)
    sub = sub[has_obs & sub["_oi"].notna()]
    return sub.groupby("KeyRecordNumber")["_oi"].max()


def _locf_block(long: pd.DataFrame, landmark: str) -> pd.DataFrame:
    """LOCF landmark block: last non-null value at or before ``landmark`` per episode.

    Returns a frame indexed by KeyRecordNumber with one ``L_<col>`` column per landmark
    measure (NaN where the episode has no observation of that measure up to the landmark).
    """
    cutoff = _OIDX[landmark]
    window = [tp for tp in TIMEPOINT_ORDER if _OIDX[tp] <= cutoff]
    sub = long[long["TIME_Name"].isin(window)].copy()
    sub["_oi"] = sub["TIME_Name"].map(_OIDX)
    sub = sub.sort_values("_oi")  # ascending so .last() picks the most-recent non-null
    block = {}
    for col in LANDMARK_COLS:
        block[LM_PREFIX + col] = (
            sub.dropna(subset=[col]).groupby("KeyRecordNumber")[col].last()
        )
    return pd.DataFrame(block)


def _prep_landmark(
    af: AnalysisFrame,
    target_col: str,
    eligible: set,
    lm_block: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.Series, list[str]]:
    """Build paired (X_base, X_landmark) matrices + target/groups for one (outcome, landmark) cell.

    Rows = episodes with the target + IDNumber present AND still admitted at the landmark.
    ``X_base`` is the admission feature set; ``X_landmark`` appends the LOCF landmark block.
    Both share row order, so a single grouped split applies to both.
    """
    ep = af.df
    used = ep[ep["KeyRecordNumber"].isin(eligible)].dropna(subset=[target_col, "IDNumber"]).copy()

    X_base = used[af.feature_cols].copy()
    for c in af.categorical_cols:
        if c in X_base.columns:
            X_base[c] = X_base[c].astype("category")
    for c in af.numeric_cols:
        if c in X_base.columns:
            X_base[c] = pd.to_numeric(X_base[c], errors="coerce")

    lm_aligned = lm_block.reindex(used["KeyRecordNumber"]).set_axis(used.index)
    for c in lm_aligned.columns:
        lm_aligned[c] = pd.to_numeric(lm_aligned[c], errors="coerce")
    X_lm = pd.concat([X_base, lm_aligned], axis=1)

    y_raw = used[target_col]
    groups = used["IDNumber"].astype("float64").astype("int64")
    cat_cols = [c for c in af.categorical_cols if c in X_base.columns]
    return X_base, X_lm, pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float), groups, cat_cols


# ----------------------------- fit / eval helpers ----------------------------

def _refit_all(params: dict, X: pd.DataFrame, y: np.ndarray, cat_cols: list[str],
               best_iter: int | None, *, clf: bool = False):
    params = dict(params)
    params["n_estimators"] = int(best_iter or 400)
    model = lgb.LGBMClassifier(**params) if clf else lgb.LGBMRegressor(**params)
    model.fit(X, y, categorical_feature=cat_cols)
    return model


def _eval_regression(
    X: pd.DataFrame, y_t: np.ndarray, y_raw: np.ndarray, cat_cols: list[str],
    tr: np.ndarray, cal: np.ndarray, te: np.ndarray,
    transform: str | None, clip_min: float | None, clip_max: float | None,
    *, persist: bool,
) -> tuple[dict, dict | None]:
    """Fit a regression head on the train fold, conformalise on the calibration fold, score on test.

    Returns (metrics, models).  ``models`` (median refit on the full cohort + marginal conformal
    q) is built only when ``persist`` is set.  Both the admission-only baseline and the landmark
    head are persisted so the dashboard can show their paired (value-of-observation) comparison.
    """
    med = _fit_reg(_params_lgbm_reg(), X.iloc[tr], y_t[tr], X.iloc[cal], y_t[cal], cat_cols)
    q = _conformal_q(np.abs(y_t[cal] - med.predict(X.iloc[cal])), ALPHA)

    pt = med.predict(X.iloc[te])
    pred = _clip(_inverse_transform(pt, transform), clip_min, clip_max)
    lo = _clip(_inverse_transform(pt - q, transform), clip_min, clip_max)
    hi = _clip(_inverse_transform(pt + q, transform), clip_min, clip_max)
    y_te = y_raw[te]
    metrics = {
        "r2": float(r2_score(y_te, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "mae": float(mean_absolute_error(y_te, pred)),
        "coverage_80": float(((y_te >= lo) & (y_te <= hi)).mean()),
        "pi_halfwidth_raw": float(np.mean((hi - lo) / 2.0)),  # mean PI half-width on the natural scale
        "conformal_q_transformed": float(q),
    }
    models = None
    if persist:
        # Landmark inference uses the marginal-conformal interval directly (no quantile-head
        # union), so only the median head + its conformal q are persisted — the p10/p90 heads
        # the production simulator needs are intentionally omitted here.
        models = {
            "median": _refit_all(_params_lgbm_reg(), X, y_t, cat_cols, med.best_iteration_),
            "conformal_q": float(q),
            "feature_cols": list(X.columns),
        }
    return metrics, models


def _eval_multiclass(
    X: pd.DataFrame, y_codes: np.ndarray, groups: pd.Series, cat_cols: list[str],
    class_codes: tuple[int, ...], tr: np.ndarray, cal: np.ndarray, te: np.ndarray,
    *, persist: bool,
) -> tuple[dict, dict | None]:
    """Fit the AIS multiclass head, calibrate APS sets on the calibration fold, score on test."""
    n_classes = len(class_codes)
    code_arr = np.array(class_codes)

    # inner group split of the train fold for early stopping
    g_tr = groups.iloc[tr]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
    i_tr, i_val = next(inner.split(X.iloc[tr], y_codes[tr], groups=g_tr))
    X_tr_full = X.iloc[tr]
    clf = _fit_clf(
        _params_lgbm_clf(n_classes),
        X_tr_full.iloc[i_tr], y_codes[tr][i_tr],
        X_tr_full.iloc[i_val], y_codes[tr][i_val], cat_cols,
    )

    aps_q = _conformal_q(_aps_scores(np.asarray(clf.predict_proba(X.iloc[cal]), float), y_codes[cal]), ALPHA)

    proba_te = np.asarray(clf.predict_proba(X.iloc[te]), float)
    pred_idx = np.asarray(clf.predict(X.iloc[te]), dtype=int)
    sets = [_aps_prediction_set(proba_te[i], aps_q) for i in range(len(te))]
    y_te = y_codes[te]
    metrics = {
        "accuracy": float(accuracy_score(y_te, pred_idx)),
        "kappa_quadratic": float(cohen_kappa_score(y_te, pred_idx, weights="quadratic")),
        "ordinal_mae": float(mean_absolute_error(code_arr[y_te], code_arr[pred_idx])),
        "aps_coverage_80": float(np.mean([y_te[i] in sets[i] for i in range(len(te))])),
        "aps_avg_set_size": float(np.mean([len(s) for s in sets])),
        "aps_q_hat": float(aps_q),
    }
    models = None
    if persist:
        final = _refit_all(_params_lgbm_clf(n_classes), X, y_codes, cat_cols,
                           clf.best_iteration_, clf=True)
        models = {
            "clf": final,
            "aps_q_hat": float(aps_q),
            "class_codes": [int(c) for c in class_codes],
            "class_labels": None,  # filled by caller from spec
            "feature_cols": list(X.columns),
        }
    return metrics, models


# ----------------------------- per-outcome driver ----------------------------

def _eval_cell(spec: OutcomeSpec, X: pd.DataFrame, y_t, y_raw, y_codes, groups, cat_cols,
               tr: np.ndarray, cal: np.ndarray, te: np.ndarray) -> tuple[dict, dict | None]:
    """Eval + persist one head on matrix ``X`` for ``spec`` (dispatches by task).

    Shared by the admission baseline, the full landmark block, and every single-add head so all
    three are fit / conformalised identically (same split, same persist contract).
    """
    if spec.task == "regression":
        return _eval_regression(X, y_t, y_raw, cat_cols, tr, cal, te,
                                spec.transform, spec.clip_min, spec.clip_max, persist=True)
    m, models = _eval_multiclass(X, y_codes, groups, cat_cols, spec.class_codes,
                                 tr, cal, te, persist=True)
    if models is not None:
        models["class_labels"] = list(spec.class_labels)
    return m, models


def _run_outcome(spec: OutcomeSpec, af: AnalysisFrame, lm_blocks: dict[str, pd.DataFrame],
                 max_oi: pd.Series) -> tuple[dict, dict]:
    """Fit every landmark (paired baseline + landmark model) for one outcome.

    Returns (metrics_by_landmark, models_by_landmark).
    """
    metrics_by_lm: dict[str, dict] = {}
    models_by_lm: dict[str, dict] = {}
    code_to_idx = {int(c): i for i, c in enumerate(spec.class_codes)} if spec.task == "multiclass" else {}

    for lm in LANDMARKS:
        eligible = set(max_oi[max_oi >= _OIDX[lm]].index)
        X_base, X_lm, y_raw, groups, cat_cols = _prep_landmark(
            af, spec.target_col, eligible, lm_blocks[lm]
        )
        n = len(X_base)
        if n < MIN_COHORT:
            print(f"   [{spec.key}/{lm}]  SKIP  n={n} < {MIN_COHORT}")
            continue

        # one grouped split (identical rows for base & landmark)
        tr_dev, te = _grouped_holdout(X_base, y_raw, groups, test_size=0.2)
        g_dev = groups.iloc[tr_dev]
        tr_in, cal_in = _grouped_holdout(X_base.iloc[tr_dev], y_raw[tr_dev], g_dev, test_size=0.2)
        tr, cal = tr_dev[tr_in], tr_dev[cal_in]
        n_test_pat = int(groups.iloc[te].nunique())

        if spec.task == "regression":
            y_t = _apply_transform(y_raw, spec.transform)
            y_codes = None
        else:
            y_t = None
            y_codes = np.array([code_to_idx[int(v)] for v in y_raw.astype(int)])

        # Admission-only baseline, full-observation-block landmark, and one single-add head per
        # measure (admission + exactly that L_<measure>).  The single-add heads each carry their
        # OWN conformal q / APS set, so the value-of-information surfaces can rank measures by the
        # PI tightening each buys over the *same* baseline + split (G2).
        base_m, base_models = _eval_cell(spec, X_base, y_t, y_raw, y_codes, groups, cat_cols, tr, cal, te)
        lm_m, lm_models = _eval_cell(spec, X_lm, y_t, y_raw, y_codes, groups, cat_cols, tr, cal, te)
        base_cols = list(X_base.columns)
        single_metrics: dict[str, dict] = {}
        single_models: dict[str, dict] = {}
        for meas in LANDMARK_COLS:
            sm, smodels = _eval_cell(spec, X_lm[[*base_cols, LM_PREFIX + meas]],
                                     y_t, y_raw, y_codes, groups, cat_cols, tr, cal, te)
            single_metrics[meas] = sm
            single_models[meas] = smodels

        if spec.task == "regression":
            print(f"   [{spec.key}/{lm}]  n={n} test={len(te)}  "
                  f"R² {base_m['r2']:.3f}→{lm_m['r2']:.3f}  "
                  f"PIhw {base_m['pi_halfwidth_raw']:.1f}→{lm_m['pi_halfwidth_raw']:.1f}  "
                  f"cov {lm_m['coverage_80']:.0%}  (+{len(single_metrics)} single-add)")
        else:
            print(f"   [{spec.key}/{lm}]  n={n} test={len(te)}  "
                  f"κ {base_m['kappa_quadratic']:.3f}→{lm_m['kappa_quadratic']:.3f}  "
                  f"APSset {base_m['aps_avg_set_size']:.2f}→{lm_m['aps_avg_set_size']:.2f}  "
                  f"cov {lm_m['aps_coverage_80']:.0%}  (+{len(single_metrics)} single-add)")

        metrics_by_lm[lm] = {
            "n_eligible": n,
            "n_test": len(te),
            "n_test_patients": n_test_pat,
            "baseline": base_m,
            "landmark": lm_m,
            "single": single_metrics,
        }
        # Paired heads (admission-only vs full block) + per-measure single-add heads so the
        # dashboard can contrast baseline vs landmark and rank each measure's value per patient.
        models_by_lm[lm] = {"baseline": base_models, "landmark": lm_models, "single": single_models}
    return metrics_by_lm, models_by_lm


# ----------------------------- entry point ----------------------------

def main() -> None:
    af = build_analysis_dataset()
    long = af.longitudinal
    print("=" * 64)
    print("LANDMARK (dynamic) prediction — value of observed early recovery")
    print("=" * 64)
    print(f"episodes={len(af.df)}  landmarks={list(LANDMARKS)}  "
          f"landmark_block={len(LANDMARK_COLS)} measures (LOCF)")

    max_oi = _latest_intermediate_oidx(long)
    lm_blocks = {lm: _locf_block(long, lm) for lm in LANDMARKS}

    # Per-landmark reference distribution of each measure over the still-admitted eligible cohort
    # (outcome-independent).  Used at inference to impute a not-yet-observed measure when ranking
    # value-of-information per patient (G2).
    value_summary: dict[str, dict] = {}
    for lm in LANDMARKS:
        eligible_idx = max_oi[max_oi >= _OIDX[lm]].index
        blk = lm_blocks[lm].reindex(eligible_idx)
        per: dict[str, dict] = {}
        for meas in LANDMARK_COLS:
            col = LM_PREFIX + meas
            if col not in blk.columns:
                continue
            s = pd.to_numeric(blk[col], errors="coerce").dropna()
            if len(s):
                per[meas] = {
                    "median": float(s.median()),
                    "q25": float(s.quantile(0.25)),
                    "q75": float(s.quantile(0.75)),
                    "n": len(s),
                }
        value_summary[lm] = per

    all_metrics: dict[str, dict] = {}
    all_models: dict[str, dict] = {}
    for spec in OUTCOMES:
        print(f"\n[{spec.key}]  task={spec.task}")
        m, models = _run_outcome(spec, af, lm_blocks, max_oi)
        all_metrics[spec.key] = {"task": spec.task, "transform": spec.transform,
                                 "clip_min": spec.clip_min, "clip_max": spec.clip_max,
                                 "by_landmark": m}
        all_models[spec.key] = {"task": spec.task, "transform": spec.transform,
                                "clip_min": spec.clip_min, "clip_max": spec.clip_max,
                                "by_landmark": models}

    lm_dir = OUT / "landmark"
    lm_dir.mkdir(parents=True, exist_ok=True)
    # Bundle shape (consumed by dashboard/compute.py::predict_landmark / landmark_voi):
    #   outcomes[key] = {task, transform, clip_min, clip_max, by_landmark}
    #   by_landmark[L] = {"baseline": head, "landmark": head, "single": {measure: head}}
    #                     # L present only if not skipped (eligible n >= MIN_COHORT)
    #   head (regression)  = {median, conformal_q, feature_cols}
    #   head (multiclass)  = {clf, aps_q_hat, class_codes, class_labels, feature_cols}
    #   baseline.feature_cols == feature_cols_base (30); landmark == base + all 10 L_<measure>;
    #   single[m].feature_cols == base + the one L_<measure> (each measure has its own conformal q).
    # lm_value_summary[L][measure] = {median, q25, q75, n} over the still-admitted eligible cohort,
    #   for imputing a not-yet-observed measure when ranking value-of-information per patient (G2).
    bundle = {
        "landmarks": list(LANDMARKS),
        "landmark_days": LANDMARK_DAYS,
        "landmark_cols": list(LANDMARK_COLS),
        "lm_prefix": LM_PREFIX,
        "timepoint_order": list(TIMEPOINT_ORDER),  # for real-patient LOCF reconstruction at inference
        "feature_cols_base": list(af.feature_cols),
        "numeric_cols": list(af.numeric_cols),
        "categorical_cols": list(af.categorical_cols),
        "lm_value_summary": value_summary,
        "outcomes": all_models,
    }
    joblib.dump(bundle, lm_dir / "bundle.joblib")

    payload = {
        "landmarks": list(LANDMARKS),
        "landmark_days": LANDMARK_DAYS,
        "landmark_cols": list(LANDMARK_COLS),
        "random_state": RANDOM_STATE,
        "alpha": ALPHA,
        "outcomes": all_metrics,
    }
    (OUT / "landmark_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {lm_dir / 'bundle.joblib'} and {OUT / 'landmark_metrics.json'}")


if __name__ == "__main__":
    main()
