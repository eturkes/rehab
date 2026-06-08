"""Neuro-functional dissociation modeling (G11) — does function track neurology in recovery?

Spinal-cord-injury recovery has two largely-separate axes: *neurological* recovery (motor/
sensory score gain) and *functional* recovery (ADL independence gain).  SCI trials report them
separately because they dissociate — a patient can regain strength without translating it into
independence (poor functional translation) or, far more commonly, reach independence through
compensation/adaptation despite limited neurological gain (the core value of rehabilitation).

This module quantifies and predicts that dissociation directly.  On this cohort the two axes are
only weakly coupled — Pearson r ≈ 0.27 for total-motor↔SCIM-total, and as low as ≈0.12 for
LEMS↔mobility (lower-limb strength barely predicts mobility independence, which comes via
wheelchair skills) — so dissociation is the norm, not the exception (~39-50% of episodes
off-diagonal).

The dissociation score
----------------------
Because the correlation is low, any residual-based score is dominated by the functional axis; a
genuine *dissociation* must weight both axes equally.  We use the **standardized contrast**

    D = z(Δfunction) − z(Δneuro)            (z = cohort standardization of each Δ)

so D is symmetric (corr ≈ ±0.6 with each axis), mean 0.  **D > 0 = functional over-achiever**
(function outpaces neurology — compensation/adaptation); **D < 0 = functional under-achiever**
(neurology outpaces function — a flag for functional-translation focus).  Δneuro is the G9
``y_delta_*`` (discharge − first-non-null-admission score, already on the episode frame); Δfunction
is the matching discharge-slot SCIM − its admission baseline feature.  No leakage: the target uses
discharge values, the features are admission-only (the admission baselines legitimately enter both,
exactly as in G9's Δ-score heads).

Three domain-paired axes (user choice: domain-paired ⊇ the global score)
-----------------------------------------------------------------------
* ``uems_selfcare``     — UEMS (upper-limb motor)   ↔ SCIM self-care     (r ≈ 0.21)
* ``lems_mobility``     — LEMS (lower-limb motor)    ↔ SCIM mobility      (r ≈ 0.12, the striking decoupling)
* ``totalmotor_total``  — total motor               ↔ SCIM total         (r ≈ 0.27, the global headline)

Two heads per axis (user choice: both)
--------------------------------------
* **over_achiever** (binary, calibrated): P(D > 0).  No ``class_weight`` (the cohort is
  near-balanced ~0.47-0.52 → Platt-calibratable) — *the* calibrated direction probability.
* **magnitude** (continuous regression on D) + a marginal cross-conformal 80% PI.  Direction +
  magnitude + uncertainty in one signed head.  Unlike the G4/G10 binary-vs-balanced-magnitude
  CRUX, the magnitude here is a *regression* (no class weighting), so the two heads are honestly
  comparable: surface the binary as the calibrated probability and the magnitude as the signed
  point estimate + PI; sign(magnitude) ≈ the binary direction by construction.

Methodology (robust for small cohorts; few heads)
-------------------------------------------------
Identical to :mod:`rehab_sci.models.conversion` / :mod:`~.level_descent` (whose binary plumbing is
imported verbatim): grouped 5-fold CV by ``IDNumber`` → out-of-fold (OOF) predictions drive every
reported metric, the Platt calibrator (binary), and the marginal cross-conformal q (the magnitude
PI half-width = the (1−α) quantile of the OOF |residual| pool — the small-sample analogue of
production's single split-conformal fold).  Final heads refit on the full cohort reusing the OOF
calibrator / conformal q (conservative).  SHAP drivers are *descriptive* in-sample.

Diagnostic + inference layer, like conversion/level_descent/independence/topography: writes a
tracked identifier-free ``models/dissociation_metrics.json`` + a git-ignored
``models/dissociation/bundle.joblib`` and **never touches** ``train.py``'s production artifacts
(byte-repro of ``training_metrics.json`` preserved).

Persisted bundle shape (consumed by dashboard/compute.py::predict_dissociation in Part 2)
-----------------------------------------------------------------------------------------
    feature_cols, numeric_cols, categorical_cols, alpha
    axes (ordered keys)
    axis_meta[key] = {neuro_delta_col, func_dis_col, func_adm_col, labels…, zparams:{mu_n,sd_n,mu_f,sd_f}}
    heads[key] = {
        "over_achiever": {clf, calibrator, feature_cols, base_rate},
        "magnitude":     {reg, conformal_q, feature_cols, d_sd},
    }
``_apply_platt`` (logit → calibrator) is mirrored in compute.py so the dashboard never imports this
module (which pulls shap via conversion → train).  Back-translate a magnitude D to functional points
in the UI via ``zparams`` (D ≈ Δfunction-equivalent × sd_f) for readability.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from rehab_sci.data.dataset import AnalysisFrame, build_analysis_dataset
from rehab_sci.models.conformal import _conformal_q
from rehab_sci.models.conversion import (
    ALPHA,
    N_CAL_BINS,
    N_SPLITS,
    _apply_platt,
    _calibration_curve,
    _fit_platt,
    _oof_binary,
    _params_binary,
    _refit,
    _shap_top,
    _typed_X,
)
from rehab_sci.models.train import RANDOM_STATE, _fit_reg, _params_lgbm_reg

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models"

MIN_COHORT = 120  # skip an axis whose paired-delta cohort is too small to model

# The three domain-paired dissociation axes.  ``neuro_delta`` is a G9 ``y_delta_*`` column already
# on the episode frame; the functional Δ is ``func_dis`` − ``func_adm`` (discharge SCIM − admission
# baseline feature).  Labels here seed the metrics file / console; ui_strings.yaml owns the UI.
AXES: tuple[dict, ...] = (
    {
        "key": "uems_selfcare",
        "neuro_delta": "y_delta_uems", "func_dis": "y_discharge_scim_self_care", "func_adm": "SCIM_self_care",
        "neuro_en": "UEMS (upper-limb motor)", "func_en": "SCIM self-care",
        "neuro_ja": "UEMS（上肢運動スコア）", "func_ja": "SCIM セルフケア",
    },
    {
        "key": "lems_mobility",
        "neuro_delta": "y_delta_lems", "func_dis": "y_discharge_scim_mobility", "func_adm": "SCIM_mobility",
        "neuro_en": "LEMS (lower-limb motor)", "func_en": "SCIM mobility",
        "neuro_ja": "LEMS（下肢運動スコア）", "func_ja": "SCIM 移動",
    },
    {
        "key": "totalmotor_total",
        "neuro_delta": "y_delta_totalmotor", "func_dis": "y_discharge_scim", "func_adm": "SCIM_total",
        "neuro_en": "Total motor", "func_en": "SCIM total",
        "neuro_ja": "運動スコア合計", "func_ja": "SCIM 合計",
    },
)


# ----------------------------- target construction ----------------------------

def _axis_deltas(axis: dict, ep: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (Δneuro, Δfunction) aligned to ``ep`` rows; NaN where either side is undefined.

    Δneuro = the G9 ``y_delta_*`` (discharge − first-non-null-admission score).  Δfunction =
    discharge-slot SCIM subscale − its admission baseline feature (the same first-non-null value
    the model sees).  Both are discharge − admission, so the standardized contrast is consistent.
    """
    dneuro = pd.to_numeric(ep[axis["neuro_delta"]], errors="coerce").to_numpy()
    dfunc = (
        pd.to_numeric(ep[axis["func_dis"]], errors="coerce")
        - pd.to_numeric(ep[axis["func_adm"]], errors="coerce")
    ).to_numpy()
    return dneuro, dfunc


def _landscape(dneuro: np.ndarray, dfunc: np.ndarray, D: np.ndarray) -> dict:
    """Descriptive dissociation landscape for one axis over its paired-delta cohort.

    Carries the coupling strength (Pearson/Spearman), the cohort regression line, the marginal Δ
    moments, the over-achiever base rate (D>0), and the median-split quadrant counts (concordant
    vs the two dissociated corners) — the headline "x% dissociated" stat.  The live scatter is
    recomputed by the dashboard from the episode frame; only these compact summaries are stored.
    """
    b1, b0 = (float(v) for v in np.polyfit(dneuro, dfunc, 1))
    mn, mf = float(np.median(dneuro)), float(np.median(dfunc))
    hi_n, hi_f = dneuro > mn, dfunc > mf
    quad = {
        "concordant_high": int((hi_n & hi_f).sum()),
        "concordant_low": int((~hi_n & ~hi_f).sum()),
        "neuro_over": int((hi_n & ~hi_f).sum()),   # neurology outpaces function (under-achiever side)
        "func_over": int((~hi_n & hi_f).sum()),    # function outpaces neurology (over-achiever side)
    }
    return {
        "n": len(D),
        "pearson_r": float(stats.pearsonr(dneuro, dfunc)[0]),
        "spearman_r": float(stats.spearmanr(dneuro, dfunc)[0]),
        "slope": b1,
        "intercept": b0,
        "neuro_mean": float(dneuro.mean()), "neuro_sd": float(dneuro.std()),
        "func_mean": float(dfunc.mean()), "func_sd": float(dfunc.std()),
        "over_achiever_rate": float((D > 0).mean()),
        "quadrants": quad,
        "dissociated_share": float((quad["neuro_over"] + quad["func_over"]) / len(D)),
    }


# ----------------------------- regression plumbing (magnitude head) ----------------------------

def _oof_reg(X, y, groups, cat_cols) -> tuple[np.ndarray, int]:
    """Grouped-CV out-of-fold regression predictions + median best-iteration (mirrors _oof_binary)."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.full(len(y), np.nan)
    iters: list[int] = []
    for tr, te in gkf.split(X, y, groups=groups):
        inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
        i_tr, i_val = next(inner.split(X.iloc[tr], y[tr], groups=groups.iloc[tr]))
        model = _fit_reg(
            _params_lgbm_reg(),
            X.iloc[tr].iloc[i_tr], y[tr][i_tr],
            X.iloc[tr].iloc[i_val], y[tr][i_val], cat_cols,
        )
        oof[te] = np.asarray(model.predict(X.iloc[te]), float)
        iters.append(int(model.best_iteration_ or 400))
    assert not np.isnan(oof).any(), "GroupKFold left an episode unpredicted"
    return oof, int(np.median(iters))


def _refit_reg(params: dict, X: pd.DataFrame, y: np.ndarray, cat_cols: list[str], best_iter: int):
    """Refit a regressor on the full cohort at a fixed iteration count (mirrors conversion._refit)."""
    params = dict(params)
    params["n_estimators"] = int(best_iter or 400)
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y, categorical_feature=cat_cols)
    return model


# ----------------------------- per-axis heads ----------------------------

def _run_over_achiever(X, y, groups, cat_cols) -> tuple[dict, dict]:
    """Calibrated binary head: P(D > 0) = functional over-achiever.  (metrics, persisted-model)."""
    oof, best_iter = _oof_binary(X, y, groups, cat_cols)
    cal = _fit_platt(oof, y)
    oof_cal = _apply_platt(cal, oof)
    base = float(y.mean())
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
        "shap_top": _shap_top(final, X),
    }
    model = {"clf": final, "calibrator": cal, "feature_cols": list(X.columns), "base_rate": base}
    return metrics, model


def _run_magnitude(X, D, groups, cat_cols) -> tuple[dict, dict]:
    """Continuous regression head on the signed dissociation D + marginal conformal PI."""
    oof, best_iter = _oof_reg(X, D, groups, cat_cols)
    resid = np.abs(D - oof)
    q = float(_conformal_q(resid, ALPHA))
    final = _refit_reg(_params_lgbm_reg(), X, D, cat_cols, best_iter)
    metrics = {
        "n": len(D),
        "d_mean": float(D.mean()),
        "d_sd": float(D.std()),
        "r2": float(r2_score(D, oof)),
        "rmse": float(np.sqrt(np.mean((D - oof) ** 2))),
        "mae": float(np.mean(resid)),
        "conformal_q": q,
        "conformal_coverage_80": float(np.mean(resid <= q)),
        "pi_halfwidth": q,
        "shap_top": _shap_top(final, X),
    }
    model = {"reg": final, "conformal_q": q, "feature_cols": list(X.columns), "d_sd": float(D.std())}
    return metrics, model


def _run_axis(axis: dict, ep: pd.DataFrame, af: AnalysisFrame):
    """Fit + score both heads for one axis on its paired-delta cohort.  None if too small."""
    dneuro, dfunc = _axis_deltas(axis, ep)
    mask = np.isfinite(dneuro) & np.isfinite(dfunc) & ep["IDNumber"].notna().to_numpy()
    if int(mask.sum()) < MIN_COHORT:
        return None
    cohort = ep[mask].copy()
    X = _typed_X(cohort, af)
    groups = cohort["IDNumber"].astype("float64").astype("int64")
    cat_cols = [c for c in af.categorical_cols if c in X.columns]

    dn, df_ = dneuro[mask], dfunc[mask]
    mu_n, sd_n = float(dn.mean()), float(dn.std())
    mu_f, sd_f = float(df_.mean()), float(df_.std())
    D = (df_ - mu_f) / sd_f - (dn - mu_n) / sd_n  # standardized contrast: >0 = functional over-achiever

    over_m, over_mod = _run_over_achiever(X, (D > 0).astype(int), groups, cat_cols)
    mag_m, mag_mod = _run_magnitude(X, D, groups, cat_cols)

    zparams = {"mu_n": mu_n, "sd_n": sd_n, "mu_f": mu_f, "sd_f": sd_f}
    metrics = {
        "neuro_en": axis["neuro_en"], "func_en": axis["func_en"],
        "neuro_ja": axis["neuro_ja"], "func_ja": axis["func_ja"],
        "zparams": zparams,
        "landscape": _landscape(dn, df_, D),
        "over_achiever": over_m,
        "magnitude": mag_m,
    }
    models = {
        "meta": {
            "neuro_delta_col": axis["neuro_delta"], "func_dis_col": axis["func_dis"], "func_adm_col": axis["func_adm"],
            "neuro_en": axis["neuro_en"], "func_en": axis["func_en"],
            "neuro_ja": axis["neuro_ja"], "func_ja": axis["func_ja"],
            "zparams": zparams,
        },
        "over_achiever": over_mod,
        "magnitude": mag_mod,
    }
    return metrics, models


# ----------------------------- entry point ----------------------------

def main() -> None:
    af = build_analysis_dataset()
    ep = af.df
    print("=" * 72)
    print("NEURO-FUNCTIONAL DISSOCIATION (G11) — does function track neurology in recovery?")
    print("=" * 72)
    print("D = z(Δfunction) − z(Δneuro);  D>0 = functional over-achiever (compensation/adaptation)")

    axes_metrics: dict[str, dict] = {}
    heads: dict[str, dict] = {}
    axis_meta: dict[str, dict] = {}
    ordered_keys: list[str] = []
    for axis in AXES:
        res = _run_axis(axis, ep, af)
        if res is None:
            print(f"\n[{axis['key']:<17}] SKIPPED — cohort < {MIN_COHORT}")
            continue
        m, mods = res
        axes_metrics[axis["key"]] = m
        heads[axis["key"]] = {"over_achiever": mods["over_achiever"], "magnitude": mods["magnitude"]}
        axis_meta[axis["key"]] = mods["meta"]
        ordered_keys.append(axis["key"])
        ls, ov, mg = m["landscape"], m["over_achiever"], m["magnitude"]
        print(f"\n[{axis['key']:<17}] {m['neuro_en']} ↔ {m['func_en']}  n={ls['n']}")
        print(f"   coupling : r={ls['pearson_r']:+.2f} ρ={ls['spearman_r']:+.2f}  "
              f"dissociated={ls['dissociated_share']:.0%}  over-achiever base={ls['over_achiever_rate']:.0%}")
        print(f"   binary   : AUC {ov['auc']:.3f}  Brier {ov['brier']:.3f} (base {ov['brier_baseline']:.3f})")
        print(f"   magnitude: R² {mg['r2']:.3f}  RMSE {mg['rmse']:.2f}  "
              f"PI±{mg['conformal_q']:.2f}z  cov {mg['conformal_coverage_80']:.0%}")

    out_dir = OUT / "dissociation"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "feature_cols": list(af.feature_cols),
        "numeric_cols": list(af.numeric_cols),
        "categorical_cols": list(af.categorical_cols),
        "alpha": ALPHA,
        "axes": ordered_keys,
        "axis_meta": axis_meta,
        "heads": heads,
    }
    joblib.dump(bundle, out_dir / "bundle.joblib")

    payload = {
        "random_state": RANDOM_STATE,
        "alpha": ALPHA,
        "n_splits": N_SPLITS,
        "axes": axes_metrics,
    }
    (OUT / "dissociation_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {out_dir / 'bundle.joblib'} and {OUT / 'dissociation_metrics.json'}")


if __name__ == "__main__":
    main()
