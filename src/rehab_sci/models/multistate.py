"""AIS multi-state recovery modeling (G6) — neurological-grade *dynamics* over early recovery.

Where G4 (:mod:`rehab_sci.models.conversion`) models the admission->discharge AIS *endpoint*,
this module models the **trajectory between**: how the AIS grade moves across the dense
early-recovery grid (``0day``..``6m``).  Two layers, mirroring the user-chosen design:

Population multi-state Markov (the cohort-dynamics centerpiece)
--------------------------------------------------------------
A time-inhomogeneous discrete-time Markov chain over the five AIS states (A=1..E=5) on the
fixed grid.  For each grid step ``k`` (``WINDOW[k] -> WINDOW[k+1]``) the empirical transition
matrix ``P_k`` is estimated from the **pairwise-complete** episodes — those with an observed
grade at *both* adjacent slots (a missing-at-random assumption within the observed-at-both
subset).  Rows with zero observed departures fall back to the identity (assume stable) so the
chain conserves probability mass.  Forward-multiplying ``P_0 P_1 ... P_{k-1}`` from an
admission-grade point mass yields, **stratified by admission grade**:

* **state-occupancy / prevalence curves** ``pi_k(g)`` — P(in grade g at slot k);
* **first-passage conversion curves** P(reached grade >= X by slot k) — states >= X made
  absorbing, so the curve is monotone non-decreasing (thresholds: >=C motor-incomplete,
  >=D ambulatory-capable, and "any >=1-grade improvement" to >= admission+1);
* **median day to first improvement** (interpolated 0.5-crossing of the any-improvement curve);
* **expected days in each state** over the window (trapezoidal occupancy-time).

This is a genuine multi-state object: it answers "where does an AIS-C admission end up over six
months, and when?" purely from cohort dynamics, with no covariates.

Covariate improve-by-6m head (the personalized / driver layer)
--------------------------------------------------------------
A LightGBM **binary** head predicting P(>=1-grade AIS improvement anywhere in the window) on the
room-to-improve admission cohort (A-D, >=2 in-window AIS observations; ~690 episodes, near-
balanced ~49% positive).  Grouped-CV out-of-fold predictions drive the metrics and a Platt
(sigmoid) calibrator; the final head is refit on the full cohort reusing the OOF calibrator
(conservative, mirroring ``conversion.py``).  Global SHAP importances are *descriptive*
in-sample drivers ("which admission features drive fast vs slow conversion"), not an OOS claim.
No ``class_weight`` (the target is near-balanced -> calibrated probabilities, per the §0b lesson).

The binary-head plumbing (typed feature matrix, params, OOF, Platt, calibration curve, SHAP) is
reused verbatim from :mod:`rehab_sci.models.conversion` to avoid duplication.

Diagnostic + inference layer, like :mod:`~.landmark` / :mod:`~.conversion` / :mod:`~.temporal`:
writes its own tracked ``models/multistate_metrics.json`` (identifier-free) + a git-ignored
``models/multistate/bundle.joblib`` and **never touches** ``train.py``'s production artifacts, so
``training_metrics.json`` byte-repro is preserved.

Persisted bundle shape (consumed by dashboard/compute.py::predict_multistate)
-----------------------------------------------------------------------------
    window, window_days, states, state_labels, conv_thresholds, adm_col,
    feature_cols, numeric_cols, categorical_cols,
    P_step[k]                = (5,5) per-step transition matrix,
    occupancy_by_adm[g0]     = (K,5) state-occupancy curve from admission grade g0,
    conversion_by_adm[g0]    = {label: (K,) first-passage curve} (labels: improve / ge_C / ge_D),
    sojourn_by_adm[g0]       = (5,) expected days per state,
    median_day_to_improve[g0]= float | None,
    improve_head             = {clf, calibrator, adm_grades, feature_cols, base_rate}.
The Platt ``calibrator`` is applied with the same logit transform used in conversion.py; the
dashboard mirrors ``_apply_platt`` (compute.py) so it never imports this module (which pulls shap).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from rehab_sci.constants import AIS_ORD_TO_LETTER
from rehab_sci.data.dataset import AnalysisFrame, build_analysis_dataset
from rehab_sci.models.conversion import (
    N_CAL_BINS,
    _apply_platt,
    _calibration_curve,
    _fit_platt,
    _oof_binary,
    _params_binary,
    _refit,
    _shap_top,
    _typed_X,
)
from rehab_sci.models.train import RANDOM_STATE

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models"

ADM_COL = "AIS_ord"  # admission AIS as ordinal (1=A .. 5=E); == the 0day grade on the episode frame

# AIS state space (ordinal) and the dense early-recovery grid (mirrors the G3 phenotype window).
STATES: tuple[int, ...] = (1, 2, 3, 4, 5)
WINDOW: tuple[str, ...] = ("0day", "72h", "2w", "4w", "6w", "2m", "3m", "4m", "5m", "6m")
WINDOW_DAYS: dict[str, int] = {
    "0day": 0, "72h": 3, "2w": 14, "4w": 28, "6w": 42,
    "2m": 60, "3m": 90, "4m": 120, "5m": 150, "6m": 180,
}

# First-passage conversion thresholds surfaced as curves: >=C (motor-incomplete), >=D (ambulatory).
CONV_THRESHOLDS: tuple[int, ...] = (3, 4)
THRESH_LABEL: dict[int, str] = {3: "ge_C", 4: "ge_D"}

# Covariate improve-head cohort: admitted A-D (room to improve; E excluded) with enough panel.
IMPROVE_ADM_GRADES: tuple[int, ...] = (1, 2, 3, 4)
MIN_WINDOW_OBS = 2  # >=2 in-window AIS observations to define an improvement target


# ============================ population multi-state Markov ============================

def _ais_grid(long: pd.DataFrame) -> pd.DataFrame:
    """Wide AIS-grade grid: index=KeyRecordNumber, columns=WINDOW slots, values=AIS ordinal.

    NaN where the episode has no observed grade at that slot.  One value per (episode, slot)
    by construction (the long frame is rectangular), so ``aggfunc="first"`` is a no-op guard.
    """
    sub = long[long["TIME_Name"].isin(WINDOW)][["KeyRecordNumber", "TIME_Name", "AIS_ord"]].copy()
    sub["g"] = pd.to_numeric(sub["AIS_ord"], errors="coerce")
    sub = sub.dropna(subset=["g"])
    grid = sub.pivot_table(index="KeyRecordNumber", columns="TIME_Name", values="g", aggfunc="first")
    return grid.reindex(columns=list(WINDOW))


def _transition_matrices(grid: pd.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Empirical per-step transition probability + count matrices on the WINDOW grid.

    For step ``k``, ``counts[g, g']`` tallies episodes observed at *both* ``WINDOW[k]`` and
    ``WINDOW[k+1]`` with grade ``g`` then ``g'`` (pairwise-complete).  ``P_k`` row-normalizes
    ``counts``; a row with zero observed departures falls back to the identity (assume stable).
    """
    s = len(STATES)
    eye = np.eye(s)
    probs: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    for k in range(len(WINDOW) - 1):
        a = grid[WINDOW[k]].to_numpy(dtype=float)
        b = grid[WINDOW[k + 1]].to_numpy(dtype=float)
        m = ~np.isnan(a) & ~np.isnan(b)
        ai = np.rint(a[m]).astype(int) - 1
        bi = np.rint(b[m]).astype(int) - 1
        cnt = np.zeros((s, s))
        np.add.at(cnt, (ai, bi), 1.0)
        p = np.zeros((s, s))
        for g in range(s):
            tot = cnt[g].sum()
            p[g] = cnt[g] / tot if tot > 0 else eye[g]
        probs.append(p)
        counts.append(cnt)
    return probs, counts


def _occupancy(probs: list[np.ndarray], pi0: np.ndarray) -> np.ndarray:
    """Forward-propagate an initial distribution over the grid -> (K, S) occupancy (rows sum to 1)."""
    occ = np.zeros((len(probs) + 1, len(pi0)))
    occ[0] = pi0
    for k, p in enumerate(probs):
        occ[k + 1] = occ[k] @ p
    return occ


def _absorbing_above(probs: list[np.ndarray], thresh: int) -> list[np.ndarray]:
    """Copy of the per-step matrices with states >= ``thresh`` made absorbing (row -> identity)."""
    s = len(STATES)
    hi = thresh - 1  # 0-based index of the first absorbing state
    out: list[np.ndarray] = []
    for p in probs:
        q = p.copy()
        q[hi:s] = 0.0
        for g in range(hi, s):
            q[g, g] = 1.0
        out.append(q)
    return out


def _conversion_curve(probs: list[np.ndarray], g0: int, thresh: int) -> np.ndarray:
    """First-passage P(reached grade >= ``thresh`` by slot k) from admission grade ``g0``.

    Monotone non-decreasing: states >= thresh are absorbing, so their cumulative mass only grows.
    """
    pi0 = np.zeros(len(STATES))
    pi0[g0 - 1] = 1.0
    occ = _occupancy(_absorbing_above(probs, thresh), pi0)
    return occ[:, thresh - 1:].sum(axis=1)


def _median_day_to_event(curve: np.ndarray) -> float | None:
    """Interpolated day at which a first-passage curve first crosses 0.5 (None if it never does)."""
    days = np.array([WINDOW_DAYS[w] for w in WINDOW], dtype=float)
    if curve[-1] < 0.5:
        return None
    k = int(np.argmax(curve >= 0.5))
    if k == 0:
        return float(days[0])
    y0, y1 = float(curve[k - 1]), float(curve[k])
    d0, d1 = float(days[k - 1]), float(days[k])
    frac = (0.5 - y0) / (y1 - y0) if y1 > y0 else 0.0
    return float(d0 + frac * (d1 - d0))


def _expected_days_in_state(occ: np.ndarray) -> np.ndarray:
    """Trapezoidal expected days spent in each state over the window, given an occupancy curve."""
    days = np.array([WINDOW_DAYS[w] for w in WINDOW], dtype=float)
    return np.trapezoid(occ, x=days, axis=0)


def _population_dynamics(grid: pd.DataFrame) -> dict:
    """Assemble per-admission-grade occupancy / conversion / sojourn from the empirical chain."""
    probs, counts = _transition_matrices(grid)

    occupancy: dict[int, np.ndarray] = {}
    conversion: dict[int, dict[str, np.ndarray]] = {}
    sojourn: dict[int, np.ndarray] = {}
    med_improve: dict[int, float | None] = {}

    for g0 in STATES:
        pi0 = np.zeros(len(STATES))
        pi0[g0 - 1] = 1.0
        occ = _occupancy(probs, pi0)
        occupancy[g0] = occ
        sojourn[g0] = _expected_days_in_state(occ)

        curves: dict[str, np.ndarray] = {}
        if g0 < len(STATES):  # room to improve
            imp = _conversion_curve(probs, g0, g0 + 1)
            curves["improve"] = imp
            med_improve[g0] = _median_day_to_event(imp)
        for x in CONV_THRESHOLDS:
            if x > g0:
                curves[THRESH_LABEL[x]] = _conversion_curve(probs, g0, x)
        conversion[g0] = curves

    # Marginal occupancy from the empirical admission (0day) grade distribution.
    adm0 = grid[WINDOW[0]].dropna()
    pi_emp = np.array([(np.rint(adm0) == g).sum() for g in STATES], dtype=float)
    pi_emp = pi_emp / pi_emp.sum() if pi_emp.sum() > 0 else pi_emp
    occupancy_overall = _occupancy(probs, pi_emp)

    # Pooled one-step transition matrix (all steps summed) for the headline Methods heatmap.
    pooled_n = np.sum(counts, axis=0)
    pooled = np.zeros_like(pooled_n)
    for g in range(len(STATES)):
        tot = pooled_n[g].sum()
        pooled[g] = pooled_n[g] / tot if tot > 0 else np.eye(len(STATES))[g]

    return {
        "P_step": probs,
        "P_step_n": counts,
        "P_pooled": pooled,
        "P_pooled_n": pooled_n,
        "occupancy_by_adm": occupancy,
        "occupancy_overall": occupancy_overall,
        "conversion_by_adm": conversion,
        "sojourn_by_adm": sojourn,
        "median_day_to_improve": med_improve,
        "adm_distribution": pi_emp,
    }


def _landscape(grid: pd.DataFrame, ep: pd.DataFrame) -> dict:
    """Descriptive within-window AIS dynamics summary (improve/stable/decline by admission grade)."""
    adm = pd.to_numeric(ep.set_index("KeyRecordNumber")[ADM_COL], errors="coerce")
    nobs = grid.notna().sum(axis=1)
    wmax = grid.max(axis=1)
    wmin = grid.min(axis=1)
    eligible = nobs[nobs >= MIN_WINDOW_OBS].index
    a = adm.reindex(eligible)
    mx = wmax.reindex(eligible)
    mn = wmin.reindex(eligible)
    keep = a.notna() & mx.notna()
    a, mx, mn = a[keep], mx[keep], mn[keep]
    improved = (mx > a)
    declined = (mn < a)

    by_grade: dict[str, dict] = {}
    for g in STATES:
        gm = np.rint(a) == g
        if gm.any():
            by_grade[AIS_ORD_TO_LETTER[g]] = {
                "n": int(gm.sum()),
                "improve_rate": float(improved[gm].mean()),
                "decline_rate": float(declined[gm].mean()),
            }
    return {
        "n_eligible": int(keep.sum()),
        "improve_rate": float(improved.mean()),
        "stable_rate": float(((~improved) & (~declined)).mean()),
        "decline_rate": float(declined.mean()),
        "by_admission_grade": by_grade,
    }


# ============================ covariate improve-by-6m head ============================

def _improve_cohort(ep: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Episodes admitted A-D with a real IDNumber and >= MIN_WINDOW_OBS in-window AIS observations."""
    nobs = grid.notna().sum(axis=1)
    wmax = grid.max(axis=1)
    adm = pd.to_numeric(ep[ADM_COL], errors="coerce")
    coh = ep[adm.isin(IMPROVE_ADM_GRADES) & ep["IDNumber"].notna()].copy()
    coh["_wobs"] = coh["KeyRecordNumber"].map(nobs).fillna(0).astype(int)
    coh["_wmax"] = coh["KeyRecordNumber"].map(wmax)
    return coh[(coh["_wobs"] >= MIN_WINDOW_OBS) & coh["_wmax"].notna()].copy()


def _run_improve_head(ep: pd.DataFrame, grid: pd.DataFrame, af: AnalysisFrame) -> tuple[dict, dict]:
    """Fit + score the binary 'improves >=1 grade within the window' head; return (metrics, model)."""
    cohort = _improve_cohort(ep, grid)
    X = _typed_X(cohort, af)
    adm = pd.to_numeric(cohort[ADM_COL], errors="coerce").to_numpy()
    y = (cohort["_wmax"].to_numpy() > adm).astype(int)
    groups = cohort["IDNumber"].astype("float64").astype("int64")
    cat_cols = [c for c in af.categorical_cols if c in X.columns]

    oof, best_iter = _oof_binary(X, y, groups, cat_cols)
    cal = _fit_platt(oof, y)
    oof_cal = _apply_platt(cal, oof)

    base = float(y.mean())
    adm_int = adm.astype(int)
    rate_by_grade = {
        AIS_ORD_TO_LETTER[g]: {"rate": float(y[adm_int == g].mean()), "n": int((adm_int == g).sum())}
        for g in IMPROVE_ADM_GRADES if (adm_int == g).any()
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
        "adm_grades": list(IMPROVE_ADM_GRADES),
        "shap_top": _shap_top(final, X),
    }
    model = {
        "clf": final,
        "calibrator": cal,
        "adm_grades": list(IMPROVE_ADM_GRADES),
        "feature_cols": list(X.columns),
        "base_rate": base,
    }
    return metrics, model


# ============================ serialization helpers ============================

def _curves_to_lists(conv: dict[int, dict[str, np.ndarray]]) -> dict[str, dict[str, list]]:
    """JSON-able conversion curves keyed by AIS letter -> label -> list."""
    return {
        AIS_ORD_TO_LETTER[g0]: {lab: c.tolist() for lab, c in curves.items()}
        for g0, curves in conv.items()
    }


def _by_letter(d: dict[int, np.ndarray]) -> dict[str, list]:
    return {AIS_ORD_TO_LETTER[g0]: arr.tolist() for g0, arr in d.items()}


# ============================ entry point ============================

def main() -> None:
    af = build_analysis_dataset()
    ep = af.df
    long = af.longitudinal
    grid = _ais_grid(long)

    print("=" * 64)
    print("AIS MULTI-STATE RECOVERY (G6) — neurological-grade dynamics, 0day..6m")
    print("=" * 64)
    print(f"episodes with >=1 in-window AIS grade: {grid.notna().any(axis=1).sum()}  "
          f"(>=2 obs: {(grid.notna().sum(axis=1) >= 2).sum()})")

    pop = _population_dynamics(grid)
    landscape = _landscape(grid, ep)
    print(f"within-window: improve {landscape['improve_rate']:.1%}  "
          f"stable {landscape['stable_rate']:.1%}  decline {landscape['decline_rate']:.1%}  "
          f"(n={landscape['n_eligible']})")
    for g0 in STATES:
        md = pop["median_day_to_improve"].get(g0)
        occ_end = pop["occupancy_by_adm"][g0][-1]
        print(f"  adm {AIS_ORD_TO_LETTER[g0]}: 6m occupancy "
              f"{{{', '.join(f'{AIS_ORD_TO_LETTER[s]}:{occ_end[i]:.2f}' for i, s in enumerate(STATES))}}}  "
              f"median d->improve {md if md is None else round(md)}")

    print("\n[improve head]  P(>=1-grade AIS improvement within 0day..6m), admission A-D")
    imp_metrics, imp_model = _run_improve_head(ep, grid, af)
    print(f"   n={imp_metrics['n']} pos={imp_metrics['n_pos']} base={imp_metrics['base_rate']:.0%}  "
          f"AUC {imp_metrics['auc']:.3f}  Brier {imp_metrics['brier']:.3f} "
          f"(base {imp_metrics['brier_baseline']:.3f})")
    print("   top drivers: " + ", ".join(d["feature"] for d in imp_metrics["shap_top"][:6]))

    # ---- persist git-ignored bundle (numpy arrays + fitted head) for the dashboard ----
    out_dir = OUT / "multistate"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "window": list(WINDOW),
        "window_days": WINDOW_DAYS,
        "states": list(STATES),
        "state_labels": [AIS_ORD_TO_LETTER[g] for g in STATES],
        "conv_thresholds": list(CONV_THRESHOLDS),
        "thresh_label": dict(THRESH_LABEL),
        "adm_col": ADM_COL,
        "feature_cols": list(af.feature_cols),
        "numeric_cols": list(af.numeric_cols),
        "categorical_cols": list(af.categorical_cols),
        "P_step": pop["P_step"],
        "P_pooled": pop["P_pooled"],
        "occupancy_by_adm": pop["occupancy_by_adm"],
        "occupancy_overall": pop["occupancy_overall"],
        "conversion_by_adm": pop["conversion_by_adm"],
        "sojourn_by_adm": pop["sojourn_by_adm"],
        "median_day_to_improve": pop["median_day_to_improve"],
        "improve_head": imp_model,
    }
    joblib.dump(bundle, out_dir / "bundle.joblib")

    # ---- tracked, identifier-free metrics ----
    payload = {
        "random_state": RANDOM_STATE,
        "adm_col": ADM_COL,
        "window": list(WINDOW),
        "window_days": WINDOW_DAYS,
        "states": list(STATES),
        "state_labels": [AIS_ORD_TO_LETTER[g] for g in STATES],
        "conv_thresholds": list(CONV_THRESHOLDS),
        "min_window_obs": MIN_WINDOW_OBS,
        "landscape": landscape,
        "transition": {
            "P_step": [p.tolist() for p in pop["P_step"]],
            "P_step_n": [c.tolist() for c in pop["P_step_n"]],
            "P_pooled": pop["P_pooled"].tolist(),
            "P_pooled_n": pop["P_pooled_n"].tolist(),
            "step_labels": [f"{WINDOW[k]}->{WINDOW[k + 1]}" for k in range(len(WINDOW) - 1)],
        },
        "occupancy_by_adm": _by_letter(pop["occupancy_by_adm"]),
        "occupancy_overall": pop["occupancy_overall"].tolist(),
        "conversion_by_adm": _curves_to_lists(pop["conversion_by_adm"]),
        "sojourn_by_adm": _by_letter(pop["sojourn_by_adm"]),
        "median_day_to_improve": {AIS_ORD_TO_LETTER[g0]: v for g0, v in pop["median_day_to_improve"].items()},
        "adm_distribution": pop["adm_distribution"].tolist(),
        "improve_head": imp_metrics,
    }
    (OUT / "multistate_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {out_dir / 'bundle.joblib'} and {OUT / 'multistate_metrics.json'}")


if __name__ == "__main__":
    main()
