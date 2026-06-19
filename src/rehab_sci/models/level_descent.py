"""Neurological-level descent modeling (G10) — admission->discharge change in ISNCSCI *levels*.

Where G9's Δ-score heads predict the change in the ISNCSCI summary *scores* (ΔUEMS, ΔLEMS, …),
this module targets the complementary, equally-standard SCI endpoint: recovery of the
neurological *level* of injury — the anatomical boundary descending caudally as previously
impaired segments regain function.  A patient can gain motor *score* without the motor *level*
descending (a weak muscle strengthens but no new myotome crosses grade 3) and vice versa, so
"level conversion" is reported separately from "score recovery" in SCI trials.

The five modelled levels (each a categorical ``cord_level``, ordinalised by the loader as
``*_ord``: C1=0 … S45=28; smaller = more rostral = more severe):

* ``nli``           — neurological level of injury (the most rostral of the four component levels)
* ``right_motor`` / ``left_motor``     — bilateral motor levels
* ``right_sensory`` / ``left_sensory`` — bilateral sensory levels

Descent direction & the INT ceiling
------------------------------------
Δ = (discharge level ordinal) − (admission level ordinal); **positive Δ = caudal descent =
neurological improvement** (the level moves to a more caudal, less-severe segment).  Full
neurological recovery is recorded as the raw value ``INT`` (intact), which the loader's ``*_ord``
maps to NaN — so naively those best-outcome episodes would silently drop.  Here the discharge
``INT`` is lifted to ``INT_ORD`` (= len(cord order) = 29, one beyond S45) so a cure is the Δ
*ceiling* rather than a dropped row.  The admission baseline reuses the loader ``*_ord`` (INT/
missing → NaN), so admission-INT episodes (already intact, no room to descend) are excluded from
the cohort — only episodes with a defined admission level (ord 0..28) enter.

Two heads per level (10 total)
------------------------------
* **descent** (binary, calibrated): P(level descends ≥1 segment) = P(Δ ≥ 1).  No
  ``class_weight`` (the cohort is near-balanced ~50% descend → the probabilities are Platt-
  calibratable) — surface this as *the* calibrated descent probability.
* **magnitude** (multiclass + APS): improvement size {0, +1, ≥+2} (``MAG_CAP``=2); deterioration
  (Δ<0) folds into class 0.  Uses ``class_weight="balanced"`` (via ``_params_lgbm_clf``) so its
  class probabilities are *uncalibrated/inflated* for the minority classes — surface the APS set
  / argmax class, never as a competing probability (mirrors the G4 binary-vs-magnitude CRUX).

Features
--------
The standard 30 admission features (``af.feature_cols``) — which already include every level's own
admission ``*_ord`` — so the level's own baseline (the dominant predictor of its own descent, à la
G8's ``adm_self`` / G9's baseline score) is present without adding any column.  No leakage: the
target is the *change*, never the discharge level itself.

(A G8-style enrichment — concatenating the modality-matched per-segment admission grades, 20/112/132
columns — was tried and **rejected**: it left descent AUC flat-to-worse on every head (within OOF
noise).  Granular grades nail segment *state* (autocorrelation) but not a level boundary's *threshold
crossing*, which depends on recovery dynamics the static grades don't encode.  See .agent/memory.md.)

Methodology (robust for small cohorts; few heads)
-------------------------------------------------
Identical to :mod:`rehab_sci.models.conversion` (whose binary plumbing is imported verbatim):
grouped 5-fold CV by ``IDNumber`` → out-of-fold (OOF) predictions drive every reported metric, the
Platt calibrator (binary), and the cross-conformal APS q (magnitude); final heads refit on the full
cohort reusing the OOF calibrator / APS q (conservative).  SHAP drivers are *descriptive* in-sample.
**No conformal PI on the binary head** (binary APS degenerates — see .agent/memory.md §0b); its
uncertainty surface is the calibrated probability + reliability curve.

Diagnostic + inference layer, like conversion/independence/multistate/topography: writes a tracked
identifier-free ``models/level_descent_metrics.json`` + a git-ignored
``models/level_descent/bundle.joblib`` and **never touches** ``train.py``'s production artifacts
(byte-repro of ``training_metrics.json`` preserved).

Persisted bundle shape (consumed by dashboard/compute.py::predict_level_descent in Part 2)
------------------------------------------------------------------------------------------
    feature_cols, numeric_cols, categorical_cols, int_ord, mag_cap, levels (ordered keys)
    level_meta[key] = {raw, ord, label_en, label_ja}
    heads[key] = {
        "descent":   {clf, calibrator, feature_cols, base_rate},
        "magnitude": {clf, aps_q_hat, class_codes, mag_cap, feature_cols},
    }
``_apply_platt`` (logit → calibrator) is mirrored in compute.py so the dashboard never imports this
module (which pulls shap via conversion → train).  INT-aware level→ord mapping is reconstructed at
inference from ``int_ord`` + the loader's cord order.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)

from rehab_sci.data.dataset import AnalysisFrame, build_analysis_dataset
from rehab_sci.data.loader import _CORD_ORDER
from rehab_sci.models.conformal import _aps_prediction_set, _aps_scores, _conformal_q
from rehab_sci.models.conversion import (
    ALPHA,
    N_CAL_BINS,
    N_SPLITS,
    _apply_platt,
    _calibration_curve,
    _fit_platt,
    _oof_binary,
    _oof_multiclass,
    _params_binary,
    _refit,
    _shap_top,
    _typed_X,
)
from rehab_sci.models.train import RANDOM_STATE, _params_lgbm_clf

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models"

INT_ORD = len(_CORD_ORDER)  # 29: "Intact" / full neurological recovery — the Δ ceiling beyond S45
MAG_CAP = 2  # ordinal improvement magnitude clipped to {0, +1, >=+2}
MIN_COHORT = 120  # skip a level whose room-to-descend cohort is too small to model

# The five ISNCSCI levels modelled.  ``raw`` = the categorical cord-level column (carries the raw
# ``INT`` token for intact); ``ord`` = the loader's ordinal (C1=0..S45=28; INT/missing -> NaN),
# already an admission feature so the level's own baseline is in ``feature_cols``.
LEVELS: tuple[dict, ...] = (
    {"key": "nli", "raw": "NLI", "ord": "NLI_ord"},
    {"key": "right_motor", "raw": "RightMotorLevel", "ord": "RightMotorLevel_ord"},
    {"key": "left_motor", "raw": "LeftMotorLevel", "ord": "LeftMotorLevel_ord"},
    {"key": "right_sensory", "raw": "RightSensoryLevel", "ord": "RightSensoryLevel_ord"},
    {"key": "left_sensory", "raw": "LeftSensoryLevel", "ord": "LeftSensoryLevel_ord"},
)

# Coarse admission-region strata (by admission ordinal) for the descriptive landscape.
REGIONS: tuple[tuple[str, int, int], ...] = (
    ("cervical", 0, 7),    # C1..C8
    ("thoracic", 8, 19),   # T1..T12
    ("lumbar", 20, 24),    # L1..L5
    ("sacral", 25, 28),    # S1..S45
)


# ----------------------------- target construction ----------------------------

def _level_delta(level: dict, ep: pd.DataFrame, disc: pd.DataFrame, idx_kr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (admission ordinal, INT-aware discharge ordinal), both aligned to ``ep`` rows.

    Admission = the loader's first-non-null ``*_ord`` (the model's own feature): C1=0..S45=28,
    INT/missing → NaN (so admission-INT episodes, having no room to descend, drop from the cohort).
    Discharge = the discharge-slot ``*_ord`` with raw ``INT`` lifted to ``INT_ORD`` (full recovery
    becomes the Δ ceiling, not a dropped NaN).  ``disc`` is the discharge slot indexed by
    KeyRecordNumber; align by ``ep["KeyRecordNumber"]`` — never ``ep.index`` (.agent/memory.md §0b).
    """
    adm = pd.to_numeric(ep[level["ord"]], errors="coerce").to_numpy()
    dord = pd.to_numeric(disc[level["ord"]].reindex(idx_kr), errors="coerce").to_numpy()
    draw = disc[level["raw"]].reindex(idx_kr).astype("string").str.strip()
    is_int = (draw == "INT").fillna(False).to_numpy()  # nullable-string -> clean bool mask
    dis = np.where(is_int, float(INT_ORD), dord)
    return adm, dis


def _landscape(a: np.ndarray, delta: np.ndarray) -> dict:
    """Descriptive descent landscape for one level over its room-to-descend cohort."""
    vals, cnts = np.unique(delta.astype(int), return_counts=True)
    by_region: dict[str, dict] = {}
    for name, lo, hi in REGIONS:
        rm = (a >= lo) & (a <= hi)
        if rm.any():
            by_region[name] = {
                "n": int(rm.sum()),
                "descent_rate": float((delta[rm] >= 1).mean()),
                "mean_delta": float(delta[rm].mean()),
            }
    return {
        "n": len(delta),
        "any_descent_rate": float((delta >= 1).mean()),
        "stable_rate": float((delta == 0).mean()),
        "deteriorate_rate": float((delta <= -1).mean()),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "delta_distribution": {int(v): int(c) for v, c in zip(vals, cnts, strict=True)},
        "descent_by_admission_region": by_region,
    }


# ----------------------------- per-level heads ----------------------------

def _run_descent(X, y, groups, cat_cols) -> tuple[dict, dict]:
    """Calibrated binary head: P(level descends ≥1 segment).  Returns (metrics, persisted-model)."""
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
    model = {
        "clf": final,
        "calibrator": cal,
        "feature_cols": list(X.columns),
        "base_rate": base,
    }
    return metrics, model


def _run_magnitude(X, delta, groups, cat_cols) -> tuple[dict, dict]:
    """Ordinal magnitude head {0,+1,≥+2} (balanced) + APS.  Returns (metrics, persisted-model)."""
    mag = np.clip(delta, 0, MAG_CAP).astype(int)  # deterioration folds into class 0
    n_classes = MAG_CAP + 1
    oof, best_iter = _oof_multiclass(X, mag, groups, cat_cols, n_classes)
    pred = oof.argmax(axis=1)
    q = float(_conformal_q(_aps_scores(oof, mag), ALPHA))
    sets = [_aps_prediction_set(oof[i], q) for i in range(len(mag))]
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
        "mag_cap": MAG_CAP,
    }
    model = {
        "clf": final,
        "aps_q_hat": q,
        "class_codes": list(range(n_classes)),
        "mag_cap": MAG_CAP,
        "feature_cols": list(X.columns),
    }
    return metrics, model


def _run_level(level: dict, ep: pd.DataFrame, af: AnalysisFrame, disc: pd.DataFrame, idx_kr: np.ndarray):
    """Fit + score both heads for one level on its room-to-descend cohort.  None if too small."""
    adm, dis = _level_delta(level, ep, disc, idx_kr)
    mask = np.isfinite(adm) & np.isfinite(dis) & ep["IDNumber"].notna().to_numpy()
    if int(mask.sum()) < MIN_COHORT:
        return None
    cohort = ep[mask].copy()
    X = _typed_X(cohort, af)
    groups = cohort["IDNumber"].astype("float64").astype("int64")
    cat_cols = [c for c in af.categorical_cols if c in X.columns]
    a = adm[mask]
    delta = dis[mask] - a

    descent_m, descent_mod = _run_descent(X, (delta >= 1).astype(int), groups, cat_cols)
    mag_m, mag_mod = _run_magnitude(X, delta, groups, cat_cols)

    label_en = af.schema.by_raw(level["raw"]).label("en")
    label_ja = af.schema.by_raw(level["raw"]).label("ja")
    metrics = {
        "raw_col": level["raw"],
        "label_en": label_en,
        "label_ja": label_ja,
        "landscape": _landscape(a, delta),
        "descent": descent_m,
        "magnitude": mag_m,
    }
    models = {
        "meta": {"raw": level["raw"], "ord": level["ord"], "label_en": label_en, "label_ja": label_ja},
        "descent": descent_mod,
        "magnitude": mag_mod,
    }
    return metrics, models


# ----------------------------- entry point ----------------------------

def main() -> None:
    af = build_analysis_dataset()
    ep = af.df
    idx_kr = ep["KeyRecordNumber"].to_numpy()
    disc = af.longitudinal[af.longitudinal["TIME_Name"] == "discharge"].set_index("KeyRecordNumber")

    print("=" * 70)
    print("NEUROLOGICAL-LEVEL DESCENT (G10) — admission->discharge ISNCSCI level recovery")
    print("=" * 70)
    print(f"INT (full recovery) ceiling ordinal = {INT_ORD}  |  magnitude classes = {{0,+1,>=+2}}")

    levels_metrics: dict[str, dict] = {}
    heads: dict[str, dict] = {}
    level_meta: dict[str, dict] = {}
    ordered_keys: list[str] = []
    for level in LEVELS:
        res = _run_level(level, ep, af, disc, idx_kr)
        if res is None:
            print(f"\n[{level['key']:<13}] SKIPPED — cohort < {MIN_COHORT}")
            continue
        m, mods = res
        levels_metrics[level["key"]] = m
        heads[level["key"]] = {"descent": mods["descent"], "magnitude": mods["magnitude"]}
        level_meta[level["key"]] = mods["meta"]
        ordered_keys.append(level["key"])
        d, g, ls = m["descent"], m["magnitude"], m["landscape"]
        print(f"\n[{level['key']:<13}] n={d['n']}  descend>=1 base={d['base_rate']:.0%}  "
              f"meanΔ={ls['mean_delta']:+.2f} seg")
        print(f"   descent : AUC {d['auc']:.3f}  Brier {d['brier']:.3f} (base {d['brier_baseline']:.3f})")
        print(f"   magnitude: classes={g['class_counts']}  κ {g['kappa_quadratic']:.3f}  "
              f"APS cov {g['aps_coverage_80']:.0%} set {g['aps_avg_set_size']:.2f}")

    out_dir = OUT / "level_descent"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "feature_cols": list(af.feature_cols),
        "numeric_cols": list(af.numeric_cols),
        "categorical_cols": list(af.categorical_cols),
        "int_ord": INT_ORD,
        "mag_cap": MAG_CAP,
        "levels": ordered_keys,
        "level_meta": level_meta,
        "heads": heads,
    }
    joblib.dump(bundle, out_dir / "bundle.joblib")

    payload = {
        "random_state": RANDOM_STATE,
        "alpha": ALPHA,
        "n_splits": N_SPLITS,
        "int_ord": INT_ORD,
        "mag_cap": MAG_CAP,
        "levels": levels_metrics,
    }
    (OUT / "level_descent_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {out_dir / 'bundle.joblib'} and {OUT / 'level_descent_metrics.json'}")


if __name__ == "__main__":
    main()
