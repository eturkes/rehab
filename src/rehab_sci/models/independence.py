"""Functional-independence profile (G7) — per-SCIM-item discharge independence prediction.

Where the production SCIM heads predict the *aggregate* discharge score (total + 3 subscales),
this module predicts, for each individual SCIM-III ADL item, the calibrated probability that the
patient is **functionally independent** in that specific function at discharge — reframing a
continuous score into the actionable question clinicians and patients actually ask ("will I feed
myself / manage my bladder / walk?").

Label — functional independence (aids/devices allowed, no human assistance)
---------------------------------------------------------------------------
A binary label per item: independent iff the discharge item score is at/above a per-item
threshold that the SCIM-III rubric places at the boundary "independent (no human help), adaptive
devices / set-up permitted".  The thresholds (mapped to the rubric, validated against the
observed score distributions) live in :data:`ITEMS`; the full mapping + rationale is in
AGENT_NOTES §3.  Notable choices: dressing/bathing/grooming/feeding at >=2 (independent with
devices, simple clothes); bladder >=9 / bowel >=8 (self-manages, no caregiver); toilet >=3; the
0--2 transfer items + bed-mobility at their top score (the scale lumps human-assist with
device-use, so no clean "with-aids" middle exists); the three **walking** items (indoors /
moderate / outdoors) at >=4 = ambulates independently with or without aids — wheelchair
independence does NOT count, so these read as distinct "will I walk?" milestones.  Respiration is
**excluded** (>=95% independent at discharge — too imbalanced to model); its cohort rate is noted
in the metrics.

Heads (one calibrated binary LightGBM per modelable item; 18 items)
-------------------------------------------------------------------
Each head is fit on its at-risk cohort (episodes with an observed discharge score for that item +
a real IDNumber — no admission-grade gating; independence is predicted for everyone) using the
**binary plumbing reused verbatim from** :mod:`rehab_sci.models.conversion`: LightGBM (no
``class_weight`` — the items are near-balanced, so probabilities stay calibratable) + Platt
(sigmoid) recalibration on grouped-CV out-of-fold predictions.  The uncertainty surface is the
**calibrated probability + its reliability curve** (raw vs calibrated), the correct read for a
binary head.  A conformal *prediction set* is deliberately NOT surfaced: an APS set over the two
classes degenerates here — ~(1 - accuracy) of the nonconformity scores pin at exactly 1.0, which
drags the 80%-coverage q to ~0.98-1.0 and forces a {both = uncertain} set for ~99% of patients
(verified; see AGENT_NOTES §0b).  Descriptive in-sample SHAP drivers rank what predicts each
function (not an OOS claim).

Diagnostic + inference layer, like :mod:`~.conversion` / :mod:`~.multistate` / :mod:`~.landmark`:
writes its own tracked ``models/independence_metrics.json`` (identifier-free) + a git-ignored
``models/independence/bundle.joblib`` and **never touches** ``train.py``'s production artifacts,
so the byte-repro of ``training_metrics.json`` is preserved.

Persisted bundle shape (consumed by dashboard/compute.py::predict_independence)
-------------------------------------------------------------------------------
    feature_cols, numeric_cols, categorical_cols, alpha
    items = [{key, col, thr, domain}, ...]            # ordered registry (display order)
    heads[key] = {clf, calibrator, thr, col, domain, feature_cols, base_rate}
The Platt ``calibrator`` is a 1-feature ``LogisticRegression`` over the LightGBM logit; apply it
with the same logit transform (``_apply_platt`` is mirrored in compute.py to avoid importing this
module — and thus shap — into the dashboard process).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from rehab_sci.constants import AIS_ORD_TO_LETTER
from rehab_sci.data.dataset import AnalysisFrame, build_analysis_dataset
from rehab_sci.models.conversion import (
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
from rehab_sci.models.train import RANDOM_STATE

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models"

N_CAL_BINS = 5  # quantile bins for the reliability curve (small cohorts -> few bins)
MIN_MINORITY = 25  # defensive: skip an item if either class has < this many discharge observations

ADM_COL = "AIS_ord"  # admission AIS ordinal (1=A .. 5=E) — for the per-grade independence landscape
DISCHARGE_TP = "discharge"  # TIME_Name of the discharge slot in the longitudinal frame

# Item registry: (key, raw discharge column, independence threshold, SCIM domain).
# ``thr`` = the minimum item score counting as *functional* independence (aids/devices allowed,
# no human assistance) per the SCIM-III rubric.  Ordered by domain for display.  See AGENT_NOTES
# §3 for the full threshold mapping and the data-validated rationale.
ITEMS: tuple[dict, ...] = (
    # self-care (0--20 subscale)
    {"key": "feeding", "col": "Feeding", "thr": 2, "domain": "self_care"},
    {"key": "bath_upper", "col": "UpperBath", "thr": 2, "domain": "self_care"},
    {"key": "bath_lower", "col": "LowerBath", "thr": 2, "domain": "self_care"},
    {"key": "dress_upper", "col": "UpperClothes", "thr": 2, "domain": "self_care"},
    {"key": "dress_lower", "col": "LowerClothes", "thr": 2, "domain": "self_care"},
    {"key": "grooming", "col": "Grooming", "thr": 2, "domain": "self_care"},
    # respiration & sphincter management (0--40 subscale; respiration excluded as near-universal)
    {"key": "bladder", "col": "Bladder", "thr": 9, "domain": "sphincter"},
    {"key": "bowel", "col": "Bowel", "thr": 8, "domain": "sphincter"},
    {"key": "toilet", "col": "Toilet", "thr": 3, "domain": "sphincter"},
    # mobility — transfers & bed (room/toilet)
    {"key": "bed_mobility", "col": "MobilityBED", "thr": 6, "domain": "mobility"},
    {"key": "transfer_bed_wc", "col": "TransferBED_WC", "thr": 2, "domain": "mobility"},
    {"key": "transfer_wc_toilet", "col": "TransferWC_TOILET", "thr": 2, "domain": "mobility"},
    {"key": "transfer_wc_car", "col": "TransferWC_CAR", "thr": 2, "domain": "mobility"},
    {"key": "transfer_floor_wc", "col": "TransferFLOOR_WC", "thr": 1, "domain": "mobility"},
    # ambulation — independent walking (aids allowed, no human assistance)
    {"key": "walk_indoor", "col": "MobilityIndoor", "thr": 4, "domain": "ambulation"},
    {"key": "walk_moderate", "col": "MobilityModerateDistance", "thr": 4, "domain": "ambulation"},
    {"key": "walk_outdoor", "col": "MobilityOutdoor", "thr": 4, "domain": "ambulation"},
    {"key": "stair", "col": "Stair", "thr": 2, "domain": "ambulation"},
)

# Documented exclusion: near-universal at discharge, too imbalanced to model.  Its observed rate
# is recorded in the metrics so the surface can show it as a flat cohort prior if desired.
EXCLUDED: tuple[dict, ...] = (
    {"key": "respiration", "col": "Respiration", "thr": 8, "domain": "sphincter",
     "reason": "near-universal (>=95% independent at discharge) — too imbalanced to model"},
)


# ----------------------------- discharge-item attachment ----------------------------

def _attach_discharge_items(af: AnalysisFrame) -> pd.DataFrame:
    """Episode frame + the discharge-timepoint raw score for every registry (and excluded) item.

    The per-item discharge scores live in the longitudinal frame, not on the episode frame, so
    they are pulled from the ``discharge`` slot and reindexed onto the episode order.
    """
    ep = af.df.copy()
    discharge = af.longitudinal[af.longitudinal["TIME_Name"] == DISCHARGE_TP].set_index("KeyRecordNumber")
    krs = ep["KeyRecordNumber"]
    for it in (*ITEMS, *EXCLUDED):
        raw = pd.to_numeric(discharge[it["col"]], errors="coerce").reindex(krs)
        ep[f"item_{it['key']}"] = raw.to_numpy()
    return ep


# ----------------------------- per-item head ----------------------------

def _run_item(spec: dict, ep: pd.DataFrame, af: AnalysisFrame) -> tuple[dict, dict] | None:
    """Fit + score one calibrated binary independence head; return (metrics, persisted-model).

    Returns ``None`` if the item's minority class is below :data:`MIN_MINORITY` (defensive — no
    registry item triggers this on the current data).
    """
    item_col = f"item_{spec['key']}"
    score = pd.to_numeric(ep[item_col], errors="coerce")
    cohort = ep[score.notna() & ep["IDNumber"].notna()].copy()
    y = (pd.to_numeric(cohort[item_col], errors="coerce") >= spec["thr"]).astype(int).to_numpy()
    if min(int(y.sum()), int((1 - y).sum())) < MIN_MINORITY:
        return None

    X = _typed_X(cohort, af)
    groups = cohort["IDNumber"].astype("float64").astype("int64")
    cat_cols = [c for c in af.categorical_cols if c in X.columns]

    oof, best_iter = _oof_binary(X, y, groups, cat_cols)
    cal = _fit_platt(oof, y)
    oof_cal = _apply_platt(cal, oof)

    base = float(y.mean())
    adm_int = pd.to_numeric(cohort[ADM_COL], errors="coerce").to_numpy()
    rate_by_grade = {
        AIS_ORD_TO_LETTER[g]: {"rate": float(y[adm_int == g].mean()), "n": int((adm_int == g).sum())}
        for g in (1, 2, 3, 4, 5) if (adm_int == g).any()
    }

    final = _refit(_params_binary(), X, y, cat_cols, best_iter)
    metrics = {
        "n": len(y),
        "n_pos": int(y.sum()),
        "base_rate": base,
        "threshold": spec["thr"],
        "col": spec["col"],
        "domain": spec["domain"],
        "auc": float(roc_auc_score(y, oof)),
        "brier": float(brier_score_loss(y, oof_cal)),
        "brier_raw": float(brier_score_loss(y, oof)),
        "brier_baseline": float(base * (1.0 - base)),
        "logloss": float(log_loss(y, np.clip(oof_cal, 1e-6, 1 - 1e-6))),
        "calibration_raw": _calibration_curve(oof, y, N_CAL_BINS),
        "calibration": _calibration_curve(oof_cal, y, N_CAL_BINS),
        "rate_by_admission_grade": rate_by_grade,
        "shap_top": _shap_top(final, X),
    }
    model = {
        "clf": final,
        "calibrator": cal,
        "thr": spec["thr"],
        "col": spec["col"],
        "domain": spec["domain"],
        "feature_cols": list(X.columns),
        "base_rate": base,
    }
    return metrics, model


def _excluded_rates(ep: pd.DataFrame) -> dict:
    """Discharge independence rate for the documented-excluded items (for the metrics record)."""
    out: dict[str, dict] = {}
    for it in EXCLUDED:
        v = pd.to_numeric(ep[f"item_{it['key']}"], errors="coerce").dropna()
        out[it["key"]] = {
            "col": it["col"],
            "n": len(v),
            "independent_rate": float((v >= it["thr"]).mean()) if len(v) else None,
            "reason": it["reason"],
        }
    return out


# ----------------------------- entry point ----------------------------

def main() -> None:
    af = build_analysis_dataset()
    ep = _attach_discharge_items(af)
    print("=" * 64)
    print("FUNCTIONAL-INDEPENDENCE PROFILE (G7) — per-SCIM-item discharge independence")
    print("=" * 64)

    heads_metrics: dict[str, dict] = {}
    heads_models: dict[str, dict] = {}
    skipped: list[str] = []
    for spec in ITEMS:
        res = _run_item(spec, ep, af)
        if res is None:
            skipped.append(spec["key"])
            print(f"[skip:{spec['key']:18s}] minority class < {MIN_MINORITY}")
            continue
        m, mod = res
        heads_metrics[spec["key"]] = m
        heads_models[spec["key"]] = mod
        print(f"[{spec['domain']:10s}] {spec['key']:18s} (>= {spec['thr']}, {spec['col']}) "
              f"n={m['n']:3d} base={m['base_rate']:.0%}  AUC {m['auc']:.3f}  "
              f"Brier {m['brier']:.3f} (base {m['brier_baseline']:.3f})")

    aucs = {k: m["auc"] for k, m in heads_metrics.items()}
    summary = {
        "n_items_modeled": len(heads_models),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "mean_auc": float(np.mean(list(aucs.values()))) if aucs else None,
        "best_item": max(aucs, key=aucs.get) if aucs else None,
        "worst_item": min(aucs, key=aucs.get) if aucs else None,
        "domains": sorted({it["domain"] for it in ITEMS}),
    }
    print(f"\nmodeled {summary['n_items_modeled']} items  mean AUC {summary['mean_auc']:.3f}  "
          f"best={summary['best_item']} worst={summary['worst_item']}")

    out_dir = OUT / "independence"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "feature_cols": list(af.feature_cols),
        "numeric_cols": list(af.numeric_cols),
        "categorical_cols": list(af.categorical_cols),
        "discharge_timepoint": DISCHARGE_TP,
        "items": [dict(it) for it in ITEMS if it["key"] in heads_models],
        "heads": heads_models,
    }
    joblib.dump(bundle, out_dir / "bundle.joblib")

    payload = {
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "definition": "functional independence (aids/devices allowed, no human assistance)",
        "discharge_timepoint": DISCHARGE_TP,
        "items": [dict(it) for it in ITEMS],
        "excluded": _excluded_rates(ep),
        "heads": heads_metrics,
        "summary": summary,
    }
    (OUT / "independence_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {out_dir / 'bundle.joblib'} and {OUT / 'independence_metrics.json'}")


if __name__ == "__main__":
    main()
