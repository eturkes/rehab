"""Recovery topography map (G8) — per-segment ISNCSCI functional-recovery atlas.

Where the production heads predict aggregate discharge function (SCIM, AIS, LOS) and G7 the
per-SCIM-item ADL independence, this module predicts, for **every ISNCSCI segment**, the
calibrated probability of reaching a *functional milestone* at discharge — a body-map atlas of
"which muscles will have functional strength, and where will sensation return."  It is the
impairment/neurology complement to G7's function-level profile, and it mines the single largest
*unused* signal in the dataset: the 132 per-segment ISNCSCI columns that the 30 admission
features collapse into just five totals.

Segments + milestones (132 total)
---------------------------------
* **Motor** — 20 key muscles ({side}KeyMuscle{level}, level in C5-T1 & L2-S1, L/R), grade 0-5.
  Milestone = P(grade >= 3) = *antigravity* (the muscle can move against gravity — the key
  functional motor threshold).
* **Sensory** — 56 light-touch + 56 pin-prick dermatomes ({side}{LightTouch|PinPrick}{level},
  level in C2..S45, L/R), grade 0-2.  Milestone = P(grade >= 1) = *protective/preserved*
  sensation (the closest analogue to motor antigravity: functionally useful, not necessarily
  normal).  Dermatomes far from the injury sit near the ceiling (preserved in almost everyone);
  a head with too few minority cases is *degenerate* — it gets an honest constant base-rate
  probability for the map but no fitted model (mirrors G7 excluding the near-universal items).

Method — 132 independent calibrated heads, each fed its OWN admission grade
---------------------------------------------------------------------------
A diagnostic settled the architecture decisively: the 30 aggregate admission features (motor/
sensory totals, neurological level, AIS) predict a *specific* segment's discharge state barely
better than chance (per-segment OOF AUC ~0.43-0.63), because the dominant predictor of a segment's
fate is **its own admission grade** — a per-segment/diagonal relationship, not the cross-segment
correlation a low-rank/structured model would exploit (the discharge matrix is also not low-rank).
Adding the segment's own admission grade lifts OOF AUC to ~0.85-0.94.  So each segment gets an
independent LightGBM head on ``[the 30 admission features] + [that segment's own admission grade
``adm_self``]`` (LOCF over the admission-fallback timepoints), with Platt (sigmoid) calibration —
the same battle-tested plumbing as ``conversion``/``independence`` (helpers imported verbatim).
Including ``adm_self`` is still admission-only prediction (no leakage); it is simply the
per-segment admission ISNCSCI that the curated 30 features omit.

Methodology (small per-segment cohorts; many heads)
---------------------------------------------------
Grouped 5-fold CV by ``IDNumber`` -> out-of-fold (OOF) predictions drive every reported metric and
the Platt calibrator; the head is refit on the full cohort for the bundle (conservative, mirroring
``conversion.py``).  SHAP drivers are *descriptive* in-sample importances.  No conformal/APS layer:
a prediction set over a binary head degenerates (see .agent/memory.md 0b), so the calibrated probability
+ reliability curve is the uncertainty surface (as in G7).

Diagnostic + inference layer, like :mod:`~.conversion` / :mod:`~.independence` / :mod:`~.multistate`:
writes a tracked identifier-free ``models/topography_metrics.json`` + a git-ignored
``models/topography/bundle.joblib`` and **never touches** ``train.py``'s production artifacts, so
``training_metrics.json`` byte-repro is preserved.

Persisted bundle shape (consumed by dashboard/compute.py::predict_topography in Part 2)
---------------------------------------------------------------------------------------
    feature_cols, numeric_cols, categorical_cols, admission_fallback, discharge_timepoint
    segments: [{key, modality, side, level, cord_order, thr}]   (body-map display order)
    heads[seg_key] = {clf, calibrator, feature_cols, adm_self_col, thr, base_rate, degenerate}
                     (degenerate heads carry only {thr, base_rate, degenerate=True} -> constant prob)
The Platt ``calibrator`` is a 1-feature ``LogisticRegression`` over the LightGBM logit; apply it at
inference with the same logit transform (``_apply_platt`` is mirrored in compute.py to avoid
importing this module — and thus shap — into the dashboard process).  Each head's feature row is
``[the 30 admission features] + [adm_self = the patient's own admission grade for ``adm_self_col``]``.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from rehab_sci.data.dataset import ADMISSION_FALLBACK, build_analysis_dataset
from rehab_sci.models.conversion import (
    _apply_platt,
    _calibration_curve,
    _fit_platt,
    _oof_binary,
    _params_binary,
    _refit,
    _shap_top,
    _typed_X,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models"

N_SPLITS = 5            # grouped-CV folds (matches conversion._oof_binary's GroupKFold)
N_CAL_BINS = 5          # quantile bins for the per-segment reliability curve (small cohorts)
POOLED_BINS = 10        # bins for the pooled per-modality reliability curve
MIN_MINORITY = 12       # a head needs >= this many minority-class episodes to fit; else degenerate
DISCHARGE_TP = "discharge"

# Milestone thresholds per modality (see module docstring).
THRESHOLDS = {"motor": 3, "light_touch": 1, "pin_prick": 1}
MOTOR_LEVELS = ("C5", "C6", "C7", "C8", "T1", "L2", "L3", "L4", "L5", "S1")
SENSORY_LEVELS = (
    "C2", "C3", "C4", "C5", "C6", "C7", "C8", "T1", "T2", "T3", "T4", "T5", "T6",
    "T7", "T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "S1", "S2", "S3", "S45",
)
SIDES = ("Right", "Left")
MODALITY_TEMPLATE = {
    "motor": "{side}KeyMuscle{level}",
    "light_touch": "{side}LightTouch{level}",
    "pin_prick": "{side}PinPrick{level}",
}
MODALITIES = ("motor", "light_touch", "pin_prick")

# Rostro-caudal cord order for every anatomical level (spatial axis for the body map).
_CORD_SEQUENCE = (
    [f"C{i}" for i in range(1, 9)]
    + [f"T{i}" for i in range(1, 13)]
    + [f"L{i}" for i in range(1, 6)]
    + ["S1", "S2", "S3", "S4", "S45", "S5"]
)
LEVEL_ORDER = {lv: i for i, lv in enumerate(_CORD_SEQUENCE)}


# ----------------------------- segment registry ----------------------------

def _build_registry() -> list[dict]:
    """The 132 ISNCSCI segments in body-map display order: motor, then LT, then PP; rostro-caudal."""
    reg: list[dict] = []
    for modality, levels in (("motor", MOTOR_LEVELS), ("light_touch", SENSORY_LEVELS), ("pin_prick", SENSORY_LEVELS)):
        for side in SIDES:
            for level in levels:
                reg.append({
                    "key": MODALITY_TEMPLATE[modality].format(side=side, level=level),
                    "modality": modality,
                    "side": side,
                    "level": level,
                    "cord_order": LEVEL_ORDER[level],
                    "thr": THRESHOLDS[modality],
                })
    reg.sort(key=lambda s: (MODALITIES.index(s["modality"]), s["cord_order"], s["side"]))
    return reg


# ----------------------------- per-segment ISNCSCI matrices ----------------------------

def _timepoint_matrix(long: pd.DataFrame, tp: str, seg_keys: list[str]) -> pd.DataFrame:
    return (long[long["TIME_Name"] == tp].set_index("KeyRecordNumber")
            .reindex(columns=seg_keys).apply(pd.to_numeric, errors="coerce"))


def _discharge_matrix(long: pd.DataFrame, seg_keys: list[str]) -> pd.DataFrame:
    """Graded discharge ISNCSCI matrix (index=KeyRecordNumber, cols=segment keys), NaN where unobserved."""
    return _timepoint_matrix(long, DISCHARGE_TP, seg_keys)


def _admission_matrix(long: pd.DataFrame, seg_keys: list[str]) -> pd.DataFrame:
    """Admission ISNCSCI matrix: first non-null over the admission-fallback timepoints, per segment.

    Mirrors ``build_episode_frame``'s admission backfill (0day, then 72h/2w/4w fill the gaps).
    """
    A: pd.DataFrame | None = None
    for tp in ADMISSION_FALLBACK:
        slc = _timepoint_matrix(long, tp, seg_keys)
        A = slc if A is None else A.combine_first(slc)
    return A if A is not None else pd.DataFrame(columns=seg_keys)


# ----------------------------- per-segment head ----------------------------

def _run_segment(seg: dict, ep: pd.DataFrame, af, disch_col: pd.Series, adm_self: pd.Series) -> tuple[dict, dict]:
    """Fit + score one segment's calibrated binary milestone head; return (metrics, persisted-model)."""
    d = pd.to_numeric(disch_col, errors="coerce")
    sel = d.notna() & ep["IDNumber"].notna()
    cohort = ep[sel]
    y = (d[sel] >= seg["thr"]).astype(int).to_numpy()
    n, n_pos = len(y), int(y.sum())
    base = float(y.mean()) if n else 0.0
    rec = {k: seg[k] for k in ("key", "modality", "side", "level", "cord_order", "thr")}
    rec.update({"n": n, "n_pos": n_pos, "base_rate": base, "brier_baseline": float(base * (1 - base))})

    minority = min(n_pos, n - n_pos)
    if minority < MIN_MINORITY:  # too few minority cases to fit a stable CV -> honest constant head
        rec["degenerate"] = True
        rec["auc"] = None
        model = {k: seg[k] for k in ("key", "modality", "side", "level", "cord_order", "thr")}
        model.update({"degenerate": True, "base_rate": base})
        return rec, model

    X = _typed_X(cohort, af).copy()
    X["adm_self"] = adm_self[sel].to_numpy()
    groups = cohort["IDNumber"].astype("float64").astype("int64")
    cat_cols = [c for c in af.categorical_cols if c in X.columns]

    oof, best_iter = _oof_binary(X, y, groups, cat_cols)
    cal = _fit_platt(oof, y)
    oof_cal = _apply_platt(cal, oof)
    final = _refit(_params_binary(), X, y, cat_cols, best_iter)

    rec.update({
        "degenerate": False,
        "auc": float(roc_auc_score(y, oof)),
        "brier": float(brier_score_loss(y, oof_cal)),
        "brier_raw": float(brier_score_loss(y, oof)),
        "calibration": _calibration_curve(oof_cal, y, N_CAL_BINS),
        "calibration_raw": _calibration_curve(oof, y, N_CAL_BINS),
        "shap_top": _shap_top(final, X),
    })
    model = {k: seg[k] for k in ("key", "modality", "side", "level", "cord_order", "thr")}
    model.update({
        "degenerate": False,
        "clf": final,
        "calibrator": cal,
        "feature_cols": list(X.columns),
        "adm_self_col": seg["key"],
        "base_rate": base,
        # OOF positive-class probs for the pooled per-modality reliability curve (not persisted in bundle)
        "_oof_cal": oof_cal,
        "_y": y,
    })
    return rec, model


# ----------------------------- entry point ----------------------------

def main() -> None:
    af = build_analysis_dataset()
    ep = af.df
    registry = _build_registry()
    seg_keys = [s["key"] for s in registry]

    print("=" * 72)
    print("RECOVERY TOPOGRAPHY MAP (G8) — per-segment ISNCSCI functional-recovery atlas")
    print("=" * 72)

    # Align the per-segment matrices to ``ep`` BY KeyRecordNumber, then relabel to ep's RangeIndex.
    # ``ep`` is ``build_episode_frame(...).reset_index()`` -> KeyRecordNumber is a column, the index is
    # positional, so reindexing the KeyRecordNumber-indexed matrices on ``ep.index`` would misalign them.
    krn = ep["KeyRecordNumber"].to_numpy()
    disch_df = _discharge_matrix(af.longitudinal, seg_keys).reindex(krn).set_axis(ep.index)
    adm_df = _admission_matrix(af.longitudinal, seg_keys).reindex(krn).set_axis(ep.index)

    segments_out: list[dict] = []
    heads: dict[str, dict] = {}
    pooled: dict[str, dict] = {m: {"prob": [], "y": []} for m in MODALITIES}
    print(f"\nfitting {len(registry)} per-segment heads (30 admission features + own admission grade) ...")
    for i, seg in enumerate(registry):
        rec, model = _run_segment(seg, ep, af, disch_df[seg["key"]], adm_df[seg["key"]])
        segments_out.append(rec)
        mod = seg["modality"]
        if model.get("degenerate"):
            heads[seg["key"]] = {k: model[k] for k in ("thr", "base_rate", "degenerate", "modality",
                                                       "side", "level", "cord_order", "key")}
        else:
            pooled[mod]["prob"].append(model.pop("_oof_cal"))
            pooled[mod]["y"].append(model.pop("_y"))
            heads[seg["key"]] = model
        if (i + 1) % 20 == 0 or (i + 1) == len(registry):
            print(f"  {i + 1}/{len(registry)} heads done")

    # ---- per-modality summary ----
    modality_summary: dict[str, dict] = {}
    for mod in MODALITIES:
        segs = [s for s in segments_out if s["modality"] == mod]
        fit = [s for s in segs if not s["degenerate"]]
        pc = pooled[mod]
        cal_curve = {}
        if pc["prob"]:
            cal_curve = _calibration_curve(np.concatenate(pc["prob"]), np.concatenate(pc["y"]), POOLED_BINS)
        modality_summary[mod] = {
            "n_segments": len(segs),
            "n_modelable": len(fit),
            "n_degenerate": len(segs) - len(fit),
            "mean_auc": float(np.mean([s["auc"] for s in fit])) if fit else None,
            "mean_brier": float(np.mean([s["brier"] for s in fit])) if fit else None,
            "mean_brier_baseline": float(np.mean([s["brier_baseline"] for s in fit])) if fit else None,
            "mean_base_rate": float(np.mean([s["base_rate"] for s in segs])),
            "expected_count_cohort_mean": float(np.sum([s["base_rate"] for s in segs])),  # ~Sum base rates
            "calibration": cal_curve,
        }
        ms = modality_summary[mod]
        print(f"  [{mod}] modelable {ms['n_modelable']}/{ms['n_segments']}  "
              f"meanAUC {ms['mean_auc']}  expected {ms['expected_count_cohort_mean']:.1f}/{len(segs)}")

    # ---- aggregate descriptive drivers per modality (tally mean|SHAP| across heads) ----
    drivers_by_modality: dict[str, list] = {}
    for mod in MODALITIES:
        tally: dict[str, float] = {}
        cnt = 0
        for s in segments_out:
            if s["modality"] == mod and not s["degenerate"]:
                cnt += 1
                for d in s["shap_top"]:
                    tally[d["feature"]] = tally.get(d["feature"], 0.0) + d["mean_abs"]
        ranked = sorted(tally.items(), key=lambda kv: -kv[1])[:12]
        drivers_by_modality[mod] = [{"feature": f, "mean_abs": v / max(cnt, 1)} for f, v in ranked]

    # ---- persist bundle (git-ignored) ----
    out_dir = OUT / "topography"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "feature_cols": list(af.feature_cols),
        "numeric_cols": list(af.numeric_cols),
        "categorical_cols": list(af.categorical_cols),
        "admission_fallback": list(ADMISSION_FALLBACK),
        "discharge_timepoint": DISCHARGE_TP,
        "segments": registry,
        "heads": heads,
    }
    joblib.dump(bundle, out_dir / "bundle.joblib")

    # ---- persist tracked identifier-free metrics ----
    payload = {
        "random_state": _params_binary()["random_state"],
        "n_splits": N_SPLITS,
        "min_minority": MIN_MINORITY,
        "thresholds": THRESHOLDS,
        "feature_note": "30 admission features + per-segment own admission grade (adm_self)",
        "cohort": {
            "n_segments": len(seg_keys),
            "mean_n_by_modality": {
                mod: float(np.mean([s["n"] for s in segments_out if s["modality"] == mod]))
                for mod in MODALITIES
            },
        },
        "modality_summary": modality_summary,
        "drivers_by_modality": drivers_by_modality,
        "segments": segments_out,
    }
    (OUT / "topography_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {out_dir / 'bundle.joblib'} and {OUT / 'topography_metrics.json'}")


if __name__ == "__main__":
    main()
