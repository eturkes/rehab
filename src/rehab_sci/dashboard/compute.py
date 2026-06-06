"""Pure computation helpers for model inference, conformal PI, and SHAP.

Shared by simulator, patient, and report callbacks. No Dash or Plotly deps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rehab_sci.constants import AIS_ORD_TO_LETTER
from rehab_sci.dashboard.state import (
    CONVERSION_BUNDLE,
    EP,
    FEATURE_SPEC,
    INDEPENDENCE_BUNDLE,
    LANDMARK_BUNDLE,
    LONG,
    MULTISTATE_BUNDLE,
    OUTCOME_BUNDLES,
    PHENOTYPE_DATA,
    SIM_DEFAULTS,
    TRAJECTORY_BUNDLE,
)
from rehab_sci.data.episodes import episode_admission_features
from rehab_sci.data.phenotypes import (
    MEASURES,
    WINDOW,
    WINDOW_DAYS,
    build_gmm_data,
    predict_proba,
)
from rehab_sci.models.outcomes import OUTCOMES, OutcomeSpec


# ---------- conformal q resolution ----------
def resolve_group_q(q_by_group: dict | None, marginal: float, X: pd.DataFrame) -> float:
    """Resolve Mondrian q for a single-row input.

    Priority: AIS group -> paralysis group -> marginal.
    """
    if q_by_group is None or len(X) == 0:
        return marginal
    row = X.iloc[0]
    ais_qs = q_by_group.get("ais", {})
    if ais_qs and "AIS_ord" in X.columns:
        ais_val = row["AIS_ord"]
        if pd.notna(ais_val):
            letter = AIS_ORD_TO_LETTER.get(int(ais_val))
            if letter and letter in ais_qs:
                return float(ais_qs[letter])
    para_qs = q_by_group.get("paralysis", {})
    if para_qs and "対麻痺_四肢麻痺" in X.columns:
        para_val = row["対麻痺_四肢麻痺"]
        if pd.notna(para_val) and str(para_val) in para_qs:
            return float(para_qs[str(para_val)])
    return marginal


def resolve_conformal_q(fspec: dict, X: pd.DataFrame) -> float:
    return resolve_group_q(
        fspec.get("conformal_q_by_group"),
        float(fspec.get("conformal_q_transformed", 0.0)),
        X,
    )


def resolve_aps_q(fspec: dict, X: pd.DataFrame) -> float:
    return resolve_group_q(
        fspec.get("aps_q_by_group"),
        float(fspec.get("aps_q_hat", 1.0)),
        X,
    )


# ---------- trajectory ----------
def predict_trajectory(X: pd.DataFrame) -> dict | None:
    """Predict SCIM-total at each trajectory timepoint for a single-row input."""
    if TRAJECTORY_BUNDLE is None:
        return None
    tps = TRAJECTORY_BUNDLE["timepoints"]
    models = TRAJECTORY_BUNDLE["models"]
    conf = TRAJECTORY_BUNDLE["conformal"]
    clip_min = TRAJECTORY_BUNDLE.get("clip_min", 0.0)
    clip_max = TRAJECTORY_BUNDLE.get("clip_max", 100.0)

    out_tps: list[str] = []
    out_pred: list[float] = []
    out_lo: list[float] = []
    out_hi: list[float] = []

    for tp in tps:
        tp_models = models[tp]
        tp_conf = conf[tp]
        pred_t = float(tp_models["median"].predict(X)[0])
        pred = max(clip_min, min(clip_max, pred_t))
        q_t = resolve_group_q(
            tp_conf["q_by_group"], float(tp_conf["q_transformed"]), X,
        )
        lo_conf = max(clip_min, min(clip_max, pred_t - q_t))
        hi_conf = max(clip_min, min(clip_max, pred_t + q_t))
        lo_q = max(clip_min, min(clip_max, float(tp_models["p10"].predict(X)[0])))
        hi_q = max(clip_min, min(clip_max, float(tp_models["p90"].predict(X)[0])))

        out_tps.append(tp)
        out_pred.append(pred)
        out_lo.append(min(lo_conf, lo_q))
        out_hi.append(max(hi_conf, hi_q))

    return {"timepoints": out_tps, "pred": out_pred, "lo": out_lo, "hi": out_hi}


# ---------- APS ----------
def aps_prediction_set(proba_row: np.ndarray, q_hat: float) -> list[int]:
    order = np.argsort(-proba_row)
    cumsum = 0.0
    pred_set: list[int] = []
    for j in order:
        pred_set.append(int(j))
        cumsum += proba_row[j]
        if cumsum >= q_hat:
            break
    return sorted(pred_set)


# ---------- scalar transforms ----------
def inv_transform_scalar(x: float, transform: str | None) -> float:
    if transform == "log1p":
        return float(np.expm1(x))
    return float(x)


def clip_scalar(x: float, lo: float | None, hi: float | None) -> float:
    if lo is not None and x < lo:
        x = lo
    if hi is not None and x > hi:
        x = hi
    return x


def format_value(col: str, value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    if isinstance(value, float):
        return f"{value:.0f}"
    return str(value)


# ---------- reference predictions (What-if, PDF report) ----------
def compute_ref_predictions(X: pd.DataFrame) -> dict:
    """Compute predictions for all outcomes on a single-row X."""
    outcomes: dict[str, dict] = {}
    for spec in OUTCOMES:
        bundle = OUTCOME_BUNDLES[spec.key]
        fspec = bundle["feature_spec"]
        if spec.task == "regression":
            transform = fspec.get("transform")
            cmin = fspec.get("clip_min")
            cmax = fspec.get("clip_max")
            q_t = resolve_conformal_q(fspec, X)
            pred_t = float(bundle["median"].predict(X)[0])
            pred_p10_t = float(bundle["p10"].predict(X)[0])
            pred_p90_t = float(bundle["p90"].predict(X)[0])
            pred = clip_scalar(inv_transform_scalar(pred_t, transform), cmin, cmax)
            lo_conf = clip_scalar(inv_transform_scalar(pred_t - q_t, transform), cmin, cmax)
            hi_conf = clip_scalar(inv_transform_scalar(pred_t + q_t, transform), cmin, cmax)
            lo_q = clip_scalar(inv_transform_scalar(pred_p10_t, transform), cmin, cmax)
            hi_q = clip_scalar(inv_transform_scalar(pred_p90_t, transform), cmin, cmax)
            outcomes[spec.key] = {
                "task": "regression",
                "pred": pred,
                "lo": min(lo_conf, lo_q),
                "hi": max(hi_conf, hi_q),
            }
        else:
            class_labels = list(fspec.get("class_labels", spec.class_labels))
            proba = np.asarray(bundle["clf"].predict_proba(X)[0], dtype=float)
            pred_idx = int(np.argmax(proba))
            outcomes[spec.key] = {
                "task": "multiclass",
                "pred_class": class_labels[pred_idx],
                "pred_prob": float(proba[pred_idx]),
                "proba": proba.tolist(),
            }
    return outcomes


# ---------- simulator input collection ----------
def collect_sim_inputs(num_vals, num_ids, cat_vals, cat_ids) -> pd.DataFrame:
    # Blank fields stay unknown (NaN) — LightGBM uses native missing handling,
    # matching training (features keep NaN; see train.py:_prep). No imputation.
    row: dict[str, object] = {}
    for ident, v in zip(num_ids, num_vals, strict=False):
        row[ident["col"]] = v
    for ident, v in zip(cat_ids, cat_vals, strict=False):
        row[ident["col"]] = v
    X = pd.DataFrame([{c: row.get(c) for c in FEATURE_SPEC["feature_cols"]}])
    for c in FEATURE_SPEC["categorical_cols"]:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for c in FEATURE_SPEC["numeric_cols"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


# ---------- SHAP for single-row inference ----------
def shap_for_row_regression(X: pd.DataFrame, model) -> tuple[np.ndarray, float]:
    import shap

    expl = shap.TreeExplainer(model)
    values = expl.shap_values(X)
    base = (
        float(expl.expected_value)
        if np.isscalar(expl.expected_value)
        else float(expl.expected_value[0])
    )
    return values[0], base


def shap_for_row_class(
    X: pd.DataFrame, clf, class_idx: int, n_classes: int,
) -> tuple[np.ndarray, float]:
    import shap

    expl = shap.TreeExplainer(clf)
    raw = expl.shap_values(X)
    if isinstance(raw, list):
        per_class = np.asarray(raw[class_idx])[0]
        base_arr = expl.expected_value
        base = (
            float(base_arr)
            if np.isscalar(base_arr)
            else float(np.asarray(base_arr).ravel()[class_idx])
        )
    else:
        arr = np.asarray(raw)
        if arr.ndim == 3 and arr.shape[0] == n_classes and arr.shape[-1] != n_classes:
            arr = np.transpose(arr, (1, 2, 0))
        per_class = arr[0, :, class_idx]
        base_arr = expl.expected_value
        if np.isscalar(base_arr):
            base = float(base_arr)
        else:
            base = float(np.asarray(base_arr).ravel()[class_idx])
    return per_class, base


# ---------- episode helpers ----------
def episode_row_for_model(key_record: int) -> pd.DataFrame:
    """Build a one-row model input from an episode's admission features."""
    feat = episode_admission_features(EP, key_record, FEATURE_SPEC["feature_cols"])
    for c in FEATURE_SPEC["feature_cols"]:
        if feat.get(c) is None:
            feat[c] = SIM_DEFAULTS.get(c)
    X = pd.DataFrame([{c: feat.get(c) for c in FEATURE_SPEC["feature_cols"]}])
    for c in FEATURE_SPEC["categorical_cols"]:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for c in FEATURE_SPEC["numeric_cols"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def episode_has_admission(key_record: int) -> bool:
    feat = episode_admission_features(EP, key_record, FEATURE_SPEC["feature_cols"])
    return any(v is not None for v in feat.values())


def get_observed_for_outcome(key_record: int, spec: OutcomeSpec) -> float | None:
    row = EP.loc[EP["KeyRecordNumber"] == key_record]
    if row.empty or spec.target_col not in row.columns:
        return None
    v = row.iloc[0][spec.target_col]
    return None if pd.isna(v) else float(v)


# ---------- landmark (dynamic) prediction ----------
# Sharpen a discharge prognosis once early-recovery scores are observed by landmark L.
# Compares the admission-only baseline head against the landmark head (admission + observed
# block) on the marginal-conformal interval — the same paired contrast as the Methods curve.

def _landmark_input(X_base: pd.DataFrame, observed: dict, feature_cols: list[str]) -> pd.DataFrame:
    """Build a one-row model input over ``feature_cols`` from base features + observed block.

    A column named ``<lm_prefix><measure>`` is filled from ``observed[measure]`` (NaN when
    absent); every other column is taken from ``X_base``.  Re-casts categoricals so LightGBM
    realigns category levels by value (matching the production inference path).
    """
    b = LANDMARK_BUNDLE
    prefix = b["lm_prefix"]
    cat_cols = set(b["categorical_cols"])
    base_row = X_base.iloc[0].to_dict() if len(X_base) else {}
    row: dict[str, object] = {}
    for c in feature_cols:
        row[c] = observed.get(c[len(prefix):]) if c.startswith(prefix) else base_row.get(c)
    X = pd.DataFrame([{c: row.get(c) for c in feature_cols}])
    for c in feature_cols:
        if c in cat_cols:
            X[c] = X[c].astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _predict_landmark_head(
    head: dict, X: pd.DataFrame, task: str,
    transform: str | None, cmin: float | None, cmax: float | None,
) -> dict:
    if task == "regression":
        pred_t = float(head["median"].predict(X)[0])
        q = float(head["conformal_q"])
        return {
            "task": "regression",
            "pred": clip_scalar(inv_transform_scalar(pred_t, transform), cmin, cmax),
            "lo": clip_scalar(inv_transform_scalar(pred_t - q, transform), cmin, cmax),
            "hi": clip_scalar(inv_transform_scalar(pred_t + q, transform), cmin, cmax),
        }
    class_labels = list(head["class_labels"])
    proba = np.asarray(head["clf"].predict_proba(X)[0], dtype=float)
    pred_idx = int(np.argmax(proba))
    aps_idx = aps_prediction_set(proba, float(head["aps_q_hat"]))
    return {
        "task": "multiclass",
        "pred_class": class_labels[pred_idx],
        "pred_prob": float(proba[pred_idx]),
        "proba": proba.tolist(),
        "class_labels": class_labels,
        "aps_set": [class_labels[i] for i in aps_idx],
    }


def predict_landmark(
    outcome_key: str, landmark: str, X_base: pd.DataFrame, observed: dict,
) -> dict | None:
    """Paired admission-only baseline vs landmark prediction for one outcome at landmark ``L``.

    Returns ``{"task", "baseline": {...}, "landmark": {...}}`` or ``None`` when the bundle is
    absent or this (outcome, landmark) cell was not modelled (too few eligible episodes).
    """
    b = LANDMARK_BUNDLE
    if b is None:
        return None
    oc = b["outcomes"].get(outcome_key)
    if oc is None:
        return None
    cell = oc["by_landmark"].get(landmark)
    if cell is None:
        return None
    out = {"task": oc["task"]}
    for which in ("baseline", "landmark"):
        head = cell[which]
        X = _landmark_input(X_base, observed, head["feature_cols"])
        out[which] = _predict_landmark_head(
            head, X, oc["task"], oc["transform"], oc.get("clip_min"), oc.get("clip_max"),
        )
    return out


def landmark_voi(
    outcome_key: str, landmark: str, X_base: pd.DataFrame, observed: dict,
) -> dict | None:
    """Per-measure value-of-information for one patient at landmark ``L`` (G2).

    Runs the admission-only baseline head and every single-add head (admission + exactly one
    ``L_<measure>``).  Each measure is scored at the patient's REAL value when present in
    ``observed`` (``which="observed"`` — value realised), else at the eligible-cohort median from
    the bundle's ``lm_value_summary`` (``which="prescriptive"`` — value if obtained).  Because
    every single-add head carries its own marginal conformal ``q`` / APS set, the PI tightening (or
    APS-set shrink) a measure buys over the *same* baseline is well-defined and measure-specific.

    Returns ``{"task", "landmark", "baseline", "measures": [...]}`` sorted by that tightening
    (largest first), or ``None`` when the bundle / cell / single-add heads are absent.
    """
    b = LANDMARK_BUNDLE
    if b is None:
        return None
    oc = b["outcomes"].get(outcome_key)
    if oc is None:
        return None
    cell = oc["by_landmark"].get(landmark)
    if cell is None or not cell.get("single"):
        return None
    task, transform = oc["task"], oc["transform"]
    cmin, cmax = oc.get("clip_min"), oc.get("clip_max")
    value_summary = (b.get("lm_value_summary") or {}).get(landmark, {})

    base_head = cell["baseline"]
    base_pred = _predict_landmark_head(
        base_head, _landmark_input(X_base, {}, base_head["feature_cols"]),
        task, transform, cmin, cmax,
    )

    measures: list[dict] = []
    for meas, head in cell["single"].items():
        obs_val = observed.get(meas)
        if obs_val is not None and not (isinstance(obs_val, float) and np.isnan(obs_val)):
            val: float | None = float(obs_val)
            which = "observed"
        else:
            vs = value_summary.get(meas)
            val = float(vs["median"]) if vs else None
            which = "prescriptive"
        X = _landmark_input(X_base, {meas: val} if val is not None else {}, head["feature_cols"])
        pred = _predict_landmark_head(head, X, task, transform, cmin, cmax)
        entry: dict = {"measure": meas, "which": which, "value": val}
        if task == "regression":
            base_hw = (base_pred["hi"] - base_pred["lo"]) / 2.0
            hw = (pred["hi"] - pred["lo"]) / 2.0
            entry.update({
                "pred": pred["pred"], "lo": pred["lo"], "hi": pred["hi"], "halfwidth": hw,
                "d_halfwidth": base_hw - hw,            # > 0 = PI tightening over baseline
                "d_point": pred["pred"] - base_pred["pred"],
            })
        else:
            base_size = len(base_pred["aps_set"])
            entry.update({
                "pred_class": pred["pred_class"], "pred_prob": pred["pred_prob"],
                "aps_set": pred["aps_set"], "set_size": len(pred["aps_set"]),
                "d_setsize": base_size - len(pred["aps_set"]),   # > 0 = APS set shrinks
                "changed_class": pred["pred_class"] != base_pred["pred_class"],
            })
        measures.append(entry)

    rank_key = "d_halfwidth" if task == "regression" else "d_setsize"
    measures.sort(key=lambda e: e[rank_key], reverse=True)
    return {"task": task, "landmark": landmark, "baseline": base_pred, "measures": measures}


def _episode_timepoint_oidx(key_record: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Episode rows restricted to the landmark measures, plus the timepoint order-index map."""
    b = LANDMARK_BUNDLE
    oidx = {tp: i for i, tp in enumerate(b["timepoint_order"])}
    measures = [m for m in b["landmark_cols"] if m in LONG.columns]
    sub = LONG[LONG["KeyRecordNumber"] == key_record][["TIME_Name", *measures]].copy()
    sub["_oi"] = sub["TIME_Name"].map(oidx)
    return sub.dropna(subset=["_oi"]), oidx


def landmark_observed_for_episode(key_record: int, landmark: str) -> dict[str, float]:
    """Real LOCF observed block for one episode: last non-null value of each landmark measure
    at an intermediate timepoint on or before ``landmark`` (mirrors training-time construction)."""
    if LANDMARK_BUNDLE is None:
        return {}
    sub, oidx = _episode_timepoint_oidx(key_record)
    cutoff = oidx.get(landmark)
    if cutoff is None or sub.empty:
        return {}
    sub = sub[sub["_oi"] <= cutoff].sort_values("_oi")
    observed: dict[str, float] = {}
    for col in LANDMARK_BUNDLE["landmark_cols"]:
        if col in sub.columns:
            nn = sub[col].dropna()
            if not nn.empty:
                observed[col] = float(nn.iloc[-1])
    return observed


def episode_landmark_eligibility(key_record: int) -> dict[str, bool]:
    """Per-landmark still-admitted eligibility: True when the episode has a tracked observation
    at an intermediate timepoint at or after L (its latest intermediate obs reaches L)."""
    if LANDMARK_BUNDLE is None:
        return {}
    landmarks = LANDMARK_BUNDLE["landmarks"]
    sub, oidx = _episode_timepoint_oidx(key_record)
    measures = [m for m in LANDMARK_BUNDLE["landmark_cols"] if m in sub.columns]
    if sub.empty or not measures:
        return {lm: False for lm in landmarks}
    has_obs = sub[measures].notna().any(axis=1)
    reached = sub.loc[has_obs, "_oi"]
    max_oi = int(reached.max()) if not reached.empty else -1
    return {lm: (max_oi >= oidx[lm]) for lm in landmarks}


# ---------- observed-trajectory phenotype prognosis (G3 part 2) ----------
def _phenotype_episode_obs(key_record: int) -> dict[str, list[tuple[str, float]]]:
    """This episode's observed ``(timepoint, value)`` pairs within the phenotyping window,
    per measure, in chronological window order (empty list for an unobserved measure)."""
    order = {tp: i for i, tp in enumerate(WINDOW)}
    sub = LONG[(LONG["KeyRecordNumber"] == key_record) & (LONG["TIME_Name"].isin(WINDOW))]
    out: dict[str, list[tuple[str, float]]] = {}
    for m in MEASURES:
        if m not in sub.columns:
            out[m] = []
            continue
        rows = sub[["TIME_Name", m]].dropna(subset=[m])
        pairs = [(str(tp), float(v)) for tp, v in zip(rows["TIME_Name"], rows[m], strict=True)]
        out[m] = sorted(pairs, key=lambda pv: order.get(pv[0], 99))
    return out


def phenotype_cutoff_options(key_record: int, min_cells: int = 2) -> list[str]:
    """Window timepoints (chronological) eligible as observation-cutoffs for phenotype
    membership: each timepoint at which a *new* observation appears once the episode has
    accumulated at least ``min_cells`` observed SCIM/motor cells.  Membership changes only
    when an observation is added, so redundant cutoffs are skipped; the final entry is the
    full observed window.  Empty when the phenotype model is unavailable or the episode is
    too sparsely observed to define a trajectory."""
    if PHENOTYPE_DATA is None:
        return []
    per_tp: dict[str, int] = {}
    for pairs in _phenotype_episode_obs(key_record).values():
        for tp, _ in pairs:
            per_tp[tp] = per_tp.get(tp, 0) + 1
    cutoffs: list[str] = []
    cum = 0
    for tp in WINDOW:
        if tp not in per_tp:
            continue
        cum += per_tp[tp]
        if cum >= min_cells:
            cutoffs.append(tp)
    return cutoffs


def predict_phenotype_membership(key_record: int, cutoff: str) -> dict | None:
    """Soft phenotype membership for one episode using only observations on/before ``cutoff``,
    plus the membership-weighted conditioned-outcome mix.

    Builds the single-individual growth-mixture design from the observed (SCIM, motor) cells up
    to ``cutoff`` and applies the fitted model's E-step (``predict_proba``).  Because the
    per-individual responsibilities are independent, the full-window membership reproduces the
    bundle's stored posterior for an in-cohort episode.  The conditioned-outcome mix is the
    membership-weighted blend of the per-phenotype cohort summaries (renormalized over the
    phenotypes that carry each statistic).  Returns ``None`` when the model is unavailable or no
    cell is observed up to ``cutoff``."""
    if PHENOTYPE_DATA is None:
        return None
    params = PHENOTYPE_DATA["params"]
    summaries = PHENOTYPE_DATA["summaries"]
    K = int(PHENOTYPE_DATA["k"])
    cutoff_day = WINDOW_DAYS[cutoff]
    win_upto = [tp for tp in WINDOW if WINDOW_DAYS[tp] <= cutoff_day]
    sub = LONG[(LONG["KeyRecordNumber"] == key_record) & (LONG["TIME_Name"].isin(win_upto))]
    data = build_gmm_data(sub, [int(key_record)], PHENOTYPE_DATA["degree"])
    if data.N == 0:
        return None
    membership = predict_proba(data, params)[0]  # (K,)

    def _wmean(field: str) -> float | None:
        num = den = 0.0
        for k in range(K):
            v = summaries[k].get(field)
            if v is not None:
                num += membership[k] * v
                den += membership[k]
        return (num / den) if den > 0 else None

    grades = ["A", "B", "C", "D", "E"]
    ais_num = dict.fromkeys(grades, 0.0)
    ais_den = 0.0
    for k in range(K):
        dist = summaries[k].get("ais_distribution") or {}
        if dist:
            for g in grades:
                ais_num[g] += membership[k] * dist.get(g, 0.0)
            ais_den += membership[k]
    ais_mix = {g: (ais_num[g] / ais_den if ais_den > 0 else 0.0) for g in grades}

    obs_upto = {
        m: [(tp, v) for tp, v in pairs if WINDOW_DAYS[tp] <= cutoff_day]
        for m, pairs in _phenotype_episode_obs(key_record).items()
    }
    return {
        "membership": [float(x) for x in membership],
        "dominant": int(np.argmax(membership)),
        "exp_discharge_scim": _wmean("median_discharge_scim"),
        "exp_los": _wmean("mean_los"),
        "ais_mix": ais_mix,
        "observed": obs_upto,
        "cutoff": cutoff,
        "n_obs": sum(len(v) for v in obs_upto.values()),
        "k": K,
    }


# ---------- AIS-grade conversion (G4) ----------
# Admission->discharge AIS *transition* inference for one admission row, gated by admission grade.
# Two CALIBRATED binary endpoints (motor_incomplete A/B->>=C, ambulatory A-C->>=D) plus an ORDINAL
# magnitude head {0,+1,>=+2} on A-D.  The two head families are NOT numerically comparable (see
# AGENT_NOTES s3 CRUX): the binary heads are Platt-calibrated probabilities, the magnitude head is
# class_weight="balanced" so its class scores are uncalibrated — surface it as its APS set /
# most-likely class, never as a competing probability.  `_apply_platt` is mirrored here (logit ->
# calibrator.predict_proba) so the dashboard never imports models.conversion (which pulls shap).

def _apply_platt(calibrator, prob: float) -> float:
    """Platt (sigmoid) recalibration over the LightGBM logit — mirrors models.conversion._apply_platt."""
    p = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    logit = float(np.log(p / (1.0 - p)))
    return float(calibrator.predict_proba([[logit]])[0, 1])


def _conversion_input(X: pd.DataFrame) -> pd.DataFrame:
    """One-row model input over the conversion bundle's feature universe, dtyped like training
    (every head shares the 30 admission features; re-cast so LightGBM realigns category levels)."""
    b = CONVERSION_BUNDLE
    base = X.iloc[0].to_dict() if len(X) else {}
    cat = set(b["categorical_cols"])
    Xc = pd.DataFrame([{c: base.get(c) for c in b["feature_cols"]}])
    for c in b["feature_cols"]:
        if c in cat:
            Xc[c] = Xc[c].astype("category")
        else:
            Xc[c] = pd.to_numeric(Xc[c], errors="coerce")
    return Xc


def mag_short_label(code: int, mag_cap: int) -> str:
    """Compact, language-neutral label for an ordinal improvement-magnitude class."""
    if code <= 0:
        return "0"
    if code >= mag_cap:
        return f"≥+{mag_cap}"
    return f"+{code}"


def predict_conversion(X: pd.DataFrame) -> dict | None:
    """Admission->discharge AIS conversion for one admission row (see header).

    Returns ``None`` when the bundle is absent.  Requires the admission grade (``AIS_ord``) to be
    present — cohort membership is otherwise undefined; ``ais_ord=None`` then yields an all-N/A
    result the surfaces render as a "needs admission grade" prompt.  Each endpoint applies only if
    the admission grade is in its at-risk ``adm_grades``; the magnitude head only for A-D.
    """
    if CONVERSION_BUNDLE is None:
        return None
    b = CONVERSION_BUNDLE
    ais_raw = X.iloc[0].get(b["adm_col"]) if len(X) else None
    ais_ord = None if ais_raw is None or pd.isna(ais_raw) else int(ais_raw)
    Xc = _conversion_input(X)

    endpoints: dict[str, dict] = {}
    for key, e in b["endpoints"].items():
        applicable = ais_ord is not None and ais_ord in e["adm_grades"]
        entry: dict = {
            "applicable": applicable,
            "base_rate": float(e["base_rate"]),
            "discharge_min": int(e["discharge_min"]),
            "discharge_min_letter": AIS_ORD_TO_LETTER[int(e["discharge_min"])],
            "adm_grades": list(e["adm_grades"]),
        }
        if applicable:
            raw = float(e["clf"].predict_proba(Xc[e["feature_cols"]])[0, 1])
            entry["prob"] = _apply_platt(e["calibrator"], raw)
        endpoints[key] = entry

    m = b["magnitude"]
    mag_applicable = ais_ord is not None and ais_ord in m["adm_grades"]
    magnitude: dict = {
        "applicable": mag_applicable,
        "mag_cap": int(m["mag_cap"]),
        "class_codes": [int(c) for c in m["class_codes"]],
    }
    if mag_applicable:
        proba = np.asarray(m["clf"].predict_proba(Xc[m["feature_cols"]])[0], dtype=float)
        aps_idx = aps_prediction_set(proba, float(m["aps_q_hat"]))
        codes = [int(c) for c in m["class_codes"]]
        magnitude.update({
            "proba": proba.tolist(),
            "pred_code": codes[int(np.argmax(proba))],
            "aps_codes": [codes[i] for i in aps_idx],
        })

    return {
        "ais_ord": ais_ord,
        "adm_letter": AIS_ORD_TO_LETTER.get(ais_ord) if ais_ord is not None else None,
        "any_applicable": mag_applicable or any(e["applicable"] for e in endpoints.values()),
        "endpoints": endpoints,
        "magnitude": magnitude,
    }


# ---------- AIS multi-state recovery (G6) ----------
# AIS-grade *trajectory* inference over the 0day-6m grid (complement to G4's admission->discharge
# endpoint).  Two layers: (1) population multi-state Markov curves — occupancy, first-passage
# conversion, sojourn, median-day-to-improve — looked up by ADMISSION GRADE alone (pure cohort
# dynamics, identical for any patient of that grade); (2) the calibrated improve-by-6m covariate
# head — the one feature-driven quantity, so only it personalizes.  `_apply_platt` is the same
# Platt mirror used for conversion, so the dashboard never imports models.multistate (pulls shap).

# Threshold ordinal -> first-passage curve label (admission grade < threshold => curve is informative).
_MS_THRESH_ORD = {"ge_C": 3, "ge_D": 4}


def _multistate_input(X: pd.DataFrame) -> pd.DataFrame:
    """One-row improve-head input over the multi-state bundle's feature universe, dtyped like
    training (the 30 admission features; re-cast so LightGBM realigns category levels by value)."""
    b = MULTISTATE_BUNDLE
    base = X.iloc[0].to_dict() if len(X) else {}
    cat = set(b["categorical_cols"])
    Xc = pd.DataFrame([{c: base.get(c) for c in b["feature_cols"]}])
    for c in b["feature_cols"]:
        if c in cat:
            Xc[c] = Xc[c].astype("category")
        else:
            Xc[c] = pd.to_numeric(Xc[c], errors="coerce")
    return Xc


def multistate_observed_grades(key_record: int) -> list[tuple[str, int]]:
    """This episode's observed AIS grade (ordinal 1=A..5=E) at each window timepoint, in
    chronological window order.  Drives the patient-card overlay of the individual's *own*
    trajectory atop the admission-grade cohort dynamics.  Empty when AIS is never recorded."""
    if MULTISTATE_BUNDLE is None:
        return []
    col = MULTISTATE_BUNDLE["adm_col"]
    order = {tp: i for i, tp in enumerate(WINDOW)}
    sub = LONG[(LONG["KeyRecordNumber"] == key_record) & (LONG["TIME_Name"].isin(WINDOW))]
    if col not in sub.columns:
        return []
    rows = sub[["TIME_Name", col]].dropna(subset=[col])
    pairs = [(str(tp), round(float(v))) for tp, v in zip(rows["TIME_Name"], rows[col], strict=True)]
    return sorted(pairs, key=lambda pv: order.get(pv[0], 99))


def predict_multistate(X: pd.DataFrame) -> dict | None:
    """AIS multi-state recovery for one admission row (see header).

    Returns ``None`` when the bundle is absent.  Requires the admission grade (``AIS_ord``) —
    multi-state membership is otherwise undefined; ``ais_ord=None`` yields an ``applicable=False``
    result the surfaces render as a "needs admission grade" prompt.  The occupancy / conversion /
    sojourn / median-day curves are looked up by admission grade (cohort dynamics, not
    personalized); only ``improve`` (P(>=1-grade improvement by 6m), calibrated) is feature-driven
    and applies on the A-D room-to-improve cohort.  For an AIS-D admission the improvement endpoint
    is specifically P(reach AIS E) — the surfaces frame it accordingly.
    """
    if MULTISTATE_BUNDLE is None:
        return None
    b = MULTISTATE_BUNDLE
    ais_raw = X.iloc[0].get(b["adm_col"]) if len(X) else None
    ais_ord = None if ais_raw is None or pd.isna(ais_raw) else int(ais_raw)
    result: dict = {
        "ais_ord": ais_ord,
        "adm_letter": AIS_ORD_TO_LETTER.get(ais_ord) if ais_ord is not None else None,
        "window": list(b["window"]),
        "window_days": [int(b["window_days"][w]) for w in b["window"]],
        "state_labels": list(b["state_labels"]),
        "applicable": ais_ord is not None and ais_ord in b["occupancy_by_adm"],
    }
    if not result["applicable"]:
        return result
    result["occupancy"] = np.asarray(b["occupancy_by_adm"][ais_ord], dtype=float).tolist()
    # Only first-passage curves above the admission grade are informative (a threshold at/below the
    # admission grade is trivially already-reached); always surface the any-improvement curve.  An
    # AIS-E admission is the ceiling: its conversion dict is empty (no improvement possible), so the
    # surfaces show the cohort occupancy drift but flag the absent improvement endpoint.
    conv = b["conversion_by_adm"][ais_ord]
    result["at_ceiling"] = "improve" not in conv
    result["conversion"] = {
        lab: [float(x) for x in conv[lab]]
        for lab in ("improve", "ge_C", "ge_D")
        if lab in conv and (lab == "improve" or _MS_THRESH_ORD[lab] > ais_ord)
    }
    result["sojourn"] = [float(x) for x in b["sojourn_by_adm"][ais_ord]]
    mdi = b["median_day_to_improve"].get(ais_ord)
    result["median_day_to_improve"] = None if mdi is None else float(mdi)

    ih = b["improve_head"]
    imp: dict = {"applicable": ais_ord in ih["adm_grades"], "base_rate": float(ih["base_rate"])}
    if imp["applicable"]:
        Xc = _multistate_input(X)
        raw = float(ih["clf"].predict_proba(Xc[ih["feature_cols"]])[0, 1])
        imp["prob"] = _apply_platt(ih["calibrator"], raw)
        imp["to_e"] = ais_ord == 4  # admission D -> "improvement" means reaching AIS E
    result["improve"] = imp
    return result


# ---------- functional-independence profile (G7) ----------
# Per-SCIM-item discharge independence inference for one admission row.  Unlike conversion /
# multistate, this is NOT admission-grade gated — functional independence is predicted for everyone;
# each of the 18 calibrated binary heads emits P(functionally independent in that act at discharge).
# `_apply_platt` is the same Platt mirror used for conversion, so the dashboard never imports
# models.independence (which pulls shap via conversion -> train).

def _independence_input(X: pd.DataFrame) -> pd.DataFrame:
    """One-row input over the independence bundle's feature universe, dtyped like training (all 18
    heads share the 30 admission features; re-cast so LightGBM realigns category levels by value)."""
    b = INDEPENDENCE_BUNDLE
    base = X.iloc[0].to_dict() if len(X) else {}
    cat = set(b["categorical_cols"])
    Xc = pd.DataFrame([{c: base.get(c) for c in b["feature_cols"]}])
    for c in b["feature_cols"]:
        if c in cat:
            Xc[c] = Xc[c].astype("category")
        else:
            Xc[c] = pd.to_numeric(Xc[c], errors="coerce")
    return Xc


def predict_independence(X: pd.DataFrame) -> dict | None:
    """Per-SCIM-item discharge functional-independence profile for one admission row.

    Returns ``None`` when the bundle is absent.  Runs every calibrated per-item head (no
    admission-grade gating — independence is predicted for everyone) and returns, in display order,
    each item's calibrated P(independent) + its cohort base rate, the ordered domain list, and the
    expected number of independent functions (Σ of the calibrated probabilities — a clean profile
    summary that rises monotonically with admission AIS).
    """
    if INDEPENDENCE_BUNDLE is None:
        return None
    b = INDEPENDENCE_BUNDLE
    Xc = _independence_input(X)
    items: list[dict] = []
    total = 0.0
    for it in b["items"]:
        head = b["heads"][it["key"]]
        raw = float(head["clf"].predict_proba(Xc[head["feature_cols"]])[0, 1])
        prob = _apply_platt(head["calibrator"], raw)
        total += prob
        items.append({
            "key": it["key"],
            "col": head["col"],
            "domain": head["domain"],
            "thr": int(head["thr"]),
            "prob": prob,
            "base_rate": float(head["base_rate"]),
        })
    domains: list[str] = []
    for it in items:
        if it["domain"] not in domains:
            domains.append(it["domain"])
    return {
        "items": items,
        "domains": domains,
        "expected_count": total,
        "n_items": len(items),
    }


def independence_observed_for_episode(key_record: int) -> dict[str, bool]:
    """Realized discharge functional independence per item for one episode (discharge item score
    >= the head's independence threshold).  Drives the patient-card overlay of achieved-vs-predicted
    independence.  Only items with an observed discharge score appear; empty when the episode has
    no discharge record."""
    if INDEPENDENCE_BUNDLE is None:
        return {}
    b = INDEPENDENCE_BUNDLE
    sub = LONG[(LONG["KeyRecordNumber"] == key_record) & (LONG["TIME_Name"] == b["discharge_timepoint"])]
    if sub.empty:
        return {}
    row = sub.iloc[0]
    out: dict[str, bool] = {}
    for it in b["items"]:
        if it["col"] not in sub.columns:
            continue
        v = pd.to_numeric(pd.Series([row[it["col"]]]), errors="coerce").iloc[0]
        if pd.notna(v):
            out[it["key"]] = bool(v >= it["thr"])
    return out
