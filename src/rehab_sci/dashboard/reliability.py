"""Input reliability + out-of-distribution assessment for the simulator.

Pure functions (no Dash). Given a single-row model input, report:

* **completeness** — how much of the prediction's evidence the user actually
  supplied, weighted by each feature's LightGBM gain importance, so omitting a
  high-impact field (e.g. TotalMotor) lowers reliability more than a minor one;
* **OOD signals** — whether supplied values are typical of the training cohort:
  hard ``range_violations`` (outside the observed min/max → the model is
  extrapolating) and soft ``atypical`` values (outside q05–q95) plus a joint
  RMS-z score.

The conformal PI half-width is a fixed calibrated scalar, so it does NOT widen
as input grows sparser. This completeness/OOD signal is the separate cue that
tells the user how far to trust a partial-input prediction; the PI itself stays
calibrated under the cohort's (high) natural missingness.

Reference statistics come straight from ``feature_spec`` (``ranges`` /
``categories``), so no extra artifact or retrain is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# z for the 95th percentile: q05..q95 spans ~2*_Z90 robust sigmas under normality.
_Z90 = 1.6448536269514722
_ATYP_Z = _Z90  # |z| beyond this (i.e. outside q05..q95) counts as atypical
_RMS_Z_MEDIUM = 2.0  # joint-atypicality threshold for a "medium" OOD verdict

# gain importances are immutable per fitted model; cache by outcome key.
_IMPORTANCE_CACHE: dict[str, dict[str, float]] = {}


def _gain_importance(bundle: dict) -> dict[str, float]:
    """Per-feature LightGBM gain importance keyed by feature name (cached)."""
    key = bundle["key"]
    cached = _IMPORTANCE_CACHE.get(key)
    if cached is not None:
        return cached
    model = bundle.get("median") or bundle.get("clf")
    booster = model.booster_
    gains = booster.feature_importance(importance_type="gain")
    names = booster.feature_name()
    imp = {n: float(g) for n, g in zip(names, gains, strict=False)}
    _IMPORTANCE_CACHE[key] = imp
    return imp


def _supplied(value: object) -> bool:
    """True when a cell holds a real user-supplied value (not blank / NaN)."""
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is not None


def assess_input(X: pd.DataFrame, bundle: dict, feature_spec: dict) -> dict:
    """Assess a single-row model input for completeness and OOD.

    ``feature_spec`` is the global feature spec (``ranges`` / ``categories`` /
    ``numeric_cols`` / ``categorical_cols`` / ``feature_cols``); ``bundle`` is
    the selected outcome bundle (its fitted model supplies importances).
    """
    feature_cols = feature_spec["feature_cols"]
    numeric_cols = set(feature_spec["numeric_cols"])
    ranges = feature_spec["ranges"]
    categories = feature_spec["categories"]
    imp = _gain_importance(bundle)
    row = X.iloc[0]

    total_w = sum(imp.get(c, 0.0) for c in feature_cols)
    supplied_w = 0.0
    n_supplied = 0
    range_violations: list[dict] = []
    atypical: list[dict] = []
    z2_vals: list[float] = []

    for c in feature_cols:
        val = row.get(c)
        if not _supplied(val):
            continue
        n_supplied += 1
        supplied_w += imp.get(c, 0.0)
        if c in numeric_cols:
            rng = ranges.get(c)
            if rng is None:
                continue
            fval = float(val)
            if fval < rng["min"] or fval > rng["max"]:
                range_violations.append(
                    {"feature": c, "value": fval, "min": rng["min"], "max": rng["max"]}
                )
                continue  # extrapolation already flagged; skip the soft check
            span = rng["q95"] - rng["q05"]
            if span > 0:
                z = (fval - rng["median"]) / (span / (2 * _Z90))
                z2_vals.append(z * z)
                if abs(z) > _ATYP_Z:
                    atypical.append({"feature": c, "value": fval, "z": float(z)})
        else:
            allowed = [str(a) for a in categories.get(c, [])]
            if allowed and str(val) not in allowed:
                range_violations.append(
                    {"feature": c, "value": val, "min": None, "max": None}
                )

    n_total = len(feature_cols)
    completeness = (
        supplied_w / total_w
        if total_w > 0
        else (n_supplied / n_total if n_total else 0.0)
    )
    rms_z = float(np.sqrt(np.mean(z2_vals))) if z2_vals else 0.0

    if range_violations:
        ood_level = "high"
    elif rms_z > _RMS_Z_MEDIUM or len(atypical) >= 2:
        ood_level = "medium"
    else:
        ood_level = "low"

    if completeness >= 0.8:
        reliability_level = "high"
    elif completeness >= 0.5:
        reliability_level = "medium"
    else:
        reliability_level = "low"

    return {
        "n_supplied": n_supplied,
        "n_total": n_total,
        "completeness": completeness,
        "completeness_plain": n_supplied / n_total if n_total else 0.0,
        "range_violations": range_violations,
        "atypical": atypical,
        "rms_z": rms_z,
        "ood_level": ood_level,
        "reliability_level": reliability_level,
    }
