"""Patient similarity via Gower distance on admission features.

Gower distance handles mixed numeric/categorical features and missing values:
- Numeric: Manhattan distance normalized by observed range.
- Categorical: 0 if match, 1 if mismatch.
- Missing: feature excluded from both numerator and denominator for that pair.

The ``find_nearest`` function returns the K closest historical episodes with
demographics, actual outcomes, and a 0–1 similarity score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def gower_distance_one_vs_all(
    query: dict,
    candidates: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    ranges: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Gower distance from a single query to every row in *candidates*.

    Returns ``(distances, weights)`` — both shape ``(len(candidates),)``.
    ``weights[i]`` is the number of features with non-null values in both
    the query and candidate *i* (the Gower denominator).
    """
    n = len(candidates)
    total_weight = np.zeros(n)
    total_dist = np.zeros(n)

    for col in numeric_cols:
        if col not in candidates.columns:
            continue
        qv = query.get(col)
        if qv is None or (isinstance(qv, float) and np.isnan(qv)):
            continue
        rng_info = ranges.get(col, {})
        r = rng_info.get("max", 1.0) - rng_info.get("min", 0.0)
        if r <= 0:
            r = 1.0
        cv = pd.to_numeric(candidates[col], errors="coerce").values
        valid = ~np.isnan(cv)
        d = np.abs(float(qv) - cv) / r
        total_dist += np.where(valid, d, 0.0)
        total_weight += valid.astype(float)

    for col in categorical_cols:
        if col not in candidates.columns:
            continue
        qv = query.get(col)
        if qv is None or (isinstance(qv, float) and np.isnan(qv)):
            continue
        cv = candidates[col]
        valid = cv.notna().values
        d = (cv.astype(str).values != str(qv)).astype(float)
        total_dist += np.where(valid, d, 0.0)
        total_weight += valid.astype(float)

    safe_weight = np.maximum(total_weight, 1.0)
    return total_dist / safe_weight, total_weight


MIN_FEATURE_OVERLAP = 5


def find_nearest(
    ep: pd.DataFrame,
    key_record: int,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    ranges: dict[str, dict],
    k: int = 10,
) -> list[dict]:
    """Return the *k* nearest episodes to *key_record* by Gower distance.

    Candidates with fewer than :data:`MIN_FEATURE_OVERLAP` mutually non-null
    features are excluded (prevents vacuous distance=0 for data-sparse
    episodes).

    Each result dict contains:
    ``key_record``, ``id_number``, ``distance``, ``similarity``, ``age``,
    ``sex``, ``paralysis``, ``ais_admit``, ``y_discharge_scim``,
    ``y_discharge_ais``, ``los_days``.
    """
    row = ep.loc[ep["KeyRecordNumber"] == key_record]
    if row.empty:
        return []
    query = row.iloc[0].to_dict()

    candidates = ep[ep["KeyRecordNumber"] != key_record].copy()
    has_any = candidates[feature_cols].notna().any(axis=1)
    candidates = candidates[has_any].copy()
    if candidates.empty:
        return []

    dists, weights = gower_distance_one_vs_all(
        query, candidates, numeric_cols, categorical_cols, ranges
    )

    # Exclude candidates with too few overlapping features.
    sufficient = weights >= MIN_FEATURE_OVERLAP
    if not sufficient.any():
        return []
    dists = np.where(sufficient, dists, np.inf)

    # Select top-k from valid candidates.
    n_valid = int(sufficient.sum())
    k = min(k, n_valid)
    if k == 0:
        return []
    top_idx = np.argpartition(dists, k)[:k]
    top_idx = top_idx[np.argsort(dists[top_idx])]

    results: list[dict] = []
    for idx in top_idx:
        if not np.isfinite(dists[idx]):
            continue
        r = candidates.iloc[idx]
        pid = r.get("IDNumber")
        ais_raw = r.get("y_discharge_ais")
        results.append({
            "key_record": int(r["KeyRecordNumber"]),
            "id_number": None if pd.isna(pid) else int(pid),
            "distance": float(dists[idx]),
            "similarity": float(1.0 - min(dists[idx], 1.0)),
            "age": None if pd.isna(r.get("年齢")) else float(r.get("年齢")),
            "sex": None if pd.isna(r.get("性別")) else str(r.get("性別")),
            "paralysis": (
                None if pd.isna(r.get("対麻痺_四肢麻痺"))
                else str(r.get("対麻痺_四肢麻痺"))
            ),
            "ais_admit": None if pd.isna(r.get("AIS")) else str(r.get("AIS")),
            "y_discharge_scim": (
                None if pd.isna(r.get("y_discharge_scim"))
                else float(r.get("y_discharge_scim"))
            ),
            "y_discharge_ais": (
                None if pd.isna(ais_raw) else int(ais_raw)
            ),
            "los_days": (
                None if pd.isna(r.get("LOS_days"))
                else float(r.get("LOS_days"))
            ),
        })

    return results
