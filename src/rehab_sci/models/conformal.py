"""Split-conformal & APS prediction-set helpers (Mondrian per-AIS / per-paralysis).

Pure helpers carved from train.py; the trainers import them.  Imports beyond
numpy/pandas/AIS_ORD_TO_LETTER are pruned by ruff --fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rehab_sci.constants import AIS_ORD_TO_LETTER

AIS_ORD_COL = "AIS_ord"
PARALYSIS_COL = "対麻痺_四肢麻痺"
MONDRIAN_MIN_N = 8


def _conformal_q(residuals: np.ndarray, alpha: float) -> float:
    q_idx = int(np.ceil((len(residuals) + 1) * (1 - alpha))) - 1
    q_idx = max(0, min(q_idx, len(residuals) - 1))
    return float(np.sort(residuals)[q_idx])


def _compute_mondrian_q(
    residuals_t: np.ndarray,
    X_cal: pd.DataFrame,
    alpha: float,
) -> dict:
    """Per-AIS-grade and per-paralysis-class conformal quantiles.

    Groups with fewer than ``MONDRIAN_MIN_N`` calibration samples are omitted;
    inference falls back to the next coarser group or to the marginal.
    """
    result: dict = {}
    if AIS_ORD_COL in X_cal.columns:
        ais_vals = pd.to_numeric(X_cal[AIS_ORD_COL], errors="coerce").to_numpy()
        ais_qs: dict[str, float] = {}
        for code, letter in AIS_ORD_TO_LETTER.items():
            mask = ais_vals == code
            if int(mask.sum()) >= MONDRIAN_MIN_N:
                ais_qs[letter] = _conformal_q(residuals_t[mask], alpha)
        result["ais"] = ais_qs
    if PARALYSIS_COL in X_cal.columns:
        para_vals = X_cal[PARALYSIS_COL].astype(str).to_numpy()
        para_qs: dict[str, float] = {}
        for label in ("TETRA", "PARA", "NONE"):
            mask = para_vals == label
            if int(mask.sum()) >= MONDRIAN_MIN_N:
                para_qs[label] = _conformal_q(residuals_t[mask], alpha)
        result["paralysis"] = para_qs
    return result


def _resolve_mondrian_q_array(
    marginal_q: float,
    q_by_group: dict,
    X: pd.DataFrame,
) -> np.ndarray:
    """Per-row conformal q: AIS group -> paralysis group -> marginal."""
    n = len(X)
    q_arr = np.full(n, marginal_q)
    resolved = np.zeros(n, dtype=bool)
    ais_qs = q_by_group.get("ais", {})
    if ais_qs and AIS_ORD_COL in X.columns:
        ais_vals = pd.to_numeric(X[AIS_ORD_COL], errors="coerce").to_numpy()
        for code, letter in AIS_ORD_TO_LETTER.items():
            if letter in ais_qs:
                mask = ais_vals == code
                q_arr[mask] = ais_qs[letter]
                resolved |= mask
    para_qs = q_by_group.get("paralysis", {})
    if para_qs and PARALYSIS_COL in X.columns:
        para_vals = X[PARALYSIS_COL].astype(str).to_numpy()
        for label, q in para_qs.items():
            mask = (~resolved) & (para_vals == label)
            q_arr[mask] = q
    return q_arr


def _mondrian_test_coverage(
    y_raw: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    X: pd.DataFrame,
) -> dict:
    """Per-group coverage on the test set using Mondrian PI bounds."""
    covered = (y_raw >= lo) & (y_raw <= hi)
    result: dict = {}
    if AIS_ORD_COL in X.columns:
        ais_vals = pd.to_numeric(X[AIS_ORD_COL], errors="coerce").to_numpy()
        ais_cov: dict = {}
        for code, letter in AIS_ORD_TO_LETTER.items():
            mask = ais_vals == code
            n = int(mask.sum())
            if n > 0:
                ais_cov[letter] = {"n": n, "coverage": round(float(covered[mask].mean()), 4)}
        result["ais"] = ais_cov
    if PARALYSIS_COL in X.columns:
        para_vals = X[PARALYSIS_COL].astype(str).to_numpy()
        para_cov: dict = {}
        for label in ("TETRA", "PARA", "NONE"):
            mask = para_vals == label
            n = int(mask.sum())
            if n > 0:
                para_cov[label] = {"n": n, "coverage": round(float(covered[mask].mean()), 4)}
        result["paralysis"] = para_cov
    return result


def _aps_scores(proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """APS nonconformity scores for conformal classification sets.

    For each sample: sort class probabilities descending, accumulate until the
    true class is included.  The score is the total accumulated mass.
    """
    n = len(y_true)
    scores = np.zeros(n)
    for i in range(n):
        order = np.argsort(-proba[i])
        cumsum = 0.0
        for j in order:
            cumsum += proba[i, j]
            if j == y_true[i]:
                scores[i] = cumsum
                break
    return scores


def _aps_prediction_set(proba_row: np.ndarray, q_hat: float) -> list[int]:
    """Class indices in the APS prediction set for one sample."""
    order = np.argsort(-proba_row)
    cumsum = 0.0
    pred_set: list[int] = []
    for j in order:
        pred_set.append(int(j))
        cumsum += proba_row[j]
        if cumsum >= q_hat:
            break
    return sorted(pred_set)


def _aps_test_metrics(
    proba: np.ndarray,
    y_true: np.ndarray,
    q_arr: np.ndarray,
    X: pd.DataFrame,
) -> dict:
    """Coverage and avg set size on test set using per-row Mondrian APS q."""
    n = len(y_true)
    covered = np.zeros(n, dtype=bool)
    sizes = np.zeros(n, dtype=int)
    for i in range(n):
        pset = _aps_prediction_set(proba[i], q_arr[i])
        covered[i] = y_true[i] in pset
        sizes[i] = len(pset)
    result: dict = {
        "coverage": round(float(covered.mean()), 4),
        "avg_set_size": round(float(sizes.mean()), 3),
        "n": n,
    }
    if AIS_ORD_COL in X.columns:
        ais_vals = pd.to_numeric(X[AIS_ORD_COL], errors="coerce").to_numpy()
        ais_cov: dict = {}
        for code, letter in AIS_ORD_TO_LETTER.items():
            mask = ais_vals == code
            ng = int(mask.sum())
            if ng > 0:
                ais_cov[letter] = {
                    "n": ng,
                    "coverage": round(float(covered[mask].mean()), 4),
                    "avg_set_size": round(float(sizes[mask].mean()), 3),
                }
        result["ais"] = ais_cov
    if PARALYSIS_COL in X.columns:
        para_vals = X[PARALYSIS_COL].astype(str).to_numpy()
        para_cov: dict = {}
        for label in ("TETRA", "PARA", "NONE"):
            mask = para_vals == label
            ng = int(mask.sum())
            if ng > 0:
                para_cov[label] = {
                    "n": ng,
                    "coverage": round(float(covered[mask].mean()), 4),
                    "avg_set_size": round(float(sizes[mask].mean()), 3),
                }
        result["paralysis"] = para_cov
    return result
