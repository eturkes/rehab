"""TreeSHAP interaction-value encoding + top feature-pair ranking helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _encode_cats_for_shap(X: pd.DataFrame) -> pd.DataFrame:
    """Encode category-dtype columns to integer codes for shap_interaction_values."""
    out = X.copy()
    for c in out.columns:
        if out[c].dtype.name == "category":
            out[c] = out[c].cat.codes.astype(float)
            out[c] = out[c].replace(-1, float("nan"))
    return out


def _top_interactions(
    shap_interaction: np.ndarray,
    feature_names: list[str],
    top_n: int = 25,
) -> list[dict]:
    """Rank feature pairs by mean |SHAP interaction| (regression: 3-D input)."""
    # shap_interaction shape: (n_samples, n_features, n_features)
    abs_mean = np.abs(shap_interaction).mean(axis=0)  # (p, p)
    n_feat = len(feature_names)
    pairs = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            pairs.append((feature_names[i], feature_names[j], float(abs_mean[i, j])))
    pairs.sort(key=lambda x: -x[2])
    return [
        {"feat_a": a, "feat_b": b, "abs_mean_interaction": v}
        for a, b, v in pairs[:top_n]
    ]


def _top_interactions_multiclass(
    shap_interaction: np.ndarray,
    feature_names: list[str],
    top_n: int = 25,
) -> list[dict]:
    """Rank feature pairs by mean |SHAP interaction| (multiclass: 4-D input)."""
    # shap_interaction shape: (n_samples, n_features, n_features, K)
    abs_mean = np.abs(shap_interaction).mean(axis=(0, 3))  # (p, p)
    n_feat = len(feature_names)
    pairs = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            pairs.append((feature_names[i], feature_names[j], float(abs_mean[i, j])))
    pairs.sort(key=lambda x: -x[2])
    return [
        {"feat_a": a, "feat_b": b, "abs_mean_interaction": v}
        for a, b, v in pairs[:top_n]
    ]
