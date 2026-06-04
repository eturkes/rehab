"""Recovery archetype discovery via k-means clustering on predicted trajectories.

Clusters patients by the *shape* of their model-predicted SCIM-III recovery
trajectory (9 intermediate timepoints + discharge).  Each cluster defines a
recovery archetype — a characteristic curve pattern (e.g. rapid early recovery,
gradual steady improvement, limited recovery plateau).

The predicted trajectories are used rather than raw observations because:
- Complete coverage for all training-eligible episodes (no missing data).
- The trajectory models already encode the relationship between admission
  features and recovery shape, so clusters capture systematic variation rather
  than observation noise.

Archetype assignments are validated against observed outcomes and demographics
to confirm clinical meaningfulness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 20260518


def build_trajectory_matrix(
    ep: pd.DataFrame,
    trajectory_bundle: dict,
    discharge_model,
    feature_cols: list[str],
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Predict 10-point recovery trajectory (9 intermediate + discharge) for all eligible episodes.

    Returns
    -------
    X_model : pd.DataFrame
        Model input rows (one per eligible episode), indexed by position.
    traj_matrix : np.ndarray
        Shape ``(n_eligible, 10)`` — predicted SCIM at each timepoint.
    timepoint_labels : list[str]
        Column labels for ``traj_matrix`` (9 trajectory + ``"discharge"``).
    """
    eligible_mask = ep["IDNumber"].notna()
    ep_eligible = ep[eligible_mask].copy()
    if ep_eligible.empty:
        return pd.DataFrame(), np.empty((0, 10)), []

    X = ep_eligible[feature_cols].copy()
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for c in numeric_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    tps = trajectory_bundle["timepoints"]
    models = trajectory_bundle["models"]
    clip_min = trajectory_bundle.get("clip_min", 0.0)
    clip_max = trajectory_bundle.get("clip_max", 100.0)

    n = len(X)
    n_cols = len(tps) + 1
    traj = np.full((n, n_cols), np.nan)

    for j, tp in enumerate(tps):
        pred = models[tp]["median"].predict(X)
        traj[:, j] = np.clip(pred, clip_min, clip_max)

    pred_dis = discharge_model.predict(X)
    traj[:, -1] = np.clip(pred_dis, clip_min, clip_max)

    timepoint_labels = [*tps, "discharge"]
    return X, traj, timepoint_labels


def find_best_k(
    traj_matrix: np.ndarray,
    k_range: tuple[int, int] = (3, 6),
) -> tuple[int, dict[int, float]]:
    """Evaluate k-means for each k in range; return best k by silhouette score.

    Returns ``(best_k, {k: silhouette_score})``.
    """
    scores: dict[int, float] = {}
    traj_std = StandardScaler().fit_transform(traj_matrix)

    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(traj_std)
        scores[k] = float(silhouette_score(traj_std, labels))

    best_k = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best_k, scores


def cluster_trajectories(
    traj_matrix: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Run k-means on standardized trajectory matrix.

    Returns ``(labels, centroids_original_scale, scaler)``.
    """
    scaler = StandardScaler()
    traj_std = scaler.fit_transform(traj_matrix)

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(traj_std)

    centroids_std = km.cluster_centers_
    centroids = scaler.inverse_transform(centroids_std)

    return labels, centroids, scaler


def order_archetypes_by_discharge(
    labels: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-label archetypes so archetype 0 has the lowest discharge SCIM (last column).

    Returns ``(reordered_labels, reordered_centroids, sort_order)``.
    """
    discharge_col = centroids[:, -1]
    sort_order = np.argsort(discharge_col)
    remap = np.empty_like(sort_order)
    for new_idx, old_idx in enumerate(sort_order):
        remap[old_idx] = new_idx

    return remap[labels], centroids[sort_order], sort_order


def archetype_summary(
    ep_eligible: pd.DataFrame,
    labels: np.ndarray,
) -> list[dict]:
    """Compute per-archetype demographics and outcome summary.

    Returns a list of dicts (one per archetype, ordered by label) with keys:
    ``n``, ``mean_age``, ``pct_tetra``, ``ais_distribution``,
    ``mean_discharge_scim``, ``median_discharge_scim``, ``mean_los``.
    """
    df = ep_eligible.copy()
    df["_archetype"] = labels
    summaries: list[dict] = []

    for arch_id in sorted(df["_archetype"].unique()):
        sub = df[df["_archetype"] == arch_id]
        n = len(sub)

        age = pd.to_numeric(sub["年齢"], errors="coerce")
        para = sub["対麻痺_四肢麻痺"].astype(str)
        ais = sub["AIS"].dropna().astype(str)
        scim = pd.to_numeric(sub.get("y_discharge_scim"), errors="coerce").dropna()
        los = pd.to_numeric(sub.get("LOS_days"), errors="coerce").dropna()

        ais_dist = {}
        if len(ais) > 0:
            vc = ais.value_counts(normalize=True)
            for grade in ["A", "B", "C", "D", "E"]:
                ais_dist[grade] = float(vc.get(grade, 0.0))

        summaries.append({
            "id": int(arch_id),
            "n": n,
            "mean_age": float(age.mean()) if len(age.dropna()) > 0 else None,
            "pct_tetra": float((para == "TETRA").sum() / n * 100) if n > 0 else 0.0,
            "ais_distribution": ais_dist,
            "mean_discharge_scim": float(scim.mean()) if len(scim) > 0 else None,
            "median_discharge_scim": float(scim.median()) if len(scim) > 0 else None,
            "mean_los": float(los.mean()) if len(los) > 0 else None,
        })

    return summaries


def assign_single(
    X_row: pd.DataFrame,
    trajectory_bundle: dict,
    discharge_model,
    scaler: StandardScaler,
    centroids_std: np.ndarray,
) -> int:
    """Assign a single patient (one-row DataFrame) to the nearest archetype."""
    tps = trajectory_bundle["timepoints"]
    models = trajectory_bundle["models"]
    clip_min = trajectory_bundle.get("clip_min", 0.0)
    clip_max = trajectory_bundle.get("clip_max", 100.0)

    vec = np.empty(len(tps) + 1)
    for j, tp in enumerate(tps):
        pred = float(models[tp]["median"].predict(X_row)[0])
        vec[j] = max(clip_min, min(clip_max, pred))

    pred_dis = float(discharge_model.predict(X_row)[0])
    vec[-1] = max(clip_min, min(clip_max, pred_dis))

    vec_std = scaler.transform(vec.reshape(1, -1))
    dists = np.linalg.norm(centroids_std - vec_std, axis=1)
    return int(np.argmin(dists))
