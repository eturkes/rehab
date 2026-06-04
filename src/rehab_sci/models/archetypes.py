"""Compute recovery archetypes and persist artifacts.

Run with::

    uv run python -m rehab_sci.models.archetypes

Requires: trained trajectory models (``models/trajectory/bundle.joblib``)
and the SCIM-total discharge model (``models/scim_total/lgbm_median.joblib``).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from rich.console import Console
from rich.table import Table

from rehab_sci.data.archetypes import (
    archetype_summary,
    build_trajectory_matrix,
    cluster_trajectories,
    find_best_k,
    order_archetypes_by_discharge,
)
from rehab_sci.data.dataset import build_analysis_dataset

console = Console()
ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models"


def main() -> None:
    console.rule("[bold]Recovery archetype discovery[/bold]")

    af = build_analysis_dataset()
    ep = af.df

    traj_bundle = joblib.load(MODELS_DIR / "trajectory" / "bundle.joblib")
    discharge_model = joblib.load(MODELS_DIR / "scim_total" / "lgbm_median.joblib")

    console.print("[cyan]Building predicted trajectory matrix...[/cyan]")
    _X, traj_matrix, tp_labels = build_trajectory_matrix(
        ep,
        traj_bundle,
        discharge_model,
        af.feature_cols,
        af.categorical_cols,
        af.numeric_cols,
    )
    console.print(f"  Eligible episodes: {traj_matrix.shape[0]}")
    console.print(f"  Timepoints: {tp_labels}")

    console.print("\n[cyan]Evaluating k-means (k=3..5)...[/cyan]")
    best_k, sil_scores = find_best_k(traj_matrix, k_range=(3, 5))

    sil_table = Table(title="Silhouette scores")
    sil_table.add_column("k", style="bold")
    sil_table.add_column("Silhouette", justify="right")
    sil_table.add_column("", justify="center")
    for k, s in sorted(sil_scores.items()):
        marker = "← best" if k == best_k else ""
        sil_table.add_row(str(k), f"{s:.4f}", marker)
    console.print(sil_table)

    console.print(f"\n[cyan]Clustering with k={best_k}...[/cyan]")
    labels, centroids, scaler = cluster_trajectories(traj_matrix, best_k)

    labels, centroids, _sort_order = order_archetypes_by_discharge(labels, centroids)

    ep_eligible = ep[ep["IDNumber"].notna()].copy()
    key_records = ep_eligible["KeyRecordNumber"].values
    assignments = {int(kr): int(lbl) for kr, lbl in zip(key_records, labels, strict=True)}

    summaries = archetype_summary(ep_eligible, labels)

    summary_table = Table(title="Archetype summary")
    summary_table.add_column("Archetype", style="bold")
    summary_table.add_column("n", justify="right")
    summary_table.add_column("Age (mean)", justify="right")
    summary_table.add_column("% Tetra", justify="right")
    summary_table.add_column("AIS dist.", justify="left")
    summary_table.add_column("SCIM dis. (med)", justify="right")
    summary_table.add_column("LOS (mean d)", justify="right")

    for s in summaries:
        ais_str = " ".join(f"{g}={p*100:.0f}%" for g, p in s["ais_distribution"].items())
        summary_table.add_row(
            str(s["id"]),
            str(s["n"]),
            f"{s['mean_age']:.1f}" if s["mean_age"] else "–",
            f"{s['pct_tetra']:.0f}",
            ais_str,
            f"{s['median_discharge_scim']:.0f}" if s["median_discharge_scim"] else "–",
            f"{s['mean_los']:.0f}" if s["mean_los"] else "–",
        )
    console.print(summary_table)

    centroid_table = Table(title="Archetype centroids (predicted SCIM)")
    centroid_table.add_column("Archetype", style="bold")
    for tp in tp_labels:
        centroid_table.add_column(tp, justify="right")
    for i, row in enumerate(centroids):
        centroid_table.add_row(str(i), *(f"{v:.1f}" for v in row))
    console.print(centroid_table)

    scaler_centroids_std = scaler.transform(centroids)

    out_dir = MODELS_DIR / "archetypes"
    out_dir.mkdir(exist_ok=True)
    artifact = {
        "k": best_k,
        "assignments": assignments,
        "centroids": centroids,
        "centroids_std": scaler_centroids_std,
        "scaler": scaler,
        "timepoint_labels": tp_labels,
        "silhouette_scores": sil_scores,
        "summaries": summaries,
    }
    joblib.dump(artifact, out_dir / "archetypes.joblib")
    console.print(f"\n[green]Saved → {out_dir / 'archetypes.joblib'}[/green]")

    metrics_path = MODELS_DIR / "training_metrics.json"
    with metrics_path.open(encoding="utf-8") as f:
        metrics = json.load(f)
    metrics["archetypes"] = {
        "k": best_k,
        "silhouette_scores": {str(k): v for k, v in sil_scores.items()},
        "summaries": summaries,
        "centroids": centroids.tolist(),
        "timepoint_labels": tp_labels,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    console.print(f"[green]Updated → {metrics_path}[/green]")

    console.rule("[bold green]Done[/bold green]")


if __name__ == "__main__":
    main()
