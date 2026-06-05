"""Observed-trajectory phenotyping (G3) — fit + persist a growth mixture model.

Run with::

    uv run python -m rehab_sci.models.phenotypes

Discovers recovery *phenotypes* from the **observed** early-recovery trajectories (SCIM-total
+ total motor score) via a multivariate growth mixture model (see ``data/phenotypes.py``) —
the data-driven counterpart to the model-*predicted* recovery archetypes (``models/archetypes.py``).

Diagnostic + inference layer, NOT a production training step: it writes its own tracked
``models/phenotype_metrics.json`` (aggregates only — no row-level identifiers) and a git-ignored
``models/phenotypes/phenotypes.joblib`` (full params + per-episode assignments for the dashboard /
part-2 patient surface), and never touches ``train.py``'s artifacts, so production byte-repro is
preserved.

Cohort: episodes with >= ``MIN_SCIM_OBS`` observed SCIM-total points in the 0day-6m window and a
non-null IDNumber (trainable / joinable to demographics).  Motor-score cells are used wherever
present, so the model is genuinely bivariate while remaining anchored on the primary functional
measure.  K (number of phenotypes) and the polynomial degree are chosen by BIC.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from rich.console import Console
from rich.table import Table

from rehab_sci.data.dataset import build_analysis_dataset
from rehab_sci.data.phenotypes import (
    MEASURES,
    WINDOW,
    WINDOW_DAYS,
    build_gmm_data,
    class_means,
    class_support,
    diagnostics,
    order_by_discharge,
    phenotype_summary,
    select,
)

console = Console()
ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models"

MIN_SCIM_OBS = 3              # min observed SCIM points in the window to enter the cohort
K_RANGE = (2, 3, 4, 5)       # candidate phenotype counts
DEGREES = (1, 2)             # candidate fixed-effect polynomial degrees (linear / quadratic)
N_RESTARTS = 10
SEED = 20260518


def _cohort_keys(ep, long) -> list[int]:
    """Episodes with >= MIN_SCIM_OBS observed SCIM points in the window and a real IDNumber."""
    win = list(WINDOW)
    counts = (
        long[long["TIME_Name"].isin(win)]
        .dropna(subset=["SCIM_total"])
        .groupby("KeyRecordNumber")
        .size()
    )
    eligible = set(counts[counts >= MIN_SCIM_OBS].index)
    trainable = set(ep[ep["IDNumber"].notna()]["KeyRecordNumber"])
    return sorted(int(k) for k in eligible & trainable)


def main() -> None:
    console.rule("[bold]Observed-trajectory phenotyping (growth mixture model)[/bold]")

    af = build_analysis_dataset()
    ep, long = af.df, af.longitudinal

    cohort = _cohort_keys(ep, long)
    console.print(f"Cohort (>= {MIN_SCIM_OBS} SCIM obs, real IDNumber): [bold]{len(cohort)}[/bold] episodes")

    data_by_degree = {d: build_gmm_data(long, cohort, d) for d in DEGREES}
    console.print(f"Measures: {MEASURES}  |  window: {WINDOW}")

    console.print("\n[cyan]Selecting K x degree by BIC...[/cyan]")

    def _progress(row: dict) -> None:
        console.print(
            f"  deg={row['degree']} K={row['K']}: BIC={row['bic']:.0f} "
            f"loglik={row['loglik']:.0f} ent={row['relative_entropy']:.3f} "
            f"min_share={row['min_class_share']:.3f}"
        )

    best_key, fits, table = select(
        data_by_degree, K_RANGE, DEGREES, n_restarts=N_RESTARTS, seed=SEED, progress=_progress
    )
    best_k, best_degree = best_key

    sel_table = Table(title="Model selection (BIC)")
    for col in ("K", "degree", "loglik", "n_params", "BIC", "rel.entropy", "min share"):
        sel_table.add_column(col, justify="right")
    for row in sorted(table, key=lambda r: (r["degree"], r["K"])):
        marker = "  ← best" if (row["K"], row["degree"]) == best_key else ""
        sel_table.add_row(
            str(row["K"]), str(row["degree"]), f"{row['loglik']:.0f}",
            str(row["n_params"]), f"{row['bic']:.0f}",
            f"{row['relative_entropy']:.3f}", f"{row['min_class_share']:.3f}{marker}",
        )
    console.print(sel_table)
    console.print(f"\n[green]Best: K={best_k}, degree={best_degree}[/green]")

    params, resp, _loglik = fits[best_key]
    data = data_by_degree[best_degree]

    # Per-class observed-support edges (computed on the pre-order labels) drive BOTH the
    # recovery ranking (so it ignores out-of-support extrapolation) and curve truncation.
    pre_labels = resp.argmax(1)
    pre_assign = {int(k): int(lbl) for k, lbl in zip(data.keys, pre_labels, strict=True)}
    support = class_support(long, pre_assign, best_k)   # (K, M) last observed-support window idx

    params, resp, order = order_by_discharge(params, resp, support)
    support = support[order]                             # realign support to the reordered labels

    labels = resp.argmax(1)
    assignments = {int(k): int(lbl) for k, lbl in zip(data.keys, labels, strict=True)}
    ep_cohort = ep[ep["KeyRecordNumber"].isin(data.keys)].copy()
    summaries = phenotype_summary(ep_cohort, assignments, best_k)
    diag = diagnostics(resp)
    curves = class_means(params)  # (K, M, T)

    # ---- console summaries ----
    summary_table = Table(title="Phenotype summary")
    for col in ("Phenotype", "n", "Age", "%Tetra", "AIS dist.", "Disch.SCIM(med)", "LOS(mean d)"):
        summary_table.add_column(col, justify="left")
    for s in summaries:
        ais_str = " ".join(f"{g}={p*100:.0f}" for g, p in s["ais_distribution"].items() if p > 0)
        summary_table.add_row(
            str(s["id"]), str(s["n"]),
            f"{s['mean_age']:.1f}" if s["mean_age"] else "–",
            f"{s['pct_tetra']:.0f}", ais_str,
            f"{s['median_discharge_scim']:.0f}" if s["median_discharge_scim"] is not None else "–",
            f"{s['mean_los']:.0f}" if s["mean_los"] is not None else "–",
        )
    console.print(summary_table)

    curve_table = Table(title="Phenotype class means (0day → end of observed support)")
    curve_table.add_column("Phenotype", justify="right")
    for m in MEASURES:
        curve_table.add_column(f"{m} (start→support)", justify="left")
    for k in range(best_k):
        cells = []
        for m in range(len(MEASURES)):
            last = int(support[k, m])
            cells.append(f"{curves[k, m, 0]:.0f} → {curves[k, m, last]:.0f} @{WINDOW[last]}")
        curve_table.add_row(str(k), *cells)
    console.print(curve_table)

    console.print(
        f"\nRelative entropy (separation): [bold]{diag['relative_entropy']:.3f}[/bold]  "
        f"| APPA: {[round(a, 3) for a in diag['appa']]}  "
        f"| class shares: {[round(s, 3) for s in diag['class_shares']]}"
    )

    # ---- persist git-ignored bundle (full params + per-episode assignments) ----
    out_dir = MODELS_DIR / "phenotypes"
    out_dir.mkdir(exist_ok=True)
    bundle = {
        "k": best_k,
        "degree": best_degree,
        "measures": MEASURES,
        "window": WINDOW,
        "window_days": WINDOW_DAYS,
        "params": params,                 # GMMParams (pickled dataclass)
        "assignments": assignments,       # KeyRecordNumber -> phenotype id (hard)
        "posterior": resp,                # (N, K) soft membership
        "keys": list(data.keys),
        "class_means": curves,            # (K, M, T) raw fitted means
        "class_support": support,         # (K, M) last observed-support window index per measure
        "summaries": summaries,
        "diagnostics": diag,
    }
    joblib.dump(bundle, out_dir / "phenotypes.joblib")
    console.print(f"\n[green]Saved → {out_dir / 'phenotypes.joblib'}[/green]")

    # ---- persist tracked metrics (aggregates only; NO identifiers) ----
    metrics = {
        "k": best_k,
        "degree": best_degree,
        "cohort_n": len(data.keys),
        "min_scim_obs": MIN_SCIM_OBS,
        "measures": list(MEASURES),
        "window": list(WINDOW),
        "window_days": [WINDOW_DAYS[tp] for tp in WINDOW],
        "n_restarts": N_RESTARTS,
        "seed": SEED,
        "selection_table": sorted(table, key=lambda r: (r["degree"], r["K"])),
        "class_means": curves.tolist(),   # (K, M, T) raw fitted means
        "class_support": support.tolist(),  # (K, M) last observed-support window index per measure
        "min_coverage": 0.20,             # support threshold: fraction of class observed at a window
        "summaries": summaries,
        "diagnostics": diag,
    }
    metrics_path = MODELS_DIR / "phenotype_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    console.print(f"[green]Updated → {metrics_path}[/green]")

    console.rule("[bold green]Done[/bold green]")


if __name__ == "__main__":
    main()
