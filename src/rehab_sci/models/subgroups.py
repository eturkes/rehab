"""Subgroup discovery + effect sizes for the discharge SCIM outcome.

For every categorical admission feature, the function ``run_all_subgroups`` computes:
  - Mann–Whitney U (2-level) or Kruskal–Wallis H (3+-level) test
  - Cliff's δ (rank-based effect size, paired 2-level) — robust to non-normality
  - Cohen's d (mean diff / pooled SD) for 2-level
  - η² (Kruskal-Wallis effect size) for k-level
  - Bonferroni-Holm and Benjamini-Hochberg adjusted p-values
  - Per-group N, median, IQR

Numeric features are stratified into quantile bins (default 4) before running the same tests
so we capture monotone dose-response patterns automatically.

Output is a pandas DataFrame; nothing is persisted to disk by this module (the caller saves a
small ``models/subgroups.json`` with only aggregate stats).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models"


def cliffs_delta(a: Iterable[float], b: Iterable[float]) -> float:
    """Cliff's δ: P(X>Y) - P(X<Y). Bounded in [-1, 1]."""
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    # vectorized via numpy broadcasting (mem cheap for n≈500)
    diff = np.sign(a[:, None] - b[None, :])
    return float(diff.mean())


def cohens_d(a: Iterable[float], b: Iterable[float]) -> float:
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def kruskal_eta_squared(h: float, k: int, n: int) -> float:
    """Effect size for Kruskal–Wallis (Tomczak & Tomczak 2014)."""
    if n - k <= 0:
        return float("nan")
    return float((h - k + 1) / (n - k))


def _adjust_p(p: list[float]) -> tuple[list[float], list[float]]:
    """Return (Holm, BH) adjusted p-values matching the input order."""
    p_arr = np.array(p, dtype=float)
    n = len(p_arr)
    if n == 0:
        return [], []
    order = np.argsort(p_arr)
    sorted_p = p_arr[order]

    # Holm step-down: running max of (n - k + 1) * p_(k) over the sorted p-values.
    holm_sorted = np.maximum.accumulate(sorted_p * (n - np.arange(n)))
    holm_sorted = np.minimum(holm_sorted, 1.0)
    holm = np.empty_like(sorted_p)
    holm[order] = holm_sorted

    # Benjamini–Hochberg
    bh_sorted = sorted_p * n / (np.arange(n) + 1)
    bh_sorted = np.minimum.accumulate(bh_sorted[::-1])[::-1]
    bh_sorted = np.minimum(bh_sorted, 1.0)
    bh = np.empty_like(sorted_p)
    bh[order] = bh_sorted

    return holm.tolist(), bh.tolist()


def _summary(g: pd.Series) -> dict:
    g = g.dropna().astype(float)
    if g.empty:
        return {"n": 0, "median": None, "iqr_low": None, "iqr_high": None, "mean": None, "sd": None}
    return {
        "n": int(g.size),
        "median": float(g.median()),
        "iqr_low": float(g.quantile(0.25)),
        "iqr_high": float(g.quantile(0.75)),
        "mean": float(g.mean()),
        "sd": float(g.std(ddof=1)) if g.size > 1 else 0.0,
    }


def run_one(df: pd.DataFrame, feature: str, outcome: str, kind: str) -> dict:
    """Run one feature-outcome comparison; ``kind`` ∈ {"categorical","numeric_quartile"}."""
    sub = df[[feature, outcome]].dropna()
    if kind == "numeric_quartile":
        try:
            sub = sub.assign(_q=pd.qcut(sub[feature], q=4, duplicates="drop"))
        except ValueError:
            return {"feature": feature, "skipped": True, "reason": "could not quartile-bin"}
        group_col = "_q"
    else:
        group_col = feature

    groups = list(sub.groupby(group_col, dropna=True, observed=True))
    groups = [(k, v[outcome]) for k, v in groups if len(v) >= 5]
    if len(groups) < 2:
        return {"feature": feature, "skipped": True, "reason": "<2 groups with n≥5"}

    names = [str(k) for k, _ in groups]
    values = [v.values for _, v in groups]

    result: dict = {
        "feature": feature,
        "kind": kind,
        "groups": [
            {"label": names[i], **_summary(pd.Series(values[i]))}
            for i in range(len(groups))
        ],
        "n_total": int(sum(len(v) for v in values)),
    }
    if len(groups) == 2:
        stat, p = stats.mannwhitneyu(values[0], values[1], alternative="two-sided")
        result.update(
            test="mann_whitney_u",
            statistic=float(stat),
            p_value=float(p),
            cliffs_delta=cliffs_delta(values[0], values[1]),
            cohens_d=cohens_d(values[0], values[1]),
        )
    else:
        stat, p = stats.kruskal(*values)
        n_total = sum(len(v) for v in values)
        result.update(
            test="kruskal_wallis",
            statistic=float(stat),
            p_value=float(p),
            eta_squared=kruskal_eta_squared(float(stat), len(groups), n_total),
        )
        # pairwise Cliff's δ matrix (compressed to a list of {a,b,d})
        pair = []
        for i, j in itertools.combinations(range(len(groups)), 2):
            pair.append(
                {
                    "a": names[i],
                    "b": names[j],
                    "cliffs_delta": cliffs_delta(values[i], values[j]),
                    "cohens_d": cohens_d(values[i], values[j]),
                }
            )
        result["pairwise"] = pair
    return result


def run_all_subgroups(
    df: pd.DataFrame,
    outcome: str,
    categorical_features: list[str],
    numeric_features: list[str],
) -> dict:
    results: list[dict] = []
    for f in categorical_features:
        if f in df.columns:
            results.append(run_one(df, f, outcome, kind="categorical"))
    for f in numeric_features:
        if f in df.columns:
            results.append(run_one(df, f, outcome, kind="numeric_quartile"))

    keep = [r for r in results if not r.get("skipped")]
    p_vals = [r["p_value"] for r in keep]
    holm, bh = _adjust_p(p_vals)
    for r, h, b in zip(keep, holm, bh, strict=True):
        r["p_holm"] = h
        r["p_bh"] = b
    return {"results": results, "n_tested": len(keep), "outcome": outcome}


def main() -> None:
    from rehab_sci.data.dataset import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        build_analysis_dataset,
    )

    af = build_analysis_dataset()
    out = run_all_subgroups(
        af.df, af.outcome_col, CATEGORICAL_FEATURES, NUMERIC_FEATURES
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "subgroups.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )

    # console summary, ranked by BH-adjusted p then effect size
    rows = []
    for r in out["results"]:
        if r.get("skipped"):
            continue
        eff = (
            r.get("cliffs_delta")
            if r.get("cliffs_delta") is not None
            else r.get("eta_squared")
        )
        rows.append(
            {
                "feature": r["feature"],
                "kind": r["kind"],
                "test": r["test"],
                "p_bh": r["p_bh"],
                "effect": eff,
                "n_groups": len(r["groups"]),
                "n_total": r["n_total"],
            }
        )
    summary = pd.DataFrame(rows).sort_values(["p_bh", "feature"]).reset_index(drop=True)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
