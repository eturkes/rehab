"""Plotly figures for the Methods tab — calibration and performance visualizations."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rehab_sci.dashboard.figures._common import _hex_to_rgba
from rehab_sci.dashboard.i18n import col_label, level_label, t
from rehab_sci.dashboard.theme import (
    INK,
    PALETTE_AIS,
    PALETTE_CATEGORICAL,
    PALETTE_INDEPENDENCE_DOMAIN,
    PALETTE_TOPOGRAPHY_MODALITY,
)
from rehab_sci.schema import Schema


def fig_pred_vs_observed(
    shap_pack: dict,
    schema: Schema,
    lang: str,
    *,
    clip_min: float = 0.0,
    clip_max: float | None = 100.0,
    axis_label: str | None = None,
) -> go.Figure | None:
    y_true = shap_pack.get("y_test")
    y_pred = shap_pack.get("y_pred")
    if y_true is None or y_pred is None:
        return None
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if axis_label is None:
        axis_label = "SCIM-III"
    obs_lbl = "実測値" if lang == "ja" else "Observed"
    pred_lbl = "予測値" if lang == "ja" else "Predicted"
    lo = clip_min if clip_min is not None else float(min(y_true.min(), y_pred.min()) - 1)
    hi = clip_max if clip_max is not None else float(max(y_true.max(), y_pred.max()) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        line=dict(color=INK["200"], dash="dash", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    residuals = y_pred - y_true
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        marker=dict(
            size=6, opacity=0.7,
            color=residuals,
            colorscale=[[0, "#117a8b"], [0.5, INK["100"]], [1, "#a3354e"]],
            cmid=0,
            colorbar=dict(
                title=dict(text=("誤差" if lang == "ja" else "Error"), side="right"),
                thickness=10, len=0.6,
            ),
        ),
        hovertemplate=(
            f"{obs_lbl}: %{{x:.1f}}<br>{pred_lbl}: %{{y:.1f}}"
            "<extra></extra>"
        ),
        showlegend=False,
    ))
    from sklearn.metrics import r2_score as _r2
    r2 = _r2(y_true, y_pred)
    fig.add_annotation(
        x=0.03, y=0.97, xref="paper", yref="paper",
        text=f"R² = {r2:.3f}  (n={len(y_true)})",
        showarrow=False, font=dict(size=12, color=INK["700"]),
        xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
    )
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=20, t=28, b=44),
        xaxis=dict(title=f"{obs_lbl} ({axis_label})", range=[lo, hi]),
        yaxis=dict(title=f"{pred_lbl} ({axis_label})", range=[lo, hi],
                   scaleanchor="x", scaleratio=1),
    )
    return fig


def fig_residual_hist(
    shap_pack: dict,
    schema: Schema,
    lang: str,
    *,
    axis_label: str | None = None,
) -> go.Figure | None:
    y_true = shap_pack.get("y_test")
    y_pred = shap_pack.get("y_pred")
    if y_true is None or y_pred is None:
        return None
    residuals = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    if axis_label is None:
        axis_label = "SCIM-III"
    res_lbl = "残差" if lang == "ja" else "Residual"
    freq_lbl = "頻度" if lang == "ja" else "Frequency"
    mu = float(residuals.mean())
    sigma = float(residuals.std())
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals,
        marker=dict(color=PALETTE_CATEGORICAL[0], line=dict(width=0)),
        hovertemplate=f"{res_lbl}: %{{x:.1f}}<br>{freq_lbl}: %{{y}}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=INK["300"], line_width=1.5)
    fig.add_annotation(
        x=0.97, y=0.97, xref="paper", yref="paper",
        text=f"μ = {mu:.1f}   σ = {sigma:.1f}",
        showarrow=False, font=dict(size=12, color=INK["700"]),
        xanchor="right", yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
    )
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=20, t=28, b=44),
        xaxis=dict(title=f"{res_lbl} ({axis_label})"),
        yaxis=dict(title=freq_lbl),
    )
    return fig


def fig_confusion_matrix(
    shap_pack: dict,
    schema: Schema,
    lang: str,
) -> go.Figure | None:
    y_true = shap_pack.get("y_test")
    y_pred = shap_pack.get("y_pred")
    labels = shap_pack.get("class_labels")
    if y_true is None or y_pred is None or labels is None:
        return None
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    k = len(labels)
    cm = np.zeros((k, k), dtype=int)
    for t_idx, p_idx in zip(y_true, y_pred, strict=False):
        cm[t_idx, p_idx] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm / row_sums * 100, 0)
    actual_lbl = "実際" if lang == "ja" else "Actual"
    pred_lbl = "予測" if lang == "ja" else "Predicted"
    text = [[f"{cm[i, j]}<br>({cm_pct[i, j]:.0f}%)" for j in range(k)] for i in range(k)]
    fig = go.Figure(go.Heatmap(
        z=cm_pct,
        x=[f"AIS {g}" for g in labels],
        y=[f"AIS {g}" for g in labels],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorscale=[[0, INK["paper"]], [1, PALETTE_CATEGORICAL[0]]],
        showscale=False,
        hovertemplate=(
            f"{actual_lbl}: %{{y}}<br>{pred_lbl}: %{{x}}<br>"
            "%{text}<extra></extra>"
        ),
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=60, r=20, t=28, b=50),
        xaxis=dict(title=pred_lbl, side="bottom"),
        yaxis=dict(title=actual_lbl, autorange="reversed"),
    )
    return fig


def fig_calibration_curve(
    shap_pack: dict,
    schema: Schema,
    lang: str,
    *,
    n_bins: int = 5,
) -> go.Figure | None:
    y_true = shap_pack.get("y_test")
    proba = shap_pack.get("y_pred_proba")
    labels = shap_pack.get("class_labels")
    if y_true is None or proba is None or labels is None:
        return None
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    k = len(labels)
    conf_lbl = "予測確率" if lang == "ja" else "Predicted probability"
    obs_lbl = "実測頻度" if lang == "ja" else "Observed frequency"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color=INK["200"], dash="dash", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    ais_colors = list(PALETTE_AIS.values())
    for c in range(k):
        p_c = proba[:, c]
        y_c = (y_true == c).astype(float)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers, bin_freqs = [], []
        for b in range(n_bins):
            mask = (p_c >= bin_edges[b]) & (p_c < bin_edges[b + 1])
            if b == n_bins - 1:
                mask = (p_c >= bin_edges[b]) & (p_c <= bin_edges[b + 1])
            if mask.sum() >= 2:
                bin_centers.append(float(p_c[mask].mean()))
                bin_freqs.append(float(y_c[mask].mean()))
        if len(bin_centers) >= 2:
            fig.add_trace(go.Scatter(
                x=bin_centers, y=bin_freqs,
                mode="lines+markers",
                name=f"AIS {labels[c]}",
                marker=dict(size=7),
                line=dict(color=ais_colors[c], width=2),
            ))
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=20, t=28, b=44),
        xaxis=dict(title=conf_lbl, range=[0, 1]),
        yaxis=dict(title=obs_lbl, range=[0, 1]),
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)", font=dict(size=11)),
    )
    return fig


# severity → semantic color (reuses this module's residual red / teal, plus an amber).
_DQ_SEV_COLOR = {"error": "#a3354e", "warn": "#d98c1f", "info": "#117a8b"}
_DQ_CAT_ORDER = ("domain", "cross_field", "longitudinal")


def fig_dataquality_overview(summary: dict, lang: str) -> go.Figure | None:
    """Stacked bar of finding counts per category, split by severity."""
    rules = (summary or {}).get("rules")
    if not rules:
        return None
    cat_label = {
        "domain": "領域" if lang == "ja" else "Domain",
        "cross_field": "項目間" if lang == "ja" else "Cross-field",
        "longitudinal": "経時" if lang == "ja" else "Longitudinal",
    }
    sev_label = {
        "error": "エラー" if lang == "ja" else "Error",
        "warn": "警告" if lang == "ja" else "Warning",
        "info": "情報" if lang == "ja" else "Info",
    }
    agg: dict[tuple[str, str], int] = {}
    for r in rules:
        key = (r["category"], r["severity"])
        agg[key] = agg.get(key, 0) + int(r["count"])
    cats = [c for c in _DQ_CAT_ORDER if any(k[0] == c for k in agg)]
    findings_lbl = "件数" if lang == "ja" else "Findings"
    fig = go.Figure()
    for sev in ("error", "warn", "info"):
        ys = [agg.get((c, sev), 0) for c in cats]
        if not any(ys):
            continue
        fig.add_trace(go.Bar(
            x=[cat_label[c] for c in cats], y=ys,
            name=sev_label[sev],
            marker=dict(color=_DQ_SEV_COLOR[sev]),
            hovertemplate=f"%{{x}} · {sev_label[sev]}<br>{findings_lbl}: %{{y}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        height=280,
        margin=dict(l=50, r=20, t=30, b=40),
        xaxis=dict(title=""),
        yaxis=dict(title=findings_lbl),
        legend=dict(orientation="h", x=0, y=1.12, xanchor="left", yanchor="bottom",
                    font=dict(size=11)),
    )
    return fig


def fig_temporal_drift(t_outcome: dict, lang: str) -> go.Figure | None:
    """Out-of-time drift across rolling-origin test years (F24).

    Left axis = point accuracy (R² for regression, accuracy for AIS) with the
    in-time random-split baseline as a dashed reference; right axis = 80% PI / APS
    coverage with the nominal-0.8 reference line.  ``t_outcome`` is one entry of
    ``temporal_metrics.json['outcomes']``.
    """
    origins = (t_outcome or {}).get("origins") or []
    if not origins:
        return None
    task = t_outcome.get("task")
    baseline = t_outcome.get("baseline", {})
    years = [o["test_year"] for o in origins]
    n_test = [o["n_test"] for o in origins]
    if task == "regression":
        point = [o["r2"] for o in origins]
        cov = [o["conformal_coverage_80"] for o in origins]
        base_point = baseline.get("r2")
        point_lbl = "R²"
        cov_lbl = ("80%予測区間カバレッジ" if lang == "ja" else "80% PI coverage")
    else:
        point = [o["accuracy"] for o in origins]
        cov = [o["aps_coverage_80"] for o in origins]
        base_point = baseline.get("accuracy")
        point_lbl = ("正解率" if lang == "ja" else "Accuracy")
        cov_lbl = ("APSカバレッジ" if lang == "ja" else "APS coverage")

    oot_lbl = "期間外" if lang == "ja" else "Out-of-time"
    base_lbl = "基準(ランダム分割)" if lang == "ja" else "Baseline (random split)"
    nominal_lbl = "名目80%" if lang == "ja" else "Nominal 80%"
    year_lbl = "テスト年" if lang == "ja" else "Test year"
    n_lbl = "症例数" if lang == "ja" else "n"
    c_point = PALETTE_CATEGORICAL[0]
    c_cov = PALETTE_CATEGORICAL[1]
    xr = [min(years) - 0.3, max(years) + 0.3]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=point, mode="lines+markers", name=f"{point_lbl} ({oot_lbl})",
        marker=dict(size=8, color=c_point), line=dict(color=c_point, width=2),
        customdata=n_test,
        hovertemplate=f"%{{x}}<br>{point_lbl}=%{{y:.3f}}<br>{n_lbl}=%{{customdata}}<extra></extra>",
        yaxis="y",
    ))
    if base_point is not None:
        fig.add_trace(go.Scatter(
            x=xr, y=[base_point, base_point], mode="lines",
            name=f"{point_lbl} · {base_lbl}",
            line=dict(color=c_point, width=1.5, dash="dash"),
            hoverinfo="skip", yaxis="y",
        ))
    fig.add_trace(go.Scatter(
        x=years, y=cov, mode="lines+markers", name=f"{cov_lbl} ({oot_lbl})",
        marker=dict(size=8, color=c_cov, symbol="square"), line=dict(color=c_cov, width=2),
        hovertemplate=f"%{{x}}<br>{cov_lbl}=%{{y:.0%}}<extra></extra>",
        yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=xr, y=[0.8, 0.8], mode="lines", name=nominal_lbl,
        line=dict(color=INK["300"], width=1.5, dash="dot"),
        hoverinfo="skip", yaxis="y2",
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=54, t=30, b=44),
        xaxis=dict(title=year_lbl, dtick=1, range=xr),
        yaxis=dict(title=point_lbl, side="left"),
        yaxis2=dict(title=cov_lbl, overlaying="y", side="right", range=[0, 1.05],
                    tickformat=".0%", showgrid=False),
        legend=dict(orientation="h", x=0, y=1.14, xanchor="left", yanchor="bottom",
                    font=dict(size=10)),
    )
    return fig


def fig_landmark_value(lm_outcome: dict, landmark_days: dict, lang: str) -> go.Figure | None:
    """Value of observation: discharge-outcome accuracy + PI sharpening vs landmark time (G1).

    Left axis = point accuracy (R² for regression, quadratic κ for AIS), landmark model (solid)
    vs the admission-only baseline on the *same* still-admitted risk set (dashed).  Right axis =
    uncertainty (mean PI half-width for regression, APS set size for AIS), same pairing.  As the
    landmark advances and more early recovery is observed, the landmark line pulls away from the
    baseline while uncertainty falls.  ``lm_outcome`` is one entry of
    ``landmark_metrics.json['outcomes']``.
    """
    by_lm = (lm_outcome or {}).get("by_landmark") or {}
    if not by_lm:
        return None
    task = lm_outcome.get("task")
    lms = list(by_lm)  # insertion order = chronological (written by the trainer)
    x = [landmark_days.get(L, i) for i, L in enumerate(lms)]
    n_test = [by_lm[L]["n_test"] for L in lms]

    if task == "regression":
        base_pt = [by_lm[L]["baseline"]["r2"] for L in lms]
        lm_pt = [by_lm[L]["landmark"]["r2"] for L in lms]
        base_u = [by_lm[L]["baseline"]["pi_halfwidth_raw"] for L in lms]
        lm_u = [by_lm[L]["landmark"]["pi_halfwidth_raw"] for L in lms]
        pt_lbl = "R²"
        u_lbl = "PI半値幅" if lang == "ja" else "PI half-width"
        u_fmt = ".1f"
    else:
        base_pt = [by_lm[L]["baseline"]["kappa_quadratic"] for L in lms]
        lm_pt = [by_lm[L]["landmark"]["kappa_quadratic"] for L in lms]
        base_u = [by_lm[L]["baseline"]["aps_avg_set_size"] for L in lms]
        lm_u = [by_lm[L]["landmark"]["aps_avg_set_size"] for L in lms]
        pt_lbl = "κ (二次)" if lang == "ja" else "κ (quadratic)"
        u_lbl = "APS集合サイズ" if lang == "ja" else "APS set size"
        u_fmt = ".2f"

    lm_word = "ランドマーク" if lang == "ja" else "Landmark"
    base_word = "基準(入院時のみ)" if lang == "ja" else "Baseline (admission-only)"
    x_lbl = "観測ランドマーク時点" if lang == "ja" else "Landmark (time observed)"
    n_lbl = "症例数" if lang == "ja" else "n"
    c_pt = PALETTE_CATEGORICAL[0]
    c_u = PALETTE_CATEGORICAL[1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=lm_pt, mode="lines+markers", name=f"{pt_lbl} · {lm_word}",
        marker=dict(size=8, color=c_pt), line=dict(color=c_pt, width=2), customdata=n_test,
        hovertemplate=f"{pt_lbl}=%{{y:.3f}}<br>{n_lbl}=%{{customdata}}<extra></extra>", yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=base_pt, mode="lines+markers", name=f"{pt_lbl} · {base_word}",
        marker=dict(size=6, color=c_pt), line=dict(color=c_pt, width=1.5, dash="dash"),
        hovertemplate=f"{pt_lbl}=%{{y:.3f}}<extra></extra>", yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=lm_u, mode="lines+markers", name=f"{u_lbl} · {lm_word}",
        marker=dict(size=8, color=c_u, symbol="square"), line=dict(color=c_u, width=2),
        hovertemplate=f"{u_lbl}=%{{y:{u_fmt}}}<extra></extra>", yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=base_u, mode="lines+markers", name=f"{u_lbl} · {base_word}",
        marker=dict(size=6, color=c_u, symbol="square"), line=dict(color=c_u, width=1.5, dash="dash"),
        hovertemplate=f"{u_lbl}=%{{y:{u_fmt}}}<extra></extra>", yaxis="y2",
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=54, t=30, b=44),
        xaxis=dict(title=x_lbl, tickmode="array", tickvals=x, ticktext=lms),
        yaxis=dict(title=pt_lbl, side="left"),
        yaxis2=dict(title=u_lbl, overlaying="y", side="right", showgrid=False, rangemode="tozero"),
        legend=dict(orientation="h", x=0, y=1.14, xanchor="left", yanchor="bottom",
                    font=dict(size=10)),
    )
    return fig


def fig_voi_scorecard(lm_outcome: dict, lang: str, measure_labels: dict) -> go.Figure | None:
    """Value-of-information scorecard: per-measure × per-landmark uncertainty reduction (G2).

    Each cell is the improvement a *single* observed measure buys over the admission-only baseline
    on the same still-admitted cohort: PI half-width reduction (regression) or APS-set shrink
    (AIS), both from each single-add head's own marginal conformal calibration.  Rows are ordered
    by mean improvement (most valuable measure on top); teal = tightening, red = (noise) widening.
    ``lm_outcome`` is one entry of ``landmark_metrics.json['outcomes']``; ``measure_labels`` maps
    each raw measure name to its localized label.
    """
    by_lm = (lm_outcome or {}).get("by_landmark") or {}
    if not by_lm:
        return None
    task = lm_outcome.get("task")
    lms = list(by_lm)  # chronological (trainer insertion order)
    measures: list[str] | None = next((list(by_lm[L]["single"]) for L in lms if by_lm[L].get("single")), None)
    if not measures:
        return None

    if task == "regression":
        u_key, cbar = "pi_halfwidth_raw", ("PI半値幅の縮小" if lang == "ja" else "PI half-width\nreduction")
        tfmt = ".1f"
    else:
        u_key, cbar = "aps_avg_set_size", ("APS集合の縮小" if lang == "ja" else "APS set\nshrink")
        tfmt = ".2f"

    def _delta(cell: dict, m: str) -> float | None:
        s = (cell.get("single") or {}).get(m)
        return None if s is None else cell["baseline"][u_key] - s[u_key]

    z = [[_delta(by_lm[L], m) for L in lms] for m in measures]
    means = [
        float(np.nanmean([v for v in row if v is not None])) if any(v is not None for v in row) else -1e9
        for row in z
    ]
    order = list(np.argsort(means)[::-1])
    measures = [measures[i] for i in order]
    z = [z[i] for i in order]
    ylabels = [measure_labels.get(m, m) for m in measures]

    x_lbl = "観測ランドマーク時点" if lang == "ja" else "Landmark (time observed)"
    fig = go.Figure(go.Heatmap(
        z=z, x=lms, y=ylabels, zmid=0,
        colorscale=[[0.0, "#c0504d"], [0.5, "#f3f4f6"], [1.0, "#117a8b"]],
        texttemplate="%{z:" + tfmt + "}", textfont=dict(size=10),
        hovertemplate="%{y} · %{x}<br>Δ=%{z:" + tfmt + "}<extra></extra>",
        colorbar=dict(title=dict(text=cbar, font=dict(size=10)), thickness=12),
    ))
    fig.update_layout(
        height=40 * len(measures) + 90,
        margin=dict(l=8, r=8, t=10, b=40),
        xaxis=dict(title=x_lbl, side="bottom"),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
    )
    return fig


# ----------------------------- AIS-grade conversion (G4) ----------------------------
_AIS_GRADE_ORDER = ("A", "B", "C", "D", "E")


def fig_conversion_landscape(conv: dict, lang: str) -> go.Figure | None:
    """Descriptive conversion landscape: ≥1-grade AIS improvement rate by admission grade.

    Visualizes the admission-gating that motivates the conversion heads — improvement is common
    from B/C, rare from D, impossible from E (ceiling).  ``conv`` is ``conversion_metrics.json``.
    """
    by_grade = (conv or {}).get("landscape", {}).get("improve_rate_by_admission_grade")
    if not by_grade:
        return None
    grades = [g for g in _AIS_GRADE_ORDER if g in by_grade]
    rates = [by_grade[g]["improve_rate"] for g in grades]
    ns = [by_grade[g]["n"] for g in grades]
    rate_lbl = "改善率 (≥1段階)" if lang == "ja" else "Improvement rate (≥1 grade)"
    adm_lbl = "入院時 AIS" if lang == "ja" else "Admission AIS"
    n_word = "症例数" if lang == "ja" else "n"
    fig = go.Figure(go.Bar(
        x=[f"AIS {g}" for g in grades], y=rates,
        marker=dict(color=[PALETTE_AIS[g] for g in grades]),
        text=[f"{r:.0%}<br>(n={n})" for r, n in zip(rates, ns, strict=True)],
        textposition="outside", textfont=dict(size=11),
        hovertemplate="%{x}<br>" + rate_lbl + ": %{y:.0%}<br>" + n_word + "=%{customdata}<extra></extra>",
        customdata=ns, showlegend=False,
    ))
    fig.update_layout(
        height=280, margin=dict(l=54, r=20, t=20, b=40),
        xaxis=dict(title=adm_lbl),
        yaxis=dict(range=[0, 1.08], tickformat=".0%", title=rate_lbl),
    )
    return fig


def fig_conversion_delta(conv: dict, lang: str) -> go.Figure | None:
    """Distribution of the AIS grade change (discharge − admission) over the dual-AIS cohort.

    Deterioration (Δ<0) crimson, stable (Δ=0) slate, improvement (Δ>0) teal.  ``conv`` is
    ``conversion_metrics.json``.
    """
    dist = (conv or {}).get("landscape", {}).get("delta_distribution")
    if not dist:
        return None
    deltas = sorted(int(k) for k in dist)
    counts = [dist[str(d)] for d in deltas]
    colors = ["#a3354e" if d < 0 else (INK["300"] if d == 0 else PALETTE_CATEGORICAL[0]) for d in deltas]
    delta_lbl = "AIS グレード変化 (退院 − 入院)" if lang == "ja" else "AIS grade change (discharge − admission)"
    n_word = "症例数" if lang == "ja" else "Episodes"
    fig = go.Figure(go.Bar(
        x=[f"{d:+d}" if d != 0 else "0" for d in deltas], y=counts,
        marker=dict(color=colors),
        text=counts, textposition="outside", textfont=dict(size=11),
        hovertemplate="Δ=%{x}<br>" + n_word + ": %{y}<extra></extra>", showlegend=False,
    ))
    fig.update_layout(
        height=260, margin=dict(l=54, r=20, t=20, b=40),
        xaxis=dict(title=delta_lbl),
        yaxis=dict(title=n_word),
    )
    return fig


def fig_conversion_reliability(em: dict, lang: str, label: str) -> go.Figure | None:
    """Reliability curve for one binary conversion endpoint: Platt-calibrated vs raw LightGBM,
    against the diagonal.  Markers sized by bin count.  ``em`` is one entry of
    ``conversion_metrics.json['endpoints']``."""
    cal, raw = (em or {}).get("calibration"), (em or {}).get("calibration_raw")
    if not cal or not raw:
        return None
    conf_lbl = "予測確率" if lang == "ja" else "Predicted probability"
    obs_lbl = "実測頻度" if lang == "ja" else "Observed frequency"
    cal_lbl = "較正後 (Platt)" if lang == "ja" else "Calibrated (Platt)"
    raw_lbl = "生 (LightGBM)" if lang == "ja" else "Raw (LightGBM)"

    def _sizes(counts: list) -> list:
        c = np.asarray(counts, dtype=float)
        return list(8 + 14 * np.sqrt(c / max(c.max(), 1.0)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color=INK["200"], dash="dash", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=raw["pred_mean"], y=raw["obs_freq"], mode="lines+markers", name=raw_lbl,
        marker=dict(size=_sizes(raw["count"]), color=INK["300"], symbol="circle-open"),
        line=dict(color=INK["300"], width=1.5, dash="dot"),
        hovertemplate=conf_lbl + "=%{x:.2f}<br>" + obs_lbl + "=%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=cal["pred_mean"], y=cal["obs_freq"], mode="lines+markers", name=cal_lbl,
        marker=dict(size=_sizes(cal["count"]), color=PALETTE_CATEGORICAL[0]),
        line=dict(color=PALETTE_CATEGORICAL[0], width=2),
        hovertemplate=conf_lbl + "=%{x:.2f}<br>" + obs_lbl + "=%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=300, margin=dict(l=50, r=20, t=34, b=44),
        title=dict(text=label, font=dict(size=12.5, color=INK["700"]), x=0.5, xanchor="center"),
        xaxis=dict(title=conf_lbl, range=[0, 1]),
        yaxis=dict(title=obs_lbl, range=[0, 1]),
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)", font=dict(size=10)),
    )
    return fig


def fig_conversion_shap(em: dict, schema: Schema, lang: str, top_n: int = 10) -> go.Figure | None:
    """Descriptive in-sample SHAP drivers for one conversion endpoint (mean |SHAP| on the full
    cohort).  ``em`` is one entry of ``conversion_metrics.json['endpoints']``."""
    items = (em or {}).get("shap_top")
    if not items:
        return None
    items = items[:top_n][::-1]
    names = [col_label(schema, r["feature"], lang) for r in items]
    vals = [r["mean_abs"] for r in items]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=PALETTE_CATEGORICAL[3]),
        hovertemplate="%{y}<br>|SHAP|: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=max(240, 22 * len(items) + 70),
        margin=dict(l=210, r=20, t=10, b=40),
        xaxis_title="mean(|SHAP|)",
    )
    return fig


def fig_conversion_confusion(mag: dict, lang: str) -> go.Figure | None:
    """Row-normalized confusion matrix for the ordinal magnitude head over {0, +1, ≥+2}.

    ``mag`` is ``conversion_metrics.json['magnitude']`` (out-of-fold predictions).
    """
    cm = (mag or {}).get("confusion")
    if not cm:
        return None
    cm = np.asarray(cm, dtype=int)
    cap = int(mag.get("mag_cap", 2))
    labels = ["0", *[f"+{i}" for i in range(1, cap)], f"≥+{cap}"][: cm.shape[0]]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm / np.maximum(row_sums, 1) * 100, 0)
    actual_lbl = "実際" if lang == "ja" else "Actual"
    pred_lbl = "予測" if lang == "ja" else "Predicted"
    text = [[f"{cm[i, j]}<br>({cm_pct[i, j]:.0f}%)" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    fig = go.Figure(go.Heatmap(
        z=cm_pct, x=labels, y=labels, text=text,
        texttemplate="%{text}", textfont=dict(size=12),
        colorscale=[[0, INK["paper"]], [1, PALETTE_CATEGORICAL[0]]], showscale=False,
        hovertemplate=f"{actual_lbl}: %{{y}}<br>{pred_lbl}: %{{x}}<br>%{{text}}<extra></extra>",
    ))
    fig.update_layout(
        height=300, margin=dict(l=60, r=20, t=28, b=50),
        xaxis=dict(title=pred_lbl, side="bottom"),
        yaxis=dict(title=actual_lbl, autorange="reversed"),
    )
    return fig


# ----------------------------- neurological-level descent (G10) -------------------------
_LEVEL_DESCENT_ORDER = ["nli", "right_motor", "left_motor", "right_sensory", "left_sensory"]


def _ld_levels(ld: dict, lang: str) -> tuple[list[str], list[str]]:
    """(keys-in-display-order, matching row labels) for the modelled levels present in ``ld``."""
    levels = (ld or {}).get("levels") or {}
    keys = [k for k in _LEVEL_DESCENT_ORDER if k in levels]
    labels = [levels[k]["label_ja" if lang == "ja" else "label_en"] for k in keys]
    return keys, labels


def fig_level_descent_scorecard(ld: dict, lang: str) -> go.Figure | None:
    """Per-level descent discrimination: horizontal OOF AUC bars over the five modelled levels, with
    a chance reference at 0.5.  Surfaces the headline — only NLI descent is well-predicted (~0.73);
    the bilateral motor/sensory heads sit near 0.62.  ``ld`` is ``level_descent_metrics.json``."""
    keys, labels = _ld_levels(ld, lang)
    if not keys:
        return None
    levels = ld["levels"]
    keys, labels = keys[::-1], labels[::-1]  # nli ends at top of the horizontal axis
    aucs = [levels[k]["descent"]["auc"] for k in keys]
    ns = [levels[k]["descent"]["n"] for k in keys]
    colors = [PALETTE_CATEGORICAL[i % len(PALETTE_CATEGORICAL)] for i in range(len(keys))]
    auc_lbl = "下降判別 (AUC)" if lang == "ja" else "Descent discrimination (AUC)"
    n_word = "症例数" if lang == "ja" else "n"
    chance_lbl = "偶然 (0.5)" if lang == "ja" else "chance (0.5)"
    fig = go.Figure(go.Bar(
        x=aucs, y=labels, orientation="h", marker=dict(color=colors),
        text=[f"{a:.3f} (n={n})" for a, n in zip(aucs, ns, strict=True)],
        textposition="outside", textfont=dict(size=11),
        hovertemplate="%{y}<br>AUC=%{x:.3f}<br>" + n_word + "=%{customdata}<extra></extra>",
        customdata=ns, showlegend=False,
    ))
    fig.add_vline(x=0.5, line=dict(color=INK["300"], dash="dash", width=1.5),
                  annotation=dict(text=chance_lbl, font=dict(size=9, color=INK["500"])),
                  annotation_position="top")
    fig.update_layout(
        height=52 * len(keys) + 80, margin=dict(l=170, r=78, t=24, b=40),
        xaxis=dict(range=[0.5, 0.85], tickformat=".2f", title=auc_lbl),
        yaxis=dict(tickfont=dict(size=11.5), showgrid=False),
    )
    return fig


def fig_level_descent_landscape(ld: dict, lang: str) -> go.Figure | None:
    """Descriptive per-level outcome composition: a stacked horizontal bar of deteriorate (Δ<0,
    crimson) / stable (Δ=0, slate) / descent (Δ≥1, teal) rates with the median Δ annotated at the
    right.  The non-trivial deterioration share reflects re-assessment noise in the cord level, not
    true ascent.  ``ld`` is ``level_descent_metrics.json``."""
    keys, labels = _ld_levels(ld, lang)
    if not keys:
        return None
    levels = ld["levels"]
    keys, labels = keys[::-1], labels[::-1]
    det = [levels[k]["landscape"]["deteriorate_rate"] for k in keys]
    stab = [levels[k]["landscape"]["stable_rate"] for k in keys]
    desc = [levels[k]["landscape"]["any_descent_rate"] for k in keys]
    med = [levels[k]["landscape"]["median_delta"] for k in keys]
    crimson, slate, teal = "#a3354e", INK["300"], PALETTE_CATEGORICAL[0]
    det_lbl = "悪化 (Δ<0)" if lang == "ja" else "Deteriorate (Δ<0)"
    stab_lbl = "不変 (Δ=0)" if lang == "ja" else "Stable (Δ=0)"
    desc_lbl = "下降 (Δ≥1)" if lang == "ja" else "Descent (Δ≥1)"
    med_word = "中央値 Δ" if lang == "ja" else "median Δ"
    fig = go.Figure()
    for vals, col, name in [(det, crimson, det_lbl), (stab, slate, stab_lbl), (desc, teal, desc_lbl)]:
        fig.add_trace(go.Bar(
            x=vals, y=labels, orientation="h", name=name, marker=dict(color=col),
            text=[f"{v:.0%}" if v >= 0.07 else "" for v in vals], textposition="inside",
            insidetextfont=dict(color="#fff", size=10),
            hovertemplate="%{y}<br>" + name + ": %{x:.0%}<extra></extra>",
        ))
    for lab, m in zip(labels, med, strict=True):
        fig.add_annotation(x=1.015, y=lab, xref="paper", yref="y", xanchor="left",
                           text=f"{med_word} {m:+.0f}", showarrow=False,
                           font=dict(size=10, color=INK["700"]))
    fig.update_layout(
        barmode="stack", height=52 * len(keys) + 96, margin=dict(l=170, r=124, t=40, b=36),
        xaxis=dict(range=[0, 1.0], tickformat=".0%"),
        yaxis=dict(tickfont=dict(size=11.5), showgrid=False),
        legend=dict(orientation="h", x=0.5, y=1.04, xanchor="center", yanchor="bottom", font=dict(size=10)),
    )
    return fig


def fig_level_descent_delta(landscape: dict, lang: str) -> go.Figure | None:
    """Distribution of one level's change Δ (discharge − admission cord ordinal) for the per-level
    drilldown: deterioration (Δ<0, crimson) tail, the stable spike at 0 (slate), and the descent /
    INT full-recovery mass (Δ>0, teal) at right.  ``landscape`` is one entry of
    ``level_descent_metrics.json['levels'][key]['landscape']``."""
    dist = (landscape or {}).get("delta_distribution")
    if not dist:
        return None
    deltas = sorted(int(k) for k in dist)
    counts = [dist[str(d)] for d in deltas]
    colors = ["#a3354e" if d < 0 else (INK["300"] if d == 0 else PALETTE_CATEGORICAL[0]) for d in deltas]
    delta_lbl = ("レベル変化 Δ (尾側=改善)" if lang == "ja"
                 else "Level change Δ (caudal = improvement)")
    n_word = "症例数" if lang == "ja" else "Episodes"
    fig = go.Figure(go.Bar(
        x=deltas, y=counts, marker=dict(color=colors),
        hovertemplate="Δ=%{x:+d}<br>" + n_word + ": %{y}<extra></extra>", showlegend=False,
    ))
    fig.add_vline(x=0, line=dict(color=INK["200"], width=1))
    fig.update_layout(
        height=260, margin=dict(l=54, r=20, t=20, b=40),
        xaxis=dict(title=delta_lbl, dtick=2, zeroline=False),
        yaxis=dict(title=n_word),
    )
    return fig


# ----------------------------- AIS multi-state recovery (G6) ----------------------------
def _ms_xlabels(ms: dict, schema: Schema, lang: str) -> list[str]:
    return [level_label(schema, "time_name", tp, lang) for tp in ms["window"]]


def fig_multistate_occupancy(ms: dict, schema: Schema, lang: str) -> go.Figure | None:
    """State-occupancy (prevalence) curves P(in AIS state g at time t), as a stacked area per
    admission grade (A-D; E is the trivial ceiling).  Reads ``multistate_metrics.json``."""
    occ = (ms or {}).get("occupancy_by_adm")
    if not occ:
        return None
    states = ms["state_labels"]
    x = _ms_xlabels(ms, schema, lang)
    grades = [g for g in ("A", "B", "C", "D") if g in occ]
    titles = [(f"入院時 AIS {g}" if lang == "ja" else f"Admission AIS {g}") for g in grades]
    fig = make_subplots(rows=2, cols=2, shared_yaxes=True, vertical_spacing=0.14,
                        horizontal_spacing=0.07, subplot_titles=titles)
    for gi, g in enumerate(grades):
        r, c = gi // 2 + 1, gi % 2 + 1
        m = np.asarray(occ[g], dtype=float)
        for si, s in enumerate(states):
            fig.add_trace(go.Scatter(
                x=x, y=m[:, si], mode="lines", name=f"AIS {s}",
                legendgroup=s, showlegend=(gi == 0),
                line=dict(width=0.5, color=PALETTE_AIS[s]),
                stackgroup=f"occ{g}", fillcolor=PALETTE_AIS[s],
                hovertemplate=f"AIS {s} · %{{x}}<br>%{{y:.0%}}<extra></extra>",
            ), row=r, col=c)
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(height=560, margin=dict(l=46, r=16, t=46, b=70),
                      legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1))
    return fig


def fig_multistate_conversion(ms: dict, schema: Schema, lang: str) -> go.Figure | None:
    """First-passage conversion curves P(reached threshold by time t) from each admission grade,
    one panel per threshold (≥1-grade improvement / ≥C / ≥D).  States ≥threshold are made
    absorbing so each curve is monotone; the y=0.5 line marks the median crossing.  Reads
    ``multistate_metrics.json``."""
    conv = (ms or {}).get("conversion_by_adm")
    if not conv:
        return None
    x = _ms_xlabels(ms, schema, lang)
    thr_labels = {
        "improve": ("≥1段階改善" if lang == "ja" else "≥1-grade improvement"),
        "ge_C": ("≥C (運動不全)" if lang == "ja" else "≥C (motor-incomplete)"),
        "ge_D": ("≥D (歩行可能)" if lang == "ja" else "≥D (ambulatory)"),
    }
    thresholds = ["improve", "ge_C", "ge_D"]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=[thr_labels[k] for k in thresholds])
    for ti, k in enumerate(thresholds):
        for g in ("A", "B", "C", "D"):
            curve = conv.get(g, {}).get(k)
            if curve is None:
                continue
            fig.add_trace(go.Scatter(
                x=x, y=curve, mode="lines+markers", name=f"AIS {g}",
                legendgroup=g, showlegend=(ti == 0),
                line=dict(color=PALETTE_AIS[g], width=2.2), marker=dict(size=5),
                hovertemplate=f"AIS {g} · %{{x}}<br>%{{y:.0%}}<extra></extra>",
            ), row=1, col=ti + 1)
        fig.add_hline(y=0.5, line=dict(color=INK["200"], dash="dot", width=1), row=1, col=ti + 1)
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(height=320, margin=dict(l=46, r=16, t=48, b=70),
                      legend=dict(orientation="h", yanchor="bottom", y=1.14, xanchor="right", x=1))
    return fig


def fig_multistate_transition(ms: dict, lang: str) -> go.Figure | None:
    """Pooled one-step AIS transition matrix (time-averaged over the grid).  Rows = from-state,
    cols = to-state; the upper-right of the diagonal is improvement, the lower-left is regression
    (real AIS re-assessment / inter-rater mass, surfaced honestly per the §3 caveat).  Cell text =
    P, hover adds the observed count.  Reads ``multistate_metrics.json['transition']``."""
    tr = (ms or {}).get("transition") or {}
    P = tr.get("P_pooled")
    if not P:
        return None
    P = np.asarray(P, dtype=float)
    N = np.asarray(tr.get("P_pooled_n", np.zeros_like(P)), dtype=float)
    states = ms["state_labels"]
    from_lbl = "遷移元 (時点 k)" if lang == "ja" else "From state (t)"
    to_lbl = "遷移先 (時点 k+1)" if lang == "ja" else "To state (t+1)"
    text = [[f"{P[i, j]:.0%}" if P[i, j] >= 0.005 else "" for j in range(P.shape[1])]
            for i in range(P.shape[0])]
    fig = go.Figure(go.Heatmap(
        z=P, x=[f"AIS {s}" for s in states], y=[f"AIS {s}" for s in states],
        customdata=N, text=text, texttemplate="%{text}", textfont=dict(size=12),
        colorscale=[[0, INK["paper"]], [1, PALETTE_CATEGORICAL[0]]], zmin=0, zmax=1,
        colorbar=dict(title="P", tickformat=".0%", thickness=12, len=0.8),
        hovertemplate=f"{from_lbl}: %{{y}}<br>{to_lbl}: %{{x}}<br>P=%{{z:.1%}}<br>n=%{{customdata:.0f}}<extra></extra>",
    ))
    fig.update_layout(height=380, margin=dict(l=82, r=20, t=18, b=56),
                      xaxis=dict(title=to_lbl, side="bottom"),
                      yaxis=dict(title=from_lbl, autorange="reversed"))
    return fig


def fig_multistate_sojourn(ms: dict, lang: str) -> go.Figure | None:
    """Expected days spent in each AIS state over the 0day-6m window, per admission grade (stacked
    horizontal bars summing to the window length).  Reads ``multistate_metrics.json``."""
    soj = (ms or {}).get("sojourn_by_adm")
    if not soj:
        return None
    states = ms["state_labels"]
    grades = [g for g in ("A", "B", "C", "D", "E") if g in soj]
    unit = "日" if lang == "ja" else "d"
    days_lbl = "期待日数" if lang == "ja" else "Expected days"
    adm_lbl = "入院時 AIS" if lang == "ja" else "Admission AIS"
    fig = go.Figure()
    for si, s in enumerate(states):
        fig.add_trace(go.Bar(
            y=[f"AIS {g}" for g in grades], x=[float(soj[g][si]) for g in grades],
            name=f"AIS {s}", orientation="h", marker=dict(color=PALETTE_AIS[s]),
            hovertemplate=f"%{{y}} · AIS {s}: %{{x:.0f}}{unit}<extra></extra>",
        ))
    fig.update_layout(barmode="stack", height=300, margin=dict(l=70, r=20, t=20, b=44),
                      xaxis=dict(title=days_lbl), yaxis=dict(title=adm_lbl, autorange="reversed"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1))
    return fig


def fig_multistate_improve_base(ms: dict, lang: str) -> go.Figure | None:
    """Observed ≥1-grade improvement rate by admission grade — the non-monotone base rate
    (B > C > A ≫ D) the covariate head must respect.  For an AIS-D admission "improvement" means
    reaching AIS E (full-normal), hence the low rate.  Reads
    ``multistate_metrics.json['improve_head']['rate_by_admission_grade']``."""
    by_grade = (ms or {}).get("improve_head", {}).get("rate_by_admission_grade")
    if not by_grade:
        return None
    grades = [g for g in ("A", "B", "C", "D") if g in by_grade]
    rates = [by_grade[g]["rate"] for g in grades]
    ns = [by_grade[g]["n"] for g in grades]
    rate_lbl = "改善率 (≥1段階)" if lang == "ja" else "Improvement rate (≥1 grade)"
    adm_lbl = "入院時 AIS" if lang == "ja" else "Admission AIS"
    labels = [f"AIS {g}" + ("→E" if g == "D" else "") for g in grades]
    fig = go.Figure(go.Bar(
        x=labels, y=rates, marker=dict(color=[PALETTE_AIS[g] for g in grades]),
        text=[f"{r:.0%}<br>(n={n})" for r, n in zip(rates, ns, strict=True)],
        textposition="outside", textfont=dict(size=11),
        hovertemplate="%{x}<br>" + rate_lbl + ": %{y:.0%}<extra></extra>", showlegend=False,
    ))
    fig.update_layout(height=280, margin=dict(l=54, r=20, t=20, b=40),
                      xaxis=dict(title=adm_lbl),
                      yaxis=dict(range=[0, 1.08], tickformat=".0%", title=rate_lbl))
    return fig


# ----------------------------- functional-independence profile (G7) ----------------------------
# Cohort metric figures over ``independence_metrics.json``: per-item discrimination/calibration
# scorecard, an all-heads reliability overlay, an item x driver heatmap, and the item x
# admission-AIS independence-rate landscape (the monotone A->E story).  Per-item reliability + SHAP
# detail is the interactive drilldown (reuses fig_conversion_reliability / fig_conversion_shap).

def _ind_modeled_items(ind: dict) -> list[dict]:
    """Registry items that have a fitted head, in display (domain-grouped) order."""
    heads = (ind or {}).get("heads") or {}
    return [it for it in ((ind or {}).get("items") or []) if it["key"] in heads]


def _ind_domain_order(items: list[dict]) -> list[str]:
    """Unique SCIM domains in registry order."""
    out: list[str] = []
    for it in items:
        if it["domain"] not in out:
            out.append(it["domain"])
    return out


def _ind_domain_color(dom: str) -> str:
    return PALETTE_INDEPENDENCE_DOMAIN.get(dom, PALETTE_CATEGORICAL[0])


def fig_independence_scorecard(ind: dict, schema: Schema, lang: str) -> go.Figure | None:
    """Per-item discrimination + calibration scorecard: AUC (left) and Brier skill score
    (1 - Brier/baseline; right), one horizontal bar per SCIM-ADL item, colored by domain.  Reads
    ``independence_metrics.json``."""
    heads = (ind or {}).get("heads")
    items = _ind_modeled_items(ind)
    if not heads or not items:
        return None
    rows = items[::-1]  # reverse: registry-first item ends up at the top of the horizontal bars
    names = [col_label(schema, heads[it["key"]]["col"], lang) for it in rows]
    colors = [_ind_domain_color(it["domain"]) for it in rows]
    aucs = [heads[it["key"]]["auc"] for it in rows]
    bases = [heads[it["key"]]["base_rate"] for it in rows]
    ns = [heads[it["key"]]["n"] for it in rows]
    skill = [
        (1.0 - heads[it["key"]]["brier"] / heads[it["key"]]["brier_baseline"])
        if heads[it["key"]]["brier_baseline"] > 0 else 0.0
        for it in rows
    ]
    auc_lbl = "判別 (AUC)" if lang == "ja" else "Discrimination (AUC)"
    skill_lbl = "Brier スキル (1 − Brier/基準)" if lang == "ja" else "Brier skill (1 − Brier/baseline)"
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[auc_lbl, skill_lbl])
    fig.add_trace(go.Bar(
        x=aucs, y=names, orientation="h", marker=dict(color=colors),
        text=[f"{a:.2f}" for a in aucs], textposition="outside", textfont=dict(size=10),
        customdata=np.stack([bases, ns], axis=-1),
        hovertemplate="%{y}<br>AUC=%{x:.3f}<br>base=%{customdata[0]:.0%} · n=%{customdata[1]}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=skill, y=names, orientation="h", marker=dict(color=colors),
        text=[f"{s:.2f}" for s in skill], textposition="outside", textfont=dict(size=10),
        hovertemplate="%{y}<br>Brier skill=%{x:.3f}<extra></extra>", showlegend=False,
    ), row=1, col=2)
    for dom in _ind_domain_order(items):  # domain legend (legend-only swatch markers)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=t(schema, f"ind_domain_{dom}", lang),
            marker=dict(color=_ind_domain_color(dom), size=11, symbol="square"),
            showlegend=True, hoverinfo="skip",
        ), row=1, col=1)
    fig.update_xaxes(range=[0.5, 1.0], tickformat=".2f", row=1, col=1)
    fig.update_xaxes(range=[0, max([*skill, 0.1]) * 1.2], tickformat=".2f", row=1, col=2)
    fig.update_layout(
        height=26 * len(rows) + 124, margin=dict(l=212, r=30, t=46, b=34),
        yaxis=dict(tickfont=dict(size=10.5)), bargap=0.25,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
    )
    return fig


def fig_independence_calibration(ind: dict, schema: Schema, lang: str) -> go.Figure | None:
    """All-heads reliability overlay: every item's Platt-calibrated curve against the diagonal,
    colored by domain.  The near-diagonal bundle confirms the per-item heads are well calibrated.
    Reads ``independence_metrics.json``."""
    heads = (ind or {}).get("heads")
    items = _ind_modeled_items(ind)
    if not heads or not items:
        return None
    conf_lbl = "予測確率" if lang == "ja" else "Predicted probability"
    obs_lbl = "実測頻度" if lang == "ja" else "Observed frequency"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color=INK["200"], dash="dash", width=1.5), showlegend=False, hoverinfo="skip",
    ))
    seen: set[str] = set()
    for it in items:
        cal = heads[it["key"]].get("calibration")
        if not cal:
            continue
        dom = it["domain"]
        label = col_label(schema, heads[it["key"]]["col"], lang)
        fig.add_trace(go.Scatter(
            x=cal["pred_mean"], y=cal["obs_freq"], mode="lines",
            name=t(schema, f"ind_domain_{dom}", lang), legendgroup=dom, showlegend=(dom not in seen),
            line=dict(color=_ind_domain_color(dom), width=1.4), opacity=0.6,
            customdata=[label] * len(cal["pred_mean"]),
            hovertemplate="%{customdata}<br>" + conf_lbl + "=%{x:.2f}<br>" + obs_lbl + "=%{y:.2f}<extra></extra>",
        ))
        seen.add(dom)
    fig.update_layout(
        height=360, margin=dict(l=54, r=20, t=18, b=46),
        xaxis=dict(title=conf_lbl, range=[0, 1]),
        yaxis=dict(title=obs_lbl, range=[0, 1]),
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)", font=dict(size=10)),
    )
    return fig


def fig_independence_shap_heatmap(ind: dict, schema: Schema, lang: str, top_features: int = 12) -> go.Figure | None:
    """Item x driver heatmap: for each item, the mean |SHAP| of the globally most-important
    admission features, row-normalized so each row shows the *relative* driver pattern within that
    item.  Reads ``independence_metrics.json``."""
    heads = (ind or {}).get("heads")
    items = _ind_modeled_items(ind)
    if not heads or not items:
        return None
    per_item: dict[str, dict] = {}
    total: dict[str, float] = {}
    for it in items:
        d = {r["feature"]: r["mean_abs"] for r in (heads[it["key"]].get("shap_top") or [])}
        per_item[it["key"]] = d
        for f, v in d.items():
            total[f] = total.get(f, 0.0) + v
    if not total:
        return None
    feats = [f for f, _ in sorted(total.items(), key=lambda kv: kv[1], reverse=True)[:top_features]]
    z: list[list[float]] = []
    zraw: list[list[float]] = []
    for it in items:
        raw = [per_item[it["key"]].get(f, 0.0) for f in feats]
        m = max(raw) if max(raw) > 0 else 1.0
        z.append([v / m for v in raw])
        zraw.append(raw)
    item_labels = [col_label(schema, heads[it["key"]]["col"], lang) for it in items]
    drv_lbl = "相対寄与度 (項目内)" if lang == "ja" else "Relative importance (within item)"
    fig = go.Figure(go.Heatmap(
        z=z, x=[col_label(schema, f, lang) for f in feats], y=item_labels, customdata=zraw,
        colorscale=[[0, INK["paper"]], [1, PALETTE_CATEGORICAL[0]]], zmin=0, zmax=1,
        colorbar=dict(title=drv_lbl, thickness=12, len=0.8),
        hovertemplate="%{y}<br>%{x}<br>|SHAP|=%{customdata:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=26 * len(items) + 150, margin=dict(l=212, r=20, t=24, b=128),
        xaxis=dict(tickangle=-40, side="bottom"),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10.5)),
    )
    return fig


def fig_independence_landscape(ind: dict, schema: Schema, lang: str) -> go.Figure | None:
    """Item x admission-AIS independence-rate landscape: observed P(independent at discharge) for
    each item by admission grade — the monotone left->right brightening per row is the core G7
    finding (independence rises with admission AIS, outdoor walking lowest throughout).  Reads
    ``independence_metrics.json``."""
    heads = (ind or {}).get("heads")
    items = _ind_modeled_items(ind)
    if not heads or not items:
        return None
    grades = ["A", "B", "C", "D", "E"]
    present: set[str] = set()
    z: list[list[float | None]] = []
    nmat: list[list[int]] = []
    for it in items:
        rbg = heads[it["key"]].get("rate_by_admission_grade") or {}
        zr: list[float | None] = []
        nr: list[int] = []
        for g in grades:
            cell = rbg.get(g)
            if cell:
                zr.append(cell["rate"])
                nr.append(cell["n"])
                present.add(g)
            else:
                zr.append(None)
                nr.append(0)
        z.append(zr)
        nmat.append(nr)
    cols = [g for g in grades if g in present]
    idx = [grades.index(g) for g in cols]
    z = [[r[i] for i in idx] for r in z]
    nmat = [[r[i] for i in idx] for r in nmat]
    text = [[(f"{z[i][j] * 100:.0f}%" if z[i][j] is not None else "") for j in range(len(cols))]
            for i in range(len(items))]
    item_labels = [col_label(schema, heads[it["key"]]["col"], lang) for it in items]
    rate_lbl = "自立率" if lang == "ja" else "Independence rate"
    adm_lbl = "入院時 AIS" if lang == "ja" else "Admission AIS"
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"AIS {g}" for g in cols], y=item_labels, customdata=nmat,
        text=text, texttemplate="%{text}", textfont=dict(size=10),
        colorscale=[[0, INK["paper"]], [1, PALETTE_CATEGORICAL[0]]], zmin=0, zmax=1,
        colorbar=dict(title=rate_lbl, tickformat=".0%", thickness=12, len=0.8),
        hovertemplate="%{y}<br>%{x}<br>" + rate_lbl + "=%{z:.0%}<br>n=%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        height=26 * len(items) + 140, margin=dict(l=212, r=20, t=22, b=46),
        xaxis=dict(title=adm_lbl, side="bottom"),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10.5)),
    )
    return fig


# ----------------------------------------------------------------------------
# G8 recovery topography map — cohort-dynamics Methods figures.  The cohort body-map atlas itself
# is the shared layout.fig_topography_bodymap (driven from the tab via compute.topography_cohort_atlas);
# here live the per-modality pooled calibration, the per-segment discrimination scorecard, and the
# per-modality SHAP drivers.  The per-segment drilldown reuses fig_conversion_{reliability,shap}.

_TOPO_MODS = ("motor", "light_touch", "pin_prick")


def _topo_short_modality(mod: str, lang: str) -> str:
    ja = {"motor": "運動", "light_touch": "触覚", "pin_prick": "痛覚"}
    en = {"motor": "Motor", "light_touch": "Light touch", "pin_prick": "Pin prick"}
    return (ja if lang == "ja" else en)[mod]


def _topo_modelable(topo: dict) -> list[dict]:
    """Non-degenerate per-segment metric records (those carrying auc/calibration/shap_top)."""
    return [s for s in (topo or {}).get("segments", []) if not s.get("degenerate") and s.get("auc") is not None]


def fig_topography_calibration(topo: dict, lang: str) -> go.Figure | None:
    """Pooled per-modality reliability: each modality's Platt-calibrated curve (all its segments'
    OOF predictions pooled) against the diagonal.  Reads ``topography_metrics.json[modality_summary]``."""
    summ = (topo or {}).get("modality_summary")
    if not summ:
        return None
    conf_lbl = "予測確率" if lang == "ja" else "Predicted probability"
    obs_lbl = "実測頻度" if lang == "ja" else "Observed frequency"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color=INK["200"], dash="dash", width=1.5), showlegend=False, hoverinfo="skip",
    ))
    for mod in _TOPO_MODS:
        cal = (summ.get(mod) or {}).get("calibration")
        if not cal:
            continue
        fig.add_trace(go.Scatter(
            x=cal["pred_mean"], y=cal["obs_freq"], mode="lines+markers",
            name=_topo_short_modality(mod, lang),
            line=dict(color=PALETTE_TOPOGRAPHY_MODALITY[mod], width=2),
            marker=dict(size=6, color=PALETTE_TOPOGRAPHY_MODALITY[mod]),
            hovertemplate=conf_lbl + "=%{x:.2f}<br>" + obs_lbl + "=%{y:.2f}<extra></extra>",
        ))
    fig.update_layout(
        height=340, margin=dict(l=54, r=20, t=18, b=46),
        xaxis=dict(title=conf_lbl, range=[0, 1]),
        yaxis=dict(title=obs_lbl, range=[0, 1]),
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)", font=dict(size=10)),
    )
    return fig


def fig_topography_scorecard(topo: dict, lang: str) -> go.Figure | None:
    """Per-segment discrimination spread by modality: a horizontal box of the per-segment OOF AUC
    (left) and Brier skill score 1 - Brier/baseline (right), one box per modality, points = the
    individual segment heads.  Surfaces that all ~120 modelable heads discriminate well.  Reads
    ``topography_metrics.json``."""
    segs = _topo_modelable(topo)
    if not segs:
        return None
    auc_lbl = "判別 (AUC)" if lang == "ja" else "Discrimination (AUC)"
    skill_lbl = "Brier スキル" if lang == "ja" else "Brier skill"
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[auc_lbl, skill_lbl])
    for mod in _TOPO_MODS:
        ms = [s for s in segs if s["modality"] == mod]
        if not ms:
            continue
        name = _topo_short_modality(mod, lang)
        col = PALETTE_TOPOGRAPHY_MODALITY[mod]
        aucs = [s["auc"] for s in ms]
        skill = [1.0 - s["brier"] / s["brier_baseline"] if s["brier_baseline"] > 0 else 0.0 for s in ms]
        labels = [f"{_topo_short_modality(mod, lang)} {s['side'][0]}·{s['level']}" for s in ms]
        fig.add_trace(go.Box(
            x=aucs, y=[name] * len(aucs), name=name, orientation="h", marker=dict(color=col),
            line=dict(color=col), boxpoints="all", jitter=0.5, pointpos=0, fillcolor=_hex_to_rgba(col, 0.25),
            marker_size=4, text=labels, hovertemplate="%{text}<br>AUC=%{x:.3f}<extra></extra>", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Box(
            x=skill, y=[name] * len(skill), name=name, orientation="h", marker=dict(color=col),
            line=dict(color=col), boxpoints="all", jitter=0.5, pointpos=0, fillcolor=_hex_to_rgba(col, 0.25),
            marker_size=4, text=labels, hovertemplate="%{text}<br>skill=%{x:.3f}<extra></extra>", showlegend=False,
        ), row=1, col=2)
    fig.update_xaxes(range=[0.5, 1.0], tickformat=".2f", row=1, col=1)
    fig.update_xaxes(range=[0, 1.0], tickformat=".2f", row=1, col=2)
    fig.update_layout(height=300, margin=dict(l=86, r=24, t=46, b=36),
                      yaxis=dict(tickfont=dict(size=11)))
    return fig


def fig_topography_drivers(topo: dict, schema: Schema, lang: str) -> go.Figure | None:
    """Per-modality SHAP drivers: grouped horizontal bars of the top admission features by mean
    |SHAP|, one bar group per feature with a bar per modality.  ``adm_self`` (the segment's own
    admission grade) dominates every modality.  Reads ``topography_metrics.json[drivers_by_modality]``."""
    drv = (topo or {}).get("drivers_by_modality")
    if not drv:
        return None
    adm_lbl = "当該分節の入院時グレード" if lang == "ja" else "Own admission grade"
    per_mod = {m: {r["feature"]: r["mean_abs"] for r in (drv.get(m) or [])} for m in _TOPO_MODS}
    peak: dict[str, float] = {}
    for d in per_mod.values():
        for f, v in d.items():
            peak[f] = max(peak.get(f, 0.0), v)
    feats = [f for f, _ in sorted(peak.items(), key=lambda kv: kv[1], reverse=True)[:8]][::-1]
    names = [adm_lbl if f == "adm_self" else col_label(schema, f, lang) for f in feats]
    fig = go.Figure()
    for mod in _TOPO_MODS:
        fig.add_trace(go.Bar(
            y=names, x=[per_mod[mod].get(f, 0.0) for f in feats], orientation="h",
            name=_topo_short_modality(mod, lang), marker=dict(color=PALETTE_TOPOGRAPHY_MODALITY[mod]),
            hovertemplate="%{y}<br>|SHAP|=%{x:.3f}<extra></extra>",
        ))
    fig.update_layout(
        height=30 * len(feats) + 96, margin=dict(l=210, r=20, t=40, b=40), barmode="group",
        xaxis_title="mean(|SHAP|)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
    )
    return fig


# ----------------------------- neuro-functional dissociation (G11) -------------------------
def _diss_axis_label(em: dict, lang: str) -> str:
    """Compact axis label for the scorecard, derived from the metrics' verbose neuro/func names."""
    neuro = em["neuro_ja" if lang == "ja" else "neuro_en"].split("（")[0].split(" (")[0]
    return f"{neuro} ↔ {em['func_ja' if lang == 'ja' else 'func_en']}"


def fig_dissociation_scorecard(diss: dict, lang: str) -> go.Figure | None:
    """Per-axis predictability of the dissociation from admission features: the calibrated
    over-achiever-direction AUC and the magnitude-regression R² as grouped horizontal bars.  ``diss``
    is ``dissociation_metrics.json``.  All three axes are well-predicted (AUC 0.93-0.95)."""
    axes = (diss or {}).get("axes") or {}
    if not axes:
        return None
    keys = list(axes)
    labels = [_diss_axis_label(axes[k], lang) for k in keys]
    auc = [axes[k]["over_achiever"]["auc"] for k in keys]
    r2 = [axes[k]["magnitude"]["r2"] for k in keys]
    auc_w = "P(機能優位) の AUC" if lang == "ja" else "P(over-achiever) AUC"
    r2_w = "大きさ D の R²" if lang == "ja" else "Magnitude D  R²"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=auc, orientation="h", name=auc_w, marker=dict(color=PALETTE_CATEGORICAL[0]),
        text=[f"{v:.2f}" for v in auc], textposition="auto", textfont=dict(size=11),
        hovertemplate="%{y}<br>" + auc_w + "=%{x:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=r2, orientation="h", name=r2_w, marker=dict(color=PALETTE_CATEGORICAL[3]),
        text=[f"{v:.2f}" for v in r2], textposition="auto", textfont=dict(size=11),
        hovertemplate="%{y}<br>" + r2_w + "=%{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=70 * len(keys) + 96, margin=dict(l=180, r=20, t=20, b=42), barmode="group",
        xaxis=dict(range=[0, 1.0], title="AUC / R²"),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
    )
    return fig
