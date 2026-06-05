"""Plotly figures for the Methods tab — calibration and performance visualizations."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from rehab_sci.dashboard.i18n import col_label
from rehab_sci.dashboard.theme import INK, PALETTE_AIS, PALETTE_CATEGORICAL
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
