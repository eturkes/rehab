"""Plotly figures for the Methods tab — calibration and performance visualizations."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

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
