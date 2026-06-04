"""Plotly figures for the Insight engine tab — SHAP importance, subgroups, dependence, interactions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from rehab_sci.dashboard.i18n import all_levels_in_order, col_label, level_label, t
from rehab_sci.dashboard.theme import INK, PALETTE_CATEGORICAL
from rehab_sci.schema import Schema


def fig_global_shap_importance(metrics: dict, schema: Schema, lang: str, top_n: int = 15) -> go.Figure:
    items = metrics["global_importance_top25"][:top_n][::-1]
    names = [col_label(schema, r["feature"], lang) for r in items]
    vals = [r["abs_mean_shap"] for r in items]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker=dict(color=PALETTE_CATEGORICAL[0]),
            hovertemplate="%{y}<br>|SHAP|: %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(360, 22 * len(items) + 80),
        xaxis_title="mean(|SHAP|) " + ("点" if lang == "ja" else "pts"),
        margin=dict(l=240, r=20, t=10, b=44),
    )
    return fig


def fig_subgroup_box(
    ep: pd.DataFrame,
    feature: str,
    schema: Schema,
    lang: str,
    outcome_col: str = "y_discharge_scim",
    outcome_label: str | None = None,
) -> go.Figure:
    spec = schema.by_raw(feature)
    cols = [feature, outcome_col]
    sub = ep[cols].dropna().copy()
    if spec and spec.dtype == "categorical" and spec.levels:
        sub["_label"] = sub[feature].astype(str).map(
            lambda v: level_label(schema, spec.levels, v, lang)
        )
        order = [level_label(schema, spec.levels, lv, lang)
                 for lv, _ in all_levels_in_order(schema, spec.levels, lang)]
        present = [o for o in order if o in sub["_label"].unique()]
        sub["_label"] = pd.Categorical(sub["_label"], categories=present, ordered=True)
        x = sub["_label"]
        xtitle = col_label(schema, feature, lang)
    else:
        sub["_q"] = pd.qcut(sub[feature], q=4, duplicates="drop")
        sub["_label"] = sub["_q"].astype(str)
        x = sub["_label"]
        xtitle = col_label(schema, feature, lang) + (" (四分位)" if lang == "ja" else " (quartile)")

    ytitle = outcome_label if outcome_label else t(schema, "chart_discharge_scim", lang)

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            x=x,
            y=sub[outcome_col],
            boxpoints="outliers",
            marker=dict(color=PALETTE_CATEGORICAL[0], size=3),
            line=dict(color=PALETTE_CATEGORICAL[0]),
            fillcolor="rgba(17,122,139,0.15)",
            hovertemplate="%{x}<br>%{y:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=400,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        margin=dict(l=60, r=20, t=10, b=80),
        xaxis_tickangle=-25,
    )
    return fig


def fig_dependence(
    shap_pack: dict,
    X_test: pd.DataFrame,
    feature: str,
    schema: Schema,
    lang: str,
    *,
    class_idx: int | None = None,
) -> go.Figure:
    if feature not in X_test.columns:
        return go.Figure()
    idx = X_test.columns.get_loc(feature)
    sv = shap_pack["shap_values"]
    if class_idx is not None and sv.ndim == 3:
        shap_vals = sv[:, idx, class_idx]
        labels = shap_pack.get("class_labels", [])
        cls_lbl = labels[class_idx] if class_idx < len(labels) else str(class_idx)
        y_label = f"SHAP (AIS {cls_lbl})"
    else:
        shap_vals = sv[:, idx]
        y_label = "SHAP"
    spec = schema.by_raw(feature)

    if spec and spec.dtype == "categorical":
        x = X_test[feature].astype(str).map(
            lambda v: level_label(schema, spec.levels, v, lang) if spec.levels else v
        )
        fig = go.Figure(
            go.Box(
                x=x,
                y=shap_vals,
                boxpoints="outliers",
                marker=dict(color=PALETTE_CATEGORICAL[1], size=3),
                line=dict(color=PALETTE_CATEGORICAL[1]),
                fillcolor="rgba(212,119,60,0.15)",
            )
        )
        fig.update_layout(xaxis_title=col_label(schema, feature, lang), yaxis_title=y_label)
    else:
        x = pd.to_numeric(X_test[feature], errors="coerce")
        fig = go.Figure(
            go.Scatter(
                x=x,
                y=shap_vals,
                mode="markers",
                marker=dict(
                    color=shap_vals,
                    colorscale="RdBu_r",
                    cmid=0,
                    showscale=True,
                    size=8,
                    opacity=0.7,
                    line=dict(width=0.5, color="#fff"),
                ),
                hovertemplate=(
                    col_label(schema, feature, lang)
                    + ": %{x}<br>" + y_label + ": %{y:.2f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            xaxis_title=col_label(schema, feature, lang),
            yaxis_title=y_label,
        )
    fig.add_hline(y=0, line=dict(color=INK["300"], dash="dash", width=1))
    fig.update_layout(height=360, margin=dict(l=60, r=20, t=10, b=44))
    return fig


def fig_interaction_heatmap(
    metrics: dict,
    schema: Schema,
    lang: str,
    *,
    top_n: int = 10,
) -> go.Figure:
    """Upper-triangle heatmap of top feature-pair interactions by mean |SHAP|."""
    items = metrics.get("global_interaction_top25", [])
    if not items:
        return go.Figure()

    # collect unique features from top_n pairs
    feats_seen: list[str] = []
    for item in items[:top_n]:
        for f in (item["feat_a"], item["feat_b"]):
            if f not in feats_seen:
                feats_seen.append(f)
    n = len(feats_seen)
    labels = [col_label(schema, f, lang) for f in feats_seen]
    feat_to_idx = {f: i for i, f in enumerate(feats_seen)}

    # build matrix (upper triangle only; diagonal = 0)
    mat = np.zeros((n, n))
    for item in items[:top_n]:
        ia = feat_to_idx.get(item["feat_a"])
        ib = feat_to_idx.get(item["feat_b"])
        if ia is not None and ib is not None:
            val = item["abs_mean_interaction"]
            mat[ia, ib] = val
            mat[ib, ia] = val

    # mask lower triangle for visual clarity
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    display = np.where(mask, mat, np.nan)

    hover_text = []
    for i in range(n):
        row = []
        for j in range(n):
            if mask[i, j]:
                row.append(f"{labels[i]} × {labels[j]}<br>mean|φ|: {mat[i, j]:.4f}")
            else:
                row.append("")
        hover_text.append(row)

    fig = go.Figure(
        go.Heatmap(
            z=display,
            x=labels,
            y=labels,
            colorscale=[[0, "rgba(17,122,139,0.05)"], [1, PALETTE_CATEGORICAL[0]]],
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
            showscale=True,
            colorbar=dict(title="mean|φ|"),
        )
    )
    fig.update_layout(
        height=max(360, 28 * n + 80),
        margin=dict(l=200, r=20, t=10, b=120),
        xaxis=dict(tickangle=-45, side="bottom"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def fig_interaction_dependence(
    shap_pack: dict,
    X_test: pd.DataFrame,
    feat_x: str,
    feat_y: str,
    schema: Schema,
    lang: str,
    *,
    class_idx: int | None = None,
) -> go.Figure:
    """Scatter of feature-X value vs SHAP interaction(X,Y), colored by feature-Y value."""
    si = shap_pack.get("shap_interaction")
    if si is None or feat_x not in X_test.columns or feat_y not in X_test.columns:
        return go.Figure()
    ix = X_test.columns.get_loc(feat_x)
    iy = X_test.columns.get_loc(feat_y)

    # slice interaction values
    if class_idx is not None and si.ndim == 4:
        int_vals = si[:, ix, iy, class_idx]
        labels = shap_pack.get("class_labels", [])
        cls_lbl = labels[class_idx] if class_idx < len(labels) else str(class_idx)
        y_label = f"SHAP φ(x,y) (AIS {cls_lbl})"
    else:
        int_vals = si[:, ix, iy]
        y_label = "SHAP φ(x,y)"

    x_label = col_label(schema, feat_x, lang)
    y_feat_label = col_label(schema, feat_y, lang)

    spec_x = schema.by_raw(feat_x)
    spec_y = schema.by_raw(feat_y)

    # x-axis values
    if spec_x and spec_x.dtype == "categorical":
        x_vals = X_test[feat_x].astype(str).map(
            lambda v: level_label(schema, spec_x.levels, v, lang) if spec_x.levels else v
        )
    else:
        x_vals = pd.to_numeric(X_test[feat_x], errors="coerce")

    # color by feat_y
    if spec_y and spec_y.dtype == "categorical":
        color_vals = X_test[feat_y].astype(str).map(
            lambda v: level_label(schema, spec_y.levels, v, lang) if spec_y.levels else v
        )
        # categorical coloring: distinct traces per category
        fig = go.Figure()
        cats = [c for c in color_vals.unique() if pd.notna(c) and str(c) != "nan"]
        for ci, cat in enumerate(sorted(cats, key=str)):
            mask_c = color_vals == cat
            fig.add_trace(
                go.Scatter(
                    x=x_vals[mask_c] if not isinstance(x_vals.iloc[0], str) else x_vals[mask_c],
                    y=int_vals[mask_c],
                    mode="markers",
                    name=str(cat),
                    marker=dict(
                        color=PALETTE_CATEGORICAL[ci % len(PALETTE_CATEGORICAL)],
                        size=8, opacity=0.7,
                        line=dict(width=0.5, color="#fff"),
                    ),
                    hovertemplate=(
                        x_label + ": %{x}<br>"
                        + y_feat_label + ": " + str(cat) + "<br>"
                        + y_label + ": %{y:.4f}<extra></extra>"
                    ),
                )
            )
    else:
        color_num = pd.to_numeric(X_test[feat_y], errors="coerce")
        fig = go.Figure(
            go.Scatter(
                x=x_vals,
                y=int_vals,
                mode="markers",
                marker=dict(
                    color=color_num,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=y_feat_label),
                    size=8, opacity=0.7,
                    line=dict(width=0.5, color="#fff"),
                ),
                hovertemplate=(
                    x_label + ": %{x}<br>"
                    + y_feat_label + ": %{marker.color:.1f}<br>"
                    + y_label + ": %{y:.4f}<extra></extra>"
                ),
            )
        )

    fig.add_hline(y=0, line=dict(color=INK["300"], dash="dash", width=1))
    fig.update_layout(
        height=360,
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=60, r=20, t=10, b=44),
    )
    return fig


# ---------- Patient explorer tab ----------
# Subscale traces share the timeline figure with SCIM_total.  Use distinct hues
# but keep them muted so the total line stays the prominent signal.
