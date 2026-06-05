"""Shared layout components: topbar, cards, sliders, prediction figures.

Used by multiple tab modules. No callbacks — pure layout factories.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html

from rehab_sci.dashboard.compute import format_value
from rehab_sci.dashboard.i18n import col_label, level_label, t
from rehab_sci.dashboard.state import FEATURE_SPEC, SCHEMA, SIM_DEFAULTS
from rehab_sci.dashboard.theme import INK, PALETTE_CATEGORICAL
from rehab_sci.models.outcomes import OutcomeSpec


# ---------- chrome / structural ----------
def topbar(lang: str) -> html.Div:
    return html.Div(
        className="topbar",
        children=[
            html.Div(
                className="topbar-left",
                children=[
                    html.H1(t(SCHEMA, "app_title", lang)),
                    html.P(t(SCHEMA, "app_subtitle", lang)),
                ],
            ),
            html.Div(
                className="lang-toggle",
                children=[
                    html.Button(
                        "日本語", id="lang-ja", n_clicks=0,
                        className="active" if lang == "ja" else "",
                    ),
                    html.Button(
                        "English", id="lang-en", n_clicks=0,
                        className="active" if lang == "en" else "",
                    ),
                ],
            ),
        ],
    )


def kpi_card(label: str, value: str, sub: str | None = None) -> html.Div:
    children = [html.H3(label), html.Div(value, className="big-number")]
    if sub:
        children.append(html.Div(sub, className="subtitle"))
    return html.Div(className="card", children=children)


def chart_card(title: str, content) -> html.Div:
    return html.Div(className="chart-card", children=[html.H2(title), content])


# ---------- simulator input widgets ----------
def input_id(prefix: str, col: str) -> dict:
    return {"type": prefix, "col": col}


def number_input_for(feature: str, lang: str, defaults: dict | None = None) -> html.Div:
    """Clearable numeric input. A blank field is left unknown (NaN) so the model
    uses its native missing-value handling; ``defaults={}`` opens it blank."""
    rng = FEATURE_SPEC["ranges"].get(feature)
    if rng is None:
        return html.Div()
    src = SIM_DEFAULTS if defaults is None else defaults
    value = src.get(feature)
    lo, hi, med = rng["min"], rng["max"], rng["median"]
    hint = (
        f"範囲 {lo:.0f}–{hi:.0f}・中央 {med:.0f}"
        if lang == "ja"
        else f"range {lo:.0f}–{hi:.0f} · median {med:.0f}"
    )
    return html.Div(
        className="sim-field",
        children=[
            html.Label(col_label(SCHEMA, feature, lang)),
            dcc.Input(
                id=input_id("num", feature),
                type="number",
                min=lo, max=hi, step=1,
                value=value,
                debounce=True,
                placeholder=("未入力 = 不明" if lang == "ja" else "blank = unknown"),
                className="sim-number",
            ),
            html.Span(hint, className="sim-hint"),
        ],
    )


def dropdown_for(feature: str, lang: str, defaults: dict | None = None) -> html.Div:
    cats = FEATURE_SPEC["categories"].get(feature, [])
    spec = SCHEMA.by_raw(feature)
    level_key = spec.levels if spec else None
    options = [
        {
            "label": level_label(SCHEMA, level_key, c, lang) if level_key else c,
            "value": c,
        }
        for c in cats
    ]
    src = SIM_DEFAULTS if defaults is None else defaults
    default = src.get(feature)
    return html.Div(
        className="sim-field",
        children=[
            html.Label(col_label(SCHEMA, feature, lang)),
            dcc.Dropdown(
                id=input_id("cat", feature),
                options=options,
                value=default,
                clearable=True,
                placeholder=("未指定" if lang == "ja" else "Unspecified"),
            ),
        ],
    )


# ---------- prediction figures (shared by simulator + patient) ----------
def fig_shap_local(values: np.ndarray, X, base: float, lang: str) -> go.Figure:
    feat_names = X.columns.tolist()
    feat_values = X.iloc[0].tolist()
    pairs = sorted(
        zip(feat_names, values, feat_values, strict=False),
        key=lambda r: -abs(r[1]),
    )[:12][::-1]
    names = [
        f"{col_label(SCHEMA, n, lang)} = {format_value(n, v)}"
        for n, _, v in pairs
    ]
    contribs = [float(s) for _, s, _ in pairs]
    colors = ["#2c8a6b" if c >= 0 else "#a3354e" for c in contribs]
    fig = go.Figure(
        go.Bar(
            x=contribs, y=names, orientation="h",
            marker=dict(color=colors),
            hovertemplate="%{y}<br>SHAP: %{x:+.2f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=INK["300"], width=1))
    fig.update_layout(
        height=max(280, 22 * len(pairs) + 60),
        margin=dict(l=260, r=20, t=10, b=44),
        xaxis_title="SHAP 寄与 (点)" if lang == "ja" else "SHAP contribution (pts)",
    )
    return fig


def fig_prediction_interval(
    pred: float, lo: float, hi: float, spec: OutcomeSpec, lang: str,
) -> go.Figure:
    label = t(SCHEMA, spec.display_key, lang)
    unit = t(SCHEMA, spec.unit_key, lang) if spec.unit_key else ""
    pi_label = t(SCHEMA, "sim_prediction_interval", lang)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[hi - lo], base=[lo], y=[pi_label], orientation="h",
        marker=dict(color="rgba(17,122,139,0.18)", line=dict(width=0)),
        hovertemplate=f"{lo:.0f}–{hi:.0f}<extra></extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[pred], y=[pi_label], mode="markers",
        marker=dict(color="#0c5a66", size=14, symbol="diamond"),
        hovertemplate=(
            ("予測中央値: %{x:.0f}" if lang == "ja" else "Predicted median: %{x:.0f}")
            + "<extra></extra>"
        ),
        showlegend=False,
    ))
    x_lo = float(spec.clip_min) if spec.clip_min is not None else min(0.0, lo)
    x_hi = float(spec.clip_max) if spec.clip_max is not None else float(max(hi, pred) * 1.1 + 1.0)
    axis_title = f"{label} ({unit})" if unit else label
    fig.update_layout(
        height=120,
        margin=dict(l=110, r=20, t=10, b=30),
        xaxis=dict(range=[x_lo, x_hi], title=axis_title),
        yaxis=dict(showticklabels=True, tickfont=dict(size=12), showgrid=False),
    )
    return fig


def fig_class_probabilities(
    proba: np.ndarray, class_labels: list[str], spec: OutcomeSpec, lang: str,
    conformal_set: list[int] | None = None,
) -> go.Figure:
    label = t(SCHEMA, spec.display_key, lang)
    bar_labels = [f"AIS {c}" for c in class_labels]
    if conformal_set is not None:
        colors = [
            PALETTE_CATEGORICAL[0] if i in conformal_set else "rgba(17,122,139,0.18)"
            for i in range(len(class_labels))
        ]
        borders = [
            "#0c5a66" if i in conformal_set else "rgba(0,0,0,0)"
            for i in range(len(class_labels))
        ]
    else:
        colors = [PALETTE_CATEGORICAL[0]] * len(class_labels)
        borders = [PALETTE_CATEGORICAL[0]] * len(class_labels)
    fig = go.Figure(go.Bar(
        x=bar_labels, y=proba,
        marker=dict(color=colors, line=dict(color=borders, width=1.5)),
        text=[f"{p:.0%}" for p in proba],
        textposition="outside",
        hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=60, r=20, t=30, b=44),
        yaxis=dict(range=[0, 1.05], tickformat=".0%",
                   title=t(SCHEMA, "sim_class_probabilities", lang)),
        xaxis=dict(title=label),
    )
    return fig
