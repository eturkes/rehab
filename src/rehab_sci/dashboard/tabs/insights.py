"""Insight engine tab — SHAP importance, subgroups, dependence, interactions."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from rehab_sci.dashboard import figures as fg
from rehab_sci.dashboard.i18n import col_label, t
from rehab_sci.dashboard.layout import chart_card
from rehab_sci.dashboard.state import (
    DEFAULT_OUTCOME,
    EP,
    METRICS,
    OUTCOME_BUNDLES,
    SCHEMA,
    SCIM_TOTAL_BUNDLE,
    SUBGROUPS,
)
from rehab_sci.dashboard.theme import INK
from rehab_sci.data.dataset import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from rehab_sci.models.outcomes import OUTCOMES, OutcomeSpec


def _insight_outcome_options(lang: str) -> list[dict]:
    return [{"label": t(SCHEMA, s.display_key, lang), "value": s.key} for s in OUTCOMES]


# ---------- layout ----------
def render_insights(lang: str) -> html.Div:
    cat_features_in_data = [c for c in CATEGORICAL_FEATURES if c in EP.columns]
    num_features_in_data = [c for c in NUMERIC_FEATURES if c in EP.columns]
    sub_options = [
        {"label": col_label(SCHEMA, c, lang), "value": c}
        for c in cat_features_in_data + num_features_in_data
    ]

    outcome_selector = html.Div(
        className="sim-outcome-selector",
        children=[
            html.Label(
                t(SCHEMA, "insight_outcome_label", lang),
                style={"fontSize": "12px", "color": INK["500"]},
            ),
            dcc.Dropdown(
                id="ins-outcome",
                options=_insight_outcome_options(lang),
                value=DEFAULT_OUTCOME,
                clearable=False,
            ),
        ],
    )

    importance_card = chart_card(
        t(SCHEMA, "insight_global_importance", lang),
        dcc.Graph(id="ins-importance-graph", config={"displayModeBar": False}),
    )

    subgroup_card = chart_card(
        t(SCHEMA, "insight_subgroup_compare", lang),
        html.Div([
            html.Div(
                style={"display": "flex", "gap": "12px", "marginBottom": "8px"},
                children=[html.Div(style={"flex": "1"}, children=[
                    html.Label(t(SCHEMA, "insight_choose_strata", lang),
                               style={"fontSize": "12px", "color": INK["500"]}),
                    dcc.Dropdown(id="ins-subgroup-feature", options=sub_options,
                                 value="対麻痺_四肢麻痺", clearable=False),
                ])],
            ),
            dcc.Graph(id="ins-subgroup-graph", config={"displayModeBar": False}),
            html.Div(id="ins-effect-size", style={"fontSize": "13px", "color": INK["700"]}),
        ]),
    )

    dep_card = chart_card(
        t(SCHEMA, "insight_pair_dependence", lang),
        html.Div([
            html.Div(
                style={"display": "flex", "gap": "12px", "marginBottom": "8px"},
                children=[
                    html.Div(style={"flex": "1"}, children=[
                        html.Label(t(SCHEMA, "insight_choose_feature", lang),
                                   style={"fontSize": "12px", "color": INK["500"]}),
                        dcc.Dropdown(id="ins-dep-feature", options=[], value=None, clearable=False),
                    ]),
                    html.Div(
                        id="ins-dep-class-wrap",
                        style={"flex": "1", "display": "none"},
                        children=[
                            html.Label(t(SCHEMA, "insight_dep_class_label", lang),
                                       style={"fontSize": "12px", "color": INK["500"]}),
                            dcc.Dropdown(id="ins-dep-class", options=[], value=None, clearable=False),
                        ],
                    ),
                ],
            ),
            dcc.Graph(id="ins-dep-graph", config={"displayModeBar": False}),
            html.Div(id="ins-dep-note"),
        ]),
    )

    interaction_card = chart_card(
        t(SCHEMA, "insight_interaction_heading", lang),
        html.Div([
            dcc.Graph(id="ins-int-heatmap", config={"displayModeBar": False}),
            html.Div(
                style={"display": "flex", "gap": "12px", "marginBottom": "8px", "marginTop": "16px"},
                children=[
                    html.Div(style={"flex": "1"}, children=[
                        html.Label(t(SCHEMA, "insight_int_feat_x", lang),
                                   style={"fontSize": "12px", "color": INK["500"]}),
                        dcc.Dropdown(id="ins-int-feat-x", options=[], value=None, clearable=False),
                    ]),
                    html.Div(style={"flex": "1"}, children=[
                        html.Label(t(SCHEMA, "insight_int_feat_y", lang),
                                   style={"fontSize": "12px", "color": INK["500"]}),
                        dcc.Dropdown(id="ins-int-feat-y", options=[], value=None, clearable=False),
                    ]),
                ],
            ),
            dcc.Graph(id="ins-int-dep-graph", config={"displayModeBar": False}),
        ]),
    )

    return html.Div([
        outcome_selector,
        html.Div(className="chart-row", children=[importance_card, subgroup_card]),
        html.Div(className="chart-row", children=[dep_card]),
        html.Div(className="chart-row", children=[interaction_card]),
    ])


# ---------- callbacks ----------
@callback(
    Output("ins-outcome", "options"),
    Input("lang-store", "data"),
)
def update_insight_outcome_options(lang):  # noqa: ANN001
    return _insight_outcome_options(lang)


@callback(
    Output("ins-importance-graph", "figure"),
    Input("ins-outcome", "value"),
    Input("lang-store", "data"),
)
def update_importance(outcome_key, lang):  # noqa: ANN001
    m = METRICS["outcomes"].get(outcome_key or DEFAULT_OUTCOME, METRICS["outcomes"][DEFAULT_OUTCOME])
    return fg.fig_global_shap_importance(m, SCHEMA, lang)


@callback(
    Output("ins-subgroup-graph", "figure"),
    Output("ins-effect-size", "children"),
    Input("ins-subgroup-feature", "value"),
    Input("ins-outcome", "value"),
    Input("lang-store", "data"),
)
def update_subgroup(feature, outcome_key, lang):  # noqa: ANN001
    outcome_key = outcome_key or DEFAULT_OUTCOME
    bundle = OUTCOME_BUNDLES.get(outcome_key) or SCIM_TOTAL_BUNDLE
    spec: OutcomeSpec = bundle["spec"]
    outcome_label = t(SCHEMA, spec.display_key, lang)
    fig = fg.fig_subgroup_box(EP, feature, SCHEMA, lang,
                              outcome_col=spec.target_col, outcome_label=outcome_label)
    sg = SUBGROUPS.get(outcome_key, {})
    results = sg.get("results", [])
    info = next((r for r in results if r["feature"] == feature and not r.get("skipped")), None)
    if info is None:
        return fig, ""
    if "cliffs_delta" in info:
        eff_lbl = "Cliff's δ"
        eff = info["cliffs_delta"]
    else:
        eff_lbl = "η²"
        eff = info.get("eta_squared", float("nan"))
    p_bh = info.get("p_bh", float("nan"))
    txt_ja = f"検定: {info['test']}  ·  {eff_lbl} = {eff:+.2f}  ·  p (BH) = {p_bh:.1e}"
    txt_en = f"Test: {info['test']}  ·  {eff_lbl} = {eff:+.2f}  ·  p (BH) = {p_bh:.1e}"
    return fig, txt_ja if lang == "ja" else txt_en


@callback(
    Output("ins-dep-feature", "options"),
    Output("ins-dep-feature", "value"),
    Input("ins-outcome", "value"),
    Input("lang-store", "data"),
)
def update_dep_feature_options(outcome_key, lang):  # noqa: ANN001
    outcome_key = outcome_key or DEFAULT_OUTCOME
    m = METRICS["outcomes"].get(outcome_key, METRICS["outcomes"][DEFAULT_OUTCOME])
    items = m.get("global_importance_top25", [])[:15]
    seen, opts = set(), []
    for item in items:
        f = item["feature"]
        if f not in seen:
            seen.add(f)
            opts.append({"label": col_label(SCHEMA, f, lang), "value": f})
    val = opts[0]["value"] if opts else None
    return opts, val


@callback(
    Output("ins-dep-class-wrap", "style"),
    Output("ins-dep-class", "options"),
    Output("ins-dep-class", "value"),
    Input("ins-outcome", "value"),
)
def update_dep_class_options(outcome_key):  # noqa: ANN001
    outcome_key = outcome_key or DEFAULT_OUTCOME
    bundle = OUTCOME_BUNDLES.get(outcome_key) or SCIM_TOTAL_BUNDLE
    if bundle["task"] == "multiclass":
        spec = bundle["spec"]
        opts = [{"label": lbl, "value": i} for i, lbl in enumerate(spec.class_labels)]
        return {"flex": "1"}, opts, 0
    return {"flex": "1", "display": "none"}, [], None


@callback(
    Output("ins-dep-graph", "figure"),
    Output("ins-dep-note", "children"),
    Input("ins-dep-feature", "value"),
    Input("ins-outcome", "value"),
    Input("ins-dep-class", "value"),
    Input("lang-store", "data"),
)
def update_dependence(feature, outcome_key, class_val, lang):  # noqa: ANN001
    outcome_key = outcome_key or DEFAULT_OUTCOME
    bundle = OUTCOME_BUNDLES.get(outcome_key) or SCIM_TOTAL_BUNDLE
    if feature is None:
        return go.Figure(), ""
    shap_pack = bundle["shap"]
    class_idx = class_val if (bundle["task"] == "multiclass" and class_val is not None) else None
    return fg.fig_dependence(shap_pack, shap_pack["X_test"], feature, SCHEMA, lang, class_idx=class_idx), ""


@callback(
    Output("ins-int-heatmap", "figure"),
    Input("ins-outcome", "value"),
    Input("lang-store", "data"),
)
def update_interaction_heatmap(outcome_key, lang):  # noqa: ANN001
    m = METRICS["outcomes"].get(outcome_key or DEFAULT_OUTCOME, METRICS["outcomes"][DEFAULT_OUTCOME])
    return fg.fig_interaction_heatmap(m, SCHEMA, lang)


@callback(
    Output("ins-int-feat-x", "options"),
    Output("ins-int-feat-x", "value"),
    Output("ins-int-feat-y", "options"),
    Output("ins-int-feat-y", "value"),
    Input("ins-outcome", "value"),
    Input("lang-store", "data"),
)
def update_int_feat_options(outcome_key, lang):  # noqa: ANN001
    outcome_key = outcome_key or DEFAULT_OUTCOME
    m = METRICS["outcomes"].get(outcome_key, METRICS["outcomes"][DEFAULT_OUTCOME])
    items = m.get("global_interaction_top25", [])
    seen, opts = set(), []
    for item in items:
        for f in (item["feat_a"], item["feat_b"]):
            if f not in seen:
                seen.add(f)
                opts.append({"label": col_label(SCHEMA, f, lang), "value": f})
    val_x = items[0]["feat_a"] if items else (opts[0]["value"] if opts else None)
    val_y = items[0]["feat_b"] if items else (opts[1]["value"] if len(opts) > 1 else None)
    return opts, val_x, opts, val_y


@callback(
    Output("ins-int-dep-graph", "figure"),
    Input("ins-int-feat-x", "value"),
    Input("ins-int-feat-y", "value"),
    Input("ins-outcome", "value"),
    Input("ins-dep-class", "value"),
    Input("lang-store", "data"),
)
def update_interaction_dependence(feat_x, feat_y, outcome_key, class_val, lang):  # noqa: ANN001
    outcome_key = outcome_key or DEFAULT_OUTCOME
    bundle = OUTCOME_BUNDLES.get(outcome_key) or SCIM_TOTAL_BUNDLE
    if feat_x is None or feat_y is None:
        return go.Figure()
    shap_pack = bundle["shap"]
    class_idx = class_val if (bundle["task"] == "multiclass" and class_val is not None) else None
    return fg.fig_interaction_dependence(
        shap_pack, shap_pack["X_test"], feat_x, feat_y, SCHEMA, lang, class_idx=class_idx,
    )
