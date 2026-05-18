"""Rehabilitation Analytics & Prediction Suite — bilingual Dash app.

Run with::

    uv run python -m rehab_sci.dashboard.app

then open http://127.0.0.1:8050/.
"""

from __future__ import annotations

import json
from pathlib import Path

import dash
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, ctx, dcc, html

from rehab_sci.dashboard import figures as fg
from rehab_sci.dashboard.i18n import col_label, level_label, t
from rehab_sci.dashboard.theme import (
    INK,
    PALETTE_CATEGORICAL,
    apply_template,
)
from rehab_sci.data.dataset import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_analysis_dataset,
)
from rehab_sci.schema import load_schema

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models"


# ---------- one-time startup load ----------
apply_template()
SCHEMA = load_schema()
AF = build_analysis_dataset()
EP = AF.df
LONG = AF.longitudinal

with (MODELS_DIR / "training_metrics.json").open(encoding="utf-8") as f:
    METRICS = json.load(f)
with (MODELS_DIR / "simulator_defaults.json").open(encoding="utf-8") as f:
    SIM_DEFAULTS = json.load(f)
FEATURE_SPEC = joblib.load(MODELS_DIR / "feature_spec.joblib")
MEDIAN_MODEL = joblib.load(MODELS_DIR / "lgbm_median.joblib")
P10_MODEL = joblib.load(MODELS_DIR / "lgbm_p10.joblib")
P90_MODEL = joblib.load(MODELS_DIR / "lgbm_p90.joblib")
SHAP_PACK = joblib.load(MODELS_DIR / "shap_test.joblib")

with (MODELS_DIR / "subgroups.json").open(encoding="utf-8") as f:
    SUBGROUPS = json.load(f)


# ---------- helpers ----------
def _split_features() -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) features in display order matched to UI sections."""
    num = [c for c in FEATURE_SPEC["numeric_cols"] if c in FEATURE_SPEC["feature_cols"]]
    cat = [c for c in FEATURE_SPEC["categorical_cols"] if c in FEATURE_SPEC["feature_cols"]]
    return num, cat


def _input_id(prefix: str, col: str) -> dict:
    return {"type": prefix, "col": col}


def _format_value(col: str, value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    if isinstance(value, float):
        return f"{value:.0f}"
    return str(value)


# ---------- layout factories ----------
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
                        "日本語",
                        id="lang-ja",
                        n_clicks=0,
                        className="active" if lang == "ja" else "",
                    ),
                    html.Button(
                        "English",
                        id="lang-en",
                        n_clicks=0,
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


# ---------- TAB: overview ----------
def render_overview(lang: str) -> html.Div:
    n_episodes = len(EP)
    n_patients = EP["IDNumber"].nunique()
    mean_age = pd.to_numeric(EP["年齢"], errors="coerce").mean()
    median_scim = pd.to_numeric(EP["y_discharge_scim"], errors="coerce").median()

    ais_admit = EP["AIS"].dropna().astype(str).value_counts(normalize=True)
    severe_pct = float(ais_admit.get("A", 0.0) + ais_admit.get("B", 0.0)) * 100

    kpi_row = html.Div(
        className="card-row",
        children=[
            kpi_card(
                t(SCHEMA, "patients_n", lang),
                f"{n_patients:,}",
                t(SCHEMA, "episodes_n", lang) + f": {n_episodes:,}",
            ),
            kpi_card(
                ("平均年齢" if lang == "ja" else "Mean age"),
                f"{mean_age:.1f}",
                t(SCHEMA, "unit_years", lang),
            ),
            kpi_card(
                ("退院時 SCIM 中央値" if lang == "ja" else "Discharge SCIM median"),
                f"{median_scim:.0f}",
                "0–100",
            ),
            kpi_card(
                ("重度 (AIS A/B) 比率" if lang == "ja" else "Severe (AIS A/B) at admission"),
                f"{severe_pct:.0f}%",
                None,
            ),
        ],
    )

    row1 = html.Div(
        className="chart-row",
        children=[
            chart_card(
                t(SCHEMA, "chart_injury_sunburst", lang),
                dcc.Graph(figure=fg.fig_injury_sunburst(EP, SCHEMA, lang), config={"displayModeBar": False}),
            ),
            chart_card(
                t(SCHEMA, "chart_ais_admit_discharge", lang),
                dcc.Graph(figure=fg.fig_ais_admit_discharge_sankey(EP, SCHEMA, lang), config={"displayModeBar": False}),
            ),
        ],
    )

    row2 = html.Div(
        className="chart-row",
        children=[
            chart_card(
                t(SCHEMA, "chart_age_dist", lang),
                dcc.Graph(figure=fg.fig_age_distribution(EP, SCHEMA, lang), config={"displayModeBar": False}),
            ),
            chart_card(
                t(SCHEMA, "chart_sex_dist", lang),
                dcc.Graph(figure=fg.fig_sex_donut(EP, SCHEMA, lang), config={"displayModeBar": False}),
            ),
            chart_card(
                t(SCHEMA, "chart_mechanism", lang),
                dcc.Graph(figure=fg.fig_mechanism(EP, SCHEMA, lang), config={"displayModeBar": False}),
            ),
        ],
    )

    row3 = html.Div(
        className="chart-row",
        children=[
            chart_card(
                t(SCHEMA, "chart_discharge_scim", lang),
                dcc.Graph(figure=fg.fig_discharge_scim(EP, SCHEMA, lang), config={"displayModeBar": False}),
            ),
            chart_card(
                t(SCHEMA, "chart_recovery_curves", lang),
                dcc.Graph(figure=fg.fig_recovery_curves(LONG, SCHEMA, lang), config={"displayModeBar": False}),
            ),
        ],
    )

    return html.Div([kpi_row, row1, row2, row3])


# ---------- TAB: simulator ----------
SIM_NUMERIC_ORDER = [
    "年齢", "AIS_ord", "mFrankel_ord", "NLI_ord",
    "UEMS", "LEMS", "TotalMotor",
    "LightTouchTotal", "PinPrickTotal",
    "RightMotorLevel_ord", "LeftMotorLevel_ord",
    "RightSensoryLevel_ord", "LeftSensoryLevel_ord",
    "SCIM_total", "SCIM_self_care", "SCIM_respiration_sphincter", "SCIM_mobility",
]
SIM_CATEGORICAL_ORDER = [
    "性別", "外傷性_非外傷性", "対麻痺_四肢麻痺",
    "VAC", "DAP", "OPLL", "DISH", "糖尿病",
]


def _slider_for(feature: str, lang: str) -> html.Div:
    rng = FEATURE_SPEC["ranges"].get(feature)
    default = SIM_DEFAULTS.get(feature)
    if rng is None:
        return html.Div()
    lo = rng["min"]
    hi = rng["max"]
    if default is None:
        default = rng["median"]
    step = 1.0 if (hi - lo) > 20 else 0.5
    marks_targets = sorted({lo, rng["q05"], rng["median"], rng["q95"], hi})
    marks = {float(v): f"{v:.0f}" for v in marks_targets}
    return html.Div(
        className="sim-field",
        children=[
            html.Label(col_label(SCHEMA, feature, lang)),
            dcc.Slider(
                id=_input_id("num", feature),
                min=lo,
                max=hi,
                step=step,
                value=float(default),
                marks=marks,
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="drag",
            ),
        ],
    )


def _dropdown_for(feature: str, lang: str) -> html.Div:
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
    default = SIM_DEFAULTS.get(feature)
    return html.Div(
        className="sim-field",
        children=[
            html.Label(col_label(SCHEMA, feature, lang)),
            dcc.Dropdown(
                id=_input_id("cat", feature),
                options=options,
                value=default,
                clearable=True,
                placeholder=("未指定" if lang == "ja" else "Unspecified"),
            ),
        ],
    )


def render_simulator(lang: str) -> html.Div:
    sections = []
    sections.append(html.Div(t(SCHEMA, "sim_intro", lang),
                             style={"color": INK["500"], "marginBottom": "10px", "fontSize": "13px"}))

    sections.append(html.Div(t(SCHEMA, "sim_input_demographics", lang), className="sim-section-title"))
    for f in ["年齢", "性別"]:
        if f in FEATURE_SPEC["numeric_cols"]:
            sections.append(_slider_for(f, lang))
        else:
            sections.append(_dropdown_for(f, lang))

    sections.append(html.Div(t(SCHEMA, "sim_input_injury", lang), className="sim-section-title"))
    for f in ["外傷性_非外傷性", "対麻痺_四肢麻痺", "OPLL", "DISH", "糖尿病"]:
        sections.append(_dropdown_for(f, lang))

    sections.append(html.Div(t(SCHEMA, "sim_input_isncsci", lang), className="sim-section-title"))
    for f in ["AIS_ord", "mFrankel_ord", "NLI_ord", "RightMotorLevel_ord", "LeftMotorLevel_ord"]:
        sections.append(_slider_for(f, lang))
    for f in ["VAC", "DAP"]:
        sections.append(_dropdown_for(f, lang))

    sections.append(html.Div(t(SCHEMA, "sim_input_motor", lang), className="sim-section-title"))
    for f in ["UEMS", "LEMS", "TotalMotor"]:
        sections.append(_slider_for(f, lang))

    sections.append(html.Div(t(SCHEMA, "sim_input_sensory", lang), className="sim-section-title"))
    for f in ["LightTouchTotal", "PinPrickTotal", "RightSensoryLevel_ord", "LeftSensoryLevel_ord"]:
        sections.append(_slider_for(f, lang))

    sections.append(html.Div(("入院時の SCIM (任意)" if lang == "ja" else "Admission SCIM (optional)"),
                             className="sim-section-title"))
    for f in ["SCIM_total", "SCIM_self_care", "SCIM_respiration_sphincter", "SCIM_mobility"]:
        sections.append(_slider_for(f, lang))

    input_panel = html.Div(className="sim-input-card", children=sections)

    result_panel = html.Div(
        className="sim-result-card",
        children=[
            html.Div(id="sim-readout", className="sim-readout"),
            html.Div(
                id="sim-pi-fig",
                children=dcc.Graph(id="sim-pi-graph", config={"displayModeBar": False}),
            ),
            html.H2(
                t(SCHEMA, "sim_local_explanation", lang),
                style={"marginTop": "16px", "fontSize": "14.5px", "fontWeight": 600, "color": INK["900"]},
            ),
            dcc.Graph(id="sim-shap-graph", config={"displayModeBar": False}),
        ],
    )

    return html.Div([html.Div([input_panel, result_panel], className="sim-grid")])


# ---------- TAB: insight engine ----------
def render_insights(lang: str) -> html.Div:
    importance_card = chart_card(
        t(SCHEMA, "insight_global_importance", lang),
        dcc.Graph(figure=fg.fig_global_shap_importance(METRICS, SCHEMA, lang),
                  config={"displayModeBar": False}),
    )

    # subgroup picker
    cat_features_in_data = [c for c in CATEGORICAL_FEATURES if c in EP.columns]
    num_features_in_data = [c for c in NUMERIC_FEATURES if c in EP.columns]
    sub_options = [
        {"label": col_label(SCHEMA, c, lang), "value": c}
        for c in cat_features_in_data + num_features_in_data
    ]

    subgroup_card = chart_card(
        t(SCHEMA, "insight_subgroup_compare", lang),
        html.Div(
            [
                html.Div(
                    style={"display": "flex", "gap": "12px", "marginBottom": "8px"},
                    children=[
                        html.Div(
                            style={"flex": "1"},
                            children=[
                                html.Label(
                                    t(SCHEMA, "insight_choose_strata", lang),
                                    style={"fontSize": "12px", "color": INK["500"]},
                                ),
                                dcc.Dropdown(
                                    id="ins-subgroup-feature",
                                    options=sub_options,
                                    value="対麻痺_四肢麻痺",
                                    clearable=False,
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Graph(id="ins-subgroup-graph", config={"displayModeBar": False}),
                html.Div(id="ins-effect-size", style={"fontSize": "13px", "color": INK["700"]}),
            ]
        ),
    )

    feat_options = [
        {"label": col_label(SCHEMA, c, lang), "value": c} for c in METRICS["global_importance_top25"][:15] for c in [c["feature"]]
    ]
    # collapse to unique while preserving order
    seen, feat_opts_unique = set(), []
    for o in feat_options:
        if o["value"] not in seen:
            seen.add(o["value"])
            feat_opts_unique.append(o)
    dep_card = chart_card(
        t(SCHEMA, "insight_pair_dependence", lang),
        html.Div(
            [
                html.Div(
                    style={"marginBottom": "8px"},
                    children=[
                        html.Label(
                            t(SCHEMA, "insight_choose_feature", lang),
                            style={"fontSize": "12px", "color": INK["500"]},
                        ),
                        dcc.Dropdown(
                            id="ins-dep-feature",
                            options=feat_opts_unique,
                            value=feat_opts_unique[0]["value"] if feat_opts_unique else None,
                            clearable=False,
                        ),
                    ],
                ),
                dcc.Graph(id="ins-dep-graph", config={"displayModeBar": False}),
            ]
        ),
    )

    return html.Div(
        [
            html.Div(className="chart-row", children=[importance_card, subgroup_card]),
            html.Div(className="chart-row", children=[dep_card]),
        ]
    )


# ---------- TAB: methods ----------
def render_methods(lang: str) -> html.Div:
    cv = METRICS["cv"]
    te = METRICS["test"]
    perf_block = html.Div(
        className="methods-block",
        children=[
            html.H3(("性能" if lang == "ja" else "Performance")),
            html.P(
                f"CV  R²={cv['r2_mean']:.3f} ± {cv['r2_std']:.3f}   "
                f"RMSE={cv['rmse_mean']:.2f}   MAE={cv['mae_mean']:.2f}"
            ),
            html.P(
                f"TEST  R²={te['r2']:.3f}   RMSE={te['rmse']:.2f}   MAE={te['mae']:.2f}   "
                + ("80%予測区間カバレッジ" if lang == "ja" else "80% PI coverage")
                + f"={te['conformal_coverage_80']:.0%}   "
                + ("予測区間半幅" if lang == "ja" else "PI half-width")
                + f"=±{te['conformal_q_half_width']:.1f}"
            ),
            html.P(
                ("患者数" if lang == "ja" else "Patients")
                + f": train={te['n_train']}+calib={te['n_calib']}, test={te['n_test']}"
            ),
        ],
    )

    blocks = [
        ("methods_outcome", "methods_outcome_def"),
        ("methods_model", "methods_model_def"),
        ("methods_features", "methods_features_def"),
        ("methods_split", "methods_split_def"),
        ("methods_explainability", "methods_explainability_def"),
        ("methods_subgroup", "methods_subgroup_def"),
    ]
    md = []
    for title_key, body_key in blocks:
        md.append(
            html.Div(
                className="methods-block",
                children=[html.H3(t(SCHEMA, title_key, lang)), html.P(t(SCHEMA, body_key, lang))],
            )
        )
    md.append(perf_block)
    return html.Div(md, style={"maxWidth": "820px"})


# ---------- App ----------
def create_app() -> Dash:
    app = Dash(
        __name__,
        title="SCI Rehab Suite",
        suppress_callback_exceptions=True,
        assets_folder=str(Path(__file__).parent / "assets"),
    )
    app.layout = html.Div(
        className="app-shell",
        children=[
            dcc.Store(id="lang-store", data="ja"),
            html.Div(id="topbar-container"),
            html.Div(
                className="tabs-container",
                children=[
                    dcc.Tabs(
                        id="tabs",
                        value="overview",
                        className="dash-tabs",
                        children=[
                            dcc.Tab(value="overview", id="tab-overview", className="dash-tab",
                                    selected_className="dash-tab--selected"),
                            dcc.Tab(value="simulator", id="tab-simulator", className="dash-tab",
                                    selected_className="dash-tab--selected"),
                            dcc.Tab(value="insights", id="tab-insights", className="dash-tab",
                                    selected_className="dash-tab--selected"),
                            dcc.Tab(value="methods", id="tab-methods", className="dash-tab",
                                    selected_className="dash-tab--selected"),
                        ],
                    ),
                    html.Div(id="tab-body", className="tab-body"),
                ],
            ),
            html.Div(id="footer-note", className="footer-note"),
        ],
    )
    return app


app = create_app()
server = app.server  # for gunicorn


@callback(
    Output("lang-store", "data"),
    Input("lang-ja", "n_clicks"),
    Input("lang-en", "n_clicks"),
    State("lang-store", "data"),
    prevent_initial_call=True,
)
def update_lang(_n_ja, _n_en, _cur):  # noqa: ANN001
    return "ja" if ctx.triggered_id == "lang-ja" else "en"


@callback(
    Output("topbar-container", "children"),
    Output("tab-overview", "label"),
    Output("tab-simulator", "label"),
    Output("tab-insights", "label"),
    Output("tab-methods", "label"),
    Output("footer-note", "children"),
    Input("lang-store", "data"),
)
def update_chrome(lang):  # noqa: ANN001
    return (
        topbar(lang),
        t(SCHEMA, "tab_overview", lang),
        t(SCHEMA, "tab_simulator", lang),
        t(SCHEMA, "tab_insights", lang),
        t(SCHEMA, "tab_methods", lang),
        t(SCHEMA, "data_disclaimer", lang),
    )


@callback(
    Output("tab-body", "children"),
    Input("tabs", "value"),
    Input("lang-store", "data"),
)
def update_tab(tab, lang):  # noqa: ANN001
    if tab == "overview":
        return render_overview(lang)
    if tab == "simulator":
        return render_simulator(lang)
    if tab == "insights":
        return render_insights(lang)
    return render_methods(lang)


# ---------- simulator callback ----------
def _collect_sim_inputs(num_vals, num_ids, cat_vals, cat_ids) -> pd.DataFrame:
    row: dict[str, object] = {}
    for ident, v in zip(num_ids, num_vals, strict=False):
        row[ident["col"]] = v
    for ident, v in zip(cat_ids, cat_vals, strict=False):
        row[ident["col"]] = v
    # fill missing with defaults
    for c in FEATURE_SPEC["feature_cols"]:
        if c not in row or row[c] is None:
            row[c] = SIM_DEFAULTS.get(c)
    X = pd.DataFrame([{c: row.get(c) for c in FEATURE_SPEC["feature_cols"]}])
    for c in FEATURE_SPEC["categorical_cols"]:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for c in FEATURE_SPEC["numeric_cols"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _shap_for_row(X: pd.DataFrame) -> tuple[np.ndarray, float]:
    import shap

    expl = shap.TreeExplainer(MEDIAN_MODEL)
    values = expl.shap_values(X)
    base = (
        float(expl.expected_value)
        if np.isscalar(expl.expected_value)
        else float(expl.expected_value[0])
    )
    return values[0], base


def _fig_shap_local(values: np.ndarray, X: pd.DataFrame, base: float, lang: str) -> go.Figure:
    feat_names = X.columns.tolist()
    feat_values = X.iloc[0].tolist()
    pairs = sorted(
        zip(feat_names, values, feat_values, strict=False),
        key=lambda r: -abs(r[1]),
    )[:12][::-1]
    names = [
        f"{col_label(SCHEMA, n, lang)} = {fg.kpi_card(n,_format_value(n, v))['value'] if False else _format_value(n, v)}"
        for n, _, v in pairs
    ]
    contribs = [float(s) for _, s, _ in pairs]
    colors = ["#2c8a6b" if c >= 0 else "#a3354e" for c in contribs]
    fig = go.Figure(
        go.Bar(
            x=contribs,
            y=names,
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="%{y}<br>SHAP: %{x:+.2f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=INK["300"], width=1))
    fig.update_layout(
        height=max(280, 22 * len(pairs) + 60),
        margin=dict(l=260, r=20, t=10, b=44),
        xaxis_title=(
            "SHAP 寄与 (点)" if lang == "ja" else "SHAP contribution (pts)"
        ),
    )
    return fig


def _fig_prediction_interval(pred: float, lo: float, hi: float, lang: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[hi - lo],
            base=[lo],
            y=[t(SCHEMA, "sim_prediction_interval", lang)],
            orientation="h",
            marker=dict(color="rgba(17,122,139,0.18)", line=dict(width=0)),
            hovertemplate=f"{lo:.0f}–{hi:.0f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[pred],
            y=[t(SCHEMA, "sim_prediction_interval", lang)],
            mode="markers",
            marker=dict(color="#0c5a66", size=14, symbol="diamond"),
            hovertemplate=("予測中央値: %{x:.0f}" if lang == "ja" else "Predicted median: %{x:.0f}") + "<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=120,
        margin=dict(l=110, r=20, t=10, b=30),
        xaxis=dict(range=[0, 100], title="SCIM-III (0–100)"),
        yaxis=dict(showticklabels=True, tickfont=dict(size=12), showgrid=False),
    )
    return fig


@callback(
    Output("sim-readout", "children"),
    Output("sim-pi-graph", "figure"),
    Output("sim-shap-graph", "figure"),
    Input({"type": "num", "col": dash.ALL}, "value"),
    Input({"type": "cat", "col": dash.ALL}, "value"),
    State({"type": "num", "col": dash.ALL}, "id"),
    State({"type": "cat", "col": dash.ALL}, "id"),
    Input("lang-store", "data"),
)
def simulate(num_vals, cat_vals, num_ids, cat_ids, lang):  # noqa: ANN001
    if not num_ids and not cat_ids:
        return [], go.Figure(), go.Figure()
    X = _collect_sim_inputs(num_vals, num_ids, cat_vals, cat_ids)
    pred = float(MEDIAN_MODEL.predict(X)[0])
    pred_p10 = float(P10_MODEL.predict(X)[0])
    pred_p90 = float(P90_MODEL.predict(X)[0])
    q = float(FEATURE_SPEC.get("conformal_half_width", 0.0))
    lo = max(0.0, min(pred - q, pred_p10))
    hi = min(100.0, max(pred + q, pred_p90))

    shap_vals, base = _shap_for_row(X)
    readout = [
        html.Div(t(SCHEMA, "sim_predicted_label", lang), style={"color": INK["500"], "fontSize": "13px"}),
        html.Div(f"{pred:.0f}", className="pred"),
        html.Div("/ 100  " + t(SCHEMA, "unit_score", lang), className="pred-unit"),
        html.Div(f"{t(SCHEMA, 'sim_prediction_interval', lang)} : {lo:.0f} – {hi:.0f}", className="pi"),
    ]
    return (
        readout,
        _fig_prediction_interval(pred, lo, hi, lang),
        _fig_shap_local(shap_vals, X, base, lang),
    )


# ---------- insight engine callbacks ----------
@callback(
    Output("ins-subgroup-graph", "figure"),
    Output("ins-effect-size", "children"),
    Input("ins-subgroup-feature", "value"),
    Input("lang-store", "data"),
)
def update_subgroup(feature, lang):  # noqa: ANN001
    fig = fg.fig_subgroup_box(EP, feature, SCHEMA, lang)
    # find effect-size summary in subgroups.json
    info = next((r for r in SUBGROUPS["results"] if r["feature"] == feature and not r.get("skipped")), None)
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
    Output("ins-dep-graph", "figure"),
    Input("ins-dep-feature", "value"),
    Input("lang-store", "data"),
)
def update_dependence(feature, lang):  # noqa: ANN001
    return fg.fig_dependence(SHAP_PACK, SHAP_PACK["X_test"], feature, SCHEMA, lang)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
