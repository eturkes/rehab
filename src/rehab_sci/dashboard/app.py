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
from rehab_sci.data.episodes import (
    PATIENT_TIMELINE,
    episode_admission_features,
    list_patient_options,
    patient_meta,
    patient_timeline,
)
from rehab_sci.models.outcomes import OUTCOMES, OutcomeSpec
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
# Shared feature universe — same admission features for every outcome.
FEATURE_SPEC = joblib.load(MODELS_DIR / "feature_spec.joblib")


def _load_outcome_bundle(spec: OutcomeSpec) -> dict:
    od = MODELS_DIR / spec.key
    bundle: dict = {
        "key": spec.key,
        "spec": spec,
        "task": spec.task,
        "feature_spec": joblib.load(od / "feature_spec.joblib"),
        "shap": joblib.load(od / "shap_test.joblib"),
        "metrics": METRICS["outcomes"][spec.key],
    }
    if spec.task == "regression":
        bundle["median"] = joblib.load(od / "lgbm_median.joblib")
        bundle["p10"] = joblib.load(od / "lgbm_p10.joblib")
        bundle["p90"] = joblib.load(od / "lgbm_p90.joblib")
    elif spec.task == "multiclass":
        bundle["clf"] = joblib.load(od / "lgbm_multiclass.joblib")
    return bundle


OUTCOME_BUNDLES: dict[str, dict] = {s.key: _load_outcome_bundle(s) for s in OUTCOMES}
DEFAULT_OUTCOME = "scim_total"
SCIM_TOTAL_BUNDLE = OUTCOME_BUNDLES[DEFAULT_OUTCOME]

with (MODELS_DIR / "subgroups.json").open(encoding="utf-8") as f:
    SUBGROUPS = json.load(f)

# Patient-tab picker options (one entry per IDNumber).  Built once at startup;
# the list is small (~866 patients) and stable per process.
PATIENT_OPTIONS = list_patient_options(EP)
PATIENT_OPTIONS_BY_ID = {p.id_number: p for p in PATIENT_OPTIONS}


# ---------- helpers ----------
def _split_features() -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) features in display order matched to UI sections."""
    num = [c for c in FEATURE_SPEC["numeric_cols"] if c in FEATURE_SPEC["feature_cols"]]
    cat = [c for c in FEATURE_SPEC["categorical_cols"] if c in FEATURE_SPEC["feature_cols"]]
    return num, cat


def _input_id(prefix: str, col: str) -> dict:
    return {"type": prefix, "col": col}


AIS_ORD_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}


def _resolve_conformal_q(fspec: dict, X: pd.DataFrame) -> float:
    """Resolve Mondrian conformal q for a single-row input.

    Priority: AIS group -> paralysis group -> marginal.
    """
    q_by_group = fspec.get("conformal_q_by_group")
    marginal = float(fspec.get("conformal_q_transformed", 0.0))
    if q_by_group is None or len(X) == 0:
        return marginal
    row = X.iloc[0]
    ais_qs = q_by_group.get("ais", {})
    if ais_qs and "AIS_ord" in X.columns:
        ais_val = row["AIS_ord"]
        if pd.notna(ais_val):
            letter = AIS_ORD_TO_LETTER.get(int(ais_val))
            if letter and letter in ais_qs:
                return float(ais_qs[letter])
    para_qs = q_by_group.get("paralysis", {})
    if para_qs and "対麻痺_四肢麻痺" in X.columns:
        para_val = row["対麻痺_四肢麻痺"]
        if pd.notna(para_val):
            if str(para_val) in para_qs:
                return float(para_qs[str(para_val)])
    return marginal


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

    outcome_options = [
        {"label": t(SCHEMA, s.display_key, lang), "value": s.key}
        for s in OUTCOMES
    ]
    outcome_selector = html.Div(
        className="sim-outcome-selector",
        children=[
            html.Label(t(SCHEMA, "sim_outcome_label", lang)),
            dcc.Dropdown(
                id="sim-outcome",
                options=outcome_options,
                value=DEFAULT_OUTCOME,
                clearable=False,
                searchable=False,
            ),
        ],
    )

    result_panel = html.Div(
        className="sim-result-card",
        children=[
            outcome_selector,
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


# ---------- TAB: patient explorer ----------
def _patient_picker_options(lang: str) -> list[dict]:
    """Render the IDNumber dropdown options.

    Each label encodes age / sex / paralysis / AIS so a clinician can scan-pick.
    Labels are kept on a single line — we use AIS letter only (no parenthetical
    expansion) and short paralysis tokens so 866 entries all stay readable.
    """
    para_short = {
        "TETRA": ("四麻" if lang == "ja" else "Tetra"),
        "PARA": ("対麻" if lang == "ja" else "Para"),
        "NONE": ("無" if lang == "ja" else "None"),
    }
    opts: list[dict] = []
    for p in PATIENT_OPTIONS:
        bits: list[str] = [f"#{p.id_number}"]
        if p.age is not None:
            bits.append(
                (f"{p.age:.0f}歳" if lang == "ja" else f"{p.age:.0f}y")
            )
        if p.sex:
            sex_short = {"M": "M", "F": "F"}.get(p.sex, p.sex)
            bits.append(sex_short)
        if p.paralysis:
            bits.append(para_short.get(p.paralysis, p.paralysis))
        if p.ais_admit:
            bits.append(f"AIS {p.ais_admit}")
        if p.n_episodes > 1:
            bits.append(
                (f"×{p.n_episodes}症例" if lang == "ja" else f"×{p.n_episodes}ep")
            )
        opts.append({"label": " · ".join(bits), "value": p.id_number})
    return opts


def _episode_options_for_patient(id_number: int | None, lang: str) -> list[dict]:
    if id_number is None or id_number not in PATIENT_OPTIONS_BY_ID:
        return []
    p = PATIENT_OPTIONS_BY_ID[id_number]
    return [
        {"label": (f"症例 #{kr}" if lang == "ja" else f"Episode #{kr}"), "value": int(kr)}
        for kr in p.key_records
    ]


def _meta_strip(meta: dict, lang: str) -> html.Div:
    """One-line summary chips above the timeline."""
    if not meta:
        return html.Div()

    def chip(label: str, value: str) -> html.Span:
        return html.Span(
            className="patient-meta-chip",
            children=[html.Span(label, className="patient-meta-chip-label"),
                      html.Span(value, className="patient-meta-chip-value")],
        )

    age = ("–" if meta["age"] is None else f"{meta['age']:.0f}")
    sex = ("–" if meta["sex"] is None else level_label(SCHEMA, "sex", meta["sex"], lang))
    para = (
        "–" if meta["paralysis"] is None
        else level_label(SCHEMA, "para_tetra", meta["paralysis"], lang)
    )
    ais = (
        "–" if meta["ais_admit"] is None
        else level_label(SCHEMA, "ais", meta["ais_admit"], lang)
    )
    nli = (
        "–" if meta["nli_admit"] is None
        else level_label(SCHEMA, "cord_level", meta["nli_admit"], lang)
    )
    los = ("–" if meta["los_days"] is None else f"{meta['los_days']:.0f} "
           + t(SCHEMA, "unit_days", lang))
    discharge_scim = (
        "–" if meta["y_discharge_scim"] is None
        else f"{meta['y_discharge_scim']:.0f}"
    )
    ais_letter_map = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
    discharge_ais = (
        "–" if meta["y_discharge_ais"] is None
        else level_label(
            SCHEMA, "ais",
            ais_letter_map.get(int(meta["y_discharge_ais"]), "?"),
            lang,
        )
    )
    return html.Div(
        className="patient-meta-row",
        children=[
            chip(t(SCHEMA, "patient_meta_age", lang), age),
            chip(t(SCHEMA, "patient_meta_sex", lang), sex),
            chip(t(SCHEMA, "patient_meta_paralysis", lang), para),
            chip(t(SCHEMA, "patient_meta_ais_admit", lang), ais),
            chip(t(SCHEMA, "patient_meta_nli_admit", lang), nli),
            chip(t(SCHEMA, "patient_meta_los", lang), los),
            chip(t(SCHEMA, "patient_meta_discharge_scim", lang), discharge_scim),
            chip(t(SCHEMA, "patient_meta_discharge_ais", lang), discharge_ais),
        ],
    )


def _isncsci_table(long_df: pd.DataFrame, key_record: int, lang: str) -> html.Table:
    """Per-timepoint snapshot of AIS / NLI / motor / sensory totals."""
    pt = patient_timeline(long_df, key_record)
    # Only show timepoints where at least one of these fields has data.
    fields = ["AIS", "NLI", "UEMS", "LEMS", "LightTouchTotal", "PinPrickTotal"]
    nonempty = [tp for tp in PATIENT_TIMELINE if any(
        pd.notna(pt.loc[tp].get(f)) for f in fields if f in pt.columns
    )]
    if not nonempty:
        return html.Div(t(SCHEMA, "patient_no_data", lang), className="patient-empty-note")

    def cell(v: object, fmt: str = "{:.0f}") -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
            return "–"
        try:
            return fmt.format(float(v))
        except (TypeError, ValueError):
            return str(v)

    head = [html.Th("")] + [
        html.Th(level_label(SCHEMA, "time_name", tp, lang)) for tp in nonempty
    ]
    rows = [html.Tr(head)]
    label_map = {
        "AIS": "AIS",
        "NLI": ("NLI" if lang == "en" else "神経学的高位 (NLI)"),
        "UEMS": "UEMS",
        "LEMS": "LEMS",
        "LightTouchTotal": ("LT total" if lang == "en" else "軽触 (LT) 合計"),
        "PinPrickTotal": ("PP total" if lang == "en" else "ピンプリック (PP) 合計"),
    }
    for field in fields:
        if field not in pt.columns:
            continue
        tds = [html.Td(label_map.get(field, field), className="patient-isncsci-rowhead")]
        for tp in nonempty:
            v = pt.loc[tp].get(field)
            if field in ("AIS",):
                disp = "–" if pd.isna(v) else level_label(SCHEMA, "ais", str(v), lang)
            elif field in ("NLI",):
                disp = "–" if pd.isna(v) else level_label(SCHEMA, "cord_level", str(v), lang)
            else:
                disp = cell(v)
            tds.append(html.Td(disp))
        rows.append(html.Tr(tds))
    return html.Table(rows, className="patient-isncsci-table")


def _episode_row_for_model(key_record: int) -> pd.DataFrame:
    """Build a one-row model input from an episode's admission features."""
    feat = episode_admission_features(EP, key_record, FEATURE_SPEC["feature_cols"])
    # Fall back to simulator defaults for any feature missing on this episode.
    for c in FEATURE_SPEC["feature_cols"]:
        if feat.get(c) is None:
            feat[c] = SIM_DEFAULTS.get(c)
    X = pd.DataFrame([{c: feat.get(c) for c in FEATURE_SPEC["feature_cols"]}])
    for c in FEATURE_SPEC["categorical_cols"]:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for c in FEATURE_SPEC["numeric_cols"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _episode_has_admission(key_record: int) -> bool:
    """True if the patient has any non-null admission signal for the model."""
    feat = episode_admission_features(EP, key_record, FEATURE_SPEC["feature_cols"])
    return any(v is not None for v in feat.values())


def render_patient(lang: str) -> html.Div:
    default_pid = PATIENT_OPTIONS[0].id_number if PATIENT_OPTIONS else None

    picker_card = html.Div(
        className="patient-picker-card",
        children=[
            html.Div(t(SCHEMA, "patient_intro", lang), className="patient-intro"),
            html.Div(
                className="patient-picker-field",
                children=[
                    html.Label(t(SCHEMA, "patient_picker_label", lang)),
                    dcc.Dropdown(
                        id="patient-id-dropdown",
                        options=_patient_picker_options(lang),
                        value=default_pid,
                        clearable=False,
                        searchable=True,
                        optionHeight=36,
                    ),
                ],
            ),
            html.Div(
                className="patient-picker-field",
                children=[
                    html.Label(t(SCHEMA, "patient_episode_label", lang)),
                    dcc.RadioItems(
                        id="patient-episode-radio",
                        options=_episode_options_for_patient(default_pid, lang),
                        value=(
                            int(PATIENT_OPTIONS_BY_ID[default_pid].key_records[0])
                            if default_pid is not None
                            else None
                        ),
                        inline=True,
                        className="patient-episode-radio",
                    ),
                ],
            ),
            html.Div(
                className="patient-picker-field",
                children=[
                    html.Label(t(SCHEMA, "patient_strata_label", lang)),
                    dcc.RadioItems(
                        id="patient-strata-radio",
                        options=[
                            {"label": t(SCHEMA, "patient_strata_para", lang), "value": "para"},
                            {"label": t(SCHEMA, "patient_strata_para_ais", lang), "value": "para_ais"},
                        ],
                        value="para_ais",
                        inline=True,
                        className="patient-strata-radio",
                    ),
                ],
            ),
        ],
    )

    timeline_card = chart_card(
        t(SCHEMA, "patient_timeline_title", lang),
        html.Div(
            [
                html.Div(id="patient-meta-strip"),
                dcc.Graph(id="patient-timeline-graph", config={"displayModeBar": False}),
            ]
        ),
    )

    isncsci_card = chart_card(
        t(SCHEMA, "patient_isncsci_title", lang),
        html.Div(id="patient-isncsci-table", className="patient-isncsci-wrapper"),
    )

    prediction_card = chart_card(
        t(SCHEMA, "patient_prediction_title", lang),
        html.Div(
            [
                html.Div(id="patient-pred-readout", className="patient-pred-readout"),
                dcc.Graph(id="patient-pred-graph", config={"displayModeBar": False}),
                html.H2(
                    t(SCHEMA, "patient_local_drivers", lang),
                    style={"marginTop": "16px", "fontSize": "14px",
                           "fontWeight": 600, "color": INK["900"]},
                ),
                dcc.Graph(id="patient-shap-graph", config={"displayModeBar": False}),
                html.Div(id="patient-pred-note", className="patient-pred-note"),
            ]
        ),
    )

    return html.Div(
        className="patient-grid",
        children=[
            picker_card,
            html.Div(
                className="patient-content",
                children=[
                    timeline_card,
                    html.Div(className="chart-row", children=[isncsci_card, prediction_card]),
                ],
            ),
        ],
    )


# ---------- TAB: insight engine ----------
def render_insights(lang: str) -> html.Div:
    # Insight engine is anchored on the headline outcome (SCIM-III total).  A
    # per-outcome variant is on the F4 backlog — until then the global SHAP /
    # dependence plots reflect the SCIM-total model.
    scim_metrics = METRICS["outcomes"][DEFAULT_OUTCOME]
    importance_card = chart_card(
        t(SCHEMA, "insight_global_importance", lang),
        dcc.Graph(figure=fg.fig_global_shap_importance(scim_metrics, SCHEMA, lang),
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
        {"label": col_label(SCHEMA, c, lang), "value": c}
        for c in scim_metrics["global_importance_top25"][:15]
        for c in [c["feature"]]
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
def _perf_block_regression(spec: OutcomeSpec, info: dict, lang: str) -> html.Div:
    cv = info["cv"]
    te = info["test"]
    units = (" " + t(SCHEMA, spec.unit_key, lang)) if spec.unit_key else ""
    children = [
        html.H4(t(SCHEMA, spec.display_key, lang)),
        html.P(
            f"CV  R²={cv['r2_mean']:.3f} ± {cv['r2_std']:.3f}   "
            f"RMSE={cv['rmse_mean']:.2f}{units}   "
            f"MAE={cv['mae_mean']:.2f}{units}"
        ),
        html.P(
            f"TEST  R²={te['r2']:.3f}   RMSE={te['rmse']:.2f}{units}   "
            f"MAE={te['mae']:.2f}{units}   "
            + ("80%予測区間カバレッジ" if lang == "ja" else "80% PI coverage")
            + f"={te['conformal_coverage_80']:.0%}"
        ),
        html.P(
            ("患者数" if lang == "ja" else "Patients")
            + f": train={te['n_train']}+calib={te['n_calib']}, test={te['n_test']}"
        ),
    ]
    mondrian = te.get("mondrian_coverage", {})
    if mondrian:
        parts: list[str] = []
        ais_cov = mondrian.get("ais", {})
        if ais_cov:
            ais_strs = [f"{g}={d['coverage']:.0%}(n={d['n']})" for g, d in sorted(ais_cov.items())]
            prefix = "AIS別" if lang == "ja" else "Per-AIS"
            parts.append(f"{prefix}: {', '.join(ais_strs)}")
        para_cov = mondrian.get("paralysis", {})
        if para_cov:
            para_strs = [f"{g}={d['coverage']:.0%}(n={d['n']})" for g, d in sorted(para_cov.items())]
            prefix = "麻痺別" if lang == "ja" else "Per-paralysis"
            parts.append(f"{prefix}: {', '.join(para_strs)}")
        if parts:
            label = "Mondrian 80%カバレッジ" if lang == "ja" else "Mondrian 80% coverage"
            children.append(
                html.P(
                    f"{label}:  {'   '.join(parts)}",
                    style={"fontSize": "12px", "color": INK["500"]},
                )
            )
    return html.Div(className="methods-perf-card", children=children)


def _perf_block_multiclass(spec: OutcomeSpec, info: dict, lang: str) -> html.Div:
    cv = info["cv"]
    te = info["test"]
    ord_lbl = "順序MAE" if lang == "ja" else "ordinal MAE"
    return html.Div(
        className="methods-perf-card",
        children=[
            html.H4(t(SCHEMA, spec.display_key, lang)),
            html.P(
                f"CV  acc={cv['accuracy_mean']:.3f} ± {cv['accuracy_std']:.3f}   "
                f"κ_quad={cv['kappa_quadratic_mean']:.3f} ± {cv['kappa_quadratic_std']:.3f}   "
                f"{ord_lbl}={cv['ordinal_mae_mean']:.3f}"
            ),
            html.P(
                f"TEST  acc={te['accuracy']:.3f}   κ_quad={te['kappa_quadratic']:.3f}   "
                f"{ord_lbl}={te['ordinal_mae']:.3f}"
            ),
            html.P(
                ("患者数" if lang == "ja" else "Patients")
                + f": train={te['n_train']}+val={te['n_val']}, test={te['n_test']}"
            ),
        ],
    )


def render_methods(lang: str) -> html.Div:
    perf_children: list = [
        html.H3(t(SCHEMA, "methods_per_outcome_heading", lang)),
    ]
    for spec in OUTCOMES:
        info = METRICS["outcomes"][spec.key]
        if info["task"] == "regression":
            perf_children.append(_perf_block_regression(spec, info, lang))
        else:
            perf_children.append(_perf_block_multiclass(spec, info, lang))
    perf_block = html.Div(className="methods-block", children=perf_children)

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
                            dcc.Tab(value="patient", id="tab-patient", className="dash-tab",
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
    Output("tab-patient", "label"),
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
        t(SCHEMA, "tab_patient", lang),
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
    if tab == "patient":
        return render_patient(lang)
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


def _shap_for_row_regression(X: pd.DataFrame, model) -> tuple[np.ndarray, float]:  # noqa: ANN001
    import shap

    expl = shap.TreeExplainer(model)
    values = expl.shap_values(X)
    base = (
        float(expl.expected_value)
        if np.isscalar(expl.expected_value)
        else float(expl.expected_value[0])
    )
    return values[0], base


def _shap_for_row_class(X: pd.DataFrame, clf, class_idx: int, n_classes: int) -> tuple[np.ndarray, float]:  # noqa: ANN001
    import shap

    expl = shap.TreeExplainer(clf)
    raw = expl.shap_values(X)
    if isinstance(raw, list):
        per_class = np.asarray(raw[class_idx])[0]
        base_arr = expl.expected_value
        base = (
            float(base_arr)
            if np.isscalar(base_arr)
            else float(np.asarray(base_arr).ravel()[class_idx])
        )
    else:
        arr = np.asarray(raw)
        if arr.ndim == 3 and arr.shape[0] == n_classes and arr.shape[-1] != n_classes:
            arr = np.transpose(arr, (1, 2, 0))
        per_class = arr[0, :, class_idx]
        base_arr = expl.expected_value
        if np.isscalar(base_arr):
            base = float(base_arr)
        else:
            base = float(np.asarray(base_arr).ravel()[class_idx])
    return per_class, base


def _inv_transform_scalar(x: float, transform: str | None) -> float:
    if transform == "log1p":
        return float(np.expm1(x))
    return float(x)


def _clip_scalar(x: float, lo: float | None, hi: float | None) -> float:
    if lo is not None and x < lo:
        x = lo
    if hi is not None and x > hi:
        x = hi
    return x


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


def _fig_prediction_interval(
    pred: float, lo: float, hi: float, spec: OutcomeSpec, lang: str
) -> go.Figure:
    label = t(SCHEMA, spec.display_key, lang)
    unit = t(SCHEMA, spec.unit_key, lang) if spec.unit_key else ""
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
            hovertemplate=(
                ("予測中央値: %{x:.0f}" if lang == "ja" else "Predicted median: %{x:.0f}")
                + "<extra></extra>"
            ),
            showlegend=False,
        )
    )
    # x-axis: anchor at clip_min if known; clip_max if known, else stretch around the PI.
    x_lo = float(spec.clip_min) if spec.clip_min is not None else min(0.0, lo)
    if spec.clip_max is not None:
        x_hi = float(spec.clip_max)
    else:
        x_hi = float(max(hi, pred) * 1.1 + 1.0)
    axis_title = f"{label} ({unit})" if unit else label
    fig.update_layout(
        height=120,
        margin=dict(l=110, r=20, t=10, b=30),
        xaxis=dict(range=[x_lo, x_hi], title=axis_title),
        yaxis=dict(showticklabels=True, tickfont=dict(size=12), showgrid=False),
    )
    return fig


def _fig_class_probabilities(
    proba: np.ndarray, class_labels: list[str], spec: OutcomeSpec, lang: str
) -> go.Figure:
    label = t(SCHEMA, spec.display_key, lang)
    bar_labels = [f"AIS {c}" for c in class_labels]
    fig = go.Figure(
        go.Bar(
            x=bar_labels,
            y=proba,
            marker=dict(color=PALETTE_CATEGORICAL[0]),
            text=[f"{p:.0%}" for p in proba],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=60, r=20, t=30, b=44),
        yaxis=dict(range=[0, 1.05], tickformat=".0%",
                   title=t(SCHEMA, "sim_class_probabilities", lang)),
        xaxis=dict(title=label),
    )
    return fig


def _simulate_regression(bundle: dict, X: pd.DataFrame, lang: str):
    spec: OutcomeSpec = bundle["spec"]
    fspec = bundle["feature_spec"]
    transform = fspec.get("transform")
    q_t = _resolve_conformal_q(fspec, X)
    clip_min = fspec.get("clip_min")
    clip_max = fspec.get("clip_max")

    pred_t = float(bundle["median"].predict(X)[0])
    pred_p10_t = float(bundle["p10"].predict(X)[0])
    pred_p90_t = float(bundle["p90"].predict(X)[0])
    pred = _clip_scalar(_inv_transform_scalar(pred_t, transform), clip_min, clip_max)
    lo_conf = _clip_scalar(_inv_transform_scalar(pred_t - q_t, transform), clip_min, clip_max)
    hi_conf = _clip_scalar(_inv_transform_scalar(pred_t + q_t, transform), clip_min, clip_max)
    lo_q = _clip_scalar(_inv_transform_scalar(pred_p10_t, transform), clip_min, clip_max)
    hi_q = _clip_scalar(_inv_transform_scalar(pred_p90_t, transform), clip_min, clip_max)
    # widen with raw quantile head if it sits outside the conformal half-width
    lo = min(lo_conf, lo_q)
    hi = max(hi_conf, hi_q)

    shap_vals, base = _shap_for_row_regression(X, bundle["median"])
    label = t(SCHEMA, spec.display_key, lang)
    unit = t(SCHEMA, spec.unit_key, lang) if spec.unit_key else ""
    pi_label = t(SCHEMA, "sim_prediction_interval", lang)
    range_suffix = ""
    if spec.clip_max is not None and spec.clip_min is not None:
        range_suffix = f"/ {spec.clip_max:.0f}"
    readout = [
        html.Div(label, style={"color": INK["500"], "fontSize": "13px"}),
        html.Div(f"{pred:.0f}", className="pred"),
        html.Div(f"{range_suffix}  {unit}".strip(), className="pred-unit"),
        html.Div(f"{pi_label} : {lo:.0f} – {hi:.0f}", className="pi"),
    ]
    return (
        readout,
        _fig_prediction_interval(pred, lo, hi, spec, lang),
        _fig_shap_local(shap_vals, X, base, lang),
    )


def _simulate_multiclass(bundle: dict, X: pd.DataFrame, lang: str):
    spec: OutcomeSpec = bundle["spec"]
    fspec = bundle["feature_spec"]
    class_labels = list(fspec.get("class_labels", spec.class_labels))
    clf = bundle["clf"]
    proba = np.asarray(clf.predict_proba(X)[0], dtype=float)
    pred_idx = int(np.argmax(proba))
    pred_class = class_labels[pred_idx]
    pred_prob = float(proba[pred_idx])

    shap_vals, base = _shap_for_row_class(X, clf, pred_idx, len(class_labels))
    label = t(SCHEMA, spec.display_key, lang)
    pcls_label = t(SCHEMA, "sim_predicted_class_label", lang)
    readout = [
        html.Div(label, style={"color": INK["500"], "fontSize": "13px"}),
        html.Div(f"AIS {pred_class}", className="pred"),
        html.Div(f"{pred_prob:.0%}", className="pred-unit"),
        html.Div(f"{pcls_label} : AIS {pred_class}", className="pi"),
    ]
    return (
        readout,
        _fig_class_probabilities(proba, class_labels, spec, lang),
        _fig_shap_local(shap_vals, X, base, lang),
    )


@callback(
    Output("sim-readout", "children"),
    Output("sim-pi-graph", "figure"),
    Output("sim-shap-graph", "figure"),
    Input({"type": "num", "col": dash.ALL}, "value"),
    Input({"type": "cat", "col": dash.ALL}, "value"),
    State({"type": "num", "col": dash.ALL}, "id"),
    State({"type": "cat", "col": dash.ALL}, "id"),
    Input("sim-outcome", "value"),
    Input("lang-store", "data"),
)
def simulate(num_vals, cat_vals, num_ids, cat_ids, outcome_key, lang):  # noqa: ANN001
    if not num_ids and not cat_ids:
        return [], go.Figure(), go.Figure()
    bundle = OUTCOME_BUNDLES.get(outcome_key) or SCIM_TOTAL_BUNDLE
    X = _collect_sim_inputs(num_vals, num_ids, cat_vals, cat_ids)
    if bundle["task"] == "regression":
        return _simulate_regression(bundle, X, lang)
    return _simulate_multiclass(bundle, X, lang)


# ---------- patient explorer callbacks ----------
@callback(
    Output("patient-id-dropdown", "options"),
    Output("patient-episode-radio", "options"),
    Input("patient-id-dropdown", "value"),
    Input("lang-store", "data"),
)
def update_patient_picker(id_number, lang):  # noqa: ANN001
    return (
        _patient_picker_options(lang),
        _episode_options_for_patient(id_number, lang),
    )


@callback(
    Output("patient-episode-radio", "value"),
    Input("patient-id-dropdown", "value"),
    State("patient-episode-radio", "value"),
)
def reset_episode_on_patient_change(id_number, current):  # noqa: ANN001
    if id_number is None or id_number not in PATIENT_OPTIONS_BY_ID:
        return None
    krs = [int(k) for k in PATIENT_OPTIONS_BY_ID[id_number].key_records]
    if current in krs:
        return current
    return krs[0]


def _compute_patient_tab(key_record, strata, lang):  # noqa: ANN001
    """Pure-function payload for the patient tab. Returns a 7-tuple matching the callback's Outputs.

    Split out from the @callback so it is directly callable for tests and probes.
    """
    if key_record is None:
        empty = go.Figure()
        return html.Div(), empty, html.Div(), [], empty, empty, ""
    key_record = int(key_record)

    meta = patient_meta(EP, key_record)
    meta_strip = _meta_strip(meta, lang)
    timeline_fig = fg.fig_patient_scim_timeline(LONG, EP, key_record, strata, SCHEMA, lang)
    isncsci_table = _isncsci_table(LONG, key_record, lang)

    if not _episode_has_admission(key_record):
        return (
            meta_strip,
            timeline_fig,
            isncsci_table,
            [html.Div(t(SCHEMA, "patient_no_admission_note", lang),
                      className="patient-pred-empty")],
            go.Figure(),
            go.Figure(),
            "",
        )

    X = _episode_row_for_model(key_record)
    b = SCIM_TOTAL_BUNDLE
    bfspec = b["feature_spec"]
    q = _resolve_conformal_q(bfspec, X)
    pred = float(b["median"].predict(X)[0])
    pred_p10 = float(b["p10"].predict(X)[0])
    pred_p90 = float(b["p90"].predict(X)[0])
    lo = max(0.0, min(pred - q, pred_p10))
    hi = min(100.0, max(pred + q, pred_p90))
    observed = meta.get("y_discharge_scim")

    readout_children: list = [
        html.Div(t(SCHEMA, "patient_prediction_predicted", lang),
                 style={"color": INK["500"], "fontSize": "13px"}),
        html.Div(f"{pred:.0f}", className="pred"),
        html.Div("/ 100 " + t(SCHEMA, "unit_score", lang), className="pred-unit"),
        html.Div(
            f"{t(SCHEMA, 'sim_prediction_interval', lang)} : {lo:.0f} – {hi:.0f}",
            className="pi",
        ),
    ]
    note: list = []
    if observed is None:
        note.append(html.Div(t(SCHEMA, "patient_no_outcome_note", lang),
                             className="patient-pred-empty"))
    else:
        residual = pred - observed
        readout_children.append(
            html.Div(
                f"{t(SCHEMA, 'patient_prediction_observed', lang)} : {observed:.0f} · "
                f"{t(SCHEMA, 'patient_prediction_residual', lang)} : {residual:+.0f}",
                className="pi",
            )
        )

    pred_fig = fg.fig_patient_prediction(pred, lo, hi, observed, SCHEMA, lang)
    shap_vals, base = _shap_for_row_regression(X, SCIM_TOTAL_BUNDLE["median"])
    shap_fig = _fig_shap_local(shap_vals, X, base, lang)

    return meta_strip, timeline_fig, isncsci_table, readout_children, pred_fig, shap_fig, note


@callback(
    Output("patient-meta-strip", "children"),
    Output("patient-timeline-graph", "figure"),
    Output("patient-isncsci-table", "children"),
    Output("patient-pred-readout", "children"),
    Output("patient-pred-graph", "figure"),
    Output("patient-shap-graph", "figure"),
    Output("patient-pred-note", "children"),
    Input("patient-episode-radio", "value"),
    Input("patient-strata-radio", "value"),
    Input("lang-store", "data"),
)
def update_patient_tab(key_record, strata, lang):  # noqa: ANN001
    return _compute_patient_tab(key_record, strata, lang)


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
    shap_pack = SCIM_TOTAL_BUNDLE["shap"]
    return fg.fig_dependence(shap_pack, shap_pack["X_test"], feature, SCHEMA, lang)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
