"""Patient explorer tab — real-patient predictions, similarity, PDF report."""

from __future__ import annotations

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from rehab_sci.constants import AIS_ORD_TO_LETTER
from rehab_sci.dashboard import figures as fg
from rehab_sci.dashboard.compute import (
    aps_prediction_set,
    clip_scalar,
    compute_ref_predictions,
    episode_has_admission,
    episode_landmark_eligibility,
    episode_row_for_model,
    get_observed_for_outcome,
    inv_transform_scalar,
    landmark_observed_for_episode,
    landmark_voi,
    phenotype_cutoff_options,
    predict_conversion,
    predict_landmark,
    predict_phenotype_membership,
    predict_trajectory,
    resolve_aps_q,
    resolve_conformal_q,
    shap_for_row_class,
    shap_for_row_regression,
)
from rehab_sci.dashboard.figures import (
    ARCHETYPE_NAMES_EN,
    ARCHETYPE_NAMES_JA,
    PALETTE_ARCHETYPE,
    PHENOTYPE_NAMES_EN,
    PHENOTYPE_NAMES_JA,
)
from rehab_sci.dashboard.i18n import level_label, t
from rehab_sci.dashboard.layout import (
    chart_card,
    conversion_readout,
    fig_class_probabilities,
    fig_conversion_endpoints,
    fig_conversion_magnitude,
    fig_landmark_compare,
    fig_shap_local,
    fig_voi_patient,
    landmark_readout,
    voi_readout,
)
from rehab_sci.dashboard.report import generate_patient_report
from rehab_sci.dashboard.state import (
    ARCHETYPE_DATA,
    CONVERSION_BUNDLE,
    DEFAULT_OUTCOME,
    EP,
    FEATURE_SPEC,
    LANDMARK_BUNDLE,
    LONG,
    OUTCOME_BUNDLES,
    PATIENT_OPTIONS,
    PATIENT_OPTIONS_BY_ID,
    PHENOTYPE_DATA,
    SCHEMA,
    SCIM_TOTAL_BUNDLE,
)
from rehab_sci.dashboard.theme import INK
from rehab_sci.data.episodes import PATIENT_TIMELINE, patient_meta, patient_timeline
from rehab_sci.data.similarity import find_nearest
from rehab_sci.models.outcomes import OUTCOMES, OutcomeSpec


# ---------- layout helpers ----------
def _patient_picker_options(lang: str) -> list[dict]:
    para_short = {
        "TETRA": "四麻" if lang == "ja" else "Tetra",
        "PARA": "対麻" if lang == "ja" else "Para",
        "NONE": "無" if lang == "ja" else "None",
    }
    opts: list[dict] = []
    for p in PATIENT_OPTIONS:
        bits: list[str] = [f"#{p.id_number}"]
        if p.age is not None:
            bits.append(f"{p.age:.0f}歳" if lang == "ja" else f"{p.age:.0f}y")
        if p.sex:
            bits.append({"M": "M", "F": "F"}.get(p.sex, p.sex))
        if p.paralysis:
            bits.append(para_short.get(p.paralysis, p.paralysis))
        if p.ais_admit:
            bits.append(f"AIS {p.ais_admit}")
        if p.n_episodes > 1:
            bits.append(f"×{p.n_episodes}症例" if lang == "ja" else f"×{p.n_episodes}ep")
        opts.append({"label": " · ".join(bits), "value": p.id_number})
    return opts


def _episode_options_for_patient(id_number: int | None, lang: str) -> list[dict]:
    if id_number is None or id_number not in PATIENT_OPTIONS_BY_ID:
        return []
    p = PATIENT_OPTIONS_BY_ID[id_number]
    return [
        {"label": f"症例 #{kr}" if lang == "ja" else f"Episode #{kr}", "value": int(kr)}
        for kr in p.key_records
    ]


def _meta_strip(meta: dict, lang: str) -> html.Div:
    if not meta:
        return html.Div()

    def chip(label: str, value: str) -> html.Span:
        return html.Span(
            className="patient-meta-chip",
            children=[
                html.Span(label, className="patient-meta-chip-label"),
                html.Span(value, className="patient-meta-chip-value"),
            ],
        )

    age = "–" if meta["age"] is None else f"{meta['age']:.0f}"
    sex = "–" if meta["sex"] is None else level_label(SCHEMA, "sex", meta["sex"], lang)
    para = "–" if meta["paralysis"] is None else level_label(SCHEMA, "para_tetra", meta["paralysis"], lang)
    ais = "–" if meta["ais_admit"] is None else level_label(SCHEMA, "ais", meta["ais_admit"], lang)
    nli = "–" if meta["nli_admit"] is None else level_label(SCHEMA, "cord_level", meta["nli_admit"], lang)
    los = "–" if meta["los_days"] is None else f"{meta['los_days']:.0f} " + t(SCHEMA, "unit_days", lang)
    discharge_scim = "–" if meta["y_discharge_scim"] is None else f"{meta['y_discharge_scim']:.0f}"
    discharge_ais = (
        "–" if meta["y_discharge_ais"] is None
        else level_label(SCHEMA, "ais", AIS_ORD_TO_LETTER.get(int(meta["y_discharge_ais"]), "?"), lang)
    )
    chips = [
        chip(t(SCHEMA, "patient_meta_age", lang), age),
        chip(t(SCHEMA, "patient_meta_sex", lang), sex),
        chip(t(SCHEMA, "patient_meta_paralysis", lang), para),
        chip(t(SCHEMA, "patient_meta_ais_admit", lang), ais),
        chip(t(SCHEMA, "patient_meta_nli_admit", lang), nli),
        chip(t(SCHEMA, "patient_meta_los", lang), los),
        chip(t(SCHEMA, "patient_meta_discharge_scim", lang), discharge_scim),
        chip(t(SCHEMA, "patient_meta_discharge_ais", lang), discharge_ais),
    ]

    if ARCHETYPE_DATA is not None:
        kr = meta.get("key_record")
        arch_id = ARCHETYPE_DATA["assignments"].get(kr if isinstance(kr, int) else None)
        if arch_id is not None:
            names = ARCHETYPE_NAMES_JA if lang == "ja" else ARCHETYPE_NAMES_EN
            arch_color = PALETTE_ARCHETYPE[arch_id % len(PALETTE_ARCHETYPE)]
            chips.append(html.Span(
                className="patient-meta-chip archetype-chip",
                style={"borderColor": arch_color, "color": arch_color},
                children=[
                    html.Span(t(SCHEMA, "patient_archetype_label", lang), className="patient-meta-chip-label"),
                    html.Span(names[arch_id], className="patient-meta-chip-value"),
                ],
            ))

    return html.Div(className="patient-meta-row", children=chips)


def _isncsci_table(long_df: pd.DataFrame, key_record: int, lang: str) -> html.Table:
    pt = patient_timeline(long_df, key_record)
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

    head = [html.Th("")] + [html.Th(level_label(SCHEMA, "time_name", tp, lang)) for tp in nonempty]
    rows = [html.Tr(head)]
    label_map = {
        "AIS": "AIS",
        "NLI": "NLI" if lang == "en" else "神経学的高位 (NLI)",
        "UEMS": "UEMS", "LEMS": "LEMS",
        "LightTouchTotal": "LT total" if lang == "en" else "軽触 (LT) 合計",
        "PinPrickTotal": "PP total" if lang == "en" else "ピンプリック (PP) 合計",
    }
    for field in fields:
        if field not in pt.columns:
            continue
        tds = [html.Td(label_map.get(field, field), className="patient-isncsci-rowhead")]
        for tp in nonempty:
            v = pt.loc[tp].get(field)
            if field == "AIS":
                disp = "–" if pd.isna(v) else level_label(SCHEMA, "ais", str(v), lang)
            elif field == "NLI":
                disp = "–" if pd.isna(v) else level_label(SCHEMA, "cord_level", str(v), lang)
            else:
                disp = cell(v)
            tds.append(html.Td(disp))
        rows.append(html.Tr(tds))
    return html.Table(rows, className="patient-isncsci-table")


def _landmark_obs_note(observed: dict, landmark: str, lang: str) -> html.Div:
    """One-line summary of the real early-recovery scores feeding the landmark prediction."""
    if not observed:
        return html.Div(
            t(SCHEMA, "lm_no_obs_note", lang).format(L=landmark),
            className="lm-obs-empty",
        )
    sep = "、" if lang == "ja" else ", "
    parts = [f"{t(SCHEMA, f'lm_measure_{m.lower()}', lang)} {v:.0f}" for m, v in observed.items()]
    return html.Div(
        f"{t(SCHEMA, 'lm_observed_measures', lang)}: {sep.join(parts)}",
        className="lm-obs-list",
    )


def _patient_landmark_card(lang: str) -> html.Div | None:
    """Real-data dynamic-prediction card: at a chosen landmark the patient's own observed scores
    sharpen the admission-only prognosis.  Omitted entirely when the landmark bundle is absent."""
    if LANDMARK_BUNDLE is None:
        return None
    return chart_card(
        t(SCHEMA, "lm_card_heading", lang),
        html.Div([
            html.Div(t(SCHEMA, "lm_card_intro", lang), className="lm-card-intro"),
            html.Div(
                className="lm-landmark-select",
                children=[
                    html.Label(t(SCHEMA, "lm_landmark_select", lang)),
                    dcc.Dropdown(
                        id="patient-lm-landmark", options=[], value=None,
                        placeholder=t(SCHEMA, "lm_select_prompt", lang),
                        clearable=True, searchable=False,
                    ),
                ],
            ),
            html.Div(id="patient-lm-note", className="lm-obs-note"),
            dcc.Graph(id="patient-lm-graph", config={"displayModeBar": False}),
            html.Div(id="patient-lm-readout"),
            html.Hr(className="voi-divider"),
            html.Div(t(SCHEMA, "voi_card_subheading", lang), className="voi-subheading"),
            html.Div(t(SCHEMA, "voi_card_intro", lang), className="lm-card-intro"),
            dcc.Graph(id="patient-voi-graph", config={"displayModeBar": False}),
            html.Div(id="patient-voi-readout"),
            html.Div(t(SCHEMA, "lm_caption", lang), className="sim-caveat"),
        ]),
    )


# ---------- observed-trajectory phenotype (G3 part 2) ----------
_PHENO_MEASURE_LABELS = {"SCIM_total": "pheno_measure_scim", "TotalMotor": "pheno_measure_motor"}


def _phenotype_readout(res: dict, lang: str) -> html.Div:
    """Dominant phenotype + membership-weighted conditioned prognosis for one patient."""
    names = PHENOTYPE_NAMES_JA if lang == "ja" else PHENOTYPE_NAMES_EN
    dom = res["dominant"]
    dom_pct = res["membership"][dom] * 100

    def _stat(label: str, val) -> html.Div:
        txt = "–" if val is None else f"{val:.0f}"
        return html.Div(className="pheno-stat", children=[
            html.Span(label, className="pheno-stat-label"),
            html.Span(txt, className="pheno-stat-value"),
        ])

    ais_sorted = sorted(res["ais_mix"].items(), key=lambda kv: kv[1], reverse=True)
    ais_txt = " · ".join(f"{g} {v * 100:.0f}%" for g, v in ais_sorted if v >= 0.01) or "–"
    cutoff_lbl = level_label(SCHEMA, "time_name", res["cutoff"], lang)
    return html.Div(className="pheno-readout", children=[
        html.Div(className="pheno-dominant", children=[
            html.Span(t(SCHEMA, "pheno_dominant_label", lang) + ": ", className="pheno-stat-label"),
            html.Span(f"{names[dom]} ({dom_pct:.0f}%)", className="pheno-dominant-value"),
        ]),
        html.Div(t(SCHEMA, "pheno_cond_heading", lang), className="pheno-cond-heading"),
        html.Div(className="pheno-stat-row", children=[
            _stat(t(SCHEMA, "pheno_cond_scim", lang), res["exp_discharge_scim"]),
            _stat(t(SCHEMA, "pheno_cond_los", lang), res["exp_los"]),
            html.Div(className="pheno-stat", children=[
                html.Span(t(SCHEMA, "pheno_cond_ais", lang), className="pheno-stat-label"),
                html.Span(ais_txt, className="pheno-stat-value"),
            ]),
        ]),
        html.Div(
            t(SCHEMA, "pheno_obs_count", lang).format(n=res["n_obs"], cutoff=cutoff_lbl),
            className="pheno-obs-count",
        ),
    ])


def _patient_phenotype_card(lang: str) -> html.Div | None:
    """Observed-trajectory phenotype card: the patient's own early SCIM/motor curve is matched
    to the cohort recovery phenotypes; membership + conditioned prognosis sharpen as the
    observation cutoff advances.  Omitted entirely when the phenotype bundle is absent."""
    if PHENOTYPE_DATA is None:
        return None
    return chart_card(
        t(SCHEMA, "pheno_card_heading", lang),
        html.Div([
            html.Div(t(SCHEMA, "pheno_card_intro", lang), className="lm-card-intro"),
            html.Div(
                className="lm-landmark-select",
                children=[
                    html.Label(t(SCHEMA, "pheno_cutoff_select", lang)),
                    dcc.Dropdown(
                        id="patient-pheno-cutoff", options=[], value=None,
                        placeholder=t(SCHEMA, "pheno_select_prompt", lang),
                        clearable=False, searchable=False,
                    ),
                ],
            ),
            html.Div(id="patient-pheno-readout"),
            html.Div(t(SCHEMA, "pheno_membership_title", lang), className="pheno-subtitle"),
            dcc.Graph(id="patient-pheno-membership", config={"displayModeBar": False}),
            html.Div(t(SCHEMA, "pheno_overlay_title", lang), className="pheno-subtitle"),
            dcc.Graph(id="patient-pheno-graph", config={"displayModeBar": False}),
            html.Div(t(SCHEMA, "pheno_caption", lang), className="sim-caveat"),
        ]),
    )


def _patient_conversion_card(lang: str) -> html.Div | None:
    """AIS-grade conversion card: the patient's admission row drives the calibrated endpoint
    probabilities + ordinal magnitude set.  Omitted entirely when the conversion bundle is absent."""
    if CONVERSION_BUNDLE is None:
        return None
    return chart_card(
        t(SCHEMA, "conv_card_heading", lang),
        html.Div([
            html.Div(t(SCHEMA, "conv_card_intro", lang), className="lm-card-intro"),
            html.Div(id="patient-conv-readout"),
            html.Div(t(SCHEMA, "conv_endpoints_heading", lang), className="pheno-subtitle"),
            dcc.Graph(id="patient-conv-endpoints-graph", config={"displayModeBar": False}),
            html.Div(t(SCHEMA, "conv_magnitude_heading", lang), className="pheno-subtitle"),
            dcc.Graph(id="patient-conv-mag-graph", config={"displayModeBar": False}),
            html.Div(t(SCHEMA, "conv_mag_caption", lang), className="sim-caveat"),
            html.Div(t(SCHEMA, "conv_caption", lang), className="sim-caveat"),
        ]),
    )


# ---------- layout ----------
def render_patient(lang: str) -> html.Div:
    default_pid = PATIENT_OPTIONS[0].id_number if PATIENT_OPTIONS else None

    picker_card = html.Div(
        className="patient-picker-card",
        children=[
            html.Div(t(SCHEMA, "patient_intro", lang), className="patient-intro"),
            html.Div(className="patient-picker-field", children=[
                html.Label(t(SCHEMA, "patient_picker_label", lang)),
                dcc.Dropdown(
                    id="patient-id-dropdown",
                    options=_patient_picker_options(lang),
                    value=default_pid, clearable=False, searchable=True, optionHeight=36,
                ),
            ]),
            html.Div(className="patient-picker-field", children=[
                html.Label(t(SCHEMA, "patient_episode_label", lang)),
                dcc.RadioItems(
                    id="patient-episode-radio",
                    options=_episode_options_for_patient(default_pid, lang),
                    value=(
                        int(PATIENT_OPTIONS_BY_ID[default_pid].key_records[0])
                        if default_pid is not None else None
                    ),
                    inline=True, className="patient-episode-radio",
                ),
            ]),
            html.Div(className="patient-picker-field", children=[
                html.Label(t(SCHEMA, "patient_strata_label", lang)),
                dcc.RadioItems(
                    id="patient-strata-radio",
                    options=[
                        {"label": t(SCHEMA, "patient_strata_para", lang), "value": "para"},
                        {"label": t(SCHEMA, "patient_strata_para_ais", lang), "value": "para_ais"},
                    ],
                    value="para_ais", inline=True, className="patient-strata-radio",
                ),
            ]),
            html.Div(className="patient-picker-field", children=[
                html.Label(t(SCHEMA, "patient_outcome_label", lang)),
                dcc.Dropdown(
                    id="patient-outcome",
                    options=[{"label": t(SCHEMA, s.display_key, lang), "value": s.key} for s in OUTCOMES],
                    value=DEFAULT_OUTCOME, clearable=False,
                ),
            ]),
        ],
    )

    timeline_card = chart_card(
        t(SCHEMA, "patient_timeline_title", lang),
        html.Div([
            html.Div(id="patient-meta-strip"),
            dcc.Graph(id="patient-timeline-graph", config={"displayModeBar": False}),
        ]),
    )

    isncsci_card = chart_card(
        t(SCHEMA, "patient_isncsci_title", lang),
        html.Div(id="patient-isncsci-table", className="patient-isncsci-wrapper"),
    )

    prediction_card = chart_card(
        t(SCHEMA, "patient_prediction_title", lang),
        html.Div([
            html.Div(id="patient-pred-readout", className="patient-pred-readout"),
            dcc.Graph(id="patient-pred-graph", config={"displayModeBar": False}),
            html.H2(
                t(SCHEMA, "patient_local_drivers", lang),
                style={"marginTop": "16px", "fontSize": "14px", "fontWeight": 600, "color": INK["900"]},
            ),
            dcc.Graph(id="patient-shap-graph", config={"displayModeBar": False}),
            html.Div(id="patient-pred-note", className="patient-pred-note"),
            html.Div(className="patient-action-row", children=[
                html.Button(t(SCHEMA, "whatif_button", lang), id="patient-whatif-btn", n_clicks=0, className="whatif-btn"),
                html.Button(t(SCHEMA, "report_download_button", lang), id="patient-report-btn", n_clicks=0, className="report-btn"),
            ]),
        ]),
    )

    similarity_card = chart_card(
        t(SCHEMA, "patient_similarity_heading", lang),
        html.Div([
            dcc.Graph(id="patient-sim-graph", config={"displayModeBar": False}),
            html.Div(id="patient-sim-table"),
        ]),
    )

    content_children = [
        timeline_card,
        html.Div(className="chart-row", children=[isncsci_card, prediction_card]),
    ]
    lm_card = _patient_landmark_card(lang)
    if lm_card is not None:
        content_children.append(lm_card)
    pheno_card = _patient_phenotype_card(lang)
    if pheno_card is not None:
        content_children.append(pheno_card)
    conv_card = _patient_conversion_card(lang)
    if conv_card is not None:
        content_children.append(conv_card)
    content_children.append(similarity_card)

    return html.Div(
        className="patient-grid",
        children=[
            picker_card,
            html.Div(className="patient-content", children=content_children),
        ],
    )


# ---------- prediction helpers ----------
def _patient_regression(bundle: dict, X: pd.DataFrame, key_record: int, lang: str):
    spec: OutcomeSpec = bundle["spec"]
    fspec = bundle["feature_spec"]
    transform = fspec.get("transform")
    cmin = fspec.get("clip_min")
    cmax = fspec.get("clip_max")
    q_t = resolve_conformal_q(fspec, X)

    pred_t = float(bundle["median"].predict(X)[0])
    pred_p10_t = float(bundle["p10"].predict(X)[0])
    pred_p90_t = float(bundle["p90"].predict(X)[0])
    pred = clip_scalar(inv_transform_scalar(pred_t, transform), cmin, cmax)
    lo_conf = clip_scalar(inv_transform_scalar(pred_t - q_t, transform), cmin, cmax)
    hi_conf = clip_scalar(inv_transform_scalar(pred_t + q_t, transform), cmin, cmax)
    lo_q = clip_scalar(inv_transform_scalar(pred_p10_t, transform), cmin, cmax)
    hi_q = clip_scalar(inv_transform_scalar(pred_p90_t, transform), cmin, cmax)
    lo = min(lo_conf, lo_q)
    hi = max(hi_conf, hi_q)

    observed = get_observed_for_outcome(key_record, spec)
    label = t(SCHEMA, spec.display_key, lang)
    unit = t(SCHEMA, spec.unit_key, lang) if spec.unit_key else ""
    range_suffix = ""
    if spec.clip_max is not None and spec.clip_min is not None:
        range_suffix = f"/ {spec.clip_max:.0f}"

    readout: list = [
        html.Div(label, style={"color": INK["500"], "fontSize": "13px"}),
        html.Div(f"{pred:.0f}", className="pred"),
        html.Div(f"{range_suffix}  {unit}".strip(), className="pred-unit"),
        html.Div(
            f"{t(SCHEMA, 'sim_prediction_interval', lang)} : {lo:.0f} – {hi:.0f}",
            className="pi",
        ),
    ]
    note: list = []
    if observed is None:
        note.append(html.Div(t(SCHEMA, "patient_no_outcome_note", lang), className="patient-pred-empty"))
    else:
        residual = pred - observed
        readout.append(html.Div(
            f"{t(SCHEMA, 'patient_prediction_observed', lang)} : {observed:.0f} · "
            f"{t(SCHEMA, 'patient_prediction_residual', lang)} : {residual:+.0f}",
            className="pi",
        ))

    axis_label = f"{label} ({unit})" if unit else label
    pred_fig = fg.fig_patient_prediction(
        pred, lo, hi, observed, SCHEMA, lang,
        clip_min=spec.clip_min or 0.0, clip_max=spec.clip_max, axis_label=axis_label,
    )
    shap_vals, base = shap_for_row_regression(X, bundle["median"])
    shap_fig = fig_shap_local(shap_vals, X, base, lang)
    return readout, pred_fig, shap_fig, note


def _patient_multiclass(bundle: dict, X: pd.DataFrame, key_record: int, lang: str):
    spec: OutcomeSpec = bundle["spec"]
    fspec = bundle["feature_spec"]
    class_labels = list(fspec.get("class_labels", spec.class_labels))
    clf = bundle["clf"]
    proba = np.asarray(clf.predict_proba(X)[0], dtype=float)
    pred_idx = int(np.argmax(proba))
    pred_class = class_labels[pred_idx]
    pred_prob = float(proba[pred_idx])

    aps_q = resolve_aps_q(fspec, X)
    conformal_set = aps_prediction_set(proba, aps_q)
    set_letters = [class_labels[i] for i in conformal_set]

    observed_ord = get_observed_for_outcome(key_record, spec)
    label = t(SCHEMA, spec.display_key, lang)
    pcls_label = t(SCHEMA, "sim_predicted_class_label", lang)
    set_label = t(SCHEMA, "sim_conformal_set", lang)

    readout: list = [
        html.Div(label, style={"color": INK["500"], "fontSize": "13px"}),
        html.Div(f"AIS {pred_class}", className="pred"),
        html.Div(f"{pred_prob:.0%}", className="pred-unit"),
        html.Div(f"{pcls_label} : AIS {pred_class}", className="pi"),
        html.Div(f"{set_label} : {{{', '.join(set_letters)}}}", className="pi"),
    ]
    note: list = []
    if observed_ord is not None:
        obs_letter = AIS_ORD_TO_LETTER.get(int(observed_ord), "?")
        readout.append(html.Div(
            f"{t(SCHEMA, 'patient_prediction_observed', lang)} : AIS {obs_letter}",
            className="pi",
        ))
    else:
        note.append(html.Div(t(SCHEMA, "patient_no_outcome_note", lang), className="patient-pred-empty"))

    pred_fig = fig_class_probabilities(proba, class_labels, spec, lang, conformal_set)
    shap_vals, base = shap_for_row_class(X, clf, pred_idx, len(class_labels))
    shap_fig = fig_shap_local(shap_vals, X, base, lang)
    return readout, pred_fig, shap_fig, note


def _build_similarity_section(
    key_record: int, bundle: dict, X: pd.DataFrame, lang: str,
) -> tuple[go.Figure, html.Div]:
    neighbors = find_nearest(
        EP, key_record,
        feature_cols=FEATURE_SPEC["feature_cols"],
        numeric_cols=FEATURE_SPEC["numeric_cols"],
        categorical_cols=FEATURE_SPEC["categorical_cols"],
        ranges=FEATURE_SPEC["ranges"],
        k=10,
    )
    if not neighbors:
        return go.Figure(), html.Div()

    spec: OutcomeSpec = bundle["spec"]
    for n in neighbors:
        nr = EP.loc[EP["KeyRecordNumber"] == n["key_record"]]
        if not nr.empty:
            val = nr.iloc[0].get(spec.target_col)
            n["outcome_val"] = None if pd.isna(val) else float(val)
        else:
            n["outcome_val"] = None

    if spec.task == "regression":
        fspec = bundle["feature_spec"]
        transform = fspec.get("transform")
        cmin = fspec.get("clip_min")
        cmax = fspec.get("clip_max")
        q_t = resolve_conformal_q(fspec, X)
        pred_t = float(bundle["median"].predict(X)[0])
        pred = clip_scalar(inv_transform_scalar(pred_t, transform), cmin, cmax)
        lo_conf = clip_scalar(inv_transform_scalar(pred_t - q_t, transform), cmin, cmax)
        hi_conf = clip_scalar(inv_transform_scalar(pred_t + q_t, transform), cmin, cmax)
        lo_q = clip_scalar(inv_transform_scalar(float(bundle["p10"].predict(X)[0]), transform), cmin, cmax)
        hi_q = clip_scalar(inv_transform_scalar(float(bundle["p90"].predict(X)[0]), transform), cmin, cmax)
        lo = min(lo_conf, lo_q)
        hi = max(hi_conf, hi_q)
        observed = get_observed_for_outcome(key_record, spec)
        label = t(SCHEMA, spec.display_key, lang)
        unit = t(SCHEMA, spec.unit_key, lang) if spec.unit_key else ""
        axis_label = f"{label} ({unit})" if unit else label
        for n in neighbors:
            n["y_discharge_scim"] = n["outcome_val"]
        sim_fig = fg.fig_neighbor_outcomes(
            neighbors, pred, lo, hi, observed, SCHEMA, lang,
            clip_min=spec.clip_min or 0.0, clip_max=spec.clip_max, axis_label=axis_label,
        )
    else:
        clf = bundle["clf"]
        proba = np.asarray(clf.predict_proba(X)[0], dtype=float)
        observed_ord = get_observed_for_outcome(key_record, spec)
        for n in neighbors:
            n["y_discharge_ais"] = int(n["outcome_val"]) if n["outcome_val"] is not None else None
        sim_fig = fg.fig_neighbor_ais_distribution(
            neighbors, list(proba), int(observed_ord) if observed_ord is not None else None,
            SCHEMA, lang,
        )

    sim_lbl = "類似度" if lang == "ja" else "Similarity"
    age_lbl = "年齢" if lang == "ja" else "Age"
    out_lbl = t(SCHEMA, spec.display_key, lang)
    header = html.Tr([html.Th("ID"), html.Th(age_lbl), html.Th("AIS"), html.Th(out_lbl), html.Th(sim_lbl)])
    rows = [header]
    for n in neighbors:
        ov = n.get("outcome_val")
        if spec.task == "multiclass" and ov is not None:
            ov_txt = f"AIS {AIS_ORD_TO_LETTER.get(int(ov), '?')}"
        elif ov is not None:
            ov_txt = f"{ov:.0f}"
        else:
            ov_txt = "–"
        rows.append(html.Tr([
            html.Td(str(n["id_number"]) if n["id_number"] else "–"),
            html.Td(f"{n['age']:.0f}" if n["age"] is not None else "–"),
            html.Td(n["ais_admit"] or "–"),
            html.Td(ov_txt),
            html.Td(f"{n['similarity']:.0%}"),
        ]))
    sim_table = html.Table(rows, className="patient-sim-table")

    with_count = sum(1 for n in neighbors if n.get("outcome_val") is not None)
    total = len(neighbors)
    summary_text = (
        f"{total}{('名の類似患者を特定' if lang == 'ja' else ' similar patients identified')}"
        f" ({with_count}{('名に実測データあり' if lang == 'ja' else ' with observed outcome')})"
    )
    return sim_fig, html.Div([html.Div(summary_text, className="patient-sim-summary"), sim_table])


def _compute_patient_tab(key_record, strata, outcome_key, lang):
    if key_record is None:
        empty = go.Figure()
        return html.Div(), empty, html.Div(), [], empty, empty, "", empty, html.Div()
    key_record = int(key_record)

    meta = patient_meta(EP, key_record)
    meta_strip_el = _meta_strip(meta, lang)
    isncsci_table = _isncsci_table(LONG, key_record, lang)

    if not episode_has_admission(key_record):
        timeline_fig = fg.fig_patient_scim_timeline(LONG, EP, key_record, strata, SCHEMA, lang)
        return (
            meta_strip_el, timeline_fig, isncsci_table,
            [html.Div(t(SCHEMA, "patient_no_admission_note", lang), className="patient-pred-empty")],
            go.Figure(), go.Figure(), "", go.Figure(), html.Div(),
        )

    bundle = OUTCOME_BUNDLES.get(outcome_key) or SCIM_TOTAL_BUNDLE
    X = episode_row_for_model(key_record)

    traj = predict_trajectory(X)
    if traj is not None:
        ep_row = EP.loc[EP["KeyRecordNumber"] == key_record]
        baseline_scim_val = (
            float(ep_row.iloc[0]["baseline_scim"])
            if not ep_row.empty and pd.notna(ep_row.iloc[0].get("baseline_scim"))
            else None
        )
        if baseline_scim_val is not None:
            traj["timepoints"] = ["0day"] + traj["timepoints"]
            traj["pred"] = [baseline_scim_val] + traj["pred"]
            traj["lo"] = [baseline_scim_val] + traj["lo"]
            traj["hi"] = [baseline_scim_val] + traj["hi"]
        scim_bundle = OUTCOME_BUNDLES.get("scim_total")
        if scim_bundle is not None:
            fspec = scim_bundle["feature_spec"]
            q_dis = resolve_conformal_q(fspec, X)
            pred_dis_t = float(scim_bundle["median"].predict(X)[0])
            pred_dis = max(0.0, min(100.0, pred_dis_t))
            lo_dis_c = max(0.0, min(100.0, pred_dis_t - q_dis))
            hi_dis_c = max(0.0, min(100.0, pred_dis_t + q_dis))
            lo_dis_q = max(0.0, min(100.0, float(scim_bundle["p10"].predict(X)[0])))
            hi_dis_q = max(0.0, min(100.0, float(scim_bundle["p90"].predict(X)[0])))
            traj["timepoints"].append("discharge")
            traj["pred"].append(pred_dis)
            traj["lo"].append(min(lo_dis_c, lo_dis_q))
            traj["hi"].append(max(hi_dis_c, hi_dis_q))

    timeline_fig = fg.fig_patient_scim_timeline(LONG, EP, key_record, strata, SCHEMA, lang, trajectory=traj)

    if bundle["task"] == "regression":
        readout, pred_fig, shap_fig, note = _patient_regression(bundle, X, key_record, lang)
    else:
        readout, pred_fig, shap_fig, note = _patient_multiclass(bundle, X, key_record, lang)

    sim_fig, sim_table = _build_similarity_section(key_record, bundle, X, lang)
    return meta_strip_el, timeline_fig, isncsci_table, readout, pred_fig, shap_fig, note, sim_fig, sim_table


# ---------- callbacks ----------
@callback(
    Output("patient-id-dropdown", "options"),
    Output("patient-episode-radio", "options"),
    Input("patient-id-dropdown", "value"),
    Input("lang-store", "data"),
)
def update_patient_picker(id_number, lang):
    return _patient_picker_options(lang), _episode_options_for_patient(id_number, lang)


@callback(
    Output("patient-episode-radio", "value"),
    Input("patient-id-dropdown", "value"),
    State("patient-episode-radio", "value"),
)
def reset_episode_on_patient_change(id_number, current):
    if id_number is None or id_number not in PATIENT_OPTIONS_BY_ID:
        return None
    krs = [int(k) for k in PATIENT_OPTIONS_BY_ID[id_number].key_records]
    if current in krs:
        return current
    return krs[0]


@callback(
    Output("patient-meta-strip", "children"),
    Output("patient-timeline-graph", "figure"),
    Output("patient-isncsci-table", "children"),
    Output("patient-pred-readout", "children"),
    Output("patient-pred-graph", "figure"),
    Output("patient-shap-graph", "figure"),
    Output("patient-pred-note", "children"),
    Output("patient-sim-graph", "figure"),
    Output("patient-sim-table", "children"),
    Input("patient-episode-radio", "value"),
    Input("patient-strata-radio", "value"),
    Input("patient-outcome", "value"),
    Input("lang-store", "data"),
)
def update_patient_tab(key_record, strata, outcome_key, lang):
    return _compute_patient_tab(key_record, strata, outcome_key, lang)


# ---------- landmark (dynamic) prediction callbacks ----------
@callback(
    Output("patient-lm-landmark", "options"),
    Output("patient-lm-landmark", "value"),
    Input("patient-episode-radio", "value"),
)
def update_patient_landmark_options(key_record):
    """Offer only the landmarks this episode is still-admitted-eligible for; default to the latest."""
    if LANDMARK_BUNDLE is None or key_record is None:
        return [], None
    elig = episode_landmark_eligibility(int(key_record))
    opts = [{"label": lm, "value": lm} for lm in LANDMARK_BUNDLE["landmarks"] if elig.get(lm)]
    return opts, (opts[-1]["value"] if opts else None)


@callback(
    Output("patient-lm-graph", "figure"),
    Output("patient-lm-readout", "children"),
    Output("patient-lm-note", "children"),
    Output("patient-voi-graph", "figure"),
    Output("patient-voi-readout", "children"),
    Input("patient-lm-landmark", "value"),
    Input("patient-episode-radio", "value"),
    Input("patient-outcome", "value"),
    Input("lang-store", "data"),
)
def update_patient_landmark(landmark, key_record, outcome_key, lang):
    empty = go.Figure()
    if not landmark or key_record is None:
        msg = html.Div(t(SCHEMA, "lm_select_prompt", lang), className="lm-prompt")
        return empty, msg, "", empty, ""
    key_record = int(key_record)
    if not episode_has_admission(key_record):
        msg = html.Div(t(SCHEMA, "lm_not_modeled", lang), className="lm-prompt")
        return empty, msg, "", empty, ""
    x_base = episode_row_for_model(key_record)
    observed = landmark_observed_for_episode(key_record, landmark)
    okey = outcome_key or DEFAULT_OUTCOME
    result = predict_landmark(okey, landmark, x_base, observed)
    note = _landmark_obs_note(observed, landmark, lang)
    if result is None:
        msg = html.Div(t(SCHEMA, "lm_not_modeled", lang), className="lm-prompt")
        return empty, msg, note, empty, ""
    spec = (OUTCOME_BUNDLES.get(outcome_key) or SCIM_TOTAL_BUNDLE)["spec"]
    cmp_fig = fig_landmark_compare(result, spec, lang, landmark)
    cmp_read = landmark_readout(result, spec, lang)
    voi = landmark_voi(okey, landmark, x_base, observed)
    if voi is None:
        return cmp_fig, cmp_read, note, empty, ""
    return cmp_fig, cmp_read, note, fig_voi_patient(voi, spec, lang), voi_readout(voi, spec, lang)


# ---------- observed-trajectory phenotype callbacks ----------
@callback(
    Output("patient-pheno-cutoff", "options"),
    Output("patient-pheno-cutoff", "value"),
    Input("patient-episode-radio", "value"),
    Input("lang-store", "data"),
)
def update_patient_phenotype_options(key_record, lang):
    """Offer each observation-cutoff this episode is eligible for; default to the full window."""
    if PHENOTYPE_DATA is None or key_record is None:
        return [], None
    cuts = phenotype_cutoff_options(int(key_record))
    opts = [{"label": level_label(SCHEMA, "time_name", tp, lang), "value": tp} for tp in cuts]
    return opts, (cuts[-1] if cuts else None)


@callback(
    Output("patient-pheno-membership", "figure"),
    Output("patient-pheno-graph", "figure"),
    Output("patient-pheno-readout", "children"),
    Input("patient-pheno-cutoff", "value"),
    Input("patient-episode-radio", "value"),
    Input("lang-store", "data"),
)
def update_patient_phenotype(cutoff, key_record, lang):
    empty = go.Figure()
    if PHENOTYPE_DATA is None or not cutoff or key_record is None:
        return empty, empty, html.Div(t(SCHEMA, "pheno_ineligible", lang), className="lm-prompt")
    res = predict_phenotype_membership(int(key_record), cutoff)
    if res is None:
        return empty, empty, html.Div(t(SCHEMA, "pheno_ineligible", lang), className="lm-prompt")
    summaries = PHENOTYPE_DATA["summaries"]
    measures = PHENOTYPE_DATA["measures"]
    measure_labels = [t(SCHEMA, _PHENO_MEASURE_LABELS[m], lang) for m in measures]
    patient_obs = {i: res["observed"].get(m, []) for i, m in enumerate(measures)}
    membership_fig = fg.fig_phenotype_membership(res["membership"], summaries, SCHEMA, lang)
    overlay_fig = fg.fig_phenotype_curves(
        PHENOTYPE_DATA["class_means"], list(PHENOTYPE_DATA["window"]), summaries,
        measure_labels, SCHEMA, lang,
        class_support=PHENOTYPE_DATA["class_support"], patient_obs=patient_obs,
    )
    return membership_fig, overlay_fig, _phenotype_readout(res, lang)


# ---------- AIS-grade conversion callback ----------
@callback(
    Output("patient-conv-readout", "children"),
    Output("patient-conv-endpoints-graph", "figure"),
    Output("patient-conv-mag-graph", "figure"),
    Input("patient-episode-radio", "value"),
    Input("lang-store", "data"),
)
def update_patient_conversion(key_record, lang):
    empty = go.Figure()
    if CONVERSION_BUNDLE is None or key_record is None:
        return html.Div(t(SCHEMA, "conv_need_ais", lang), className="lm-prompt"), empty, empty
    key_record = int(key_record)
    if not episode_has_admission(key_record):
        return html.Div(t(SCHEMA, "conv_need_ais", lang), className="lm-prompt"), empty, empty
    X = episode_row_for_model(key_record)
    # Gate on the REAL admission grade — conversion cohort membership is undefined when AIS is
    # unrecorded, so override episode_row_for_model's cohort-default imputation with the raw value
    # (NaN -> the readout shows the "needs admission grade" prompt).
    raw_ais = EP.loc[EP["KeyRecordNumber"] == key_record, "AIS_ord"]
    X = X.copy()
    X["AIS_ord"] = float(raw_ais.iloc[0]) if (not raw_ais.empty and pd.notna(raw_ais.iloc[0])) else float("nan")
    result = predict_conversion(X)
    if result is None:
        return html.Div(t(SCHEMA, "conv_need_ais", lang), className="lm-prompt"), empty, empty
    return (
        conversion_readout(result, lang),
        fig_conversion_endpoints(result, lang),
        fig_conversion_magnitude(result, lang),
    )


# ---------- PDF report callback ----------
@callback(
    Output("report-download", "data"),
    Input("patient-report-btn", "n_clicks"),
    State("patient-episode-radio", "value"),
    State("patient-id-dropdown", "value"),
    State("patient-strata-radio", "value"),
    State("lang-store", "data"),
    prevent_initial_call=True,
)
def download_report(n_clicks, key_record, id_number, strata, lang):
    if not n_clicks or key_record is None:
        return dash.no_update
    key_record = int(key_record)
    if not episode_has_admission(key_record):
        return dash.no_update

    meta = patient_meta(EP, key_record)
    X = episode_row_for_model(key_record)

    ref_preds = compute_ref_predictions(X)
    for spec in OUTCOMES:
        p = ref_preds.get(spec.key, {})
        observed = get_observed_for_outcome(key_record, spec)
        p["observed"] = observed
        if p.get("task") == "multiclass":
            fspec = OUTCOME_BUNDLES[spec.key]["feature_spec"]
            aps_q = resolve_aps_q(fspec, X)
            class_labels = list(fspec.get("class_labels", spec.class_labels))
            proba = np.asarray(p.get("proba", []))
            if proba.size > 0:
                cs = aps_prediction_set(proba, aps_q)
                p["conformal_set_letters"] = [class_labels[i] for i in cs]

    traj = predict_trajectory(X)
    if traj is not None:
        ep_row = EP.loc[EP["KeyRecordNumber"] == key_record]
        baseline = (
            float(ep_row.iloc[0]["baseline_scim"])
            if not ep_row.empty and pd.notna(ep_row.iloc[0].get("baseline_scim"))
            else None
        )
        if baseline is not None:
            traj["timepoints"] = ["0day"] + traj["timepoints"]
            traj["pred"] = [baseline] + traj["pred"]
            traj["lo"] = [baseline] + traj["lo"]
            traj["hi"] = [baseline] + traj["hi"]
        scim_out = ref_preds.get("scim_total")
        if scim_out:
            traj["timepoints"].append("discharge")
            traj["pred"].append(scim_out["pred"])
            traj["lo"].append(scim_out["lo"])
            traj["hi"].append(scim_out["hi"])

    timeline_fig = fg.fig_patient_scim_timeline(LONG, EP, key_record, strata, SCHEMA, lang, trajectory=traj)

    scim_bundle = OUTCOME_BUNDLES.get("scim_total")
    shap_fig = None
    if scim_bundle is not None:
        shap_vals, base = shap_for_row_regression(X, scim_bundle["median"])
        shap_fig = fig_shap_local(shap_vals, X, base, lang)

    outcome_labels = [
        (s.key, t(SCHEMA, s.display_key, lang), t(SCHEMA, s.unit_key, lang) if s.unit_key else "")
        for s in OUTCOMES
    ]

    pdf_bytes = generate_patient_report(
        meta=meta, predictions=ref_preds,
        trajectory_fig=timeline_fig, shap_fig=shap_fig,
        outcome_labels=outcome_labels, lang=lang,
    )

    pid = meta.get("id_number", "unknown")
    filename = f"patient_{pid}_episode_{key_record}.pdf"
    return dcc.send_bytes(pdf_bytes, filename)
