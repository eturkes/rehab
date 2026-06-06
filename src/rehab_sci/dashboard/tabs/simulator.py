"""Simulator tab — hypothetical patient prediction + What-if counterfactual."""

from __future__ import annotations

import contextlib

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html

from rehab_sci.dashboard import figures as fg
from rehab_sci.dashboard.compute import (
    aps_prediction_set,
    clip_scalar,
    collect_sim_inputs,
    compute_ref_predictions,
    episode_has_admission,
    inv_transform_scalar,
    predict_conversion,
    predict_independence,
    predict_landmark,
    predict_multistate,
    predict_topography,
    predict_trajectory,
    resolve_aps_q,
    resolve_conformal_q,
    shap_for_row_class,
    shap_for_row_regression,
    topography_admission_grades,
)
from rehab_sci.dashboard.i18n import col_label, t
from rehab_sci.dashboard.layout import (
    conversion_readout,
    dropdown_for,
    fig_class_probabilities,
    fig_conversion_endpoints,
    fig_conversion_magnitude,
    fig_independence_profile,
    fig_landmark_compare,
    fig_multistate_conversion_personal,
    fig_multistate_trajectory,
    fig_prediction_interval,
    fig_shap_local,
    fig_topography_bodymap,
    independence_readout,
    landmark_readout,
    multistate_readout,
    number_input_for,
    topography_readout,
)
from rehab_sci.dashboard.reliability import assess_input
from rehab_sci.dashboard.state import (
    CONVERSION_BUNDLE,
    DEFAULT_OUTCOME,
    EP,
    FEATURE_SPEC,
    INDEPENDENCE_BUNDLE,
    LANDMARK_BUNDLE,
    MULTISTATE_BUNDLE,
    OUTCOME_BUNDLES,
    SCHEMA,
    SCIM_TOTAL_BUNDLE,
    SIM_DEFAULTS,
    TOPOGRAPHY_BUNDLE,
)
from rehab_sci.dashboard.theme import INK
from rehab_sci.data.episodes import episode_admission_features
from rehab_sci.models.outcomes import OUTCOMES, OutcomeSpec


# ---------- layout ----------
def render_simulator(lang: str, ref_data: dict | None = None) -> html.Div:
    # Open blank: the user supplies only known fields (blanks → unknown/NaN).
    # What-if mode prefills the reference episode's admission values instead.
    defaults: dict = {}
    if ref_data and ref_data.get("features"):
        defaults = {k: v for k, v in ref_data["features"].items() if v is not None}

    sections: list = []
    sections.append(html.Div(
        t(SCHEMA, "sim_intro", lang),
        style={"color": INK["500"], "marginBottom": "10px", "fontSize": "13px"},
    ))

    sections.append(html.Div(t(SCHEMA, "sim_input_demographics", lang), className="sim-section-title"))
    for f in ["年齢", "性別"]:
        if f in FEATURE_SPEC["numeric_cols"]:
            sections.append(number_input_for(f, lang, defaults))
        else:
            sections.append(dropdown_for(f, lang, defaults))

    sections.append(html.Div(t(SCHEMA, "sim_input_injury", lang), className="sim-section-title"))
    for f in ["外傷性_非外傷性", "対麻痺_四肢麻痺", "OPLL", "DISH", "糖尿病"]:
        sections.append(dropdown_for(f, lang, defaults))

    sections.append(html.Div(t(SCHEMA, "sim_input_isncsci", lang), className="sim-section-title"))
    for f in ["AIS_ord", "mFrankel_ord", "NLI_ord", "RightMotorLevel_ord", "LeftMotorLevel_ord"]:
        sections.append(number_input_for(f, lang, defaults))
    for f in ["VAC", "DAP"]:
        sections.append(dropdown_for(f, lang, defaults))

    sections.append(html.Div(t(SCHEMA, "sim_input_motor", lang), className="sim-section-title"))
    for f in ["UEMS", "LEMS", "TotalMotor"]:
        sections.append(number_input_for(f, lang, defaults))

    sections.append(html.Div(t(SCHEMA, "sim_input_sensory", lang), className="sim-section-title"))
    for f in ["LightTouchTotal", "PinPrickTotal", "RightSensoryLevel_ord", "LeftSensoryLevel_ord"]:
        sections.append(number_input_for(f, lang, defaults))

    sections.append(html.Div(
        "入院時の SCIM (任意)" if lang == "ja" else "Admission SCIM (optional)",
        className="sim-section-title",
    ))
    for f in ["SCIM_total", "SCIM_self_care", "SCIM_respiration_sphincter", "SCIM_mobility"]:
        sections.append(number_input_for(f, lang, defaults))

    actions = html.Div(
        className="sim-input-actions",
        children=[
            html.Button(
                t(SCHEMA, "sim_fill_defaults", lang), id="sim-fill-defaults",
                n_clicks=0, className="sim-action-btn",
            ),
            html.Button(
                t(SCHEMA, "sim_clear_all", lang), id="sim-clear-all",
                n_clicks=0, className="sim-action-btn sim-action-btn--ghost",
            ),
        ],
    )
    input_panel = html.Div(
        className="sim-input-card", children=[sections[0], actions, *sections[1:]],
    )

    outcome_selector = html.Div(
        className="sim-outcome-selector",
        children=[
            html.Label(t(SCHEMA, "sim_outcome_label", lang)),
            dcc.Dropdown(
                id="sim-outcome",
                options=[{"label": t(SCHEMA, s.display_key, lang), "value": s.key} for s in OUTCOMES],
                value=DEFAULT_OUTCOME,
                clearable=False,
                searchable=False,
            ),
        ],
    )

    result_panel = html.Div(
        className="sim-result-card",
        children=[
            html.Div(id="whatif-banner"),
            outcome_selector,
            html.Div(id="sim-readout", className="sim-readout"),
            html.Div(id="sim-reliability", className="sim-reliability"),
            html.Div(id="sim-pi-fig", children=dcc.Graph(id="sim-pi-graph", config={"displayModeBar": False})),
            html.Div(t(SCHEMA, "sim_pi_caveat", lang), className="sim-caveat"),
            html.H2(
                t(SCHEMA, "sim_local_explanation", lang),
                style={"marginTop": "16px", "fontSize": "14.5px", "fontWeight": 600, "color": INK["900"]},
            ),
            dcc.Graph(id="sim-shap-graph", config={"displayModeBar": False}),
            html.H2(
                t(SCHEMA, "sim_trajectory_heading", lang),
                style={"marginTop": "22px", "fontSize": "14.5px", "fontWeight": 600, "color": INK["900"]},
            ),
            dcc.Graph(id="sim-traj-graph", config={"displayModeBar": False}),
        ],
    )

    children: list = [html.Div([input_panel, result_panel], className="sim-grid")]
    lm_card = _landmark_card(lang)
    if lm_card is not None:
        children.append(lm_card)
    conv_card = _conversion_card(lang)
    if conv_card is not None:
        children.append(conv_card)
    ms_card = _multistate_card(lang)
    if ms_card is not None:
        children.append(ms_card)
    ind_card = _independence_card(lang)
    if ind_card is not None:
        children.append(ind_card)
    topo_card = _topography_card(lang, ref_data)
    if topo_card is not None:
        children.append(topo_card)
    return html.Div(children)


# ---------- AIS-grade conversion card ----------
def _conversion_card(lang: str) -> html.Div | None:
    """Hypothetical AIS-grade conversion card driven by the simulator's admission inputs.
    Omitted entirely when the conversion bundle is absent."""
    if CONVERSION_BUNDLE is None:
        return None
    return html.Div(
        className="lm-card conv-card",
        children=[
            html.H2(t(SCHEMA, "conv_card_heading", lang), className="lm-card-heading"),
            html.Div(t(SCHEMA, "conv_card_intro", lang), className="lm-card-intro"),
            html.Div(id="sim-conv-readout"),
            html.Div(t(SCHEMA, "conv_endpoints_heading", lang), className="sim-section-title"),
            dcc.Graph(id="sim-conv-endpoints-graph", config={"displayModeBar": False}),
            html.Div(t(SCHEMA, "conv_magnitude_heading", lang), className="sim-section-title"),
            dcc.Graph(id="sim-conv-mag-graph", config={"displayModeBar": False}),
            html.Div(t(SCHEMA, "conv_mag_caption", lang), className="sim-caveat"),
            html.Div(t(SCHEMA, "conv_caption", lang), className="sim-caveat"),
        ],
    )


# ---------- AIS multi-state recovery card ----------
def _multistate_card(lang: str) -> html.Div | None:
    """Hypothetical AIS multi-state recovery card driven by the simulator's admission inputs.
    The trajectory band + first-passage curves are cohort dynamics for the entered admission
    grade; only the improve-by-6m probability tracks the feature inputs.  Omitted entirely when
    the multi-state bundle is absent."""
    if MULTISTATE_BUNDLE is None:
        return None
    return html.Div(
        className="lm-card ms-card",
        children=[
            html.H2(t(SCHEMA, "ms_card_heading", lang), className="lm-card-heading"),
            html.Div(t(SCHEMA, "ms_card_intro", lang), className="lm-card-intro"),
            html.Div(id="sim-ms-readout"),
            html.Div(t(SCHEMA, "ms_trajectory_heading", lang), className="sim-section-title"),
            dcc.Graph(id="sim-ms-traj-graph", config={"displayModeBar": False}),
            html.Div(t(SCHEMA, "ms_conversion_heading", lang), className="sim-section-title"),
            dcc.Graph(id="sim-ms-conv-graph", config={"displayModeBar": False}),
            html.Div(t(SCHEMA, "ms_cohort_caption", lang), className="sim-caveat"),
            html.Div(t(SCHEMA, "ms_caption", lang), className="sim-caveat"),
        ],
    )


# ---------- functional-independence profile card ----------
def _independence_card(lang: str) -> html.Div | None:
    """Hypothetical functional-independence profile driven by the simulator's admission inputs:
    per-SCIM-item calibrated P(independent) bars with cohort base-rate diamonds.  No realized-
    outcome overlay (hypothetical patient).  Omitted entirely when the bundle is absent."""
    if INDEPENDENCE_BUNDLE is None:
        return None
    return html.Div(
        className="lm-card ind-card",
        children=[
            html.H2(t(SCHEMA, "ind_card_heading", lang), className="lm-card-heading"),
            html.Div(t(SCHEMA, "ind_card_intro", lang), className="lm-card-intro"),
            html.Div(id="sim-ind-readout"),
            html.Div(t(SCHEMA, "ind_profile_heading", lang), className="sim-section-title"),
            dcc.Graph(id="sim-ind-graph", config={"displayModeBar": False}),
            html.Div(t(SCHEMA, "ind_caption", lang), className="sim-caveat"),
        ],
    )


# ---------- recovery-topography card (seeded ISNCSCI worksheet) ----------
def _topo_worksheet(lang: str, seed: dict) -> html.Div:
    """ISNCSCI-worksheet-style editable admission-grade grid: rows = cord levels (rostro-caudal),
    columns = Motor / Light touch / Pin prick x Left/Right.  Each existing segment gets a small
    number input (motor 0-5, sensory 0-2), id ``{"type":"topo-seg","seg":<segment key>}``, pre-filled
    from ``seed`` (the reference patient's real admission exam) where present; blanks stay NaN."""
    segs = TOPOGRAPHY_BUNDLE["segments"]
    order = {s["level"]: s["cord_order"] for s in segs}
    levels = sorted(order, key=order.get)
    cell_for = {(s["modality"], s["side"], s["level"]): s for s in segs}
    cols = [("motor", "Left"), ("motor", "Right"), ("light_touch", "Left"),
            ("light_touch", "Right"), ("pin_prick", "Left"), ("pin_prick", "Right")]
    hdr = {"motor": "運動" if lang == "ja" else "Mot",
           "light_touch": "触覚" if lang == "ja" else "LT",
           "pin_prick": "痛覚" if lang == "ja" else "PP"}
    sd = {"Left": t(SCHEMA, "topo_side_left", lang), "Right": t(SCHEMA, "topo_side_right", lang)}

    def _cell(modality: str, side: str, level: str):
        seg = cell_for.get((modality, side, level))
        if seg is None:
            return html.Td("", className="topo-ws-empty")
        v = seed.get(seg["key"])
        return html.Td(dcc.Input(
            id={"type": "topo-seg", "seg": seg["key"]}, type="number", min=0,
            max=(5 if modality == "motor" else 2), step=1, debounce=True,
            value=(round(v) if v is not None else None), className="topo-ws-input",
        ))

    header = html.Thead(html.Tr(
        [html.Th("")] + [html.Th(f"{hdr[m]}{sd[s]}") for m, s in cols]
    ))
    body = html.Tbody([
        html.Tr([html.Td(lv, className="topo-ws-level")] + [_cell(m, s, lv) for m, s in cols])
        for lv in levels
    ])
    return html.Div(html.Table([header, body], className="topo-worksheet"),
                    className="topo-worksheet-scroll")


def _topography_card(lang: str, ref_data: dict | None = None) -> html.Div | None:
    """Hypothetical recovery-topography card: an editable per-segment admission worksheet (seeded
    from the What-if reference patient's real exam) + the 30-field form drive the calibrated
    P(milestone) body map.  Edit any segment's admission grade and watch the predicted discharge
    topography shift.  No realized-outcome overlay (hypothetical).  Omitted when bundle absent."""
    if TOPOGRAPHY_BUNDLE is None:
        return None
    seed: dict = {}
    if ref_data and ref_data.get("key_record") is not None:
        seed = topography_admission_grades(int(ref_data["key_record"]))
    mod_opts = [
        {"label": t(SCHEMA, "topo_modality_light_touch", lang), "value": "light_touch"},
        {"label": t(SCHEMA, "topo_modality_pin_prick", lang), "value": "pin_prick"},
    ]
    return html.Div(
        className="lm-card topo-card",
        children=[
            html.H2(t(SCHEMA, "topo_card_heading", lang), className="lm-card-heading"),
            html.Div(t(SCHEMA, "topo_card_intro", lang), className="lm-card-intro"),
            html.Div(t(SCHEMA, "topo_sim_worksheet_heading", lang), className="sim-section-title"),
            html.Div(t(SCHEMA, "topo_sim_worksheet_note", lang), className="sim-caveat"),
            html.Div(className="sim-input-actions", children=[
                html.Button(t(SCHEMA, "topo_sim_seed", lang), id="sim-topo-seed",
                            n_clicks=0, className="sim-action-btn"),
                html.Button(t(SCHEMA, "topo_sim_clear", lang), id="sim-topo-clear",
                            n_clicks=0, className="sim-action-btn sim-action-btn--ghost"),
            ]),
            _topo_worksheet(lang, seed),
            html.Div(id="sim-topo-readout"),
            html.Div(t(SCHEMA, "topo_atlas_heading", lang), className="sim-section-title"),
            dcc.RadioItems(id="sim-topo-modality", options=mod_opts, value="light_touch",
                           inline=True, style={"marginBottom": "4px", "fontSize": "13px"}),
            dcc.Graph(id="sim-topo-graph", config={"displayModeBar": False}),
            html.Div(t(SCHEMA, "topo_caption", lang), className="sim-caveat"),
        ],
    )


# ---------- landmark (dynamic) prediction card ----------
def _lm_obs_input(measure: str, lang: str) -> html.Div:
    return html.Div(
        className="lm-obs-field",
        children=[
            html.Label(t(SCHEMA, f"lm_measure_{measure.lower()}", lang), className="lm-obs-label"),
            dcc.Input(
                id={"type": "lmobs", "col": measure}, type="number", debounce=True,
                className="lm-obs-input", placeholder="—",
            ),
        ],
    )


def _landmark_card(lang: str) -> html.Div | None:
    """Hypothetical dynamic-prediction card: pick a landmark, enter observed scores, see the
    admission-only baseline sharpen.  Omitted entirely when the landmark bundle is absent."""
    if LANDMARK_BUNDLE is None:
        return None
    landmarks = LANDMARK_BUNDLE["landmarks"]
    measures = LANDMARK_BUNDLE["landmark_cols"]
    return html.Div(
        className="lm-card",
        children=[
            html.H2(t(SCHEMA, "lm_card_heading", lang), className="lm-card-heading"),
            html.Div(t(SCHEMA, "lm_card_intro", lang), className="lm-card-intro"),
            html.Div(
                className="lm-landmark-select",
                children=[
                    html.Label(t(SCHEMA, "lm_landmark_select", lang)),
                    dcc.Dropdown(
                        id="sim-lm-landmark",
                        options=[{"label": lm, "value": lm} for lm in landmarks],
                        value=None, placeholder=t(SCHEMA, "lm_select_prompt", lang),
                        clearable=True, searchable=False,
                    ),
                ],
            ),
            html.Div(t(SCHEMA, "lm_observed_inputs_heading", lang), className="sim-section-title"),
            html.Div(
                className="lm-obs-grid",
                children=[_lm_obs_input(m, lang) for m in measures],
            ),
            dcc.Graph(id="sim-lm-graph", config={"displayModeBar": False}),
            html.Div(id="sim-lm-readout"),
            html.Div(t(SCHEMA, "lm_caption", lang), className="sim-caveat"),
        ],
    )


# ---------- simulate helpers ----------
def _simulate_regression(bundle: dict, X: pd.DataFrame, lang: str):
    spec: OutcomeSpec = bundle["spec"]
    fspec = bundle["feature_spec"]
    transform = fspec.get("transform")
    q_t = resolve_conformal_q(fspec, X)
    cmin = fspec.get("clip_min")
    cmax = fspec.get("clip_max")

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

    shap_vals, base = shap_for_row_regression(X, bundle["median"])
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
    return readout, fig_prediction_interval(pred, lo, hi, spec, lang), fig_shap_local(shap_vals, X, base, lang)


def _simulate_multiclass(bundle: dict, X: pd.DataFrame, lang: str):
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

    shap_vals, base = shap_for_row_class(X, clf, pred_idx, len(class_labels))
    label = t(SCHEMA, spec.display_key, lang)
    pcls_label = t(SCHEMA, "sim_predicted_class_label", lang)
    set_label = t(SCHEMA, "sim_conformal_set", lang)
    readout = [
        html.Div(label, style={"color": INK["500"], "fontSize": "13px"}),
        html.Div(f"AIS {pred_class}", className="pred"),
        html.Div(f"{pred_prob:.0%}", className="pred-unit"),
        html.Div(f"{pcls_label} : AIS {pred_class}", className="pi"),
        html.Div(f"{set_label} : {{{', '.join(set_letters)}}}", className="pi"),
    ]
    return readout, fig_class_probabilities(proba, class_labels, spec, lang, conformal_set), fig_shap_local(shap_vals, X, base, lang)


# ---------- reliability badge ----------
def _reliability_badge(a: dict, lang: str) -> html.Div:
    pct = a["completeness"] * 100
    ood = a["ood_level"]
    ood_text = {
        "low": t(SCHEMA, "sim_ood_typical", lang),
        "medium": t(SCHEMA, "sim_ood_atypical", lang),
        "high": t(SCHEMA, "sim_ood_outrange", lang),
    }[ood]
    chip_children = [html.Span(ood_text)]
    flagged = a["range_violations"] or a["atypical"]
    if flagged:
        sep = "、" if lang == "ja" else ", "
        names = sep.join(col_label(SCHEMA, f["feature"], lang) for f in flagged[:4])
        chip_children.append(html.Span(f" · {names}", className="ood-detail"))
    fields_txt = (
        f"{a['n_supplied']}/{a['n_total']} 項目"
        if lang == "ja"
        else f"{a['n_supplied']}/{a['n_total']} fields"
    )
    return html.Div(
        className="sim-reliability-inner",
        children=[
            html.Div(
                t(SCHEMA, "sim_reliability_heading", lang),
                className="sim-reliability-title",
            ),
            html.Div(
                className="completeness-row",
                children=[
                    html.Div(
                        className="completeness-bar",
                        children=html.Div(
                            className=f"completeness-fill rel-{a['reliability_level']}",
                            style={"width": f"{pct:.0f}%"},
                        ),
                    ),
                    html.Span(f"{pct:.0f}% · {fields_txt}", className="completeness-text"),
                ],
            ),
            html.Div(
                t(SCHEMA, "sim_completeness_label", lang),
                className="completeness-sublabel",
            ),
            html.Div(className=f"ood-chip ood-{ood}", children=chip_children),
        ],
    )


# ---------- simulate callback ----------
@callback(
    Output("sim-readout", "children"),
    Output("sim-pi-graph", "figure"),
    Output("sim-shap-graph", "figure"),
    Output("sim-traj-graph", "figure"),
    Output("sim-reliability", "children"),
    Input({"type": "num", "col": dash.ALL}, "value"),
    Input({"type": "cat", "col": dash.ALL}, "value"),
    State({"type": "num", "col": dash.ALL}, "id"),
    State({"type": "cat", "col": dash.ALL}, "id"),
    Input("sim-outcome", "value"),
    Input("lang-store", "data"),
    State("patient-ref", "data"),
)
def simulate(num_vals, cat_vals, num_ids, cat_ids, outcome_key, lang, ref_data):
    if not num_ids and not cat_ids:
        return [], go.Figure(), go.Figure(), go.Figure(), []
    bundle = OUTCOME_BUNDLES.get(outcome_key) or SCIM_TOTAL_BUNDLE
    X = collect_sim_inputs(num_vals, num_ids, cat_vals, cat_ids)
    reliability = _reliability_badge(assess_input(X, bundle, FEATURE_SPEC), lang)

    ref_pred_for_outcome: dict | None = None
    if ref_data and ref_data.get("outcomes"):
        ref_pred_for_outcome = ref_data["outcomes"].get(outcome_key or DEFAULT_OUTCOME)

    if bundle["task"] == "regression":
        readout, pi_fig, shap_fig = _simulate_regression(bundle, X, lang)
        if ref_pred_for_outcome and ref_pred_for_outcome.get("task") == "regression":
            ref_p = ref_pred_for_outcome["pred"]
            ref_label = t(SCHEMA, "whatif_ref_label", lang)
            pi_fig.add_trace(go.Scatter(
                x=[ref_p], y=[t(SCHEMA, "sim_prediction_interval", lang)],
                mode="markers",
                marker=dict(color="#a3354e", size=12, symbol="circle",
                            line=dict(color="#fff", width=1.5)),
                hovertemplate=f"{ref_label}: %{{x:.0f}}<extra></extra>",
                showlegend=False,
            ))
            delta_label = t(SCHEMA, "whatif_delta", lang)
            current_pred = ref_p
            for item in readout:
                if hasattr(item, "className") and getattr(item, "className", "") == "pred":
                    with contextlib.suppress(TypeError, ValueError):
                        current_pred = float(item.children)
                    break
            delta = current_pred - ref_p
            readout.append(html.Div(
                f"{ref_label} : {ref_p:.0f} · {delta_label} : {delta:+.0f}",
                className="pi whatif-delta",
            ))
    else:
        readout, pi_fig, shap_fig = _simulate_multiclass(bundle, X, lang)
        if ref_pred_for_outcome and ref_pred_for_outcome.get("task") == "multiclass":
            ref_label = t(SCHEMA, "whatif_ref_label", lang)
            ref_cls = ref_pred_for_outcome["pred_class"]
            readout.append(html.Div(
                f"{ref_label} : AIS {ref_cls}",
                className="pi whatif-delta",
            ))

    traj = predict_trajectory(X)
    ref_traj = ref_data.get("trajectory") if ref_data else None
    if traj is not None:
        scim_val = None
        for ident, v in zip(num_ids, num_vals, strict=False):
            if ident.get("col") == "SCIM_total":
                scim_val = v
                break
        if scim_val is not None:
            traj["timepoints"] = ["0day"] + traj["timepoints"]
            traj["pred"] = [float(scim_val)] + traj["pred"]
            traj["lo"] = [float(scim_val)] + traj["lo"]
            traj["hi"] = [float(scim_val)] + traj["hi"]
        scim_bundle = OUTCOME_BUNDLES.get("scim_total")
        if scim_bundle is not None:
            fspec = scim_bundle["feature_spec"]
            q_dis = resolve_conformal_q(fspec, X)
            pred_dis_t = float(scim_bundle["median"].predict(X)[0])
            pred_dis = max(0.0, min(100.0, pred_dis_t))
            lo_c = max(0.0, min(100.0, pred_dis_t - q_dis))
            hi_c = max(0.0, min(100.0, pred_dis_t + q_dis))
            lo_q = max(0.0, min(100.0, float(scim_bundle["p10"].predict(X)[0])))
            hi_q = max(0.0, min(100.0, float(scim_bundle["p90"].predict(X)[0])))
            traj["timepoints"].append("discharge")
            traj["pred"].append(pred_dis)
            traj["lo"].append(min(lo_c, lo_q))
            traj["hi"].append(max(hi_c, hi_q))
        traj_fig = fg.fig_sim_trajectory(traj, SCHEMA, lang, ref_trajectory=ref_traj)
    else:
        traj_fig = go.Figure()
    return readout, pi_fig, shap_fig, traj_fig, reliability


# ---------- what-if counterfactual callbacks ----------
@callback(
    Output("patient-ref", "data"),
    Output("tabs", "value", allow_duplicate=True),
    Input("patient-whatif-btn", "n_clicks"),
    State("patient-episode-radio", "value"),
    State("patient-id-dropdown", "value"),
    prevent_initial_call=True,
)
def launch_whatif(n_clicks, key_record, id_number):
    if not n_clicks or key_record is None or not episode_has_admission(int(key_record)):
        return dash.no_update, dash.no_update
    key_record = int(key_record)
    feat = episode_admission_features(EP, key_record, FEATURE_SPEC["feature_cols"])
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
    ref_outcomes = compute_ref_predictions(X)
    ref_traj = predict_trajectory(X)
    if ref_traj is not None:
        scim_val = feat.get("SCIM_total")
        if scim_val is not None:
            ref_traj["timepoints"] = ["0day"] + ref_traj["timepoints"]
            ref_traj["pred"] = [float(scim_val)] + ref_traj["pred"]
            ref_traj["lo"] = [float(scim_val)] + ref_traj["lo"]
            ref_traj["hi"] = [float(scim_val)] + ref_traj["hi"]
        scim_out = ref_outcomes.get("scim_total")
        if scim_out:
            ref_traj["timepoints"].append("discharge")
            ref_traj["pred"].append(scim_out["pred"])
            ref_traj["lo"].append(scim_out["lo"])
            ref_traj["hi"].append(scim_out["hi"])
    serializable_feat = {
        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in feat.items()
    }
    ref = {
        "id_number": int(id_number) if id_number is not None else None,
        "key_record": key_record,
        "features": serializable_feat,
        "outcomes": ref_outcomes,
        "trajectory": ref_traj,
    }
    return ref, "simulator"


@callback(
    Output("whatif-banner", "children"),
    Input("patient-ref", "data"),
    Input("lang-store", "data"),
)
def update_whatif_banner(ref_data, lang):
    if not ref_data:
        return []
    tmpl = t(SCHEMA, "whatif_banner", lang)
    text = tmpl.replace("{id}", str(ref_data.get("id_number", "?")))
    text = text.replace("{kr}", str(ref_data.get("key_record", "?")))
    return html.Div(
        className="whatif-banner",
        children=[
            html.Span(text, className="whatif-banner-text"),
            html.Button(
                t(SCHEMA, "whatif_clear", lang),
                id="whatif-clear-btn", n_clicks=0,
                className="whatif-clear-btn",
            ),
        ],
    )


@callback(
    Output("patient-ref", "data", allow_duplicate=True),
    Input("whatif-clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_whatif(_n):
    return None


# ---------- fill / clear input helpers ----------
@callback(
    Output({"type": "num", "col": dash.ALL}, "value"),
    Output({"type": "cat", "col": dash.ALL}, "value"),
    Input("sim-fill-defaults", "n_clicks"),
    Input("sim-clear-all", "n_clicks"),
    State({"type": "num", "col": dash.ALL}, "id"),
    State({"type": "cat", "col": dash.ALL}, "id"),
    prevent_initial_call=True,
)
def fill_or_clear(_fill, _clear, num_ids, cat_ids):
    """Fill every field with the cohort default, or clear all to blank (NaN)."""
    if ctx.triggered_id == "sim-fill-defaults":
        return (
            [SIM_DEFAULTS.get(i["col"]) for i in num_ids],
            [SIM_DEFAULTS.get(i["col"]) for i in cat_ids],
        )
    return [None] * len(num_ids), [None] * len(cat_ids)


# ---------- landmark (dynamic) prediction callback ----------
@callback(
    Output("sim-lm-graph", "figure"),
    Output("sim-lm-readout", "children"),
    Input("sim-lm-landmark", "value"),
    Input({"type": "lmobs", "col": dash.ALL}, "value"),
    Input({"type": "num", "col": dash.ALL}, "value"),
    Input({"type": "cat", "col": dash.ALL}, "value"),
    Input("sim-outcome", "value"),
    Input("lang-store", "data"),
    State({"type": "lmobs", "col": dash.ALL}, "id"),
    State({"type": "num", "col": dash.ALL}, "id"),
    State({"type": "cat", "col": dash.ALL}, "id"),
)
def simulate_landmark(landmark, obs_vals, num_vals, cat_vals, outcome_key, lang,
                      obs_ids, num_ids, cat_ids):
    if not landmark or not num_ids:
        return go.Figure(), html.Div(t(SCHEMA, "lm_select_prompt", lang), className="lm-prompt")
    x_base = collect_sim_inputs(num_vals, num_ids, cat_vals, cat_ids)
    observed = {
        ident["col"]: v
        for ident, v in zip(obs_ids, obs_vals, strict=False)
        if v is not None and v != ""
    }
    result = predict_landmark(outcome_key or DEFAULT_OUTCOME, landmark, x_base, observed)
    if result is None:
        return go.Figure(), html.Div(t(SCHEMA, "lm_not_modeled", lang), className="lm-prompt")
    spec = (OUTCOME_BUNDLES.get(outcome_key) or SCIM_TOTAL_BUNDLE)["spec"]
    return (
        fig_landmark_compare(result, spec, lang, landmark),
        landmark_readout(result, spec, lang),
    )


# ---------- AIS-grade conversion callback ----------
@callback(
    Output("sim-conv-readout", "children"),
    Output("sim-conv-endpoints-graph", "figure"),
    Output("sim-conv-mag-graph", "figure"),
    Input({"type": "num", "col": dash.ALL}, "value"),
    Input({"type": "cat", "col": dash.ALL}, "value"),
    Input("lang-store", "data"),
    State({"type": "num", "col": dash.ALL}, "id"),
    State({"type": "cat", "col": dash.ALL}, "id"),
)
def simulate_conversion(num_vals, cat_vals, lang, num_ids, cat_ids):
    if not num_ids and not cat_ids:
        return html.Div(t(SCHEMA, "conv_need_ais", lang), className="lm-prompt"), go.Figure(), go.Figure()
    X = collect_sim_inputs(num_vals, num_ids, cat_vals, cat_ids)
    result = predict_conversion(X)
    if result is None:
        return html.Div(t(SCHEMA, "conv_need_ais", lang), className="lm-prompt"), go.Figure(), go.Figure()
    return (
        conversion_readout(result, lang),
        fig_conversion_endpoints(result, lang),
        fig_conversion_magnitude(result, lang),
    )


# ---------- AIS multi-state recovery callback ----------
@callback(
    Output("sim-ms-readout", "children"),
    Output("sim-ms-traj-graph", "figure"),
    Output("sim-ms-conv-graph", "figure"),
    Input({"type": "num", "col": dash.ALL}, "value"),
    Input({"type": "cat", "col": dash.ALL}, "value"),
    Input("lang-store", "data"),
    State({"type": "num", "col": dash.ALL}, "id"),
    State({"type": "cat", "col": dash.ALL}, "id"),
)
def simulate_multistate(num_vals, cat_vals, lang, num_ids, cat_ids):
    empty = go.Figure()
    if not num_ids and not cat_ids:
        return html.Div(t(SCHEMA, "ms_need_ais", lang), className="lm-prompt"), empty, empty
    X = collect_sim_inputs(num_vals, num_ids, cat_vals, cat_ids)
    result = predict_multistate(X)
    if result is None:
        return html.Div(t(SCHEMA, "ms_need_ais", lang), className="lm-prompt"), empty, empty
    return (
        multistate_readout(result, lang),
        fig_multistate_trajectory(result, lang),
        fig_multistate_conversion_personal(result, lang),
    )


# ---------- functional-independence profile callback ----------
@callback(
    Output("sim-ind-readout", "children"),
    Output("sim-ind-graph", "figure"),
    Input({"type": "num", "col": dash.ALL}, "value"),
    Input({"type": "cat", "col": dash.ALL}, "value"),
    Input("lang-store", "data"),
    State({"type": "num", "col": dash.ALL}, "id"),
    State({"type": "cat", "col": dash.ALL}, "id"),
)
def simulate_independence(num_vals, cat_vals, lang, num_ids, cat_ids):
    # Independence is predicted for everyone (no admission-grade gating); blanks stay NaN, so an
    # empty form yields a near-cohort-average profile.  No realized-outcome overlay (hypothetical).
    if not num_ids and not cat_ids:
        return independence_readout(None, lang), go.Figure()
    X = collect_sim_inputs(num_vals, num_ids, cat_vals, cat_ids)
    result = predict_independence(X)
    if result is None:
        return independence_readout(None, lang), go.Figure()
    return independence_readout(result, lang), fig_independence_profile(result, lang)


# ---------- recovery-topography callbacks ----------
@callback(
    Output({"type": "topo-seg", "seg": dash.ALL}, "value"),
    Input("sim-topo-seed", "n_clicks"),
    Input("sim-topo-clear", "n_clicks"),
    State({"type": "topo-seg", "seg": dash.ALL}, "id"),
    State("patient-ref", "data"),
    prevent_initial_call=True,
)
def topo_seed_or_clear(_seed, _clear, seg_ids, ref_data):
    """Seed the worksheet from the What-if reference patient's real admission exam, or clear it."""
    if ctx.triggered_id == "sim-topo-clear":
        return [None] * len(seg_ids)
    adm: dict = {}
    if ref_data and ref_data.get("key_record") is not None:
        adm = topography_admission_grades(int(ref_data["key_record"]))
    return [round(adm[i["seg"]]) if i["seg"] in adm else None for i in seg_ids]


@callback(
    Output("sim-topo-readout", "children"),
    Output("sim-topo-graph", "figure"),
    Input({"type": "num", "col": dash.ALL}, "value"),
    Input({"type": "cat", "col": dash.ALL}, "value"),
    Input({"type": "topo-seg", "seg": dash.ALL}, "value"),
    Input("sim-topo-modality", "value"),
    Input("lang-store", "data"),
    State({"type": "num", "col": dash.ALL}, "id"),
    State({"type": "cat", "col": dash.ALL}, "id"),
    State({"type": "topo-seg", "seg": dash.ALL}, "id"),
)
def simulate_topography(num_vals, cat_vals, seg_vals, modality, lang, num_ids, cat_ids, seg_ids):
    # The 30-field form supplies the shared admission features; the worksheet supplies each
    # segment's own admission grade (adm_self — the dominant predictor).  Blanks stay NaN.  No
    # realized-outcome overlay (hypothetical patient).
    if not num_ids and not cat_ids:
        return topography_readout(None, lang), go.Figure()
    X = collect_sim_inputs(num_vals, num_ids, cat_vals, cat_ids)
    adm = {
        ident["seg"]: float(v)
        for ident, v in zip(seg_ids, seg_vals, strict=False)
        if v is not None and v != ""
    }
    result = predict_topography(X, adm)
    if result is None:
        return topography_readout(None, lang), go.Figure()
    return (
        topography_readout(result, lang),
        fig_topography_bodymap(result, lang, sensory_modality=(modality or "light_touch")),
    )
