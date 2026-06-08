"""Methods tab — model documentation + per-outcome performance visualizations."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from rehab_sci.constants import AIS_ORD_TO_LETTER
from rehab_sci.dashboard import figures as fg
from rehab_sci.dashboard.compute import topography_cohort_atlas
from rehab_sci.dashboard.i18n import col_label, t
from rehab_sci.dashboard.layout import fig_topography_bodymap
from rehab_sci.dashboard.state import (
    CONVERSION,
    DATAQUALITY,
    INDEPENDENCE,
    LANDMARK,
    LEVEL_DESCENT,
    METRICS,
    MULTISTATE,
    OUTCOME_BUNDLES,
    SCHEMA,
    TEMPORAL,
    TOPOGRAPHY,
)
from rehab_sci.dashboard.theme import INK
from rehab_sci.models.outcomes import OUTCOMES, OutcomeSpec


def _perf_block_regression(spec: OutcomeSpec, info: dict, lang: str) -> html.Div:
    cv = info["cv"]
    te = info["test"]
    units = (" " + t(SCHEMA, spec.unit_key, lang)) if spec.unit_key else ""
    bundle = OUTCOME_BUNDLES.get(spec.key, {})
    shap_pack = bundle.get("shap", {})
    axis_label = t(SCHEMA, spec.display_key, lang)
    children = [
        html.H4(axis_label),
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
            children.append(html.P(
                f"{label}:  {'   '.join(parts)}",
                style={"fontSize": "12px", "color": INK["500"]},
            ))
    fig_pvo = fg.fig_pred_vs_observed(
        shap_pack, SCHEMA, lang,
        clip_min=spec.clip_min, clip_max=spec.clip_max, axis_label=axis_label,
    )
    fig_rh = fg.fig_residual_hist(shap_pack, SCHEMA, lang, axis_label=axis_label)
    if fig_pvo is not None and fig_rh is not None:
        children.append(html.Div(
            style={"display": "flex", "gap": "12px", "marginTop": "8px"},
            children=[
                dcc.Graph(figure=fig_pvo, config={"displayModeBar": False},
                          style={"flex": "1", "minWidth": "0"}),
                dcc.Graph(figure=fig_rh, config={"displayModeBar": False},
                          style={"flex": "1", "minWidth": "0"}),
            ],
        ))
    return html.Div(className="methods-perf-card", children=children)


def _perf_block_multiclass(spec: OutcomeSpec, info: dict, lang: str) -> html.Div:
    cv = info["cv"]
    te = info["test"]
    bundle = OUTCOME_BUNDLES.get(spec.key, {})
    shap_pack = bundle.get("shap", {})
    ord_lbl = "順序MAE" if lang == "ja" else "ordinal MAE"
    children = [
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
            + f": train={te['n_train']}+calib={te.get('n_calib', te.get('n_val', '?'))}, test={te['n_test']}"
        ),
    ]
    aps_q = te.get("aps_q_hat")
    if aps_q is not None:
        set_lbl = "80%予測集合" if lang == "ja" else "80% prediction set"
        cov_lbl = "カバレッジ" if lang == "ja" else "coverage"
        size_lbl = "平均集合サイズ" if lang == "ja" else "avg set size"
        children.append(html.P(
            f"APS {set_lbl}: {cov_lbl}={te['aps_coverage_80']:.0%}  "
            f"{size_lbl}={te['aps_avg_set_size']:.2f}",
        ))
        aps_mond = te.get("aps_mondrian_coverage", {})
        parts: list[str] = []
        ais_cov = aps_mond.get("ais", {})
        if ais_cov:
            ais_strs = [
                f"{g}={d['coverage']:.0%}(n={d['n']},|C|={d['avg_set_size']:.1f})"
                for g, d in sorted(ais_cov.items())
            ]
            prefix = "AIS別" if lang == "ja" else "Per-AIS"
            parts.append(f"{prefix}: {', '.join(ais_strs)}")
        para_cov = aps_mond.get("paralysis", {})
        if para_cov:
            para_strs = [
                f"{g}={d['coverage']:.0%}(n={d['n']},|C|={d['avg_set_size']:.1f})"
                for g, d in sorted(para_cov.items())
            ]
            prefix = "麻痺別" if lang == "ja" else "Per-paralysis"
            parts.append(f"{prefix}: {', '.join(para_strs)}")
        if parts:
            children.append(html.P(
                "   ".join(parts),
                style={"fontSize": "12px", "color": INK["500"]},
            ))
    fig_cm = fg.fig_confusion_matrix(shap_pack, SCHEMA, lang)
    fig_cal = fg.fig_calibration_curve(shap_pack, SCHEMA, lang)
    if fig_cm is not None and fig_cal is not None:
        children.append(html.Div(
            style={"display": "flex", "gap": "12px", "marginTop": "8px"},
            children=[
                dcc.Graph(figure=fig_cm, config={"displayModeBar": False},
                          style={"flex": "1", "minWidth": "0"}),
                dcc.Graph(figure=fig_cal, config={"displayModeBar": False},
                          style={"flex": "1", "minWidth": "0"}),
            ],
        ))
    return html.Div(className="methods-perf-card", children=children)


def _perf_block_trajectory(lang: str) -> html.Div | None:
    traj = METRICS.get("trajectory")
    if not traj:
        return None
    heading = t(SCHEMA, "methods_trajectory_heading", lang)
    rows: list[html.Tr] = [
        html.Tr([
            html.Th("時点" if lang == "ja" else "Timepoint"),
            html.Th("n"), html.Th("R²"), html.Th("RMSE"),
            html.Th("80% PI"),
        ])
    ]
    for tp in METRICS.get("trajectory_timepoints", []):
        m = traj.get(tp, {})
        rows.append(html.Tr([
            html.Td(tp),
            html.Td(str(m.get("n_total", ""))),
            html.Td(f"{m.get('r2', 0):.3f}"),
            html.Td(f"{m.get('rmse', 0):.1f}"),
            html.Td(f"{m.get('conformal_coverage_80', 0):.0%}"),
        ]))
    return html.Div(
        className="methods-perf-card",
        children=[html.H4(heading), html.Table(rows, className="patient-isncsci-table")],
    )


def _temporal_block(lang: str) -> html.Div | None:
    """F24 — out-of-time rolling-origin drift, one card per outcome."""
    tm = TEMPORAL
    if not tm or not tm.get("outcomes"):
        return None
    test_years = tm.get("config", {}).get("test_years", [])
    cov_word = "カバレッジ" if lang == "ja" else "coverage"
    children: list = [
        html.H3(t(SCHEMA, "methods_temporal_heading", lang)),
        html.P(t(SCHEMA, "methods_temporal_def", lang)),
    ]
    for spec in OUTCOMES:
        info = tm["outcomes"].get(spec.key)
        if not info or not info.get("origins"):
            continue
        s = info.get("summary", {})
        if info["task"] == "regression":
            d = s.get("r2_delta_vs_baseline")
            line = (
                f"OOT R²={s.get('r2_oot_mean'):.3f}"
                + (f" (Δ={d:+.3f})" if d is not None else "")
                + f"   {cov_word}={s.get('coverage_oot_mean'):.0%}"
            )
        else:
            acc_word = "正解率" if lang == "ja" else "acc"
            d = s.get("accuracy_delta_vs_baseline")
            line = (
                f"OOT {acc_word}={s.get('accuracy_oot_mean'):.3f}"
                + (f" (Δ={d:+.3f})" if d is not None else "")
                + f"   APS {cov_word}={s.get('aps_coverage_oot_mean'):.0%}"
            )
        card: list = [
            html.H4(t(SCHEMA, spec.display_key, lang)),
            html.P(line, style={"fontSize": "13px", "color": INK["700"]}),
        ]
        fig = fg.fig_temporal_drift(info, lang)
        if fig is not None:
            card.append(dcc.Graph(figure=fig, config={"displayModeBar": False}))
        n_or = len(info["origins"])
        if test_years and n_or < len(test_years):
            note = (
                f"テスト年 {n_or}/{len(test_years)} (残りはその年のラベル欠損によりスキップ)"
                if lang == "ja"
                else f"Test years {n_or}/{len(test_years)} (rest skipped — outcome unlabelled that year)"
            )
            card.append(html.P(note, style={"fontSize": "12px", "color": INK["500"]}))
        children.append(html.Div(className="methods-perf-card", children=card))
    return html.Div(className="methods-block", children=children)


def _landmark_block(lang: str) -> html.Div | None:
    """G1 — landmark (dynamic) prediction: value-of-observation curve, one card per outcome."""
    lm = LANDMARK
    if not lm or not lm.get("outcomes"):
        return None
    days = lm.get("landmark_days", {})
    value_word = "観測価値" if lang == "ja" else "value"
    hw_word = "PI半値幅" if lang == "ja" else "PI hw"
    children: list = [
        html.H3(t(SCHEMA, "methods_landmark_heading", lang)),
        html.P(t(SCHEMA, "methods_landmark_def", lang)),
    ]
    for spec in OUTCOMES:
        info = lm["outcomes"].get(spec.key)
        by = (info or {}).get("by_landmark") or {}
        if not by:
            continue
        lms = list(by)
        first, last = lms[0], lms[-1]
        if info["task"] == "regression":
            r0, r1 = by[first]["landmark"]["r2"], by[last]["landmark"]["r2"]
            hw0, hw1 = by[first]["landmark"]["pi_halfwidth_raw"], by[last]["landmark"]["pi_halfwidth_raw"]
            d = by[last]["landmark"]["r2"] - by[last]["baseline"]["r2"]
            line = (f"R² {first}:{r0:.2f} → {last}:{r1:.2f}    "
                    f"{value_word} ΔR²={d:+.2f}    {hw_word} {hw0:.1f}→{hw1:.1f}")
        else:
            k0, k1 = by[first]["landmark"]["kappa_quadratic"], by[last]["landmark"]["kappa_quadratic"]
            d = by[last]["landmark"]["kappa_quadratic"] - by[last]["baseline"]["kappa_quadratic"]
            line = f"κ {first}:{k0:.2f} → {last}:{k1:.2f}    {value_word} Δκ={d:+.2f}"
        card: list = [
            html.H4(t(SCHEMA, spec.display_key, lang)),
            html.P(line, style={"fontSize": "13px", "color": INK["700"]}),
        ]
        fig = fg.fig_landmark_value(info, days, lang)
        if fig is not None:
            card.append(dcc.Graph(figure=fig, config={"displayModeBar": False}))
        # G2 — per-measure value-of-information scorecard (which single observation sharpens most).
        meas_keys: set[str] = set()
        for cell in by.values():
            meas_keys |= set((cell.get("single") or {}).keys())
        if meas_keys:
            measure_labels = {m: t(SCHEMA, f"lm_measure_{m.lower()}", lang) for m in meas_keys}
            voi_fig = fg.fig_voi_scorecard(info, lang, measure_labels)
            if voi_fig is not None:
                card.append(html.P(
                    t(SCHEMA, "voi_methods_def", lang),
                    style={"fontSize": "12px", "color": INK["500"], "marginTop": "8px"},
                ))
                card.append(dcc.Graph(figure=voi_fig, config={"displayModeBar": False}))
        children.append(html.Div(className="methods-perf-card", children=card))
    children.append(html.P(
        ("各ランドマークの予測区間は周辺split-conformal (α=0.2)。検証時点数が少ないためカバレッジは80%前後で変動する。"
         "リスク集合はランドマークが進むほど縮小する (在院中の症例に限定) ため、同一集合で入院時のみ基準を併記し、観測の正味価値を示す。"
         if lang == "ja" else
         "Per-landmark PIs use marginal split-conformal (α=0.2); coverage fluctuates around 80% on the small "
         "test folds. The still-admitted risk set shrinks as the landmark advances, so the admission-only "
         "baseline is fit on the same set to isolate the net value of the observed scores."),
        style={"fontSize": "12px", "color": INK["500"]},
    ))
    return html.Div(className="methods-block", children=children)


def _conversion_endpoint_label(key: str, discharge_min: int, lang: str) -> str:
    return f"{t(SCHEMA, f'conv_endpoint_{key}', lang)} (≥{AIS_ORD_TO_LETTER[discharge_min]})"


def _conversion_endpoint_card(key: str, em: dict, lang: str) -> html.Div:
    """One binary endpoint: headline metrics + base-rate-by-grade table + reliability/SHAP figs."""
    label = _conversion_endpoint_label(key, em["discharge_min"], lang)
    metrics_line = (
        f"AUC={em['auc']:.3f}   "
        f"{t(SCHEMA, 'conv_brier', lang)}={em['brier']:.3f} / {em['brier_baseline']:.3f}   "
        f"n={em['n']} (+{em['n_pos']}, base={em['base_rate']:.0%})"
    )
    children: list = [
        html.H4(label),
        html.P(metrics_line, style={"fontSize": "13px", "color": INK["700"]}),
    ]
    rbg = em.get("rate_by_admission_grade") or {}
    if rbg:
        header = html.Tr([
            html.Th(t(SCHEMA, "conv_table_grade", lang)),
            html.Th(t(SCHEMA, "conv_table_rate", lang)),
            html.Th(t(SCHEMA, "conv_table_n", lang)),
        ])
        body = [
            html.Tr([html.Td(f"AIS {g}"), html.Td(f"{d['rate']:.0%}"), html.Td(str(d["n"]))])
            for g, d in sorted(rbg.items())
        ]
        children.append(html.Table([header, *body], className="patient-isncsci-table conv-mini-table"))
    rel = fg.fig_conversion_reliability(em, lang, label)
    shap = fg.fig_conversion_shap(em, SCHEMA, lang)
    if rel is not None and shap is not None:
        children.append(html.Div(
            style={"display": "flex", "gap": "12px", "marginTop": "8px"},
            children=[
                dcc.Graph(figure=rel, config={"displayModeBar": False}, style={"flex": "1", "minWidth": "0"}),
                dcc.Graph(figure=shap, config={"displayModeBar": False}, style={"flex": "1", "minWidth": "0"}),
            ],
        ))
    return html.Div(className="methods-perf-card", children=children)


def _conversion_block(lang: str) -> html.Div | None:
    """G4 — AIS-grade conversion: descriptive landscape + per-endpoint calibration + magnitude."""
    conv = CONVERSION
    if not conv or not conv.get("endpoints"):
        return None
    ls = conv.get("landscape", {})
    children: list = [
        html.H3(t(SCHEMA, "methods_conversion_heading", lang)),
        html.P(t(SCHEMA, "methods_conversion_def", lang)),
    ]

    # --- descriptive landscape ---
    land_children: list = [html.H4(t(SCHEMA, "methods_conversion_landscape_heading", lang))]
    summary = (
        f"{('≥1段階改善' if lang == 'ja' else '≥1-grade improvement')} {ls.get('any_improve_rate', 0):.0%}   "
        f"{('不変' if lang == 'ja' else 'stable')} {ls.get('stable_rate', 0):.0%}   "
        f"{('悪化' if lang == 'ja' else 'deterioration')} {ls.get('deteriorate_rate', 0):.0%}   "
        f"(n={ls.get('n_with_both_ais', 0)})"
    )
    land_children.append(html.P(summary, style={"fontSize": "13px", "color": INK["700"]}))
    f_land = fg.fig_conversion_landscape(conv, lang)
    f_delta = fg.fig_conversion_delta(conv, lang)
    if f_land is not None and f_delta is not None:
        land_children.append(html.Div(
            style={"display": "flex", "gap": "12px", "marginTop": "8px"},
            children=[
                dcc.Graph(figure=f_land, config={"displayModeBar": False}, style={"flex": "1", "minWidth": "0"}),
                dcc.Graph(figure=f_delta, config={"displayModeBar": False}, style={"flex": "1", "minWidth": "0"}),
            ],
        ))
    children.append(html.Div(className="methods-perf-card", children=land_children))

    # --- per-endpoint calibration + drivers ---
    children.append(html.H4(
        t(SCHEMA, "methods_conversion_endpoint_heading", lang),
        style={"marginTop": "6px"},
    ))
    for key, em in conv["endpoints"].items():
        children.append(_conversion_endpoint_card(key, em, lang))

    # --- ordinal magnitude head ---
    mag = conv.get("magnitude")
    if mag:
        ord_lbl = "順序MAE" if lang == "ja" else "ordinal MAE"
        set_lbl = "平均集合サイズ" if lang == "ja" else "avg set size"
        cov_lbl = "カバレッジ" if lang == "ja" else "coverage"
        mag_children: list = [
            html.H4(t(SCHEMA, "methods_conversion_magnitude_heading", lang)),
            html.P(
                f"acc={mag['accuracy']:.3f}   κ_quad={mag['kappa_quadratic']:.3f}   "
                f"{ord_lbl}={mag['ordinal_mae']:.3f}   n={mag['n']}",
                style={"fontSize": "13px", "color": INK["700"]},
            ),
            html.P(
                f"APS: {cov_lbl}={mag['aps_coverage_80']:.0%}   {set_lbl}={mag['aps_avg_set_size']:.2f}",
                style={"fontSize": "12px", "color": INK["500"]},
            ),
        ]
        f_cm = fg.fig_conversion_confusion(mag, lang)
        if f_cm is not None:
            mag_children.append(dcc.Graph(figure=f_cm, config={"displayModeBar": False}))
        children.append(html.Div(className="methods-perf-card", children=mag_children))

    children.append(html.P(
        t(SCHEMA, "conv_caption", lang),
        style={"fontSize": "12px", "color": INK["500"]},
    ))
    return html.Div(className="methods-block", children=children)


def _multistate_block(lang: str) -> html.Div | None:
    """G6 — AIS multi-state recovery: cohort dynamics (occupancy / first-passage / transition /
    sojourn) + the calibrated improve-by-6m covariate head (per-grade base rate + calibration +
    drivers)."""
    ms = MULTISTATE
    if not ms or not ms.get("occupancy_by_adm"):
        return None
    children: list = [
        html.H3(t(SCHEMA, "methods_multistate_heading", lang)),
        html.P(t(SCHEMA, "methods_multistate_def", lang)),
    ]

    def _fig_card(heading_key: str, caption_key: str, fig) -> html.Div | None:
        if fig is None:
            return None
        return html.Div(className="methods-perf-card", children=[
            html.H4(t(SCHEMA, heading_key, lang)),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            html.P(t(SCHEMA, caption_key, lang), style={"fontSize": "12px", "color": INK["500"]}),
        ])

    cohort_cards = [
        _fig_card("methods_ms_occupancy_heading", "methods_ms_occupancy_caption",
                  fg.fig_multistate_occupancy(ms, SCHEMA, lang)),
        _fig_card("methods_ms_conversion_heading", "methods_ms_conversion_caption",
                  fg.fig_multistate_conversion(ms, SCHEMA, lang)),
        _fig_card("methods_ms_transition_heading", "methods_ms_transition_caption",
                  fg.fig_multistate_transition(ms, lang)),
        _fig_card("methods_ms_sojourn_heading", "methods_ms_sojourn_caption",
                  fg.fig_multistate_sojourn(ms, lang)),
    ]
    children.extend(c for c in cohort_cards if c is not None)

    # --- covariate improve head: per-grade base rate + calibration + drivers ---
    ih = ms.get("improve_head")
    if ih:
        metrics_line = (
            f"AUC={ih['auc']:.3f}   "
            f"{t(SCHEMA, 'conv_brier', lang)}={ih['brier']:.3f} / {ih['brier_baseline']:.3f}   "
            f"n={ih['n']} (+{ih['n_pos']}, base={ih['base_rate']:.0%})"
        )
        imp_children: list = [
            html.H4(t(SCHEMA, "methods_ms_improve_heading", lang)),
            html.P(metrics_line, style={"fontSize": "13px", "color": INK["700"]}),
            html.P(t(SCHEMA, "methods_ms_improve_caption", lang),
                   style={"fontSize": "12px", "color": INK["500"]}),
        ]
        base = fg.fig_multistate_improve_base(ms, lang)
        if base is not None:
            imp_children.append(dcc.Graph(figure=base, config={"displayModeBar": False}))
        rel = fg.fig_conversion_reliability(ih, lang, t(SCHEMA, "methods_ms_calibration_heading", lang))
        shap = fg.fig_conversion_shap(ih, SCHEMA, lang)
        if rel is not None and shap is not None:
            imp_children.append(html.Div(
                style={"display": "flex", "gap": "12px", "marginTop": "8px"},
                children=[
                    dcc.Graph(figure=rel, config={"displayModeBar": False}, style={"flex": "1", "minWidth": "0"}),
                    dcc.Graph(figure=shap, config={"displayModeBar": False}, style={"flex": "1", "minWidth": "0"}),
                ],
            ))
        children.append(html.Div(className="methods-perf-card", children=imp_children))

    children.append(html.P(
        t(SCHEMA, "ms_caption", lang),
        style={"fontSize": "12px", "color": INK["500"]},
    ))
    return html.Div(className="methods-block", children=children)


def _independence_block(lang: str) -> html.Div | None:
    """G7 — functional-independence profile: per-item scorecard + admission-AIS landscape +
    all-heads calibration overlay + driver heatmap, plus the interactive per-item drilldown
    (raw-vs-calibrated reliability + SHAP drivers for the selected item)."""
    ind = INDEPENDENCE
    if not ind or not ind.get("heads"):
        return None
    summary = ind.get("summary", {})
    n_items = summary.get("n_items_modeled", len(ind["heads"]))
    mean_auc = summary.get("mean_auc")
    children: list = [
        html.H3(t(SCHEMA, "methods_independence_heading", lang)),
        html.P(t(SCHEMA, "methods_independence_def", lang)),
    ]
    if mean_auc is not None:
        children.append(html.P(
            f"{n_items} {'項目' if lang == 'ja' else 'items'}   "
            f"{'平均 AUC' if lang == 'ja' else 'mean AUC'}={mean_auc:.3f}",
            style={"fontSize": "13px", "color": INK["700"]},
        ))

    def _fig_card(heading_key: str, caption_key: str, fig) -> html.Div | None:
        if fig is None:
            return None
        return html.Div(className="methods-perf-card", children=[
            html.H4(t(SCHEMA, heading_key, lang)),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            html.P(t(SCHEMA, caption_key, lang), style={"fontSize": "12px", "color": INK["500"]}),
        ])

    cards = [
        _fig_card("methods_ind_scorecard_heading", "methods_ind_scorecard_caption",
                  fg.fig_independence_scorecard(ind, SCHEMA, lang)),
        _fig_card("methods_ind_landscape_heading", "methods_ind_landscape_caption",
                  fg.fig_independence_landscape(ind, SCHEMA, lang)),
        _fig_card("methods_ind_calibration_heading", "methods_ind_calibration_caption",
                  fg.fig_independence_calibration(ind, SCHEMA, lang)),
        _fig_card("methods_ind_shap_heading", "methods_ind_shap_caption",
                  fg.fig_independence_shap_heatmap(ind, SCHEMA, lang)),
    ]
    children.extend(c for c in cards if c is not None)

    # --- interactive per-item drilldown (raw-vs-calibrated reliability + SHAP, reused from G4) ---
    options = [
        {"label": col_label(SCHEMA, ind["heads"][it["key"]]["col"], lang), "value": it["key"]}
        for it in ind["items"] if it["key"] in ind["heads"]
    ]
    children.append(html.Div(className="methods-perf-card", children=[
        html.H4(t(SCHEMA, "methods_ind_drilldown_heading", lang)),
        html.P(t(SCHEMA, "methods_ind_drilldown_caption", lang),
               style={"fontSize": "12px", "color": INK["500"]}),
        dcc.Dropdown(id="methods-ind-item", options=options,
                     value=(options[0]["value"] if options else None), clearable=False,
                     style={"maxWidth": "340px", "marginBottom": "8px"}),
        html.Div(style={"display": "flex", "gap": "12px"}, children=[
            dcc.Graph(id="methods-ind-rel-graph", config={"displayModeBar": False},
                      style={"flex": "1", "minWidth": "0"}),
            dcc.Graph(id="methods-ind-shap-graph", config={"displayModeBar": False},
                      style={"flex": "1", "minWidth": "0"}),
        ]),
    ]))

    children.append(html.P(t(SCHEMA, "ind_caption", lang),
                           style={"fontSize": "12px", "color": INK["500"]}))
    return html.Div(className="methods-block", children=children)


@callback(
    Output("methods-ind-rel-graph", "figure"),
    Output("methods-ind-shap-graph", "figure"),
    Input("methods-ind-item", "value"),
    Input("lang-store", "data"),
)
def update_methods_independence_item(item_key, lang):
    """Per-item drilldown: the selected independence head's raw-vs-Platt reliability curve and
    in-sample SHAP drivers (reusing the G4 conversion reliability/SHAP figures)."""
    if not INDEPENDENCE or not item_key:
        return go.Figure(), go.Figure()
    head = INDEPENDENCE["heads"].get(item_key)
    if head is None:
        return go.Figure(), go.Figure()
    label = col_label(SCHEMA, head["col"], lang)
    rel = fg.fig_conversion_reliability(head, lang, label) or go.Figure()
    shap = fg.fig_conversion_shap(head, SCHEMA, lang) or go.Figure()
    return rel, shap


def _topo_seg_label(entry: dict, lang: str) -> str:
    """Compact 'modality · side · level' label for one topography segment."""
    short = {"motor": ("運動", "Motor"), "light_touch": ("触覚", "LT"),
             "pin_prick": ("痛覚", "PP")}[entry["modality"]][0 if lang == "ja" else 1]
    side = t(SCHEMA, "topo_side_" + ("left" if entry["side"] == "Left" else "right"), lang)
    return f"{short} · {side} · {entry['level']}"


def _topography_block(lang: str) -> html.Div | None:
    """G8 — recovery topography map: the cohort body-map atlas (modality-toggleable dermatome
    silhouette + motor myotome ladder), pooled per-modality calibration, per-segment discrimination
    scorecard, per-modality SHAP drivers, and the interactive per-segment drilldown."""
    topo = TOPOGRAPHY
    if not topo or not topo.get("segments"):
        return None
    summ = topo.get("modality_summary", {})
    children: list = [
        html.H3(t(SCHEMA, "methods_topography_heading", lang)),
        html.P(t(SCHEMA, "methods_topography_def", lang)),
    ]
    aucs = [summ.get(m, {}).get("mean_auc") for m in ("motor", "light_touch", "pin_prick")]
    if all(a is not None for a in aucs):
        children.append(html.P(
            f"132 {'セグメント' if lang == 'ja' else 'segments'}   "
            f"{'平均 AUC' if lang == 'ja' else 'mean AUC'} "
            f"{'運動' if lang == 'ja' else 'motor'}={aucs[0]:.2f} · "
            f"{'触覚' if lang == 'ja' else 'LT'}={aucs[1]:.2f} · "
            f"{'痛覚' if lang == 'ja' else 'PP'}={aucs[2]:.2f}",
            style={"fontSize": "13px", "color": INK["700"]},
        ))

    def _fig_card(heading_key: str, caption_key: str, fig) -> html.Div | None:
        if fig is None:
            return None
        return html.Div(className="methods-perf-card", children=[
            html.H4(t(SCHEMA, heading_key, lang)),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            html.P(t(SCHEMA, caption_key, lang), style={"fontSize": "12px", "color": INK["500"]}),
        ])

    # cohort body-map atlas (interactive sensory-modality toggle; motor ladder always shown)
    mod_opts = [
        {"label": t(SCHEMA, "topo_modality_light_touch", lang), "value": "light_touch"},
        {"label": t(SCHEMA, "topo_modality_pin_prick", lang), "value": "pin_prick"},
    ]
    children.append(html.Div(className="methods-perf-card", children=[
        html.H4(t(SCHEMA, "methods_topo_atlas_heading", lang)),
        html.P(t(SCHEMA, "methods_topo_atlas_caption", lang),
               style={"fontSize": "12px", "color": INK["500"]}),
        dcc.RadioItems(id="methods-topo-atlas-modality", options=mod_opts, value="light_touch",
                       inline=True, style={"marginBottom": "4px", "fontSize": "13px"}),
        dcc.Graph(id="methods-topo-atlas-graph", config={"displayModeBar": False}),
    ]))

    children.extend(c for c in [
        _fig_card("methods_topo_scorecard_heading", "methods_topo_scorecard_caption",
                  fg.fig_topography_scorecard(topo, lang)),
        _fig_card("methods_topo_calibration_heading", "methods_topo_calibration_caption",
                  fg.fig_topography_calibration(topo, lang)),
        _fig_card("methods_topo_drivers_heading", "methods_topo_drivers_caption",
                  fg.fig_topography_drivers(topo, SCHEMA, lang)),
    ] if c is not None)

    # interactive per-segment drilldown (reuses the G4 conversion reliability + SHAP figures)
    modelable = [s for s in topo["segments"] if not s.get("degenerate") and s.get("auc") is not None]
    seg_opts = [{"label": _topo_seg_label(s, lang), "value": s["key"]} for s in modelable]
    children.append(html.Div(className="methods-perf-card", children=[
        html.H4(t(SCHEMA, "methods_topo_drilldown_heading", lang)),
        html.P(t(SCHEMA, "methods_topo_drilldown_caption", lang),
               style={"fontSize": "12px", "color": INK["500"]}),
        dcc.Dropdown(id="methods-topo-seg", options=seg_opts,
                     value=(seg_opts[0]["value"] if seg_opts else None), clearable=False,
                     style={"maxWidth": "340px", "marginBottom": "8px"}),
        html.Div(style={"display": "flex", "gap": "12px"}, children=[
            dcc.Graph(id="methods-topo-rel-graph", config={"displayModeBar": False},
                      style={"flex": "1", "minWidth": "0"}),
            dcc.Graph(id="methods-topo-shap-graph", config={"displayModeBar": False},
                      style={"flex": "1", "minWidth": "0"}),
        ]),
    ]))
    children.append(html.P(t(SCHEMA, "topo_caption", lang),
                           style={"fontSize": "12px", "color": INK["500"]}))
    return html.Div(className="methods-block", children=children)


@callback(
    Output("methods-topo-atlas-graph", "figure"),
    Input("methods-topo-atlas-modality", "value"),
    Input("lang-store", "data"),
)
def update_methods_topography_atlas(modality, lang):
    """Cohort recovery-topography atlas: the body map colored by each segment's observed cohort
    milestone base rate, with the sensory silhouette toggled between light touch / pin prick."""
    atlas = topography_cohort_atlas()
    if atlas is None:
        return go.Figure()
    return fig_topography_bodymap(atlas, lang, sensory_modality=(modality or "light_touch"))


@callback(
    Output("methods-topo-rel-graph", "figure"),
    Output("methods-topo-shap-graph", "figure"),
    Input("methods-topo-seg", "value"),
    Input("lang-store", "data"),
)
def update_methods_topography_segment(seg_key, lang):
    """Per-segment drilldown: the selected segment head's raw-vs-Platt reliability curve and
    in-sample SHAP drivers (reusing the G4 conversion reliability/SHAP figures)."""
    if not TOPOGRAPHY or not seg_key:
        return go.Figure(), go.Figure()
    entry = next((s for s in TOPOGRAPHY["segments"] if s["key"] == seg_key), None)
    if entry is None or entry.get("degenerate"):
        return go.Figure(), go.Figure()
    label = _topo_seg_label(entry, lang)
    rel = fg.fig_conversion_reliability(entry, lang, label) or go.Figure()
    shap = fg.fig_conversion_shap(entry, SCHEMA, lang) or go.Figure()
    return rel, shap


_LD_LEVEL_ORDER = ["nli", "right_motor", "left_motor", "right_sensory", "left_sensory"]


def _ld_level_label(key: str, lang: str) -> str:
    lv = (LEVEL_DESCENT or {}).get("levels", {}).get(key, {})
    return lv.get("label_ja" if lang == "ja" else "label_en", key)


def _level_descent_block(lang: str) -> html.Div | None:
    """G10 — neurological-level descent: per-level descent-discrimination scorecard + outcome-
    composition landscape + the interactive per-level drilldown (calibration, drivers, magnitude
    confusion, Δ distribution)."""
    ld = LEVEL_DESCENT
    if not ld or not ld.get("levels"):
        return None
    children: list = [
        html.H3(t(SCHEMA, "methods_ld_heading", lang)),
        html.P(t(SCHEMA, "methods_ld_def", lang)),
    ]
    for heading_key, caption_key, fig in [
        ("methods_ld_scorecard_heading", "methods_ld_scorecard_caption",
         fg.fig_level_descent_scorecard(ld, lang)),
        ("methods_ld_landscape_heading", "methods_ld_landscape_caption",
         fg.fig_level_descent_landscape(ld, lang)),
    ]:
        if fig is not None:
            children.append(html.Div(className="methods-perf-card", children=[
                html.H4(t(SCHEMA, heading_key, lang)),
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
                html.P(t(SCHEMA, caption_key, lang), style={"fontSize": "12px", "color": INK["500"]}),
            ]))

    # interactive per-level drilldown (reuses the G4 conversion reliability + SHAP + confusion figs)
    keys = [k for k in _LD_LEVEL_ORDER if k in ld["levels"]]
    lvl_opts = [{"label": _ld_level_label(k, lang), "value": k} for k in keys]
    children.append(html.Div(className="methods-perf-card", children=[
        html.H4(t(SCHEMA, "methods_ld_drilldown_heading", lang)),
        html.P(t(SCHEMA, "methods_ld_drilldown_caption", lang),
               style={"fontSize": "12px", "color": INK["500"]}),
        dcc.Dropdown(id="methods-ld-level", options=lvl_opts,
                     value=(lvl_opts[0]["value"] if lvl_opts else None), clearable=False,
                     style={"maxWidth": "340px", "marginBottom": "8px"}),
        html.Div(style={"display": "flex", "gap": "12px"}, children=[
            dcc.Graph(id="methods-ld-rel-graph", config={"displayModeBar": False},
                      style={"flex": "1", "minWidth": "0"}),
            dcc.Graph(id="methods-ld-shap-graph", config={"displayModeBar": False},
                      style={"flex": "1", "minWidth": "0"}),
        ]),
        html.Div(style={"display": "flex", "gap": "12px", "marginTop": "8px"}, children=[
            dcc.Graph(id="methods-ld-cm-graph", config={"displayModeBar": False},
                      style={"flex": "1", "minWidth": "0"}),
            dcc.Graph(id="methods-ld-delta-graph", config={"displayModeBar": False},
                      style={"flex": "1", "minWidth": "0"}),
        ]),
    ]))
    children.append(html.P(t(SCHEMA, "ld_caption", lang),
                           style={"fontSize": "12px", "color": INK["500"]}))
    return html.Div(className="methods-block", children=children)


@callback(
    Output("methods-ld-rel-graph", "figure"),
    Output("methods-ld-shap-graph", "figure"),
    Output("methods-ld-cm-graph", "figure"),
    Output("methods-ld-delta-graph", "figure"),
    Input("methods-ld-level", "value"),
    Input("lang-store", "data"),
)
def update_methods_level_descent(level_key, lang):
    """Per-level drilldown: the selected level's descent-head raw-vs-Platt reliability curve and
    in-sample SHAP drivers (reusing the G4 conversion figures), plus the magnitude head's confusion
    matrix and that level's Δ distribution."""
    if not LEVEL_DESCENT or not level_key:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure()
    lv = LEVEL_DESCENT["levels"].get(level_key)
    if lv is None:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure()
    label = _ld_level_label(level_key, lang)
    rel = fg.fig_conversion_reliability(lv["descent"], lang, label) or go.Figure()
    shap = fg.fig_conversion_shap(lv["descent"], SCHEMA, lang) or go.Figure()
    cm = fg.fig_conversion_confusion(lv["magnitude"], lang) or go.Figure()
    delta = fg.fig_level_descent_delta(lv["landscape"], lang) or go.Figure()
    return rel, shap, cm, delta


def _dataquality_block(lang: str) -> html.Div | None:
    dq = DATAQUALITY
    if not dq:
        return None
    tot, src = dq["totals"], dq["source"]
    sev_label = {"error": "エラー" if lang == "ja" else "Error",
                 "warn": "警告" if lang == "ja" else "Warning",
                 "info": "情報" if lang == "ja" else "Info"}
    cat_label = {"domain": "領域" if lang == "ja" else "Domain",
                 "cross_field": "項目間" if lang == "ja" else "Cross-field",
                 "longitudinal": "経時" if lang == "ja" else "Longitudinal"}
    findings_lbl = "総検出数" if lang == "ja" else "Total findings"
    ep_lbl = "対象症例" if lang == "ja" else "Episodes flagged"
    sev_line = "   ".join(f"{sev_label.get(k, k)}={v}" for k, v in tot["by_severity"].items())
    kpi = html.P(
        f"{findings_lbl}: {tot['violations']:,}     "
        f"{ep_lbl}: {tot['episodes_flagged']} / {src['n_episodes']}     ({sev_line})",
        style={"fontSize": "13px", "color": INK["700"]},
    )
    header = html.Tr([
        html.Th("規則" if lang == "ja" else "Rule"),
        html.Th("区分" if lang == "ja" else "Category"),
        html.Th("重大度" if lang == "ja" else "Severity"),
        html.Th("行数" if lang == "ja" else "Rows"),
        html.Th("症例" if lang == "ja" else "Ep"),
    ])
    body = [
        html.Tr([
            html.Td(r["id"]),
            html.Td(cat_label.get(r["category"], r["category"])),
            html.Td(sev_label.get(r["severity"], r["severity"])),
            html.Td(f"{r['count']:,}"),
            html.Td(str(r["episodes"])),
        ])
        for r in dq["rules"]
    ]
    children: list = [
        html.H3(t(SCHEMA, "methods_dataquality_heading", lang)),
        html.P(t(SCHEMA, "methods_dataquality_def", lang)),
        kpi,
    ]
    fig = fg.fig_dataquality_overview(dq, lang)
    if fig is not None:
        children.append(dcc.Graph(figure=fig, config={"displayModeBar": False}))
    children.append(html.Table([header, *body], className="patient-isncsci-table"))
    return html.Div(className="methods-block", children=children)


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

    traj_block = _perf_block_trajectory(lang)
    if traj_block is not None:
        perf_children.append(traj_block)

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
        md.append(html.Div(
            className="methods-block",
            children=[html.H3(t(SCHEMA, title_key, lang)), html.P(t(SCHEMA, body_key, lang))],
        ))
    md.append(perf_block)
    temporal_block = _temporal_block(lang)
    if temporal_block is not None:
        md.append(temporal_block)
    landmark_block = _landmark_block(lang)
    if landmark_block is not None:
        md.append(landmark_block)
    conversion_block = _conversion_block(lang)
    if conversion_block is not None:
        md.append(conversion_block)
    multistate_block = _multistate_block(lang)
    if multistate_block is not None:
        md.append(multistate_block)
    independence_block = _independence_block(lang)
    if independence_block is not None:
        md.append(independence_block)
    topography_block = _topography_block(lang)
    if topography_block is not None:
        md.append(topography_block)
    level_descent_block = _level_descent_block(lang)
    if level_descent_block is not None:
        md.append(level_descent_block)
    dq_block = _dataquality_block(lang)
    if dq_block is not None:
        md.append(dq_block)
    return html.Div(md, style={"maxWidth": "820px"})
