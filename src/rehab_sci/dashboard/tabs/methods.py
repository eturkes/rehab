"""Methods tab — model documentation + per-outcome performance visualizations."""

from __future__ import annotations

from dash import dcc, html

from rehab_sci.constants import AIS_ORD_TO_LETTER
from rehab_sci.dashboard import figures as fg
from rehab_sci.dashboard.i18n import t
from rehab_sci.dashboard.state import (
    CONVERSION,
    DATAQUALITY,
    LANDMARK,
    METRICS,
    OUTCOME_BUNDLES,
    SCHEMA,
    TEMPORAL,
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
    dq_block = _dataquality_block(lang)
    if dq_block is not None:
        md.append(dq_block)
    return html.Div(md, style={"maxWidth": "820px"})
