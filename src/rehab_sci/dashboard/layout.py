"""Shared layout components: topbar, cards, sliders, prediction figures.

Used by multiple tab modules. No callbacks — pure layout factories.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html

from rehab_sci.dashboard.compute import format_value, mag_short_label
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


def fig_landmark_compare(
    result: dict, spec: OutcomeSpec, lang: str, landmark_label: str,
) -> go.Figure:
    """Paired admission-only vs landmark prediction for one outcome (see compute.predict_landmark).

    Regression: two floating prediction-interval bars (baseline below, landmark on top) with a
    median diamond each.  Multiclass: grouped per-class probability bars for the two heads.
    """
    base, lm = result["baseline"], result["landmark"]
    row_base = t(SCHEMA, "lm_admission_only", lang)
    row_lm = t(SCHEMA, "lm_with_obs", lang).format(L=landmark_label)
    accent, accent_dark = PALETTE_CATEGORICAL[0], "#0c5a66"
    base_fill, base_mark = "rgba(140,140,140,0.20)", "#7a7a7a"
    med_word = "予測中央値" if lang == "ja" else "Predicted median"

    if result["task"] == "regression":
        label = t(SCHEMA, spec.display_key, lang)
        unit = t(SCHEMA, spec.unit_key, lang) if spec.unit_key else ""
        fig = go.Figure()
        for row, d, fill, mark, msize in (
            (row_base, base, base_fill, base_mark, 13),
            (row_lm, lm, "rgba(17,122,139,0.22)", accent_dark, 15),
        ):
            fig.add_trace(go.Bar(
                x=[d["hi"] - d["lo"]], base=[d["lo"]], y=[row], orientation="h",
                marker=dict(color=fill, line=dict(width=0)),
                hovertemplate=f"{d['lo']:.0f}–{d['hi']:.0f} (±{(d['hi'] - d['lo']) / 2:.0f})<extra></extra>",
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[d["pred"]], y=[row], mode="markers",
                marker=dict(color=mark, size=msize, symbol="diamond"),
                hovertemplate=f"{med_word}: %{{x:.0f}}<extra></extra>", showlegend=False,
            ))
        x_lo = float(spec.clip_min) if spec.clip_min is not None else min(0.0, base["lo"], lm["lo"])
        x_hi = (float(spec.clip_max) if spec.clip_max is not None
                else float(max(base["hi"], lm["hi"]) * 1.1 + 1.0))
        axis_title = f"{label} ({unit})" if unit else label
        fig.update_layout(
            height=150, barmode="overlay",
            margin=dict(l=150, r=20, t=10, b=34),
            xaxis=dict(range=[x_lo, x_hi], title=axis_title),
            yaxis=dict(categoryarray=[row_base, row_lm], tickfont=dict(size=12), showgrid=False),
        )
        return fig

    class_labels = lm["class_labels"]
    bar_labels = [f"AIS {c}" for c in class_labels]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=row_base, x=bar_labels, y=base["proba"],
        marker=dict(color=base_mark), hovertemplate="%{x}: %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name=row_lm, x=bar_labels, y=lm["proba"],
        marker=dict(color=accent), hovertemplate="%{x}: %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(
        height=260, barmode="group",
        margin=dict(l=60, r=20, t=30, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        yaxis=dict(range=[0, 1.05], tickformat=".0%",
                   title=t(SCHEMA, "sim_class_probabilities", lang)),
        xaxis=dict(title=t(SCHEMA, spec.display_key, lang)),
    )
    return fig


def landmark_readout(result: dict, spec: OutcomeSpec, lang: str) -> html.Div:
    """Two-line baseline→landmark summary shared by the simulator and patient dynamic cards."""
    base, lm = result["baseline"], result["landmark"]
    lines: list = []
    if result["task"] == "regression":
        unit = t(SCHEMA, spec.unit_key, lang) if spec.unit_key else ""
        u = f" {unit}" if unit else ""
        pred_word = "予測中央値" if lang == "ja" else "Predicted median"
        b_hw, l_hw = (base["hi"] - base["lo"]) / 2, (lm["hi"] - lm["lo"]) / 2
        lines.append(html.Div(
            f"{pred_word}: {base['pred']:.0f} → {lm['pred']:.0f}{u}",
            className="lm-readout-line",
        ))
        lines.append(html.Div(
            f"{t(SCHEMA, 'lm_pi_halfwidth', lang)}: ±{b_hw:.0f} → ±{l_hw:.0f}{u} "
            f"(Δ {l_hw - b_hw:+.0f})",
            className="lm-readout-line lm-readout-delta",
        ))
    else:
        none_word = t(SCHEMA, "lm_none", lang)
        b_set = ", ".join(f"AIS {c}" for c in base["aps_set"]) or none_word
        l_set = ", ".join(f"AIS {c}" for c in lm["aps_set"]) or none_word
        lines.append(html.Div(
            f"{t(SCHEMA, 'sim_predicted_class_label', lang)}: "
            f"AIS {base['pred_class']} → AIS {lm['pred_class']}",
            className="lm-readout-line",
        ))
        lines.append(html.Div(
            f"{t(SCHEMA, 'lm_aps_set', lang)}: {{{b_set}}} → {{{l_set}}}",
            className="lm-readout-line lm-readout-delta",
        ))
    return html.Div(lines, className="lm-readout")


# ---------- value-of-information (G2) ----------
def _voi_label(measure: str, lang: str) -> str:
    return t(SCHEMA, f"lm_measure_{measure.lower()}", lang)


def fig_voi_patient(voi: dict, spec: OutcomeSpec, lang: str) -> go.Figure:
    """Per-patient value-of-information bars (see compute.landmark_voi).

    One horizontal bar per measure = the PI half-width reduction (regression) / APS-set shrink
    (AIS) that observing it buys over the admission-only baseline, sorted most-valuable first.
    Teal = not yet observed (what to measure next); grey = already observed (value realised).
    Bar text shows the resulting point estimate; hover carries the value used and the new PI.
    """
    measures = voi.get("measures") or []
    if not measures:
        return go.Figure()
    labels = [_voi_label(m["measure"], lang) for m in measures]
    accent, grey = PALETTE_CATEGORICAL[0], "#9aa0a6"
    colors = [accent if m["which"] == "prescriptive" else grey for m in measures]
    obs_word = t(SCHEMA, "voi_realized", lang)
    presc_word = t(SCHEMA, "voi_to_obtain", lang)
    val_word = t(SCHEMA, "voi_value_used", lang)

    def _v(x: float | None) -> str:
        return "–" if x is None else f"{x:.0f}"

    if voi["task"] == "regression":
        x = [m["d_halfwidth"] for m in measures]
        unit = t(SCHEMA, spec.unit_key, lang) if spec.unit_key else ""
        x_title = t(SCHEMA, "voi_axis_pi", lang) + (f" ({unit})" if unit else "")
        texts = [f"{m['pred']:.0f}" for m in measures]
        pred_word = "予測中央値" if lang == "ja" else "Predicted median"
        htexts = [
            f"{(presc_word if m['which'] == 'prescriptive' else obs_word)}<br>"
            f"{val_word}: {_v(m['value'])}<br>"
            f"{pred_word}: {m['pred']:.0f} (PI {m['lo']:.0f}–{m['hi']:.0f}, ±{m['halfwidth']:.0f})"
            for m in measures
        ]
    else:
        x = [m["d_setsize"] for m in measures]
        x_title = t(SCHEMA, "voi_axis_aps", lang)
        texts = [f"AIS {m['pred_class']}" for m in measures]
        pred_word = t(SCHEMA, "sim_predicted_class_label", lang)
        htexts = [
            f"{(presc_word if m['which'] == 'prescriptive' else obs_word)}<br>"
            f"{val_word}: {_v(m['value'])}<br>"
            f"{pred_word}: AIS {m['pred_class']} ({m['pred_prob']:.0%})<br>"
            f"APS {{{', '.join('AIS ' + c for c in m['aps_set'])}}} (Δ {m['d_setsize']:+d})"
            for m in measures
        ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=x, orientation="h", marker=dict(color=colors),
        text=texts, textposition="outside", textfont=dict(size=10),
        hovertext=htexts, hovertemplate="%{hovertext}<extra></extra>", showlegend=False,
    ))
    for nm, c in ((presc_word, accent), (obs_word, grey)):
        fig.add_trace(go.Bar(
            y=[None], x=[None], orientation="h", marker=dict(color=c), name=nm,
        ))
    fig.update_layout(
        height=34 * len(measures) + 70, barmode="overlay",
        margin=dict(l=8, r=40, t=26, b=40),
        xaxis=dict(title=x_title, zeroline=True, zerolinecolor=INK["200"]),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        legend=dict(orientation="h", x=0, y=1.04, xanchor="left", yanchor="bottom", font=dict(size=10)),
    )
    return fig


def voi_readout(voi: dict, spec: OutcomeSpec, lang: str) -> html.Div:
    """One/two-line prescription: the most valuable next measure to obtain (+ best already-observed)."""
    measures = voi.get("measures") or []
    base = voi["baseline"]
    presc = [m for m in measures if m["which"] == "prescriptive"]
    obs = [m for m in measures if m["which"] == "observed"]
    lines: list = []
    if voi["task"] == "regression":
        base_hw = (base["hi"] - base["lo"]) / 2.0
        if presc and presc[0]["d_halfwidth"] > 0.05:
            top = presc[0]
            lines.append(html.Div(
                f"{t(SCHEMA, 'voi_next_best', lang)}: {_voi_label(top['measure'], lang)} — "
                f"±{base_hw:.0f} → ±{top['halfwidth']:.0f} "
                f"({t(SCHEMA, 'voi_point', lang)} {base['pred']:.0f} → {top['pred']:.0f})",
                className="lm-readout-line lm-readout-delta",
            ))
        else:
            key = "voi_all_observed" if not presc else "voi_none_prescriptive"
            lines.append(html.Div(t(SCHEMA, key, lang), className="lm-readout-line"))
        best = max(obs, key=lambda m: m["d_halfwidth"], default=None)
        if best is not None and best["d_halfwidth"] > 0.05:
            lines.append(html.Div(
                f"{t(SCHEMA, 'voi_top_realized', lang)}: {_voi_label(best['measure'], lang)} (−±{best['d_halfwidth']:.0f})",
                className="lm-readout-line",
            ))
    else:
        base_size = len(base["aps_set"])
        if presc and presc[0]["d_setsize"] > 0:
            top = presc[0]
            lines.append(html.Div(
                f"{t(SCHEMA, 'voi_next_best', lang)}: {_voi_label(top['measure'], lang)} — "
                f"APS {base_size} → {top['set_size']}",
                className="lm-readout-line lm-readout-delta",
            ))
        else:
            key = "voi_all_observed" if not presc else "voi_none_prescriptive"
            lines.append(html.Div(t(SCHEMA, key, lang), className="lm-readout-line"))
    return html.Div(lines, className="lm-readout")


# ---------- AIS-grade conversion (G4; shared by simulator + patient) ----------
def _conversion_endpoint_label(key: str, letter: str, lang: str) -> str:
    """Clinical endpoint name + its discharge threshold, e.g. 'Motor-incomplete (≥C)'."""
    return f"{t(SCHEMA, f'conv_endpoint_{key}', lang)} (≥{letter})"


def fig_conversion_endpoints(result: dict, lang: str) -> go.Figure:
    """Calibrated conversion probabilities for the applicable binary endpoints (see
    compute.predict_conversion).  One horizontal bar per endpoint = P(conversion); a grey diamond
    marks that endpoint's cohort base rate so the per-patient lift is visible.  Empty when no
    endpoint applies at the admission grade (e.g. AIS D/E)."""
    items = [(k, e) for k, e in result["endpoints"].items() if e.get("applicable")]
    if not items:
        return go.Figure()
    labels = [_conversion_endpoint_label(k, e["discharge_min_letter"], lang) for k, e in items]
    probs = [e["prob"] for _, e in items]
    bases = [e["base_rate"] for _, e in items]
    accent_dark = "#0c5a66"
    base_word = "コホート基準率" if lang == "ja" else "Cohort base rate"
    prob_word = "転換確率" if lang == "ja" else "Conversion probability"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=probs, y=labels, orientation="h",
        marker=dict(color="rgba(17,122,139,0.85)", line=dict(color=accent_dark, width=1)),
        text=[f"{p:.0%}" for p in probs], textposition="outside", textfont=dict(size=12),
        hovertemplate="%{y}<br>" + prob_word + ": %{x:.1%}<extra></extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=bases, y=labels, mode="markers",
        marker=dict(color=INK["500"], size=12, symbol="diamond",
                    line=dict(color="#fff", width=1.2)),
        hovertemplate=base_word + ": %{x:.1%}<extra></extra>", showlegend=False,
    ))
    fig.update_layout(
        height=58 * len(items) + 70,
        margin=dict(l=210, r=44, t=16, b=38),
        xaxis=dict(range=[0, 1.02], tickformat=".0%", title=prob_word),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11.5), showgrid=False),
    )
    return fig


def fig_conversion_magnitude(result: dict, lang: str) -> go.Figure:
    """Ordinal improvement-magnitude head shown AS its 80% APS set / most-likely class (not as a
    competing probability — the head is class-weighted, hence uncalibrated; see the s3 CRUX).  Bars
    are the relative class scores over {0,+1,≥+2}; APS-set members are filled + bordered, the
    most-likely class carries a caret.  Empty when magnitude does not apply (admission AIS E)."""
    mag = result.get("magnitude") or {}
    if not mag.get("applicable"):
        return go.Figure()
    codes = mag["class_codes"]
    proba = mag["proba"]
    cap = mag["mag_cap"]
    aps = set(mag["aps_codes"])
    pred = mag["pred_code"]
    ticks = [mag_short_label(c, cap) for c in codes]
    accent, accent_dark = PALETTE_CATEGORICAL[0], "#0c5a66"
    colors = [accent if c in aps else "rgba(17,122,139,0.20)" for c in codes]
    borders = [accent_dark if c in aps else "rgba(0,0,0,0)" for c in codes]
    score_word = "相対尤度 (較正なし)" if lang == "ja" else "Relative likelihood (uncalibrated)"
    fig = go.Figure(go.Bar(
        x=ticks, y=proba,
        marker=dict(color=colors, line=dict(color=borders, width=1.5)),
        text=[("▲ " if c == pred else "") + f"{p:.0%}" for c, p in zip(codes, proba, strict=True)],
        textposition="outside", textfont=dict(size=11),
        hovertemplate="%{x}: %{y:.1%}<extra></extra>", showlegend=False,
    ))
    fig.update_layout(
        height=230,
        margin=dict(l=56, r=20, t=24, b=40),
        yaxis=dict(range=[0, 1.08], tickformat=".0%", title=score_word),
        xaxis=dict(title=("改善幅 (段階)" if lang == "ja" else "Improvement (grades)")),
    )
    return fig


def conversion_readout(result: dict, lang: str) -> html.Div:
    """Text summary of the conversion panel: admission grade, each applicable calibrated endpoint
    probability with its cohort base-rate lift, and the magnitude most-likely class + APS set.
    Renders a 'needs admission grade' / 'at ceiling' note when nothing applies."""
    if result["ais_ord"] is None:
        return html.Div(t(SCHEMA, "conv_need_ais", lang), className="lm-prompt")
    lines: list = [
        html.Div(
            f"{t(SCHEMA, 'conv_adm_grade', lang)}: AIS {result['adm_letter']}",
            className="conv-readout-grade",
        ),
    ]
    if not result["any_applicable"]:
        lines.append(html.Div(t(SCHEMA, "conv_at_ceiling", lang), className="lm-readout-line"))
        return html.Div(lines, className="lm-readout conv-readout")

    pp_word = "pt" if lang == "ja" else "pp"
    for key, e in result["endpoints"].items():
        if not e.get("applicable"):
            continue
        lift = (e["prob"] - e["base_rate"]) * 100
        name = _conversion_endpoint_label(key, e["discharge_min_letter"], lang)
        lines.append(html.Div(
            f"{name}: {e['prob']:.0%} "
            f"({t(SCHEMA, 'conv_base', lang)} {e['base_rate']:.0%}, {lift:+.0f}{pp_word})",
            className="lm-readout-line",
        ))

    mag = result.get("magnitude") or {}
    if mag.get("applicable"):
        cap = mag["mag_cap"]
        unit = t(SCHEMA, "conv_mag_unit", lang)
        pred_lbl = mag_short_label(mag["pred_code"], cap)
        set_lbl = ", ".join(mag_short_label(c, cap) for c in mag["aps_codes"])
        lines.append(html.Div(
            f"{t(SCHEMA, 'conv_mag_label', lang)}: "
            f"{t(SCHEMA, 'conv_most_likely', lang)} {pred_lbl} {unit} · "
            f"{t(SCHEMA, 'conv_aps_set', lang)} {{{set_lbl}}}",
            className="lm-readout-line lm-readout-delta",
        ))
    return html.Div(lines, className="lm-readout conv-readout")
