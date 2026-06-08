"""Shared layout components: topbar, cards, sliders, prediction figures.

Used by multiple tab modules. No callbacks — pure layout factories.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots

from rehab_sci.constants import AIS_ORD_TO_LETTER
from rehab_sci.dashboard.compute import format_value, mag_short_label
from rehab_sci.dashboard.i18n import col_label, level_label, t
from rehab_sci.dashboard.state import FEATURE_SPEC, SCHEMA, SIM_DEFAULTS
from rehab_sci.dashboard.theme import (
    COLORSCALE_TOPOGRAPHY,
    INK,
    PALETTE_AIS,
    PALETTE_CATEGORICAL,
    PALETTE_INDEPENDENCE_DOMAIN,
)
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
    is_delta = spec.clip_min is not None and spec.clip_min < 0  # Δ (change) outcome
    fig.update_layout(
        height=120,
        margin=dict(l=110, r=20, t=22 if is_delta else 10, b=30),
        xaxis=dict(range=[x_lo, x_hi], title=axis_title),
        yaxis=dict(showticklabels=True, tickfont=dict(size=12), showgrid=False),
    )
    if is_delta:  # mark the "no change" line so a PI crossing 0 reads as uncertain recovery
        fig.add_vline(
            x=0, line=dict(color=INK["300"], width=1.2, dash="dot"),
            annotation_text=("変化なし" if lang == "ja" else "no change"),
            annotation_position="top", annotation_font=dict(size=10, color=INK["500"]),
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


# ---------- AIS multi-state recovery (G6) ----------
_MS_TICK_VALS = [1, 2, 3, 4, 5]


def _ms_target_curves(result: dict) -> dict[int, list[float]]:
    """Map the available first-passage curves to DISTINCT target grades.  ``improve`` is
    P(reach >= admission+1); ``ge_C`` / ``ge_D`` are P(reach >=C / >=D).  For a B or C admission the
    improvement target coincides with >=C / >=D, so deduplicate by target grade (those curves are
    then identical)."""
    adm = result["ais_ord"]
    conv = result.get("conversion", {})
    out: dict[int, list[float]] = {}
    if "improve" in conv:
        out[adm + 1] = conv["improve"]
    if "ge_C" in conv:
        out.setdefault(3, conv["ge_C"])
    if "ge_D" in conv:
        out.setdefault(4, conv["ge_D"])
    return dict(sorted(out.items()))


def fig_multistate_trajectory(result: dict, lang: str, patient_obs=None) -> go.Figure:
    """Expected AIS-grade trajectory for the admission grade's cohort (occupancy-weighted mean +
    inter-quartile grade band), optionally overlaid with one patient's OWN observed grades.  y =
    AIS grade A..E.  The band is COHORT dynamics (identical for any patient of that admission
    grade); the overlay is the individual.  Empty when nothing applies."""
    if not result.get("applicable"):
        return go.Figure()
    occ = np.asarray(result["occupancy"], dtype=float)  # (T, 5)
    x = [level_label(SCHEMA, "time_name", tp, lang) for tp in result["window"]]
    grades = np.arange(1, 6)
    exp = occ @ grades
    cdf = np.cumsum(occ, axis=1)
    lo = (cdf >= 0.25).argmax(axis=1) + 1
    hi = (cdf >= 0.75).argmax(axis=1) + 1
    band_word = "コホート四分位範囲" if lang == "ja" else "Cohort IQR"
    exp_word = "コホート期待グレード" if lang == "ja" else "Cohort expected grade"
    grade_word = "AIS グレード" if lang == "ja" else "AIS grade"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(17,122,139,0.12)", name=band_word, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=exp, mode="lines+markers", name=exp_word,
                             line=dict(color=PALETTE_CATEGORICAL[0], width=2.6),
                             marker=dict(size=6),
                             hovertemplate="%{x}<br>" + exp_word + ": %{y:.2f}<extra></extra>"))
    if patient_obs:
        pat_word = "本症例 (実測)" if lang == "ja" else "This patient (observed)"
        px = [level_label(SCHEMA, "time_name", tp, lang) for tp, _ in patient_obs]
        py = [g for _, g in patient_obs]
        fig.add_trace(go.Scatter(
            x=px, y=py, mode="lines+markers", name=pat_word,
            line=dict(color=INK["900"], width=2, dash="dot"),
            marker=dict(size=11, color=INK["900"], symbol="diamond", line=dict(color="white", width=1.5)),
            customdata=[AIS_ORD_TO_LETTER[g] for _, g in patient_obs],
            hovertemplate="%{x}<br>AIS %{customdata}<extra></extra>",
        ))
    fig.update_layout(
        height=320, margin=dict(l=46, r=16, t=20, b=64),
        yaxis=dict(title=grade_word, range=[0.7, 5.3], tickvals=_MS_TICK_VALS,
                   ticktext=[AIS_ORD_TO_LETTER[v] for v in _MS_TICK_VALS]),
        xaxis=dict(tickangle=-45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def fig_multistate_conversion_personal(result: dict, lang: str) -> go.Figure:
    """First-passage conversion curves for the admission grade: P(reach each higher AIS threshold
    by time t), monotone non-decreasing.  y = probability; the y=0.5 line marks the median
    crossing.  Empty for an at-ceiling (AIS E) admission."""
    if not result.get("applicable"):
        return go.Figure()
    targets = _ms_target_curves(result)
    if not targets:
        return go.Figure()
    x = [level_label(SCHEMA, "time_name", tp, lang) for tp in result["window"]]
    reach_word = "到達確率" if lang == "ja" else "Reach probability"
    fig = go.Figure()
    for tgt, curve in targets.items():
        letter = AIS_ORD_TO_LETTER[tgt]
        fig.add_trace(go.Scatter(
            x=x, y=curve, mode="lines+markers", name=f"≥{letter}",
            line=dict(color=PALETTE_AIS[letter], width=2.4), marker=dict(size=5),
            hovertemplate=f"≥AIS {letter} · %{{x}}<br>%{{y:.0%}}<extra></extra>",
        ))
    fig.add_hline(y=0.5, line=dict(color=INK["200"], dash="dot", width=1))
    fig.update_layout(
        height=300, margin=dict(l=48, r=16, t=18, b=64),
        yaxis=dict(title=reach_word, range=[0, 1.02], tickformat=".0%"),
        xaxis=dict(tickangle=-45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def multistate_readout(result: dict, lang: str) -> html.Div:
    """Text summary of the multi-state panel: admission grade, the calibrated improve-by-6m
    probability (with cohort base-rate lift; framed as reach-AIS-E for a D admission), the median
    day to first improvement, and the expected days in the top states.  Renders a 'needs admission
    grade' / 'at ceiling' note when nothing applies."""
    if result.get("ais_ord") is None:
        return html.Div(t(SCHEMA, "ms_need_ais", lang), className="lm-prompt")
    lines: list = [html.Div(
        f"{t(SCHEMA, 'conv_adm_grade', lang)}: AIS {result['adm_letter']}",
        className="conv-readout-grade",
    )]
    if result.get("at_ceiling"):
        lines.append(html.Div(t(SCHEMA, "ms_at_ceiling", lang), className="lm-readout-line"))
        return html.Div(lines, className="lm-readout conv-readout")
    imp = result.get("improve") or {}
    pp_word = "pt" if lang == "ja" else "pp"
    if imp.get("applicable") and imp.get("prob") is not None:
        lift = (imp["prob"] - imp["base_rate"]) * 100
        label = t(SCHEMA, "ms_improve_to_e", lang) if imp.get("to_e") else t(SCHEMA, "ms_improve_any", lang)
        lines.append(html.Div(
            f"{label}: {imp['prob']:.0%} "
            f"({t(SCHEMA, 'conv_base', lang)} {imp['base_rate']:.0%}, {lift:+.0f}{pp_word})",
            className="lm-readout-line lm-readout-delta",
        ))
    mdi = result.get("median_day_to_improve")
    unit = "日" if lang == "ja" else "d"
    if mdi is not None:
        lines.append(html.Div(
            f"{t(SCHEMA, 'ms_median_day', lang)}: {mdi:.0f}{unit}",
            className="lm-readout-line",
        ))
    soj = result.get("sojourn") or []
    if soj:
        states = result["state_labels"]
        order = np.argsort(soj)[::-1][:2]
        parts = ", ".join(f"AIS {states[i]} {soj[i]:.0f}{unit}" for i in order)
        lines.append(html.Div(
            f"{t(SCHEMA, 'ms_sojourn', lang)}: {parts}",
            className="lm-readout-line",
        ))
    return html.Div(lines, className="lm-readout conv-readout")


# ---------- functional-independence profile (G7; shared by simulator + patient) ----------
def fig_independence_profile(result: dict, lang: str, observed: dict | None = None) -> go.Figure:
    """Per-SCIM-item discharge functional-independence profile: one horizontal bar = calibrated
    P(independent), colored by SCIM domain, with a grey diamond at the item's cohort base rate so
    the per-patient lift is visible.  When ``observed`` (realized discharge independence per item)
    is supplied (patient card), a green circle / crimson cross in the right gutter marks whether the
    patient actually achieved independence — a predicted-vs-realized read.  Empty when no result."""
    items = (result or {}).get("items")
    if not items:
        return go.Figure()
    rows = items[::-1]  # reverse: first registry item (feeding) ends up at the top
    names = [col_label(SCHEMA, it["col"], lang) for it in rows]
    probs = [it["prob"] for it in rows]
    bases = [it["base_rate"] for it in rows]
    colors = [PALETTE_INDEPENDENCE_DOMAIN.get(it["domain"], PALETTE_CATEGORICAL[0]) for it in rows]
    prob_word = t(SCHEMA, "ind_prob_axis", lang)
    base_word = t(SCHEMA, "ind_base_rate", lang)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=probs, y=names, orientation="h", marker=dict(color=colors),
        text=[f"{p:.0%}" for p in probs], textposition="outside", textfont=dict(size=10.5),
        hovertemplate="%{y}<br>" + prob_word + ": %{x:.0%}<extra></extra>", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=bases, y=names, mode="markers", name=base_word,
        marker=dict(color=INK["500"], size=10, symbol="diamond", line=dict(color="#fff", width=1)),
        hovertemplate=base_word + ": %{x:.0%}<extra></extra>",
    ))
    seen: set[str] = set()
    for it in rows:  # domain legend (legend-only swatch markers)
        d = it["domain"]
        if d in seen:
            continue
        seen.add(d)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=t(SCHEMA, f"ind_domain_{d}", lang),
            marker=dict(color=PALETTE_INDEPENDENCE_DOMAIN.get(d, PALETTE_CATEGORICAL[0]),
                        size=11, symbol="square"),
            hoverinfo="skip",
        ))
    if observed:
        gx = 1.08
        ach = [n for n, it in zip(names, rows, strict=True) if observed.get(it["key"]) is True]
        miss = [n for n, it in zip(names, rows, strict=True) if observed.get(it["key"]) is False]
        if ach:
            fig.add_trace(go.Scatter(
                x=[gx] * len(ach), y=ach, mode="markers", name=t(SCHEMA, "ind_achieved", lang),
                marker=dict(color="#2c8a6b", size=12, symbol="circle", line=dict(color="#fff", width=1)),
                hovertemplate=t(SCHEMA, "ind_achieved", lang) + "<extra></extra>",
            ))
        if miss:
            fig.add_trace(go.Scatter(
                x=[gx] * len(miss), y=miss, mode="markers", name=t(SCHEMA, "ind_not_achieved", lang),
                marker=dict(color="#a3354e", size=11, symbol="x", line=dict(width=0)),
                hovertemplate=t(SCHEMA, "ind_not_achieved", lang) + "<extra></extra>",
            ))
    fig.update_layout(
        height=24 * len(rows) + 122, margin=dict(l=222, r=24, t=46, b=40), barmode="overlay",
        xaxis=dict(range=[0, 1.18], tickvals=[0, 0.25, 0.5, 0.75, 1.0], tickformat=".0%", title=prob_word),
        yaxis=dict(tickfont=dict(size=10.5), showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(size=10)),
    )
    return fig


def independence_readout(result: dict, lang: str) -> html.Div:
    """Text summary of the independence profile: the expected number of independent functions
    (Σ calibrated probs), the per-domain breakdown, and the most- / least-likely items.  Renders
    the card intro as a prompt when no result (bundle absent / empty input)."""
    items = (result or {}).get("items")
    if not items:
        return html.Div(t(SCHEMA, "ind_card_intro", lang), className="lm-prompt")
    n = len(items)
    exp = result["expected_count"]
    lines: list = [html.Div(
        f"{t(SCHEMA, 'ind_expected_count', lang)}: {exp:.1f} / {n}",
        className="conv-readout-grade",
    )]
    dom_sum: dict[str, float] = {}
    dom_n: dict[str, int] = {}
    for it in items:
        dom_sum[it["domain"]] = dom_sum.get(it["domain"], 0.0) + it["prob"]
        dom_n[it["domain"]] = dom_n.get(it["domain"], 0) + 1
    parts = [
        f"{t(SCHEMA, f'ind_domain_{d}', lang)} {dom_sum[d]:.1f}/{dom_n[d]}"
        for d in result["domains"]
    ]
    lines.append(html.Div(
        f"{t(SCHEMA, 'ind_by_domain', lang)}: " + "  ·  ".join(parts),
        className="lm-readout-line",
    ))
    ordered = sorted(items, key=lambda it: it["prob"])
    least, most = ordered[0], ordered[-1]
    lines.append(html.Div(
        f"{t(SCHEMA, 'ind_most_likely', lang)}: {col_label(SCHEMA, most['col'], lang)} {most['prob']:.0%}"
        f"  ·  {t(SCHEMA, 'ind_least_likely', lang)}: {col_label(SCHEMA, least['col'], lang)} {least['prob']:.0%}",
        className="lm-readout-line lm-readout-delta",
    ))
    return html.Div(lines, className="lm-readout conv-readout")


# ---------- G8 recovery-topography body map ----------
# Anatomical (x, y) for each sensory dermatome on a stylised front-view body, RIGHT side only
# (x > 0); the Left side mirrors to -x.  Coordinate frame: x in [-62, 62], y in [0, 205] (feet=0,
# head-top~205).  Approximate, not a clinical dermatome atlas — the markers carry the data; the
# silhouette + rostro-caudal placement orient the reader (neck cluster, two arms, a torso column,
# two legs, perineal sacral cluster).  Hover gives the exact level/side/probability.
_DERMATOME_XY = {
    "C2": (7, 184), "C3": (10, 176), "C4": (14, 167),         # neck
    "C5": (33, 150), "T1": (30, 132), "C6": (40, 114),         # arm (C5 lateral, T1 medial)
    "C7": (40, 96), "C8": (35, 93),                            # hand
    "T2": (19, 153), "T3": (18, 145), "T4": (18, 137),         # chest (T4 ~ nipple)
    "T5": (18, 129), "T6": (18, 121), "T7": (18, 113),
    "T8": (18, 106), "T9": (18, 99), "T10": (18, 92),          # T10 ~ umbilicus
    "T11": (18, 86), "T12": (19, 81), "L1": (20, 75),          # L1 ~ groin
    "L2": (20, 64), "L3": (18, 48), "L4": (16, 34),            # thigh -> leg
    "L5": (14, 20), "S1": (20, 10),                            # foot (L5 dorsum, S1 lateral/sole)
    "S2": (10, 55), "S3": (8, 66), "S45": (6, 74),             # sacral / perineal cluster
}


def _poly_path(pts: list[tuple[float, float]]) -> str:
    """SVG path string for a closed polygon from (x, y) vertices."""
    head = f"M {pts[0][0]},{pts[0][1]}"
    rest = " ".join(f"L {x},{y}" for x, y in pts[1:])
    return f"{head} {rest} Z"


def _topo_body_shapes() -> list[dict]:
    """Stylised front-view humanoid silhouette as a list of add_shape kwargs (data coords)."""
    skin = dict(fillcolor=INK["50"], line=dict(color=INK["200"], width=1.2), layer="below")
    torso = [(-36, 160), (36, 160), (32, 138), (26, 96), (30, 84),
             (-30, 84), (-26, 96), (-32, 138)]
    r_arm = [(34, 159), (41, 157), (37, 88), (30, 93)]
    l_arm = [(-x, y) for x, y in r_arm]
    r_leg = [(5, 84), (28, 84), (21, 4), (11, 4)]
    l_leg = [(-x, y) for x, y in r_leg]
    return [
        dict(type="circle", x0=-15, y0=170, x1=15, y1=202, **skin),          # head
        dict(type="rect", x0=-7, y0=159, x1=7, y1=172, **skin),              # neck
        dict(type="path", path=_poly_path(torso), **skin),                  # torso
        dict(type="path", path=_poly_path(r_arm), **skin),                  # right arm
        dict(type="path", path=_poly_path(l_arm), **skin),                  # left arm
        dict(type="path", path=_poly_path(r_leg), **skin),                  # right leg
        dict(type="path", path=_poly_path(l_leg), **skin),                  # left leg
    ]


def _topo_segment_lookup(result: dict, modality: str) -> dict[tuple[str, str], dict]:
    """{(level, side): segment-record} for one modality from a predict_topography result."""
    return {(s["level"], s["side"]): s for s in result["segments"] if s["modality"] == modality}


def fig_topography_bodymap(
    result: dict,
    lang: str,
    sensory_modality: str = "light_touch",
    observed: dict | None = None,
    title: str | None = None,
) -> go.Figure:
    """The recovery-topography atlas: a stylised front-view dermatome silhouette (the chosen
    sensory modality, light-touch or pin-prick) beside a motor myotome ladder, every segment shaded
    by P(functional milestone) on the shared topography colorscale.  When ``observed`` (realized
    discharge milestones per segment) is supplied (patient card), achieved / not-achieved segments
    get a green / crimson ring.  Empty figure when ``result`` is missing."""
    if not (result or {}).get("segments"):
        return go.Figure()
    cs, ink = COLORSCALE_TOPOGRAPHY, INK
    side_word = {"Left": t(SCHEMA, "topo_side_left", lang), "Right": t(SCHEMA, "topo_side_right", lang)}
    p_word = t(SCHEMA, "topo_prob_axis", lang)
    base_word = t(SCHEMA, "topo_base_rate", lang)
    adm_word = t(SCHEMA, "topo_adm_self", lang)
    body_title = t(SCHEMA, f"topo_modality_{sensory_modality}", lang)
    ladder_title = t(SCHEMA, "topo_modality_motor", lang)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.06,
        subplot_titles=(body_title, ladder_title),
    )
    for sh in _topo_body_shapes():
        fig.add_shape(**sh, row=1, col=1)

    def _hover(rec: dict) -> str:
        adm = "—" if rec["adm_self"] is None else f"{rec['adm_self']:g}"
        return (f"{side_word[rec['side']]} {rec['level']}<br>{p_word}: {rec['prob']:.0%}"
                f"<br>{base_word}: {rec['base_rate']:.0%}<br>{adm_word}: {adm}")

    # ---- sensory dermatome silhouette (col 1) ----
    sx, sy, sc, scustom = [], [], [], []
    for s in (x for x in result["segments"] if x["modality"] == sensory_modality):
        xy = _DERMATOME_XY.get(s["level"])
        if xy is None:
            continue
        x = xy[0] if s["side"] == "Right" else -xy[0]
        sx.append(x)
        sy.append(xy[1])
        sc.append(s["prob"])
        scustom.append(_hover(s))
    fig.add_trace(go.Scatter(
        x=sx, y=sy, mode="markers", showlegend=False,
        marker=dict(size=11, color=sc, colorscale=cs, cmin=0, cmax=1,
                    line=dict(color="#fff", width=1),
                    colorbar=dict(title=dict(text=p_word, side="top"), orientation="h",
                                  y=-0.07, x=0.28, xanchor="center", len=0.5, thickness=11,
                                  tickformat=".0%", tickfont=dict(size=9), outlinewidth=0)),
        customdata=scustom, hovertemplate="%{customdata}<extra></extra>",
    ), row=1, col=1)
    # L / R orientation anchors
    fig.add_trace(go.Scatter(
        x=[-46, 46], y=[200, 200], mode="text", text=["L", "R"], showlegend=False,
        textfont=dict(size=13, color=ink["300"]), hoverinfo="skip",
    ), row=1, col=1)

    # ---- motor myotome ladder (col 2) ----
    motor = _topo_segment_lookup(result, "motor")
    levels = list(dict.fromkeys(s["level"] for s in result["segments"] if s["modality"] == "motor"))
    lx, ly, lc, ltext, lcustom = [], [], [], [], []
    for i, lv in enumerate(levels):
        yy = len(levels) - 1 - i  # C5 at top
        for col_x, sd in ((0, "Left"), (1, "Right")):
            rec = motor.get((lv, sd))
            if rec is None:
                continue
            lx.append(col_x)
            ly.append(yy)
            lc.append(rec["prob"])
            ltext.append(f"{rec['prob']:.0%}")
            lcustom.append(_hover(rec))
    fig.add_trace(go.Scatter(
        x=lx, y=ly, mode="markers+text", showlegend=False,
        marker=dict(size=30, symbol="square", color=lc, colorscale=cs, cmin=0, cmax=1,
                    line=dict(color="#fff", width=1.5)),
        text=ltext, textposition="middle center", textfont=dict(size=8.5, color=ink["900"]),
        customdata=lcustom, hovertemplate="%{customdata}<extra></extra>",
    ), row=1, col=2)

    # ---- observed achieved / not-achieved rings ----
    if observed:
        def _rings(pred, xs, ys, keys, col):
            ach_x = [x for x, k in zip(xs, keys, strict=True) if observed.get(k) is True]
            ach_y = [y for y, k in zip(ys, keys, strict=True) if observed.get(k) is True]
            mis_x = [x for x, k in zip(xs, keys, strict=True) if observed.get(k) is False]
            mis_y = [y for y, k in zip(ys, keys, strict=True) if observed.get(k) is False]
            size = 16 if col == 1 else 38
            lw = 1.8 if col == 1 else 2.6
            if ach_x:
                fig.add_trace(go.Scatter(
                    x=ach_x, y=ach_y, mode="markers", name=t(SCHEMA, "topo_achieved", lang),
                    marker=dict(size=size, symbol="circle-open", color="#2c8a6b", line=dict(width=lw)),
                    hoverinfo="skip", legendgroup="ach", showlegend=(col == 1),
                ), row=1, col=col)
            if mis_x:
                fig.add_trace(go.Scatter(
                    x=mis_x, y=mis_y, mode="markers", name=t(SCHEMA, "topo_not_achieved", lang),
                    marker=dict(size=size, symbol="circle-open", color="#a3354e", line=dict(width=lw)),
                    hoverinfo="skip", legendgroup="mis", showlegend=(col == 1),
                ), row=1, col=col)
        # segment keys aligned to the plotted marker order, per panel
        sens_keys = [s["key"] for s in result["segments"]
                     if s["modality"] == sensory_modality and _DERMATOME_XY.get(s["level"])]
        _rings(sensory_modality, sx, sy, sens_keys, 1)
        mot_keys = []
        for lv in levels:
            for sd in ("Left", "Right"):
                rec = motor.get((lv, sd))
                if rec is not None:
                    mot_keys.append(rec["key"])
        _rings("motor", lx, ly, mot_keys, 2)

    fig.update_xaxes(visible=False, range=[-62, 62], row=1, col=1)
    fig.update_yaxes(visible=False, range=[-6, 210], row=1, col=1)
    fig.update_xaxes(range=[-0.7, 1.7], tickvals=[0, 1], ticktext=["L", "R"],
                     tickfont=dict(size=11, color=ink["500"]), showgrid=False, zeroline=False, row=1, col=2)
    fig.update_yaxes(range=[-0.7, len(levels) - 0.3], tickvals=list(range(len(levels))),
                     ticktext=levels[::-1], tickfont=dict(size=10, color=ink["500"]),
                     showgrid=False, zeroline=False, row=1, col=2)
    fig.update_layout(
        height=500, margin=dict(l=8, r=8, t=40, b=58),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=13, color=ink["900"])),
                          margin=dict(t=64))
    for a in fig.layout.annotations:  # subplot titles
        a.font = dict(size=12.5, color=ink["700"])
    return fig


def topography_readout(result: dict, lang: str) -> html.Div:
    """Text summary of the topography atlas: expected count of muscles reaching antigravity (of 20)
    and dermatomes preserving protective sensation (light touch / pin prick, of 56 each), plus the
    strongest / weakest motor segment.  Prompt when no result."""
    segs = (result or {}).get("segments")
    if not segs:
        return html.Div(t(SCHEMA, "topo_card_intro", lang), className="lm-prompt")
    bm = result["by_modality"]
    side_word = {"Left": t(SCHEMA, "topo_side_left", lang), "Right": t(SCHEMA, "topo_side_right", lang)}
    lines: list = [html.Div(
        f"{t(SCHEMA, 'topo_expected_motor', lang)}: "
        f"{bm['motor']['expected_count']:.1f} / {bm['motor']['n_segments']}",
        className="conv-readout-grade",
    )]
    lines.append(html.Div(
        f"{t(SCHEMA, 'topo_expected_lt', lang)}: {bm['light_touch']['expected_count']:.0f}"
        f" / {bm['light_touch']['n_segments']}"
        f"  ·  {t(SCHEMA, 'topo_expected_pp', lang)}: {bm['pin_prick']['expected_count']:.0f}"
        f" / {bm['pin_prick']['n_segments']}",
        className="lm-readout-line",
    ))
    motor = sorted((s for s in segs if s["modality"] == "motor"), key=lambda s: s["prob"])
    if motor:
        weak, strong = motor[0], motor[-1]
        lines.append(html.Div(
            f"{t(SCHEMA, 'topo_strongest', lang)}: {side_word[strong['side']]} {strong['level']}"
            f" {strong['prob']:.0%}"
            f"  ·  {t(SCHEMA, 'topo_weakest', lang)}: {side_word[weak['side']]} {weak['level']}"
            f" {weak['prob']:.0%}",
            className="lm-readout-line lm-readout-delta",
        ))
    return html.Div(lines, className="lm-readout conv-readout")
