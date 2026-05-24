"""Overview tab — cohort KPIs, demographic charts, archetype curves with interactive filters.

Filter bar (AIS, paralysis, age range, archetype) lets clinicians slice the cohort.
The ``update_overview_content`` callback re-renders all KPIs and charts on the
filtered subset.  Empty multi-select = no filter (show all).
"""

from __future__ import annotations

import pandas as pd
from dash import Input, Output, State, callback, dcc, html

from rehab_sci.dashboard import figures as fg
from rehab_sci.dashboard.figures import ARCHETYPE_NAMES_EN, ARCHETYPE_NAMES_JA
from rehab_sci.dashboard.i18n import level_label, t
from rehab_sci.dashboard.layout import chart_card, kpi_card
from rehab_sci.dashboard.state import ARCHETYPE_DATA, EP, LONG, SCHEMA

_AGE_MIN = 10
_AGE_MAX = 95


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def render_overview(lang: str) -> html.Div:
    """Return filter bar + empty content div (populated by callback)."""
    ais_opts = [{"label": f"AIS {g}", "value": g} for g in "ABCDE"]
    para_opts = [
        {"label": level_label(SCHEMA, "para_tetra", v, lang), "value": v}
        for v in ["TETRA", "PARA", "NONE"]
    ]
    ph = "すべて" if lang == "ja" else "All"

    fields: list = [
        html.Div(className="ov-filter-field", children=[
            html.Label(t(SCHEMA, "overview_filter_ais", lang)),
            dcc.Dropdown(id="ov-filter-ais", options=ais_opts, multi=True, placeholder=ph),
        ]),
        html.Div(className="ov-filter-field", children=[
            html.Label(t(SCHEMA, "overview_filter_paralysis", lang)),
            dcc.Dropdown(id="ov-filter-para", options=para_opts, multi=True, placeholder=ph),
        ]),
        html.Div(className="ov-filter-field ov-filter-field--slider", children=[
            html.Label(t(SCHEMA, "overview_filter_age", lang)),
            dcc.RangeSlider(
                id="ov-filter-age",
                min=_AGE_MIN, max=_AGE_MAX, step=5,
                marks={v: str(v) for v in range(_AGE_MIN, _AGE_MAX + 1, 10)},
                value=[_AGE_MIN, _AGE_MAX],
                tooltip={"placement": "bottom"},
            ),
        ]),
    ]

    if ARCHETYPE_DATA is not None:
        names = ARCHETYPE_NAMES_JA if lang == "ja" else ARCHETYPE_NAMES_EN
        arch_opts = [{"label": names[i], "value": i} for i in range(ARCHETYPE_DATA["k"])]
    else:
        arch_opts = []
    arch_vis = {} if ARCHETYPE_DATA is not None else {"display": "none"}
    fields.append(
        html.Div(className="ov-filter-field", style=arch_vis, children=[
            html.Label(t(SCHEMA, "overview_filter_archetype", lang)),
            dcc.Dropdown(id="ov-filter-arch", options=arch_opts, multi=True, placeholder=ph),
        ]),
    )

    filter_bar = html.Div(className="ov-filter-bar", children=fields)
    return html.Div([filter_bar, html.Div(id="overview-content")])


# ---------------------------------------------------------------------------
# Filter logic
# ---------------------------------------------------------------------------

def _apply_filters(ais: list, para: list, age_range: list | None, arch: list):
    """AND-combine all active filters on the global EP/LONG frames.

    Returns ``(ep_filtered, long_filtered, is_filtered)``.
    """
    mask = pd.Series(True, index=EP.index)
    if ais:
        mask &= EP["AIS"].isin(ais)
    if para:
        mask &= EP["対麻痺_四肢麻痺"].isin(para)
    age_lo, age_hi = age_range if age_range else [_AGE_MIN, _AGE_MAX]
    if age_lo > _AGE_MIN or age_hi < _AGE_MAX:
        age = pd.to_numeric(EP["年齢"], errors="coerce")
        mask &= (age >= age_lo) & (age <= age_hi)
    if arch and ARCHETYPE_DATA is not None:
        arch_set = set(arch)
        arch_keys = {k for k, v in ARCHETYPE_DATA["assignments"].items() if v in arch_set}
        mask &= EP["KeyRecordNumber"].isin(arch_keys)

    ep_f = EP[mask]
    long_f = LONG[LONG["KeyRecordNumber"].isin(ep_f["KeyRecordNumber"])]
    return ep_f, long_f, int(mask.sum()) < len(EP)


def _filtered_archetype_summaries(ep_f: pd.DataFrame) -> list[dict]:
    """Rebuild per-archetype summaries on the filtered episode subset."""
    assignments = ARCHETYPE_DATA["assignments"]
    k = ARCHETYPE_DATA["k"]
    summaries: list[dict] = []
    for i in range(k):
        keys_i = {kr for kr, label in assignments.items() if label == i}
        ep_i = ep_f[ep_f["KeyRecordNumber"].isin(keys_i)]
        n = len(ep_i)
        if n == 0:
            summaries.append(
                {"id": i, "n": 0, "mean_age": None, "pct_tetra": 0, "ais_distribution": {}}
            )
            continue
        age = pd.to_numeric(ep_i["年齢"], errors="coerce")
        ais = ep_i["AIS"].dropna()
        summaries.append({
            "id": i,
            "n": n,
            "mean_age": float(age.mean()) if age.notna().any() else None,
            "pct_tetra": float((ep_i["対麻痺_四肢麻痺"] == "TETRA").sum() / n * 100),
            "ais_distribution": ais.value_counts(normalize=True).to_dict() if len(ais) else {},
        })
    return summaries


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

@callback(
    Output("overview-content", "children"),
    Input("ov-filter-ais", "value"),
    Input("ov-filter-para", "value"),
    Input("ov-filter-age", "value"),
    Input("ov-filter-arch", "value"),
    State("lang-store", "data"),
)
def update_overview_content(ais, para, age_range, arch, lang):
    ep, long, is_filtered = _apply_filters(ais or [], para or [], age_range, arch or [])

    if len(ep) == 0:
        return html.Div(t(SCHEMA, "no_data", lang), className="overview-empty")

    # --- KPIs ---
    n_ep = len(ep)
    n_pat = ep["IDNumber"].nunique()
    mean_age = pd.to_numeric(ep["年齢"], errors="coerce").mean()
    med_scim = pd.to_numeric(ep["y_discharge_scim"], errors="coerce").median()
    ais_vc = ep["AIS"].dropna().astype(str).value_counts(normalize=True)
    severe = float(ais_vc.get("A", 0) + ais_vc.get("B", 0)) * 100

    children: list = []

    if is_filtered:
        total = len(EP)
        note = (
            f"{n_ep:,} / {total:,} 症例を表示中"
            if lang == "ja"
            else f"Showing {n_ep:,} of {total:,} episodes"
        )
        children.append(html.Div(note, className="ov-filter-note"))

    children.append(
        html.Div(className="card-row", children=[
            kpi_card(
                t(SCHEMA, "patients_n", lang),
                f"{n_pat:,}",
                t(SCHEMA, "episodes_n", lang) + f": {n_ep:,}",
            ),
            kpi_card(
                "平均年齢" if lang == "ja" else "Mean age",
                f"{mean_age:.1f}" if pd.notna(mean_age) else "–",
                t(SCHEMA, "unit_years", lang),
            ),
            kpi_card(
                "退院時 SCIM 中央値" if lang == "ja" else "Discharge SCIM median",
                f"{med_scim:.0f}" if pd.notna(med_scim) else "–",
                "0–100",
            ),
            kpi_card(
                "重度 (AIS A/B) 比率" if lang == "ja" else "Severe (AIS A/B) at admission",
                f"{severe:.0f}%" if pd.notna(severe) else "–",
                None,
            ),
        ])
    )

    # --- Charts ---
    _gc = {"displayModeBar": False}

    children.append(html.Div(className="chart-row", children=[
        chart_card(
            t(SCHEMA, "chart_injury_sunburst", lang),
            dcc.Graph(figure=fg.fig_injury_sunburst(ep, SCHEMA, lang), config=_gc),
        ),
        chart_card(
            t(SCHEMA, "chart_ais_admit_discharge", lang),
            dcc.Graph(figure=fg.fig_ais_admit_discharge_sankey(ep, SCHEMA, lang), config=_gc),
        ),
    ]))

    children.append(html.Div(className="chart-row", children=[
        chart_card(
            t(SCHEMA, "chart_age_dist", lang),
            dcc.Graph(figure=fg.fig_age_distribution(ep, SCHEMA, lang), config=_gc),
        ),
        chart_card(
            t(SCHEMA, "chart_sex_dist", lang),
            dcc.Graph(figure=fg.fig_sex_donut(ep, SCHEMA, lang), config=_gc),
        ),
        chart_card(
            t(SCHEMA, "chart_mechanism", lang),
            dcc.Graph(figure=fg.fig_mechanism(ep, SCHEMA, lang), config=_gc),
        ),
    ]))

    children.append(html.Div(className="chart-row", children=[
        chart_card(
            t(SCHEMA, "chart_discharge_scim", lang),
            dcc.Graph(figure=fg.fig_discharge_scim(ep, SCHEMA, lang), config=_gc),
        ),
        chart_card(
            t(SCHEMA, "chart_recovery_curves", lang),
            dcc.Graph(figure=fg.fig_recovery_curves(long, SCHEMA, lang), config=_gc),
        ),
    ]))

    if ARCHETYPE_DATA is not None:
        summaries = (
            _filtered_archetype_summaries(ep) if is_filtered else ARCHETYPE_DATA["summaries"]
        )
        children.append(html.Div(className="chart-row", children=[
            chart_card(
                t(SCHEMA, "chart_archetype_curves", lang),
                dcc.Graph(
                    figure=fg.fig_archetype_curves(
                        ARCHETYPE_DATA["centroids"],
                        ARCHETYPE_DATA["timepoint_labels"],
                        summaries,
                        SCHEMA,
                        lang,
                    ),
                    config=_gc,
                ),
            ),
            chart_card(
                t(SCHEMA, "chart_archetype_demographics", lang),
                dcc.Graph(
                    figure=fg.fig_archetype_demographics(summaries, SCHEMA, lang),
                    config=_gc,
                ),
            ),
        ]))

    return html.Div(children)
