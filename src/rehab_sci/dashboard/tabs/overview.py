"""Findings-first cohort landing page + a clearly separated filtered explorer.

The editorial lead is bound to full-cohort metric artifacts and never changes with
the explorer filters.  The ``update_overview_content`` callback owns the filtered
section only; empty multi-select = no filter (show all).
"""

from __future__ import annotations

import pandas as pd
from dash import Input, Output, State, callback, dcc, html

from rehab_sci.dashboard import figures as fg
from rehab_sci.dashboard.figures import ARCHETYPE_NAMES_EN, ARCHETYPE_NAMES_JA
from rehab_sci.dashboard.i18n import level_label, t
from rehab_sci.dashboard.layout import chart_card, kpi_card
from rehab_sci.dashboard.state import (
    ARCHETYPE_DATA,
    DISSOCIATION,
    EP,
    LANDMARK,
    LONG,
    PHENOTYPE_DATA,
    SCHEMA,
    SUBGROUPS,
)
from rehab_sci.data.phenotypes import phenotype_summary

_AGE_MIN = 10
_AGE_MAX = 95


def _copy(key: str, lang: str, **values) -> str:
    """Localized copy with named values; YAML remains the content source of truth."""
    return t(SCHEMA, key, lang).format(**values)


def _finding_metrics() -> dict:
    """Small, testable aggregate contract for every claim in the editorial lead."""
    years = pd.to_numeric(EP.get("BusinessYear", pd.Series(dtype=float)), errors="coerce").dropna()
    facts: dict = {
        "episodes": len(EP),
        "patients": int(EP["IDNumber"].nunique()),
        "year_start": int(years.min()) if len(years) else None,
        "year_end": int(years.max()) if len(years) else None,
    }

    scim_subgroups = SUBGROUPS.get("scim_total") or {}
    motor = next(
        (
            row
            for row in scim_subgroups.get("results", [])
            if row.get("feature") == "TotalMotor" and not row.get("skipped")
        ),
        None,
    )
    if motor:
        facts["motor_strata"] = {
            "n": int(motor["n_total"]),
            "groups": motor["groups"],
            "eta_squared": float(motor["eta_squared"]),
            "p_holm": float(motor["p_holm"]),
        }

    lm = (((LANDMARK or {}).get("outcomes") or {}).get("scim_total") or {}).get(
        "by_landmark", {}
    ).get("3m")
    if lm:
        base, observed = lm["baseline"], lm["landmark"]
        base_pi = float(base["pi_halfwidth_raw"])
        obs_pi = float(observed["pi_halfwidth_raw"])
        facts["observation"] = {
            "n": int(lm["n_eligible"]),
            "n_test": int(lm["n_test"]),
            "r2_baseline": float(base["r2"]),
            "r2_observed": float(observed["r2"]),
            "r2_gain": float(observed["r2"] - base["r2"]),
            "pi_shrink": float((base_pi - obs_pi) / base_pi),
        }

    diss = (((DISSOCIATION or {}).get("axes") or {}).get("lems_mobility") or {}).get(
        "landscape"
    )
    if diss:
        facts["dissociation"] = {
            "n": int(diss["n"]),
            "pearson_r": float(diss["pearson_r"]),
            "dissociated_share": float(diss["dissociated_share"]),
        }
    return facts


def _finding_card(
    block_id: str,
    metric: str,
    unit: str,
    title: str,
    body: str,
    caveat: str,
) -> html.Article:
    return html.Article(
        id=block_id,
        className="finding-card",
        children=[
            html.Div(className="finding-card__topline", children=[
                html.Span(metric, className="finding-card__metric"),
                html.Span(unit, className="finding-card__unit"),
            ]),
            html.H3(title),
            html.P(body, className="finding-card__body"),
            html.P(caveat, className="finding-card__caveat"),
        ],
    )


def _render_findings_lead(lang: str) -> list:
    facts = _finding_metrics()
    year_range = (
        f"{facts['year_start']}–{facts['year_end']}"
        if facts["year_start"] is not None
        else "—"
    )
    scope_stats = [
        (t(SCHEMA, "episodes_n", lang), f"{facts['episodes']:,}"),
        (t(SCHEMA, "patients_n", lang), f"{facts['patients']:,}"),
        (t(SCHEMA, "overview_scope_years", lang), year_range),
    ]
    lead_head = html.Section(
        id="findings-lead",
        className="findings-lead",
        children=[
            html.H2(t(SCHEMA, "overview_lead_title", lang)),
            html.P(t(SCHEMA, "overview_lead_deck", lang)),
            html.Dl(className="scope-list", children=[
                html.Div(className="scope-stat", children=[html.Dt(label), html.Dd(value)])
                for label, value in scope_stats
            ]),
        ],
    )

    cards: list = []
    motor = facts.get("motor_strata")
    if motor:
        first, last = motor["groups"][0], motor["groups"][-1]
        cards.append(_finding_card(
            "finding-motor-stratification",
            f"{first['median']:.0f}→{last['median']:.0f}",
            t(SCHEMA, "overview_finding_motor_unit", lang),
            t(SCHEMA, "overview_finding_motor_title", lang),
            _copy(
                "overview_finding_motor_body",
                lang,
                n=motor["n"],
                effect=motor["eta_squared"],
            ),
            t(SCHEMA, "overview_finding_motor_caveat", lang),
        ))
    obs = facts.get("observation")
    if obs:
        cards.append(_finding_card(
            "finding-landmark-value",
            f"+{obs['r2_gain']:.2f}",
            "R²",
            t(SCHEMA, "overview_finding_observation_title", lang),
            _copy(
                "overview_finding_observation_body",
                lang,
                n=obs["n"],
                n_test=obs["n_test"],
                baseline=obs["r2_baseline"],
                observed=obs["r2_observed"],
                shrink=obs["pi_shrink"],
            ),
            t(SCHEMA, "overview_finding_observation_caveat", lang),
        ))
    diss = facts.get("dissociation")
    if diss:
        cards.append(_finding_card(
            "finding-neuro-functional-dissociation",
            f"{diss['dissociated_share']:.0%}",
            t(SCHEMA, "overview_finding_dissociation_unit", lang),
            t(SCHEMA, "overview_finding_dissociation_title", lang),
            _copy(
                "overview_finding_dissociation_body",
                lang,
                n=diss["n"],
                correlation=diss["pearson_r"],
            ),
            t(SCHEMA, "overview_finding_dissociation_caveat", lang),
        ))

    lead: list = [lead_head]
    if cards:
        lead.append(html.Section(className="finding-grid", children=cards))

    if motor:
        figure = fg.fig_motor_strata_finding(motor, lang)
        lead.append(html.Section(
            id="finding-motor-evidence",
            className="finding-evidence",
            children=[
                html.Div(className="finding-evidence__copy", children=[
                    html.H3(t(SCHEMA, "overview_pattern_title", lang)),
                    html.P(_copy(
                        "overview_pattern_body",
                        lang,
                        n=motor["n"],
                        effect=motor["eta_squared"],
                        low=motor["groups"][0]["median"],
                        high=motor["groups"][-1]["median"],
                    )),
                    html.P(
                        t(SCHEMA, "overview_pattern_caveat", lang),
                        className="finding-evidence__caveat",
                    ),
                ]),
                html.Div(className="finding-evidence__chart", children=[
                    dcc.Graph(figure=figure, config={"displayModeBar": False, "responsive": True})
                ]),
            ],
        ))
    return lead


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def render_overview(lang: str) -> html.Div:
    """Return full-cohort findings lead + filtered cohort explorer shell."""
    ais_opts = [{"label": f"AIS {g}", "value": g} for g in "ABCDE"]
    para_opts = [
        {"label": level_label(SCHEMA, "para_tetra", v, lang), "value": v}
        for v in ["TETRA", "PARA", "NONE"]
    ]
    ph = t(SCHEMA, "insight_all", lang)

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
    explorer = html.Section(
        id="cohort-explorer",
        className="cohort-explorer",
        children=[
            html.Div(className="overview-section-head", children=[
                html.H2(t(SCHEMA, "overview_explorer_title", lang)),
                html.P(t(SCHEMA, "overview_explorer_deck", lang)),
            ]),
            filter_bar,
            html.Div(id="overview-content"),
        ],
    )
    return html.Div(className="overview-page", children=[*_render_findings_lead(lang), explorer])


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
        note = _copy("overview_filter_count", lang, shown=n_ep, total=total)
        children.append(html.Div(note, className="ov-filter-note"))

    children.append(
        html.Div(className="card-row", children=[
            kpi_card(
                t(SCHEMA, "patients_n", lang),
                f"{n_pat:,}",
                t(SCHEMA, "episodes_n", lang) + f": {n_ep:,}",
            ),
            kpi_card(
                t(SCHEMA, "overview_kpi_mean_age", lang),
                f"{mean_age:.1f}" if pd.notna(mean_age) else "–",
                t(SCHEMA, "unit_years", lang),
            ),
            kpi_card(
                t(SCHEMA, "overview_kpi_scim_median", lang),
                f"{med_scim:.0f}" if pd.notna(med_scim) else "–",
                "0–100",
            ),
            kpi_card(
                t(SCHEMA, "overview_kpi_severe", lang),
                f"{severe:.0f}%" if pd.notna(severe) else "–",
                None,
            ),
        ])
    )

    # --- Outcome-focused charts; descriptive context stays collapsed. ---
    _gc = {"displayModeBar": False}

    children.append(html.Div(className="overview-subhead", children=[
        html.H3(t(SCHEMA, "overview_selected_outcomes_title", lang)),
        html.P(t(SCHEMA, "overview_selected_outcomes_deck", lang)),
    ]))
    children.append(html.Div(className="chart-row chart-row--spotlight", children=[
        chart_card(
            t(SCHEMA, "chart_recovery_curves", lang),
            dcc.Graph(figure=fg.fig_recovery_curves(long, SCHEMA, lang), config=_gc),
        ),
    ]))
    children.append(html.Div(className="chart-row", children=[
        chart_card(
            t(SCHEMA, "chart_ais_admit_discharge", lang),
            dcc.Graph(figure=fg.fig_ais_admit_discharge_sankey(ep, SCHEMA, lang), config=_gc),
        ),
        chart_card(
            t(SCHEMA, "chart_discharge_scim", lang),
            dcc.Graph(figure=fg.fig_discharge_scim(ep, SCHEMA, lang), config=_gc),
        ),
    ]))

    context = html.Details(className="overview-details", children=[
        html.Summary(t(SCHEMA, "overview_context_summary", lang)),
        html.P(t(SCHEMA, "overview_context_deck", lang), className="overview-details__intro"),
        html.Div(className="chart-row", children=[
            chart_card(
                t(SCHEMA, "chart_injury_treemap", lang),
                dcc.Graph(figure=fg.fig_injury_treemap(ep, SCHEMA, lang), config=_gc),
            ),
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
        ]),
    ])
    children.append(context)

    trajectory_rows: list = []
    if ARCHETYPE_DATA is not None:
        summaries = (
            _filtered_archetype_summaries(ep) if is_filtered else ARCHETYPE_DATA["summaries"]
        )
        trajectory_rows.append(html.Div(className="chart-row", children=[
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

    if PHENOTYPE_DATA is not None:
        pheno_sum = (
            phenotype_summary(ep, PHENOTYPE_DATA["assignments"], PHENOTYPE_DATA["k"])
            if is_filtered else PHENOTYPE_DATA["summaries"]
        )
        _mkey = {"SCIM_total": "pheno_measure_scim", "TotalMotor": "pheno_measure_motor"}
        measure_labels = [t(SCHEMA, _mkey.get(m, m), lang) for m in PHENOTYPE_DATA["measures"]]
        trajectory_rows.append(html.Div(className="chart-row", children=[
            chart_card(
                t(SCHEMA, "chart_phenotype_curves", lang),
                dcc.Graph(
                    figure=fg.fig_phenotype_curves(
                        PHENOTYPE_DATA["class_means"],
                        list(PHENOTYPE_DATA["window"]),
                        pheno_sum,
                        measure_labels,
                        SCHEMA,
                        lang,
                        class_support=PHENOTYPE_DATA["class_support"],
                    ),
                    config=_gc,
                ),
            ),
            chart_card(
                t(SCHEMA, "chart_phenotype_demographics", lang),
                dcc.Graph(
                    figure=fg.fig_phenotype_demographics(pheno_sum, SCHEMA, lang),
                    config=_gc,
                ),
            ),
        ]))

    if trajectory_rows:
        children.append(html.Details(className="overview-details", children=[
            html.Summary(t(SCHEMA, "overview_trajectory_summary", lang)),
            html.P(
                t(SCHEMA, "overview_trajectory_deck", lang),
                className="overview-details__intro",
            ),
            *trajectory_rows,
        ]))

    return html.Div(children)
