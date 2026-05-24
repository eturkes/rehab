"""Overview tab — cohort KPIs, demographic charts, archetype curves."""

from __future__ import annotations

import pandas as pd
from dash import dcc, html

from rehab_sci.dashboard import figures as fg
from rehab_sci.dashboard.i18n import t
from rehab_sci.dashboard.layout import chart_card, kpi_card
from rehab_sci.dashboard.state import ARCHETYPE_DATA, EP, LONG, SCHEMA


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
                "平均年齢" if lang == "ja" else "Mean age",
                f"{mean_age:.1f}",
                t(SCHEMA, "unit_years", lang),
            ),
            kpi_card(
                "退院時 SCIM 中央値" if lang == "ja" else "Discharge SCIM median",
                f"{median_scim:.0f}",
                "0–100",
            ),
            kpi_card(
                "重度 (AIS A/B) 比率" if lang == "ja" else "Severe (AIS A/B) at admission",
                f"{severe_pct:.0f}%",
                None,
            ),
        ],
    )

    _gc = {"displayModeBar": False}
    row1 = html.Div(className="chart-row", children=[
        chart_card(
            t(SCHEMA, "chart_injury_sunburst", lang),
            dcc.Graph(figure=fg.fig_injury_sunburst(EP, SCHEMA, lang), config=_gc),
        ),
        chart_card(
            t(SCHEMA, "chart_ais_admit_discharge", lang),
            dcc.Graph(figure=fg.fig_ais_admit_discharge_sankey(EP, SCHEMA, lang), config=_gc),
        ),
    ])

    row2 = html.Div(className="chart-row", children=[
        chart_card(
            t(SCHEMA, "chart_age_dist", lang),
            dcc.Graph(figure=fg.fig_age_distribution(EP, SCHEMA, lang), config=_gc),
        ),
        chart_card(
            t(SCHEMA, "chart_sex_dist", lang),
            dcc.Graph(figure=fg.fig_sex_donut(EP, SCHEMA, lang), config=_gc),
        ),
        chart_card(
            t(SCHEMA, "chart_mechanism", lang),
            dcc.Graph(figure=fg.fig_mechanism(EP, SCHEMA, lang), config=_gc),
        ),
    ])

    row3 = html.Div(className="chart-row", children=[
        chart_card(
            t(SCHEMA, "chart_discharge_scim", lang),
            dcc.Graph(figure=fg.fig_discharge_scim(EP, SCHEMA, lang), config=_gc),
        ),
        chart_card(
            t(SCHEMA, "chart_recovery_curves", lang),
            dcc.Graph(figure=fg.fig_recovery_curves(LONG, SCHEMA, lang), config=_gc),
        ),
    ])

    rows = [kpi_row, row1, row2, row3]

    if ARCHETYPE_DATA is not None:
        row4 = html.Div(className="chart-row", children=[
            chart_card(
                t(SCHEMA, "chart_archetype_curves", lang),
                dcc.Graph(
                    figure=fg.fig_archetype_curves(
                        ARCHETYPE_DATA["centroids"],
                        ARCHETYPE_DATA["timepoint_labels"],
                        ARCHETYPE_DATA["summaries"],
                        SCHEMA, lang,
                    ),
                    config=_gc,
                ),
            ),
            chart_card(
                t(SCHEMA, "chart_archetype_demographics", lang),
                dcc.Graph(
                    figure=fg.fig_archetype_demographics(
                        ARCHETYPE_DATA["summaries"], SCHEMA, lang,
                    ),
                    config=_gc,
                ),
            ),
        ])
        rows.append(row4)

    return html.Div(rows)
