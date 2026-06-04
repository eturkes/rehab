"""Rehabilitation Analytics & Prediction Suite — bilingual Dash app.

Run with::

    uv run python -m rehab_sci.dashboard.app

then open http://127.0.0.1:8050/.

This module is the entry point: it creates the Dash app instance, defines the
top-level layout shell (stores, tabs, chrome), and registers the three
app-chrome callbacks (lang toggle, topbar/tab labels, tab body dispatch).

Tab-specific layouts and callbacks live in ``dashboard.tabs.*`` and are
registered by importing them at the bottom of this file.
"""

from __future__ import annotations

from pathlib import Path

from dash import Dash, Input, Output, State, callback, ctx, dcc, html

from rehab_sci.dashboard.i18n import t
from rehab_sci.dashboard.layout import topbar
from rehab_sci.dashboard.state import SCHEMA

# Tab renderers (layout only — callbacks registered via import below).
from rehab_sci.dashboard.tabs.insights import render_insights
from rehab_sci.dashboard.tabs.methods import render_methods
from rehab_sci.dashboard.tabs.overview import render_overview
from rehab_sci.dashboard.tabs.patient import render_patient
from rehab_sci.dashboard.tabs.simulator import render_simulator


# ---------- app factory ----------
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
            dcc.Store(id="patient-ref", storage_type="session"),
            dcc.Download(id="report-download"),
            html.Div(id="topbar-container"),
            html.Div(
                className="tabs-container",
                children=[
                    dcc.Tabs(
                        id="tabs",
                        value="overview",
                        className="dash-tabs",
                        children=[
                            dcc.Tab(value="overview", id="tab-overview",
                                    className="dash-tab", selected_className="dash-tab--selected"),
                            dcc.Tab(value="simulator", id="tab-simulator",
                                    className="dash-tab", selected_className="dash-tab--selected"),
                            dcc.Tab(value="patient", id="tab-patient",
                                    className="dash-tab", selected_className="dash-tab--selected"),
                            dcc.Tab(value="insights", id="tab-insights",
                                    className="dash-tab", selected_className="dash-tab--selected"),
                            dcc.Tab(value="methods", id="tab-methods",
                                    className="dash-tab", selected_className="dash-tab--selected"),
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
server = app.server


# ---------- chrome callbacks ----------
@callback(
    Output("lang-store", "data"),
    Input("lang-ja", "n_clicks"),
    Input("lang-en", "n_clicks"),
    State("lang-store", "data"),
    prevent_initial_call=True,
)
def update_lang(_n_ja, _n_en, _cur):
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
def update_chrome(lang):
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
    State("patient-ref", "data"),
)
def update_tab(tab, lang, ref_data):
    if tab == "overview":
        return render_overview(lang)
    if tab == "simulator":
        return render_simulator(lang, ref_data)
    if tab == "patient":
        return render_patient(lang)
    if tab == "insights":
        return render_insights(lang)
    return render_methods(lang)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
