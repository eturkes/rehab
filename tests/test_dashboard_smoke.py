"""Smoke: every tab renders in both languages, and the per-patient dynamic
callbacks build their figures without raising.

Rendering each tab body headless is the cheap guard that would have caught the
s31 ``INK['600']`` Methods crash — boot-200 checks miss it because tab bodies
build lazily inside callbacks (§0b).  Invoking the dynamic callbacks directly
additionally exercises the G-series figure builders the static render never
reaches.
"""

from __future__ import annotations

import pytest

LANGS = ("ja", "en")
TABS = ("overview", "insights", "methods", "patient", "simulator")


@pytest.fixture(scope="session")
def renderers(state):
    from rehab_sci.dashboard.tabs.insights import render_insights
    from rehab_sci.dashboard.tabs.methods import render_methods
    from rehab_sci.dashboard.tabs.overview import render_overview
    from rehab_sci.dashboard.tabs.patient import render_patient
    from rehab_sci.dashboard.tabs.simulator import render_simulator

    return {
        "overview": lambda lang: render_overview(lang),
        "insights": lambda lang: render_insights(lang),
        "methods": lambda lang: render_methods(lang),
        "patient": lambda lang: render_patient(lang),
        "simulator": lambda lang: render_simulator(lang, None),
    }


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("tab", TABS)
def test_render_tab(renderers, tab, lang):
    component = renderers[tab](lang)
    assert component is not None


@pytest.mark.parametrize("lang", LANGS)
def test_patient_dynamic_callbacks(state, sample_key_record, lang):
    from rehab_sci.dashboard.tabs import patient as P

    kr = sample_key_record
    # each callback returns a tuple whose first element is the text readout component
    assert P.update_patient_conversion(kr, lang)[0] is not None
    assert P.update_patient_multistate(kr, lang)[0] is not None
    assert P.update_patient_independence(kr, lang)[0] is not None
    assert P.update_patient_topography(kr, "light_touch", lang)[0] is not None
    assert P.update_patient_level_descent(kr, lang)[0] is not None
    assert P.update_patient_dissociation(kr, lang)[0] is not None


@pytest.mark.parametrize("lang", LANGS)
def test_methods_drilldown_callbacks(state, lang):
    from rehab_sci.dashboard.tabs import methods as M

    # drive each drilldown with the first valid key from its loaded metrics; skip the
    # drilldown whose diagnostic bundle is absent (degrades gracefully).
    if state.DISSOCIATION:
        axis = next(iter(state.DISSOCIATION["axes"]))
        rel, shap = M.update_methods_dissociation_axis(axis, lang)
        assert rel is not None and shap is not None
    if state.INDEPENDENCE:
        item = next(iter(state.INDEPENDENCE["heads"]))
        rel, shap = M.update_methods_independence_item(item, lang)
        assert rel is not None and shap is not None
