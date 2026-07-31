"""Smoke: every tab renders in both languages, and the per-patient dynamic
callbacks build their figures without raising.

Rendering each tab body headless is the cheap guard that would have caught the
s31 ``INK['600']`` Methods crash — boot-200 checks miss it because tab bodies
build lazily inside callbacks (§0b).  Invoking the dynamic callbacks directly
additionally exercises the G-series figure builders the static render never
reaches.
"""

from __future__ import annotations

import numpy as np
import pytest
from dash import dcc

LANGS = ("ja", "en")
TABS = ("overview", "insights", "methods", "patient", "simulator")


def _walk(component):
    """Yield a Dash component tree without depending on serialized layout shape."""
    if component is None:
        return
    if isinstance(component, (list, tuple)):
        for child in component:
            yield from _walk(child)
        return
    yield component
    children = getattr(component, "children", None)
    if children is not None:
        yield from _walk(children)


def _component_ids(component) -> set[str]:
    return {
        value
        for node in _walk(component)
        if isinstance((value := getattr(node, "id", None)), str)
    }


def test_findings_copy_is_bilingual():
    from rehab_sci.schema import load_schema

    schema = load_schema()
    keys = [key for key in schema.ui if key.startswith(("overview_", "insight_"))]
    assert keys
    for key in keys:
        for lang in LANGS:
            assert schema.ui[key].get(lang)


def test_interaction_heatmap_leaves_unreported_pairs_blank(state):
    from rehab_sci.dashboard.figures.insights import fig_interaction_heatmap

    reported = [
        {"feat_a": "TotalMotor", "feat_b": "AIS", "abs_mean_interaction": 0.4},
        {"feat_a": "AIS", "feat_b": "年齢", "abs_mean_interaction": 0.2},
    ]
    fig = fig_interaction_heatmap(
        {"global_interaction_top25": reported}, state.SCHEMA, "en"
    )
    z = np.asarray(fig.data[0].z, dtype=float)
    finite = z[np.isfinite(z)]
    assert sorted(finite) == pytest.approx([0.2, 0.4])
    assert np.count_nonzero(np.isfinite(z)) == len(reported)


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


def test_overview_findings_are_source_bound(state):
    from rehab_sci.dashboard.tabs import overview as O

    facts = O._finding_metrics()
    assert facts["episodes"] == len(state.EP)
    assert facts["patients"] == state.EP["IDNumber"].nunique()

    motor_source = next(
        row
        for row in state.SUBGROUPS["scim_total"]["results"]
        if row.get("feature") == "TotalMotor" and not row.get("skipped")
    )
    motor = facts["motor_strata"]
    assert motor["n"] == motor_source["n_total"]
    assert motor["eta_squared"] == pytest.approx(motor_source["eta_squared"])
    assert [group["median"] for group in motor["groups"]] == [
        group["median"] for group in motor_source["groups"]
    ]

    lm_source = state.LANDMARK["outcomes"]["scim_total"]["by_landmark"]["3m"]
    observation = facts["observation"]
    assert observation["n"] == lm_source["n_eligible"]
    assert observation["n_test"] == lm_source["n_test"]
    assert observation["r2_gain"] == pytest.approx(
        lm_source["landmark"]["r2"] - lm_source["baseline"]["r2"]
    )
    assert observation["pi_shrink"] == pytest.approx(
        1
        - lm_source["landmark"]["pi_halfwidth_raw"]
        / lm_source["baseline"]["pi_halfwidth_raw"]
    )

    diss_source = state.DISSOCIATION["axes"]["lems_mobility"]["landscape"]
    dissociation = facts["dissociation"]
    assert dissociation["pearson_r"] == pytest.approx(diss_source["pearson_r"])
    assert dissociation["dissociated_share"] == pytest.approx(diss_source["dissociated_share"])


@pytest.mark.parametrize("lang", LANGS)
def test_overview_findings_precede_filtered_explorer(state, lang):
    from rehab_sci.dashboard.tabs import overview as O

    page = O.render_overview(lang)
    ids = _component_ids(page)
    assert {
        "findings-lead",
        "finding-motor-stratification",
        "finding-landmark-value",
        "finding-neuro-functional-dissociation",
        "finding-motor-evidence",
        "cohort-explorer",
        "overview-content",
    } <= ids
    explorer = next(node for node in _walk(page) if getattr(node, "id", None) == "cohort-explorer")
    assert not {
        "findings-lead",
        "finding-motor-stratification",
        "finding-landmark-value",
        "finding-neuro-functional-dissociation",
    } & _component_ids(explorer)


@pytest.mark.parametrize("lang", LANGS)
def test_overview_callback_builds_real_figures(state, lang):
    from rehab_sci.dashboard.tabs import overview as O

    most_common_ais = state.EP["AIS"].dropna().astype(str).mode().iloc[0]
    for component in (
        O.update_overview_content([], [], [10, 95], [], lang),
        O.update_overview_content([most_common_ais], [], [10, 95], [], lang),
    ):
        graphs = [node for node in _walk(component) if isinstance(node, dcc.Graph)]
        assert graphs
        assert all(len(graph.figure.data) > 0 for graph in graphs)

    empty = O.update_overview_content(["not-a-grade"], [], [10, 95], [], lang)
    assert empty.className == "overview-empty"


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
