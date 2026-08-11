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
from dash import dcc, html

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


def _split_graphs(component, hidden: bool = False) -> tuple[list, list]:
    """Partition dcc.Graph nodes into (rendered on load, sitting inside an html.Details)."""
    shown: list = []
    behind: list = []
    if component is None:
        return shown, behind
    if isinstance(component, (list, tuple)):
        for child in component:
            a, b = _split_graphs(child, hidden)
            shown += a
            behind += b
        return shown, behind
    if isinstance(component, str):
        return shown, behind
    if isinstance(component, dcc.Graph):
        (behind if hidden else shown).append(component)
    a, b = _split_graphs(
        getattr(component, "children", None), hidden or isinstance(component, html.Details)
    )
    return shown + a, behind + b


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

    ind_source = state.INDEPENDENCE["heads"]
    ladder = facts["ladder"]
    assert {row["key"] for row in ladder["items"]} == set(ind_source)
    assert [row["rate"] for row in ladder["items"]] == sorted(
        row["rate"] for row in ladder["items"]
    )  # hardest milestone first
    for row in ladder["items"]:
        assert row["rate"] == pytest.approx(ind_source[row["key"]]["base_rate"])
    assert ladder["hardest"]["rate"] <= ladder["easiest"]["rate"]

    ms_source = state.MULTISTATE["improve_head"]["rate_by_admission_grade"]
    improve = facts["improve"]
    assert {row["grade"] for row in improve["grades"]} == set(ms_source)
    for row in improve["grades"]:
        assert row["rate"] == pytest.approx(ms_source[row["grade"]]["rate"])
        assert row["n"] == ms_source[row["grade"]]["n"]
    assert improve["best"]["rate"] == max(row["rate"] for row in improve["grades"])

    lm_source = state.LANDMARK["outcomes"]["scim_total"]["by_landmark"]["3m"]
    value = facts["measure_value"]
    assert value["n"] == lm_source["n_eligible"]
    assert value["n_test"] == lm_source["n_test"]
    assert value["baseline_r2"] == pytest.approx(lm_source["baseline"]["r2"])
    for row in value["measures"]:
        assert row["r2"] == pytest.approx(lm_source["single"][row["measure"]]["r2"])
    assert value["best"]["r2"] == max(row["r2"] for row in value["measures"])

    by_landmark = state.LANDMARK["outcomes"]["scim_total"]["by_landmark"]
    certainty = facts["certainty"]
    assert certainty["labels"] == list(by_landmark)
    assert certainty["baseline"] == [
        pytest.approx(cell["baseline"]["pi_halfwidth_raw"]) for cell in by_landmark.values()
    ]
    assert certainty["observed"] == [
        pytest.approx(cell["landmark"]["pi_halfwidth_raw"]) for cell in by_landmark.values()
    ]
    assert certainty["pi_shrink"] == pytest.approx(
        1
        - lm_source["landmark"]["pi_halfwidth_raw"]
        / lm_source["baseline"]["pi_halfwidth_raw"]
    )


@pytest.mark.parametrize("lang", LANGS)
def test_overview_findings_precede_filtered_explorer(state, lang):
    from rehab_sci.dashboard.tabs import overview as O

    page = O.render_overview(lang)
    ids = _component_ids(page)
    findings = {
        "finding-discharge-milestones",
        "finding-improvement-by-grade",
        "finding-measure-value",
        "finding-certainty-curve",
    }
    assert findings | {"findings-lead", "cohort-explorer", "overview-content"} <= ids
    explorer = next(node for node in _walk(page) if getattr(node, "id", None) == "cohort-explorer")
    assert not (findings | {"findings-lead"}) & _component_ids(explorer)


@pytest.mark.parametrize("lang", LANGS)
def test_finding_blocks_stay_tight_and_keep_their_basis(state, lang):
    """Structure + focus: the visible layer is kicker / claim / scope line, and the full
    takeaway + reading + basis prose waits complete behind the block's own disclosure.

    Two regressions trip this: flattening the disclosure back into visible wall-of-text
    (the claim/scope budgets), and trimming the reviewed prose away (the length floors).
    The claim is the headline element, so it is asserted to be an ``H3`` — a fourth
    visible copy node, in particular a display figure lifted off the chart, would fail
    the unpacking below.  Says nothing about whether the prose is *correct* — claim
    soundness against the metric artifacts stays a review obligation, not a gate.
    """
    from rehab_sci.dashboard.tabs import overview as O

    page = O.render_overview(lang)
    blocks = [
        node
        for node in _walk(page)
        if str(getattr(node, "id", "")).startswith("finding-")
    ]
    assert len(blocks) == 4
    for position, block in enumerate(blocks, start=1):
        assert len(block.children) == 2  # the copy column and the chart — nothing else visible
        copy = block.children[0]
        kicker, headline, scope_p, how = copy.children
        index_span, kicker_span = kicker.children
        assert index_span.children == f"{position:02d}"  # chapters, in story order
        assert kicker_span.children
        assert isinstance(headline, html.H3)  # the claim leads the block
        claim = headline.children
        scope = scope_p.children
        assert scope_p.className == "finding-evidence__scope"
        # Glance budget: what renders on load is one claim over one scope line.
        assert claim and scope
        assert len(claim) < 170
        assert len(scope) < 125
        # The full reviewed prose stays reachable inside the block — a real disclosure,
        # closed on load, not a Div wearing the class name.
        assert isinstance(how, html.Details)
        assert getattr(how, "open", False) is not True
        assert how.className == "finding-evidence__how"
        summary, takeaway_p, reading_p, basis_p = how.children
        assert isinstance(summary, html.Summary)
        assert summary.children
        takeaway, reading, basis = takeaway_p.children, reading_p.children, basis_p.children
        assert takeaway_p.className == "finding-evidence__takeaway"
        assert basis_p.className == "finding-evidence__caveat"
        assert takeaway
        assert len(reading) > len(claim)
        assert len(basis) > 200
        # An unresolved {placeholder} would ship as literal braces.
        for text in (claim, scope, takeaway, reading, basis):
            assert "{" not in text and "}" not in text
        # The evidence figure itself stays visible, never behind the disclosure.
        shown, behind = _split_graphs(block)
        assert shown and not behind
    # The shared ground-rules note honours the same contract: a disclosure, closed on load.
    note = next(
        node for node in _walk(page)
        if getattr(node, "className", None) == "findings-lead__note"
    )
    assert isinstance(note, html.Details)
    assert getattr(note, "open", False) is not True


@pytest.mark.parametrize("lang", LANGS)
def test_overview_glance_band_mirrors_the_blocks(state, lang):
    """The at-a-glance band shows all four findings in one strip: tile ``i`` states
    finding ``i`` in one line and anchors to it, in the same order as the blocks.

    The tile is prose, never a figure lifted off its chart, so the band is gated on
    saying something rather than on matching a number: a claim inside the one-line
    budget, an anchor that resolves to a real block, and its own wording rather than a
    copy of the block headline the reader is about to reach.
    """
    from rehab_sci.dashboard.tabs import overview as O

    page = O.render_overview(lang)
    nav = next(
        node for node in _walk(page)
        if getattr(node, "className", None) == "finding-glance"
    )
    assert nav.lang == lang  # phrase-level JA wrapping needs the language declared
    tiles = nav.children
    blocks = [
        node for node in _walk(page) if str(getattr(node, "id", "")).startswith("finding-")
    ]
    assert len(tiles) == len(blocks) == 4
    block_claim = {
        block.id: next(
            node.children for node in _walk(block) if isinstance(node, html.H3)
        )
        for block in blocks
    }
    assert [tile.href for tile in tiles] == [f"#{block.id}" for block in blocks]  # same order
    for position, tile in enumerate(tiles, start=1):
        index_span, claim_span = tile.children
        assert index_span.children == f"{position:02d}"
        claim = claim_span.children
        assert claim and len(claim) < 60
        assert "{" not in claim and "}" not in claim
        assert claim != block_claim[tile.href.lstrip("#")]


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
@pytest.mark.parametrize(("tab", "max_open"), [("patient", 3), ("simulator", 3), ("methods", 0)])
def test_detail_waits_behind_a_named_disclosure(renderers, state, tab, max_open, lang):
    """These tabs open on the answer, not the model shelf: at most a handful of figures render on
    load and the rest wait inside a labelled ``Details``.  Flattening a tab back out trips this."""
    page = renderers[tab](lang)
    shown, behind = _split_graphs(page)
    assert len(shown) <= max_open
    assert behind, "the depth must stay reachable, not deleted"
    summaries = [
        node.children
        for parent in _walk(page)
        if isinstance(parent, html.Details)
        for node in _walk(parent.children)
        if isinstance(node, html.Summary)
    ]
    assert summaries
    assert all(isinstance(text, str) and text.strip() for text in summaries)


@pytest.mark.parametrize("lang", LANGS)
def test_simulator_empty_form_is_labelled_cohort_baseline(state, lang):
    """The prior a blank form produces must announce itself; one real input clears it."""
    from rehab_sci.dashboard.tabs import simulator as S

    num_ids = [{"type": "num", "col": "年齢"}]
    empty = S.simulate([None], [], num_ids, [], "scim_total", lang, None)
    filled = S.simulate([65], [], num_ids, [], "scim_total", lang, None)
    note = "sim-baseline-note"
    assert any(getattr(node, "className", None) == note for node in _walk(empty[0]))
    assert not any(getattr(node, "className", None) == note for node in _walk(filled[0]))


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
