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
from rehab_sci.dashboard.i18n import col_label, level_label, t
from rehab_sci.dashboard.layout import chart_card, kpi_card
from rehab_sci.dashboard.state import (
    ARCHETYPE_DATA,
    EP,
    INDEPENDENCE,
    LANDMARK,
    LONG,
    MULTISTATE,
    PHENOTYPE_DATA,
    SCHEMA,
    SUBGROUPS,
)
from rehab_sci.data.phenotypes import phenotype_summary

_AGE_MIN = 10
_AGE_MAX = 95

# Landmark used for every value-of-observation claim in the lead.
_VOI_LANDMARK = "3m"

# G2 single-add measures, split by what a clinician would have to do to obtain them.
# The lead reports the ranking against this split, so it has to be explicit rather than
# inferred from the measure name.
_VOI_MODALITY = {
    "SCIM_total": "function",
    "SCIM_self_care": "function",
    "SCIM_respiration_sphincter": "function",
    "SCIM_mobility": "function",
    "AIS_ord": "neuro",
    "UEMS": "neuro",
    "LEMS": "neuro",
    "TotalMotor": "neuro",
    "LightTouchTotal": "neuro",
    "PinPrickTotal": "neuro",
}


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

    facts["ladder"] = _milestone_ladder_facts()
    facts["improve"] = _improve_by_grade_facts()

    lm_outcome = ((LANDMARK or {}).get("outcomes") or {}).get("scim_total") or {}
    facts["measure_value"] = _measure_value_facts(lm_outcome)
    facts["certainty"] = _certainty_facts(lm_outcome)
    return facts


def _milestone_ladder_facts() -> dict | None:
    """Observed discharge-independence rate per SCIM-ADL milestone, hardest first (G7)."""
    heads = (INDEPENDENCE or {}).get("heads") or {}
    items = (INDEPENDENCE or {}).get("items") or []
    rows = [
        {
            "key": item["key"],
            "domain": item["domain"],
            "col": heads[item["key"]]["col"],
            "rate": float(heads[item["key"]]["base_rate"]),
            "n": int(heads[item["key"]]["n"]),
        }
        for item in items
        if item["key"] in heads
    ]
    if not rows:
        return None
    rows.sort(key=lambda row: row["rate"])
    return {
        "items": rows,
        "hardest": rows[0],
        "easiest": rows[-1],
        "n": max(row["n"] for row in rows),
        "definition": (INDEPENDENCE or {}).get("definition"),
        # Items the trainer dropped as near-universal sit ABOVE the top of the plotted
        # ladder, so the copy names them — otherwise the rendered range reads as the whole
        # instrument when its most-reached activity is missing from it.
        "excluded": [
            {"col": cell["col"], "rate": float(cell["independent_rate"]), "n": int(cell["n"])}
            for cell in ((INDEPENDENCE or {}).get("excluded") or {}).values()
            if cell.get("col") and cell.get("independent_rate") is not None and cell.get("n")
        ],
    }


def _improve_by_grade_facts() -> dict | None:
    """P(≥1 AIS-grade improvement by 6 months) per admission grade (G6 improve head)."""
    head = (MULTISTATE or {}).get("improve_head") or {}
    by_grade = head.get("rate_by_admission_grade") or {}
    rows = [
        {"grade": grade, "rate": float(cell["rate"]), "n": int(cell["n"])}
        for grade, cell in by_grade.items()
        if cell and cell.get("n")
    ]
    if not rows:
        return None
    rows.sort(key=lambda row: row["grade"])
    best = max(rows, key=lambda row: row["rate"])
    worst = min(rows, key=lambda row: row["rate"])
    flow = (MULTISTATE or {}).get("flow") or None
    return {
        "grades": rows,
        "best": best,
        "worst": worst,
        "n": int(head.get("n") or sum(row["n"] for row in rows)),
        "auc": float(head["auc"]) if head.get("auc") is not None else None,
        "flow": flow,
    }


def _measure_value_facts(lm_outcome: dict) -> dict | None:
    """Discharge-SCIM R² from adding exactly one 3-month measure, ranked (G2)."""
    cell = (lm_outcome.get("by_landmark") or {}).get(_VOI_LANDMARK) or {}
    singles = cell.get("single") or {}
    rows = [
        {
            "measure": measure,
            "modality": _VOI_MODALITY.get(measure, "neuro"),
            "r2": float(block["r2"]),
        }
        for measure, block in singles.items()
        if block and block.get("r2") is not None
    ]
    if not rows or not cell.get("baseline"):
        return None
    rows.sort(key=lambda row: row["r2"])
    best = rows[-1]
    best_other = next(
        (row for row in reversed(rows) if row["modality"] != best["modality"]), None
    )
    return {
        "measures": rows,
        "baseline_r2": float(cell["baseline"]["r2"]),
        "best": best,
        "best_other": best_other,
        "n": int(cell["n_eligible"]),
        "n_test": int(cell["n_test"]),
        "landmark": _VOI_LANDMARK,
    }


def _certainty_facts(lm_outcome: dict) -> dict | None:
    """80% PI half-width per landmark, admission-only vs with observation (G1)."""
    by_landmark = lm_outcome.get("by_landmark") or {}
    labels, base, observed, counts = [], [], [], []
    for label, cell in by_landmark.items():  # chronological (trainer insertion order)
        if not (cell.get("baseline") and cell.get("landmark")):
            continue
        labels.append(label)
        base.append(float(cell["baseline"]["pi_halfwidth_raw"]))
        observed.append(float(cell["landmark"]["pi_halfwidth_raw"]))
        counts.append(int(cell["n_test"]))
    if not labels:
        return None
    final = by_landmark[labels[-1]]
    cov_base = final["baseline"].get("coverage_80")
    cov_obs = final["landmark"].get("coverage_80")
    return {
        "labels": labels,
        "baseline": base,
        "observed": observed,
        "n_test": counts,
        "first_observed": observed[0],
        "last_observed": observed[-1],
        "last_baseline": base[-1],
        "pi_shrink": (base[-1] - observed[-1]) / base[-1] if base[-1] else 0.0,
        "r2_baseline": float(final["baseline"]["r2"]),
        "r2_observed": float(final["landmark"]["r2"]),
        # Achieved held-out coverage against the nominal 80%; the copy quotes it so the
        # interval is presented as a calibrated estimate rather than a guarantee.
        "coverage_baseline": None if cov_base is None else float(cov_base),
        "coverage_observed": None if cov_obs is None else float(cov_obs),
        "n": int(final["n_eligible"]),
        "first_landmark": labels[0],
        "landmark": labels[-1],
    }


def _finding_block(
    block_id: str,
    index: int,
    lang: str,
    kicker: str,
    claim: str,
    scope: str,
    takeaway: str,
    reading: str,
    basis: str,
    figure,
) -> html.Section:
    """One finding: numbered kicker, the claim as headline, its scope, evidence figure.

    The claim leads because the quantity behind each of these findings is a relation —
    a spread across activities, a peak against a final grade, a ranking of R², an
    interval with and without observation — and lifting either end of it into a display
    numeral strips the axis, the category and the reference line that give it meaning.
    Every figure already labels its own values in place, so the copy states the finding
    in one sentence and ``scope`` names what was measured and on whom.  ``takeaway``
    (the retained qualifier), ``reading`` (what was measured and how to read the figure)
    and ``basis`` (the denominator, the eligibility rule and every condition that limits
    the claim) keep their reviewed wording in full, one click away behind the block's own
    disclosure, so the finding still explains itself without the Methods tab.
    """
    return html.Section(
        id=block_id,
        className="finding-evidence",
        # Phrase-level line breaking (style.css) needs the content language declared, or
        # JA claims wrap per character and split katakana and numeral+counter pairs.
        lang=lang,
        children=[
            html.Div(className="finding-evidence__copy", children=[
                html.P(className="finding-kicker", children=[
                    html.Span(f"{index:02d}", className="finding-kicker__index"),
                    html.Span(kicker),
                ]),
                html.H3(claim),
                html.P(scope, className="finding-evidence__scope"),
                html.Details(className="finding-evidence__how", children=[
                    html.Summary(t(SCHEMA, "overview_finding_how_summary", lang)),
                    html.P(takeaway, className="finding-evidence__takeaway"),
                    html.P(reading),
                    html.P(basis, className="finding-evidence__caveat"),
                ]),
            ]),
            html.Div(className="finding-evidence__chart", children=[
                dcc.Graph(figure=figure, config={"displayModeBar": False, "responsive": True}),
            ]),
        ],
    )


def _glance_tile(index: int, target: str, claim: str) -> html.A:
    """One at-a-glance tile: the finding's claim, anchor-linked to its evidence block."""
    return html.A(
        className="glance-card",
        href=f"#{target}",
        children=[
            html.Span(f"{index:02d}", className="glance-card__index"),
            html.Span(claim, className="glance-card__claim"),
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
            # The shared ground rules (observed rate vs held-out estimate, single-centre
            # association, external validity) apply to every block, so they live here
            # once, collapsed.
            html.Details(className="findings-lead__note", children=[
                html.Summary(t(SCHEMA, "overview_lead_note_summary", lang)),
                html.P(t(SCHEMA, "overview_lead_note", lang)),
            ]),
        ],
    )

    # The glance band states each finding in one short line so all four fit one viewport;
    # tiles anchor-jump to their evidence blocks.
    tiles: list = []
    lead: list = [lead_head]

    ladder = facts.get("ladder")
    if ladder:
        rows = [
            {**row, "label": col_label(SCHEMA, row["col"], lang)}
            for row in ladder["items"]
        ]
        ladder_copy = dict(
            low=ladder["hardest"]["rate"],
            high=ladder["easiest"]["rate"],
            low_item=rows[0]["label"],
            high_item=rows[-1]["label"],
            count=len(rows),
        )
        ladder_basis = _copy("overview_finding_ladder_basis", lang, n=ladder["n"])
        join = "" if lang == "ja" else " "
        for cell in ladder["excluded"]:
            ladder_basis += join + _copy(
                "overview_finding_ladder_excluded",
                lang,
                item=col_label(SCHEMA, cell["col"], lang),
                rate=cell["rate"],
                n=cell["n"],
                n_ladder=ladder["n"],
            )
        if ladder["excluded"]:
            ladder_basis += join + _copy(
                "overview_finding_ladder_excluded_ceiling",
                lang,
                high=ladder["easiest"]["rate"],
            )
        tiles.append(_glance_tile(
            1,
            "finding-discharge-milestones",
            t(SCHEMA, "overview_glance_ladder", lang),
        ))
        lead.append(_finding_block(
            "finding-discharge-milestones",
            1,
            lang,
            t(SCHEMA, "overview_finding_ladder_kicker", lang),
            t(SCHEMA, "overview_finding_ladder_title", lang),
            _copy("overview_finding_ladder_scope", lang, count=len(rows), n=ladder["n"]),
            t(SCHEMA, "overview_finding_ladder_takeaway", lang),
            _copy("overview_finding_ladder_reading", lang, **ladder_copy),
            ladder_basis,
            fg.fig_milestone_ladder({"items": rows}, SCHEMA, lang),
        ))

    improve = facts.get("improve")
    flow = (improve or {}).get("flow")
    if improve and flow:
        # Which grade leads the claim is read off the artifact, never fixed: the point is
        # that reversion concentrates somewhere, not that it concentrates at A.
        reverted = {
            grade: cell["reverted_of_peaked"]
            for grade, cell in flow["by_admission_grade"].items()
            if cell.get("n_peaked") and cell.get("reverted_of_peaked") is not None
        }
        peak_grade = max(reverted, key=lambda g: reverted[g])
        hold_grade = min(reverted, key=lambda g: reverted[g])
        improve_copy = dict(
            peak_grade=peak_grade,
            peak_reverted=reverted[peak_grade],
            hold_grade=hold_grade,
            hold_reverted=reverted[hold_grade],
            n=flow["n"],
        )
        reading = t(SCHEMA, "overview_finding_improve_reading", lang)
        tiles.append(_glance_tile(
            2,
            "finding-improvement-by-grade",
            t(SCHEMA, "overview_glance_improve", lang),
        ))
        lead.append(_finding_block(
            "finding-improvement-by-grade",
            2,
            lang,
            t(SCHEMA, "overview_finding_improve_kicker", lang),
            _copy("overview_finding_improve_title", lang, **improve_copy),
            _copy("overview_finding_improve_scope", lang, **improve_copy),
            t(SCHEMA, "overview_finding_improve_takeaway", lang),
            reading,
            _copy("overview_finding_improve_basis", lang, n=flow["n"]),
            fg.fig_improve_flow(flow, lang),
        ))

    value = facts.get("measure_value")
    if value:
        rows = [
            {**row, "label": t(SCHEMA, f"lm_measure_{row['measure'].lower()}", lang)}
            for row in value["measures"]
        ]
        best_label = t(SCHEMA, f"lm_measure_{value['best']['measure'].lower()}", lang)
        basis = _copy(
            "overview_finding_measure_basis",
            lang,
            n=value["n"],
            n_test=value["n_test"],
            baseline=value["baseline_r2"],
            count=len(rows),
        )
        # The modality contrast is the point of the finding, but it only exists when both
        # modalities were modelled at this landmark.
        other = value.get("best_other")
        if other:
            basis += ("" if lang == "ja" else " ") + _copy(
                "overview_finding_measure_compare",
                lang,
                other=t(SCHEMA, f"lm_measure_{other['measure'].lower()}", lang),
                other_r2=other["r2"],
            )
        tiles.append(_glance_tile(
            3,
            "finding-measure-value",
            _copy("overview_glance_measure", lang, count=len(rows), measure=best_label),
        ))
        lead.append(_finding_block(
            "finding-measure-value",
            3,
            lang,
            t(SCHEMA, "overview_finding_measure_kicker", lang),
            _copy(
                "overview_finding_measure_title",
                lang,
                measure=best_label,
                count=len(rows),
            ),
            _copy(
                "overview_finding_measure_scope",
                lang,
                count=len(rows),
                n_test=value["n_test"],
            ),
            _copy(
                "overview_finding_measure_takeaway",
                lang,
                measure=best_label,
                count=len(rows),
                baseline=value["baseline_r2"],
                best=value["best"]["r2"],
            ),
            _copy(
                "overview_finding_measure_reading",
                lang,
                measure=best_label,
                count=len(rows),
                baseline=value["baseline_r2"],
                best=value["best"]["r2"],
            ),
            basis,
            fg.fig_measure_value({**value, "measures": rows}, lang),
        ))

    certainty = facts.get("certainty")
    if certainty:
        # Prose placeholders take the localized landmark names; raw codes stay artifact keys.
        landmarks = dict(
            first_landmark=level_label(SCHEMA, "time_name", certainty["first_landmark"], lang),
            landmark=level_label(SCHEMA, "time_name", certainty["landmark"], lang),
        )
        certainty_basis = _copy(
            "overview_finding_certainty_basis",
            lang,
            n=certainty["n"],
            n_first=certainty["n_test"][0],
            n_last=certainty["n_test"][-1],
            baseline=certainty["r2_baseline"],
            observed=certainty["r2_observed"],
            **landmarks,
        )
        # Achieved coverage is the honest qualifier on an 80% interval, so it is quoted
        # whenever the trainer recorded it rather than asserted as a property.
        cov_base, cov_obs = certainty["coverage_baseline"], certainty["coverage_observed"]
        if cov_base is not None and cov_obs is not None:
            certainty_basis += ("" if lang == "ja" else " ") + _copy(
                "overview_finding_certainty_coverage",
                lang,
                cov_baseline=cov_base,
                cov_observed=cov_obs,
                **landmarks,
            )
        tiles.append(_glance_tile(
            4,
            "finding-certainty-curve",
            _copy("overview_glance_certainty", lang, shrink=certainty["pi_shrink"]),
        ))
        lead.append(_finding_block(
            "finding-certainty-curve",
            4,
            lang,
            t(SCHEMA, "overview_finding_certainty_kicker", lang),
            _copy(
                "overview_finding_certainty_title",
                lang,
                shrink=certainty["pi_shrink"],
                **landmarks,
            ),
            _copy(
                "overview_finding_certainty_scope",
                lang,
                n=certainty["n"],
                **landmarks,
            ),
            t(SCHEMA, "overview_finding_certainty_takeaway", lang),
            _copy(
                "overview_finding_certainty_reading",
                lang,
                baseline_pi=certainty["last_baseline"],
                observed_pi=certainty["last_observed"],
                shrink=certainty["pi_shrink"],
                **landmarks,
            ),
            certainty_basis,
            fg.fig_certainty_curve(certainty, SCHEMA, lang),
        ))

    # Admission severity sets the level every finding above is measured against; it stays
    # available as the reference gradient rather than competing as a headline.
    motor = facts.get("motor_strata")
    if motor:
        lead.append(html.Details(className="overview-details", children=[
            html.Summary(t(SCHEMA, "overview_gradient_summary", lang)),
            html.P(
                _copy(
                    "overview_gradient_basis",
                    lang,
                    n=motor["n"],
                    effect=motor["eta_squared"],
                    # unlike the four headline findings, whose model cohorts all require a
                    # real IDNumber, this observed aggregate still carries the id-less rows
                    n_orphans=int(
                        (
                            EP["TotalMotor"].notna()
                            & EP["y_discharge_scim"].notna()
                            & EP["IDNumber"].isna()
                        ).sum()
                    ),
                ),
                className="overview-details__intro overview-details__intro--prose",
            ),
            dcc.Graph(
                figure=fg.fig_motor_strata_finding(motor, lang),
                config={"displayModeBar": False, "responsive": True},
            ),
        ]))
    if tiles:
        # Same contract as a finding block: the tiles are prose now, so phrase-level JA
        # line breaking needs the content language declared on their container.
        lead.insert(1, html.Nav(className="finding-glance", lang=lang, children=tiles))
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
