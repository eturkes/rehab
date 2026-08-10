# Project status and backlog

## Status

Mature SCI-recovery analytics project; all numbered features (F1–F27) and model families (G1–G11) are shipped and green. Full session-by-session history lives in `git log` and is not duplicated here.

* **Data layer** — schema-driven (`schema/*.yaml`), cp932 raw → ghost-filtered universe of **899 episodes / 866 patients**, 26-timepoint rectangular long frame ([project reference §1](../docs/project-reference.md#1-data-invariants-established--rely-on-them)). `data/quality.py` clinical-consistency report.
* **Production heads** (`train.py`, byte-reproducible) — 4 SCIM (total + 3 subscales) + AIS multiclass + LOS + the 5 G9 Δ-score heads; split-conformal / Mondrian PI, APS sets (AIS), TreeSHAP + interactions. Plus 9-timepoint trajectory models and k=3 recovery archetypes.
* **Diagnostic / inference families** — G1 landmark + G2 value-of-information, G3 growth-mixture phenotyping, G4 AIS conversion, G6 multi-state Markov + improve-by-6m, G7 18 functional-independence heads, G8 132-segment topography, G10 5 level-descent heads, G11 3-axis neuro-functional dissociation. F24 out-of-time temporal backtest (diagnostic).
* **Dashboard** (Plotly Dash, JA default / EN toggle) — findings-first landing with four source-bound findings on unfiltered data, layered: visible = numbered kicker + headline metric + claim + one takeaway + figure; reviewed reading/basis prose complete behind a per-finding disclosure (gate caps visible claim+takeaway <480 chars, floors basis >200); shared ground-rules note collapsed under the lead; collapsed admission-severity gradient (one visible deck sentence names it); separate filtered cohort explorer. Story tab order overview→insights→patient→simulator→methods. Patient tab: prognosis full-width, ISNCSCI table behind a disclosure. Simulator: F25 completeness/OOD badge + cohort-baseline note on zero-input (test-gated); "fill defaults" labelled illustrative. What-if reference overlay.
* **Quality gates** — F26 pytest harness (invariant + smoke + behavioral, ~12 s, skips when CSV/bundles absent), ruff (with the `--select F` regression gate), pip-audit.
* **Known-good** — ruff clean, pytest green (47), pip-audit clean. Stack: scikit-learn 1.9.0, shap 0.52.0, lightgbm 4.6.0, numpy 2.4.6, pytest 9.1.0 (CVE-2025-71176 floor). All 66 `models/**/*.joblib` are pickled at sklearn 1.9.0 and load clean; `landmark_metrics.json` + `temporal_metrics.json` cover all 11 outcomes. Which metrics survive a retrain or a sklearn bump byte-identical: [project reference §0b](../docs/project-reference.md#0b-lessons-and-pitfalls).

## Backlog

Milestone status (`UNPLANNED` / `IN-PROGRESS` / `IMPLEMENTED` / `REVIEWED`) selects the `/session-prompt` MODE; the active milestone is the first one not `REVIEWED`. Keep every milestone heading carrying an explicit status.

The raw data holds **no new field families** (219-col profile audit) — any new model reuses existing ISNCSCI / SCIM / AIS signal, and the standard neurological endpoints are covered (G9 Δ-score, G10 level descent, G11 dissociation). **Before starting any new feature, scope what / why / effort / files / data-dependency.**

The **new-predictive-head well is dry**: every reuse-only candidate is either a rescaling/reframing of a shipped head — recovery-fraction ≡ G9 (Δ ÷ known headroom); motor-vs-sensory ≡ G11's z-contrast method — i.e. the documented "framing not signal" trap, or cohort-infeasible (asymmetry ~93–122 borderline, ZPP ~20–39, WISCI sparse). Remaining open options, neither urgent:

### M1 — PRR descriptive insight · UNPLANNED

Content, NOT a predictive head. This cohort does **not** obey the proportional-recovery rule (admission→discharge ISNCSCI motor): Δmotor R²(initial deficit) ≈ 0.01–0.12, recovery-fraction median ~0.6 but IQR spans the full range (no ~0.7 clustering), ~29–44% non-fitters. A worthwhile Insights/Methods panel — reuses G9 targets + admission scores, no new training — that also motivates the ML Δ-heads. Effort M.
### M2 — Calibration-drift monitoring · UNPLANNED

Infrastructure — the only on-record untried head idea; track whether head calibration degrades over time, extending F24's temporal backtest. Effort M.

### M3 — Dashboard focus backlog · UNPLANNED

Deferred flags from the focus/story pass (full ranked list + anchors: `.scratch/agents/map-task-1.md` §7; expires with scratch — re-derive by re-mapping if absent). Accepted-but-deferred, roughly by value:

1. Outcome-driven prediction headings — patient + simulator fixed H2s say "discharge SCIM-III" while the outcome selector offers 11 heads; selected non-SCIM story looks broken. `tabs/patient.py` prediction card, `tabs/simulator.py` result card.
2. Simulator quick-profile intake — 25-field wall before the answer; expose ~6–8 high-importance fields, collapse the rest per clinical group; keep missingness semantics.
3. Methods per-outcome disclosures dump 11 repeated cards (tab totals ~60 cards/92 graphs behind details); add an outcome selector inside each section, render one at a time.
4. Patient/simulator duplicate question-group decks — label mode per tab (observed episode vs hypothetical profile), cross-link.
5. Smaller: archetype filter (model-derived) out of the observable explorer filter bar; scorecard task/metric labels to stop cross-outcome score comparison; data-quality badge beside scorecard; 132-cell ISNCSCI worksheet nested behind its own "advanced exam" disclosure.

Effort S–M each, independent; all copy-safe against the reviewed reading/basis prose (untouched by design).
