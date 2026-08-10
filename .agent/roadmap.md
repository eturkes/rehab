# Project status and backlog

## Status

Mature SCI-recovery analytics project; all numbered features (F1–F27) and model families (G1–G11) are shipped and green. Full session-by-session history lives in `git log` and is not duplicated here.

* **Data layer** — schema-driven (`schema/*.yaml`), cp932 raw → ghost-filtered + duplicate-registration-merged universe of **893 episodes / 866 patients** (1:1 apart from 27 id-less orphans), 26-timepoint rectangular long frame ([project reference §1](../docs/project-reference.md#1-data-invariants-established--rely-on-them)). `data/quality.py` clinical-consistency report.
* **Production heads** (`train.py`, byte-reproducible) — 4 SCIM (total + 3 subscales) + AIS multiclass + LOS + the 5 G9 Δ-score heads; split-conformal / Mondrian PI, APS sets (AIS), TreeSHAP + interactions. Plus 9-timepoint trajectory models and k=3 recovery archetypes.
* **Diagnostic / inference families** — G1 landmark + G2 value-of-information, G3 growth-mixture phenotyping, G4 AIS conversion, G6 multi-state Markov + improve-by-6m, G7 18 functional-independence heads, G8 132-segment topography, G10 5 level-descent heads, G11 3-axis neuro-functional dissociation. F24 out-of-time temporal backtest (diagnostic).
* **Dashboard** (Plotly Dash, JA default / EN toggle) — findings-first landing with four source-bound findings on unfiltered data, glance-first: one-line deck + scope stats, then a 4-tile glance band (headline metric + micro-claim, anchor→block; tile metric == block metric, test-gated) putting all four findings in one viewport; each block visible = numbered kicker + headline metric + short unit + one claim + self-annotated figure (muted-middle emphasis, +gain annotation, shaded uncertainty band w/ −shrink label); reviewed takeaway/reading/basis prose complete behind the per-finding disclosure (gate caps claim <170 / unit <110 chars, floors basis >200); shared ground-rules note collapsed under the lead; collapsed admission-severity gradient; separate filtered cohort explorer. Typography = self-hosted IBM Plex subsets (sans + mono), stacked metric-over-unit topline. Story tab order overview→insights→patient→simulator→methods. Patient tab: prognosis full-width, ISNCSCI table behind a disclosure. Simulator: F25 completeness/OOD badge + cohort-baseline note on zero-input (test-gated); "fill defaults" labelled illustrative. What-if reference overlay.
* **Quality gates** — F26 pytest harness (invariant + smoke + behavioral, ~12 s, skips when CSV/bundles absent), ruff (with the `--select F` regression gate), pip-audit.
* **Known-good** — ruff clean, pytest green (55), pip-audit clean. Stack: scikit-learn 1.9.0, shap 0.52.0, lightgbm 4.6.0, numpy 2.4.6, pytest 9.1.0 (CVE-2025-71176 floor). All 66 `models/**/*.joblib` are pickled at sklearn 1.9.0 and load clean; `landmark_metrics.json` + `temporal_metrics.json` cover all 11 outcomes. Which metrics survive a retrain or a sklearn bump byte-identical: [project reference §0b](../docs/project-reference.md#0b-lessons-and-pitfalls).

## Backlog

Milestone status (`UNPLANNED` / `IN-PROGRESS` / `IMPLEMENTED` / `REVIEWED`) selects the `/session-roadmap` MODE; the active milestone is the first one not `REVIEWED`. Keep every milestone heading carrying an explicit status. Off-spine deferrals live in `.agent/polish.md` (`/session-polish` consumes them), not here.

The raw data holds **no new field families** (219-col profile audit) — any new model reuses existing ISNCSCI / SCIM / AIS signal, and the standard neurological endpoints are covered (G9 Δ-score, G10 level descent, G11 dissociation). **Before starting any new feature, scope what / why / effort / files / data-dependency.**

The **new-predictive-head well is dry**: every reuse-only candidate is either a rescaling/reframing of a shipped head — recovery-fraction ≡ G9 (Δ ÷ known headroom); motor-vs-sensory ≡ G11's z-contrast method — i.e. the documented "framing not signal" trap, or cohort-infeasible (asymmetry ~93–122 borderline, ZPP ~20–39, WISCI sparse). Remaining open options, neither urgent:

### M1 — PRR descriptive insight · UNPLANNED

Content, NOT a predictive head. This cohort does **not** obey the proportional-recovery rule (admission→discharge ISNCSCI motor): Δmotor R²(initial deficit) ≈ 0.01–0.12, recovery-fraction median ~0.6 but IQR spans the full range (no ~0.7 clustering), ~29–44% non-fitters. A worthwhile Insights/Methods panel — reuses G9 targets + admission scores, no new training — that also motivates the ML Δ-heads. Effort M.

### M2 — Calibration-drift monitoring · UNPLANNED

Infrastructure — the only on-record untried head idea; track whether head calibration degrades over time, extending F24's temporal backtest. Effort M.
