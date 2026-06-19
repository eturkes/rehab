# roadmap.md — canonical plan & status

## Status

Mature SCI-recovery analytics project; all numbered features (F1–F26) and model families (G1–G11) are shipped and green. Full session-by-session history lives in `git log` (the former AGENT_NOTES.md §7 index) — not duplicated here.

* **Data layer** — schema-driven (`schema/*.yaml`), cp932 raw → ghost-filtered universe of **899 episodes / 866 patients**, 26-timepoint rectangular long frame (§1). `data/quality.py` clinical-consistency report.
* **Production heads** (`train.py`, byte-reproducible) — 4 SCIM (total + 3 subscales) + AIS multiclass + LOS + the 5 G9 Δ-score heads; split-conformal / Mondrian PI, APS sets (AIS), TreeSHAP + interactions. Plus 9-timepoint trajectory models and k=3 recovery archetypes.
* **Diagnostic / inference families** — G1 landmark + G2 value-of-information, G3 growth-mixture phenotyping, G4 AIS conversion, G6 multi-state Markov + improve-by-6m, G7 18 functional-independence heads, G8 132-segment topography, G10 5 level-descent heads, G11 3-axis neuro-functional dissociation. F24 out-of-time temporal backtest (diagnostic).
* **Dashboard** (Plotly Dash, JA default / EN toggle) — overview, insights, patient, methods, simulator tabs; F25 blank partial-input simulator with a completeness/OOD reliability badge; What-if reference overlay.
* **Quality gates** — F26 pytest harness (invariant + smoke + behavioral, ~11 s, skips when CSV/bundles absent), ruff (with the `--select F` regression gate), pip-audit.
* **Known-good** — ruff clean, pytest green. Model bundles carry a benign `sklearn 1.9→1.8` unpickle version warning (orthogonal to behavior).

## Backlog — take the next open step

The raw data holds **no new field families** (confirmed by the s40 audit) — any new model must reuse existing ISNCSCI / SCIM / AIS signal, and the standard neurological endpoints are already covered (G9 Δ-score, G10 level descent, G11 dissociation). Prefer a clinically or scientifically insightful head over infrastructure when one is justifiable; otherwise take F27. **Before starting any new feature, scope what / why / effort / files / data-dependency.**

1. **New reuse-only G-series head** — propose and scope a head built on existing signal. The one untried idea on record is **calibration-drift monitoring** (infrastructure). ZPP (zone-of-partial-preservation) descent is **infeasible** — the paired admission+discharge cohort is ~20 (sensory) / ~39 (motor) episodes, far below `MIN_COHORT`=120 (ZPP is recorded sparsely).
2. **F27 — dependency refresh** (size S, low urgency: no known CVEs, lint clean) — minor/patch bumps, raise the `shap<0.52` upper cap, then retrain to confirm byte-reproducibility of every `training_metrics.json` head.
