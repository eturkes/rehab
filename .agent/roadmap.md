# roadmap.md — canonical plan & status

## Status

Mature SCI-recovery analytics project; all numbered features (F1–F26) and model families (G1–G11) are shipped and green. Full session-by-session history lives in `git log` (the former AGENT_NOTES.md §7 index) — not duplicated here.

* **Data layer** — schema-driven (`schema/*.yaml`), cp932 raw → ghost-filtered universe of **899 episodes / 866 patients**, 26-timepoint rectangular long frame (§1). `data/quality.py` clinical-consistency report.
* **Production heads** (`train.py`, byte-reproducible) — 4 SCIM (total + 3 subscales) + AIS multiclass + LOS + the 5 G9 Δ-score heads; split-conformal / Mondrian PI, APS sets (AIS), TreeSHAP + interactions. Plus 9-timepoint trajectory models and k=3 recovery archetypes.
* **Diagnostic / inference families** — G1 landmark + G2 value-of-information, G3 growth-mixture phenotyping, G4 AIS conversion, G6 multi-state Markov + improve-by-6m, G7 18 functional-independence heads, G8 132-segment topography, G10 5 level-descent heads, G11 3-axis neuro-functional dissociation. F24 out-of-time temporal backtest (diagnostic).
* **Dashboard** (Plotly Dash, JA default / EN toggle) — overview, insights, patient, methods, simulator tabs; F25 blank partial-input simulator with a completeness/OOD reliability badge; What-if reference overlay.
* **Quality gates** — F26 pytest harness (invariant + smoke + behavioral, ~11 s, skips when CSV/bundles absent), ruff (with the `--select F` regression gate), pip-audit.
* **Known-good** — ruff clean, pytest green (32). Stack refreshed (F27, done): scikit-learn 1.9.0, shap 0.52.0, lightgbm 4.6.0, numpy 2.4.6, pytest 9.1.0; pip-audit clean. The prior `sklearn 1.9→1.8` bundle skew is resolved — all served bundles re-pickled at 1.9.0; a Codex-review rescan of every `models/**/*.joblib` (66) loads clean after removing one gitignored, unreferenced orphan (`conversion/lgbm_binary.joblib`, stale 1.8.0). pytest 9.1.0 closes CVE-2025-71176. F27 also corrected stale `landmark_metrics.json` + `temporal_metrics.json` (pre-G9, 6 → all 11 outcomes). Codex review confirmed the retrain: production held-out/conformal/SHAP byte-identical (fixed-seed `GroupShuffleSplit`); G-series served metrics legitimately recomputed (OOF via `GroupKFold`, no collapse — headline AUC/Brier moved ≤~0.05).

## Backlog — take the next open step

The raw data holds **no new field families** (219-col profile audit, re-confirmed during F27) — any new model reuses existing ISNCSCI / SCIM / AIS signal, and the standard neurological endpoints are covered (G9 Δ-score, G10 level descent, G11 dissociation). **Before starting any new feature, scope what / why / effort / files / data-dependency.**

The **new-predictive-head well is dry** (assessed during F27): every reuse-only candidate is either a rescaling/reframing of a shipped head — recovery-fraction ≡ G9 (Δ ÷ known headroom); motor-vs-sensory ≡ G11's z-contrast method — i.e. the documented "framing not signal" trap, or cohort-infeasible (asymmetry ~93–122 borderline, ZPP ~20–39, WISCI sparse). Remaining open options, neither urgent:

1. **PRR descriptive insight** (content, NOT a predictive head) — F27 probe found this cohort does **not** obey the proportional-recovery rule (admission→discharge ISNCSCI motor): Δmotor R²(initial deficit) ≈ 0.01–0.12, recovery-fraction median ~0.6 but IQR spans the full range (no ~0.7 clustering), ~29–44% non-fitters. A worthwhile Insights/Methods panel — reuses G9 targets + admission scores, no new training — that also motivates the ML Δ-heads. Effort M. (User deferred it in favor of F27.)
2. **Calibration-drift monitoring** (infrastructure) — the only on-record untried head idea; track whether head calibration degrades over time, extending F24's temporal backtest. Effort M.
