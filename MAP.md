# MAP.md — generated code map (do not edit by hand)

Regenerate after structural changes: `uv run python scripts/gen_map.py`.
Line numbers are 1-indexed — slice with `Read(path, offset, limit)` instead of
reading whole files.  Sources: src/rehab_sci, scripts.
Index: 47 files, 13299 source lines.

## scripts

### 01_profile_raw.py (87 lines)
Profile ALL_SCIDATA.csv to discover schema, dtypes, missingness, and factor leve…
- L15 `ROOT` (const)
- L16 `RAW` (const)
- L17 `OUT` (const)
- L20 `main()`

### gen_map.py (129 lines)
Generate MAP.md — a token-cheap navigation index of the codebase.
- L17 `ROOT` (const)
- L18 `TARGETS` (const)
- L19 `DOC_MAX` (const)
- L22 `_sig(node)`
- L35 `_is_callback(node)`
- L44 `_doc1(node)`
- L52 `_symbols(tree)`
- L82 `main()`

## src/rehab_sci

### __init__.py (0 lines)
- (no top-level symbols)

### constants.py (11 lines)
Shared domain constants — single source of truth (imports nothing from the proje…
- L10 `AIS_LETTER_TO_ORD` (const)
- L11 `AIS_ORD_TO_LETTER` (const)

### schema.py (207 lines)
Schema / bilingual translation registry.
- L20 `ROOT` (const)
- L21 `SCHEMA_DIR` (const)
- L24 `_load_yaml(name)`
- L30 `class ColumnSpec`
    methods: label
- L50 `class LevelSpec`
    methods: label
- L60 `class Schema` — Bilingual schema registry. Constructed once via :func:`load_schema`.
    methods: __init__, by_raw, all_raw, columns_in_group, columns_by_role, level_label, normalize_level, ui_str
- L133 `_expand_families(raw_columns, families)`
- L173 `_load_levels()`
- L199 `load_schema()`

## src/rehab_sci/dashboard

### __init__.py (0 lines)
- (no top-level symbols)

### app.py (136 lines)
Rehabilitation Analytics & Prediction Suite — bilingual Dash app.
- L36 `create_app()`
- L91 `update_lang(_n_ja, _n_en, _cur)` [callback]
- L105 `update_chrome(lang)` [callback]
- L123 `update_tab(tab, lang, ref_data)` [callback]

### compute.py (669 lines)
Pure computation helpers for model inference, conformal PI, and SHAP.
- L35 `resolve_group_q(q_by_group, marginal, X)` — Resolve Mondrian q for a single-row input.
- L58 `resolve_conformal_q(fspec, X)`
- L66 `resolve_aps_q(fspec, X)`
- L75 `predict_trajectory(X)` — Predict SCIM-total at each trajectory timepoint for a single-row input.
- L112 `aps_prediction_set(proba_row, q_hat)`
- L125 `inv_transform_scalar(x, transform)`
- L131 `clip_scalar(x, lo, hi)`
- L139 `format_value(col, value)`
- L148 `compute_ref_predictions(X)` — Compute predictions for all outcomes on a single-row X.
- L187 `collect_sim_inputs(num_vals, num_ids, cat_vals, cat_ids)`
- L206 `shap_for_row_regression(X, model)`
- L219 `shap_for_row_class(X, clf, class_idx, n_classes)`
- L248 `episode_row_for_model(key_record)` — Build a one-row model input from an episode's admission features.
- L264 `episode_has_admission(key_record)`
- L269 `get_observed_for_outcome(key_record, spec)`
- L282 `_landmark_input(X_base, observed, feature_cols)` — Build a one-row model input over ``feature_cols`` from base features + observed …
- L305 `_predict_landmark_head(head, X, task, transform, cmin, cmax)`
- L332 `predict_landmark(outcome_key, landmark, X_base, observed)` — Paired admission-only baseline vs landmark prediction for one outcome at landmar…
- L359 `landmark_voi(outcome_key, landmark, X_base, observed)` — Per-measure value-of-information for one patient at landmark ``L`` (G2).
- L429 `_episode_timepoint_oidx(key_record)` — Episode rows restricted to the landmark measures, plus the timepoint order-index…
- L439 `landmark_observed_for_episode(key_record, landmark)` — Real LOCF observed block for one episode: last non-null value of each landmark m…
- L458 `episode_landmark_eligibility(key_record)` — Per-landmark still-admitted eligibility: True when the episode has a tracked obs…
- L475 `_phenotype_episode_obs(key_record)` — This episode's observed ``(timepoint, value)`` pairs within the phenotyping wind…
- L491 `phenotype_cutoff_options(key_record, min_cells)` — Window timepoints (chronological) eligible as observation-cutoffs for phenotype
- L515 `predict_phenotype_membership(key_record, cutoff)` — Soft phenotype membership for one episode using only observations on/before ``cu…
- L585 `_apply_platt(calibrator, prob)` — Platt (sigmoid) recalibration over the LightGBM logit — mirrors models.conversio…
- L592 `_conversion_input(X)` — One-row model input over the conversion bundle's feature universe, dtyped like t…
- L607 `mag_short_label(code, mag_cap)` — Compact, language-neutral label for an ordinal improvement-magnitude class.
- L616 `predict_conversion(X)` — Admission->discharge AIS conversion for one admission row (see header).

### i18n.py (38 lines)
Bilingual translation helpers used by every dashboard component.
- L10 `t(schema, key, lang)`
- L14 `col_label(schema, raw, lang)`
- L21 `level_label(schema, level_key, raw_value, lang)`
- L25 `all_levels_in_order(schema, level_key, lang)` — Return (display, ja-or-en label) pairs in their YAML declaration order.
- L36 `level_key_for_column(schema, raw)`

### layout.py (555 lines)
Shared layout components: topbar, cards, sliders, prediction figures.
- L20 `topbar(lang)`
- L48 `kpi_card(label, value, sub)`
- L55 `chart_card(title, content)`
- L60 `input_id(prefix, col)`
- L64 `number_input_for(feature, lang, defaults)` — Clearable numeric input. A blank field is left unknown (NaN) so the model
- L96 `dropdown_for(feature, lang, defaults)`
- L125 `fig_shap_local(values, X, base, lang)`
- L154 `fig_prediction_interval(pred, lo, hi, spec, lang)`
- L188 `fig_class_probabilities(proba, class_labels, spec, lang, conformal_set)`
- L224 `fig_landmark_compare(result, spec, lang, landmark_label)` — Paired admission-only vs landmark prediction for one outcome (see compute.predic…
- L292 `landmark_readout(result, spec, lang)` — Two-line baseline→landmark summary shared by the simulator and patient dynamic c…
- L327 `_voi_label(measure, lang)`
- L331 `fig_voi_patient(voi, spec, lang)` — Per-patient value-of-information bars (see compute.landmark_voi).
- L397 `voi_readout(voi, spec, lang)` — One/two-line prescription: the most valuable next measure to obtain (+ best alre…
- L439 `_conversion_endpoint_label(key, letter, lang)` — Clinical endpoint name + its discharge threshold, e.g. 'Motor-incomplete (≥C)'.
- L444 `fig_conversion_endpoints(result, lang)` — Calibrated conversion probabilities for the applicable binary endpoints (see
- L481 `fig_conversion_magnitude(result, lang)` — Ordinal improvement-magnitude head shown AS its 80% APS set / most-likely class …
- L515 `conversion_readout(result, lang)` — Text summary of the conversion panel: admission grade, each applicable calibrate…

### reliability.py (141 lines)
Input reliability + out-of-distribution assessment for the simulator.
- L36 `_gain_importance(bundle)` — Per-feature LightGBM gain importance keyed by feature name (cached).
- L51 `_supplied(value)` — True when a cell holds a real user-supplied value (not blank / NaN).
- L59 `assess_input(X, bundle, feature_spec)` — Assess a single-row model input for completeness and OOD.

### report.py (328 lines)
PDF patient report generator.
- L66 `_t(key, lang)`
- L76 `class _ReportPDF`
    methods: __init__, _font, header, footer, section_heading, kv_pair
- L121 `_fig_to_png(fig, width, height)`
- L125 `_shap_fig_for_pdf(fig)` — Return a copy of the SHAP figure with margins adjusted for PDF rendering.
- L137 `_safe(v, na, fmt)`
- L146 `generate_patient_report(meta, predictions, trajectory_fig, shap_fig, outcome_labels, lang)` — Build a 2-page PDF report for one patient episode.

### state.py (132 lines)
Startup data loading and global state for the dashboard.
- L21 `ROOT` (const)
- L22 `MODELS_DIR` (const)
- L26 `SCHEMA` (const)
- L27 `AF` (const)
- L28 `EP` (const)
- L29 `LONG` (const)
- L35 `FEATURE_SPEC` (const)
- L37 `DEFAULT_OUTCOME` (const)
- L40 `_load_outcome_bundle(spec)`
- L59 `OUTCOME_BUNDLES` (const)
- L60 `SCIM_TOTAL_BUNDLE` (const)
- L70 `TRAJECTORY_BUNDLE` (const)
- L73 `ARCHETYPE_DATA` (const)
- L104 `LANDMARK_BUNDLE` (const)
- L113 `PHENOTYPE_DATA` (const)
- L127 `CONVERSION_BUNDLE` (const)
- L131 `PATIENT_OPTIONS` (const)
- L132 `PATIENT_OPTIONS_BY_ID` (const)

### theme.py (104 lines)
Plotly theme + palettes used everywhere on the dashboard.
- L9 `PALETTE_CATEGORICAL` (const)
- L20 `PALETTE_DIVERGING` (const)
- L31 `PALETTE_AIS` (const)
- L40 `PALETTE_PARA` (const)
- L46 `INK` (const)
- L58 `apply_template()` — Register and activate the medical-grade plotly template.

## src/rehab_sci/dashboard/figures

### __init__.py (101 lines)
Plotly figure factories, split by dashboard tab.
- (no top-level symbols)

### _common.py (7 lines)
Shared figure helpers (color utilities) used across the figure submodules.
- L4 `_hex_to_rgba(hex_color, alpha)`

### insights.py (326 lines)
Plotly figures for the Insight engine tab — SHAP importance, subgroups, dependen…
- L14 `fig_global_shap_importance(metrics, schema, lang, top_n)`
- L35 `fig_subgroup_box(ep, feature, schema, lang, outcome_col, outcome_label)`
- L86 `fig_dependence(shap_pack, X_test, feature, schema, lang, *, class_idx)`
- L155 `fig_interaction_heatmap(metrics, schema, lang, *, top_n)` — Upper-triangle heatmap of top feature-pair interactions by mean |SHAP|.
- L222 `fig_interaction_dependence(shap_pack, X_test, feat_x, feat_y, schema, lang, *, class_idx)` — Scatter of feature-X value vs SHAP interaction(X,Y), colored by feature-Y value.

### methods.py (635 lines)
Plotly figures for the Methods tab — calibration and performance visualizations.
- L13 `fig_pred_vs_observed(shap_pack, schema, lang, *, clip_min, clip_max, axis_label)`
- L80 `fig_residual_hist(shap_pack, schema, lang, *, axis_label)`
- L121 `fig_confusion_matrix(shap_pack, schema, lang)`
- L165 `fig_calibration_curve(shap_pack, schema, lang, *, n_bins)`
- L226 `fig_dataquality_overview(summary, lang)` — Stacked bar of finding counts per category, split by severity.
- L270 `fig_temporal_drift(t_outcome, lang)` — Out-of-time drift across rolling-origin test years (F24).
- L346 `fig_landmark_value(lm_outcome, landmark_days, lang)` — Value of observation: discharge-outcome accuracy + PI sharpening vs landmark tim…
- L421 `fig_voi_scorecard(lm_outcome, lang, measure_labels)` — Value-of-information scorecard: per-measure × per-landmark uncertainty reduction…
- L482 `fig_conversion_landscape(conv, lang)` — Descriptive conversion landscape: ≥1-grade AIS improvement rate by admission gra…
- L513 `fig_conversion_delta(conv, lang)` — Distribution of the AIS grade change (discharge − admission) over the dual-AIS c…
- L541 `fig_conversion_reliability(em, lang, label)` — Reliability curve for one binary conversion endpoint: Platt-calibrated vs raw Li…
- L586 `fig_conversion_shap(em, schema, lang, top_n)` — Descriptive in-sample SHAP drivers for one conversion endpoint (mean |SHAP| on t…
- L608 `fig_conversion_confusion(mag, lang)` — Row-normalized confusion matrix for the ordinal magnitude head over {0, +1, ≥+2}…

### overview.py (540 lines)
Plotly figures for the Overview tab — cohort demographics, injury, recovery curv…
- L16 `fig_age_distribution(ep, schema, lang)`
- L35 `fig_sex_donut(ep, schema, lang)`
- L54 `fig_mechanism(ep, schema, lang)`
- L76 `fig_discharge_scim(ep, schema, lang)`
- L103 `fig_injury_treemap(ep, schema, lang)`
- L177 `fig_ais_admit_discharge_sankey(ep, schema, lang)`
- L222 `fig_recovery_curves(long_df, schema, lang)`
- L293 `PALETTE_ARCHETYPE` (const)
- L302 `ARCHETYPE_NAMES_JA` (const)
- L305 `ARCHETYPE_NAMES_EN` (const)
- L308 `fig_archetype_curves(centroids, timepoint_labels, summaries, schema, lang)` — Archetype recovery trajectory curves with centroid lines and member count annota…
- L378 `_ais_distribution_bars(summaries, group_labels, lang)` — Stacked AIS-grade-distribution bar chart shared by archetype + phenotype demogra…
- L403 `fig_archetype_demographics(summaries, schema, lang)` — Stacked bar chart showing AIS grade distribution per archetype.
- L419 `PALETTE_PHENOTYPE` (const)
- L427 `PHENOTYPE_NAMES_JA` (const)
- L428 `PHENOTYPE_NAMES_EN` (const)
- L431 `fig_phenotype_curves(class_means, window, summaries, measure_labels, schema, lang, class_support, patient_obs)` — Observed-trajectory phenotype mean curves, one stacked panel per measure (SCIM, …
- L537 `fig_phenotype_demographics(summaries, schema, lang)` — Stacked AIS-grade distribution per observed-trajectory phenotype.

### patient.py (570 lines)
Plotly figures for the Patient explorer tab — SCIM timeline, prediction, similar…
- L28 `_subscale_label(key, lang)`
- L38 `fig_patient_scim_timeline(long_df, ep, key_record, strata, schema, lang, trajectory)` — SCIM-III timeline for a single episode against cohort percentile bands.
- L272 `fig_patient_prediction(pred, lo, hi, observed, schema, lang, clip_min, clip_max, axis_label)` — Predicted discharge outcome with 80% PI and the observed value (if any).
- L350 `fig_neighbor_outcomes(neighbors, pred, lo, hi, observed, schema, lang, *, clip_min, clip_max, axis_label)` — Strip chart of K nearest neighbors' actual outcomes on the prediction scale.
- L453 `fig_neighbor_ais_distribution(neighbors, pred_proba, observed_ais, schema, lang)` — Bar chart comparing neighbor AIS grade distribution to the model's predicted pro…
- L516 `fig_phenotype_membership(membership, summaries, schema, lang)` — Soft phenotype membership for one patient — horizontal bars over the K phenotype…

### simulator.py (124 lines)
Plotly figures for the Simulator tab — hypothetical recovery trajectory.
- L13 `fig_sim_trajectory(trajectory, schema, lang, *, ref_trajectory)` — Predicted SCIM-total recovery trajectory for a hypothetical patient (simulator).

## src/rehab_sci/dashboard/tabs

### __init__.py (0 lines)
- (no top-level symbols)

### insights.py (288 lines)
Insight engine tab — SHAP importance, subgroups, dependence, interactions.
- L25 `_insight_outcome_options(lang)`
- L30 `render_insights(lang)`
- L139 `update_insight_outcome_options(lang)` [callback]
- L148 `update_importance(outcome_key, lang)` [callback]
- L160 `update_subgroup(feature, outcome_key, lang)` [callback]
- L190 `update_dep_feature_options(outcome_key, lang)` [callback]
- L210 `update_dep_class_options(outcome_key)` [callback]
- L228 `update_dependence(feature, outcome_key, class_val, lang)` [callback]
- L243 `update_interaction_heatmap(outcome_key, lang)` [callback]
- L256 `update_int_feat_options(outcome_key, lang)` [callback]
- L279 `update_interaction_dependence(feat_x, feat_y, outcome_key, class_val, lang)` [callback]

### methods.py (496 lines)
Methods tab — model documentation + per-outcome performance visualizations.
- L23 `_perf_block_regression(spec, info, lang)`
- L85 `_perf_block_multiclass(spec, info, lang)`
- L154 `_perf_block_trajectory(lang)`
- L181 `_temporal_block(lang)` — F24 — out-of-time rolling-origin drift, one card per outcome.
- L231 `_landmark_block(lang)` — G1 — landmark (dynamic) prediction: value-of-observation curve, one card per out…
- L293 `_conversion_endpoint_label(key, discharge_min, lang)`
- L297 `_conversion_endpoint_card(key, em, lang)` — One binary endpoint: headline metrics + base-rate-by-grade table + reliability/S…
- L334 `_conversion_block(lang)` — G4 — AIS-grade conversion: descriptive landscape + per-endpoint calibration + ma…
- L404 `_dataquality_block(lang)`
- L452 `render_methods(lang)`

### overview.py (291 lines)
Overview tab — cohort KPIs, demographic charts, archetype curves with interactiv…
- L28 `render_overview(lang)` — Return filter bar + empty content div (populated by callback).
- L79 `_apply_filters(ais, para, age_range, arch)` — AND-combine all active filters on the global EP/LONG frames.
- L103 `_filtered_archetype_summaries(ep_f)` — Rebuild per-archetype summaries on the filtered episode subset.
- L141 `update_overview_content(ais, para, age_range, arch, lang)` [callback]

### patient.py (954 lines)
Patient explorer tab — real-patient predictions, similarity, PDF report.
- L77 `_patient_picker_options(lang)`
- L100 `_episode_options_for_patient(id_number, lang)`
- L110 `_meta_strip(meta, lang)`
- L163 `_isncsci_table(long_df, key_record, lang)`
- L206 `_landmark_obs_note(observed, landmark, lang)` — One-line summary of the real early-recovery scores feeding the landmark predicti…
- L221 `_patient_landmark_card(lang)` — Real-data dynamic-prediction card: at a chosen landmark the patient's own observ…
- L258 `_phenotype_readout(res, lang)` — Dominant phenotype + membership-weighted conditioned prognosis for one patient.
- L295 `_patient_phenotype_card(lang)` — Observed-trajectory phenotype card: the patient's own early SCIM/motor curve is …
- L326 `_patient_conversion_card(lang)` — AIS-grade conversion card: the patient's admission row drives the calibrated end…
- L347 `render_patient(lang)`
- L460 `_patient_regression(bundle, X, key_record, lang)`
- L516 `_patient_multiclass(bundle, X, key_record, lang)`
- L558 `_build_similarity_section(key_record, bundle, X, lang)`
- L647 `_compute_patient_tab(key_record, strata, outcome_key, lang)`
- L714 `update_patient_picker(id_number, lang)` [callback]
- L723 `reset_episode_on_patient_change(id_number, current)` [callback]
- L747 `update_patient_tab(key_record, strata, outcome_key, lang)` [callback]
- L757 `update_patient_landmark_options(key_record)` [callback] — Offer only the landmarks this episode is still-admitted-eligible for; default to…
- L777 `update_patient_landmark(landmark, key_record, outcome_key, lang)` [callback]
- L810 `update_patient_phenotype_options(key_record, lang)` [callback] — Offer each observation-cutoff this episode is eligible for; default to the full …
- L827 `update_patient_phenotype(cutoff, key_record, lang)` [callback]
- L855 `update_patient_conversion(key_record, lang)` [callback]
- L889 `download_report(n_clicks, key_record, id_number, strata, lang)` [callback]

### simulator.py (609 lines)
Simulator tab — hypothetical patient prediction + What-if counterfactual.
- L60 `render_simulator(lang, ref_data)`
- L169 `_conversion_card(lang)` — Hypothetical AIS-grade conversion card driven by the simulator's admission input…
- L191 `_lm_obs_input(measure, lang)`
- L204 `_landmark_card(lang)` — Hypothetical dynamic-prediction card: pick a landmark, enter observed scores, se…
- L241 `_simulate_regression(bundle, X, lang)`
- L276 `_simulate_multiclass(bundle, X, lang)`
- L305 `_reliability_badge(a, lang)`
- L368 `simulate(num_vals, cat_vals, num_ids, cat_ids, outcome_key, lang, ref_data)` [callback]
- L456 `launch_whatif(n_clicks, key_record, id_number)` [callback]
- L505 `update_whatif_banner(ref_data, lang)` [callback]
- L529 `clear_whatif(_n)` [callback]
- L543 `fill_or_clear(_fill, _clear, num_ids, cat_ids)` [callback] — Fill every field with the cohort default, or clear all to blank (NaN).
- L567 `simulate_landmark(landmark, obs_vals, num_vals, cat_vals, outcome_key, lang, obs_ids, num_ids, cat_ids)` [callback]
- L598 `simulate_conversion(num_vals, cat_vals, lang, num_ids, cat_ids)` [callback]

## src/rehab_sci/data

### __init__.py (0 lines)
- (no top-level symbols)

### archetypes.py (205 lines)
Recovery archetype discovery via k-means clustering on predicted trajectories.
- L26 `RANDOM_STATE` (const)
- L29 `build_trajectory_matrix(ep, trajectory_bundle, discharge_model, feature_cols, categorical_cols, numeric_cols)` — Predict 10-point recovery trajectory (9 intermediate + discharge) for all eligib…
- L81 `find_best_k(traj_matrix, k_range)` — Evaluate k-means for each k in range; return best k by silhouette score.
- L101 `cluster_trajectories(traj_matrix, k)` — Run k-means on standardized trajectory matrix.
- L121 `order_archetypes_by_discharge(labels, centroids)` — Re-label archetypes so archetype 0 has the lowest discharge SCIM (last column).
- L138 `archetype_summary(ep_eligible, labels)` — Compute per-archetype demographics and outcome summary.
- L182 `assign_single(X_row, trajectory_bundle, discharge_model, scaler, centroids_std)` — Assign a single patient (one-row DataFrame) to the nearest archetype.

### dataset.py (238 lines)
Construct the analysis-ready frame: one row per patient-episode.
- L29 `ADMISSION_FALLBACK` (const)
- L32 `ADMISSION_FEATURES` (const)
- L70 `NUMERIC_FEATURES` (const)
- L90 `CATEGORICAL_FEATURES` (const)
- L107 `class AnalysisFrame`
- L117 `_first_non_null(group, col, order)`
- L126 `build_episode_frame(longitudinal)` — Collapse the long longitudinal frame to one row per episode (KeyRecordNumber).
- L196 `_identify_ghost_episodes(ep, admission_features)` — Return KeyRecordNumbers of pure placeholder episodes.
- L213 `build_analysis_dataset()`
- L235 `_replace_nan_to_none(o)`

### episodes.py (201 lines)
Per-episode views used by the dashboard's Patient explorer tab.
- L26 `PATIENT_TIMELINE` (const)
- L33 `PATIENT_VIEW_COLS` (const)
- L50 `class PatientOption` — One row of the patient picker.
- L62 `list_patient_options(ep)` — Return one PatientOption per ``IDNumber``, sorted by IDNumber.
- L92 `episode_admission_features(ep, key_record, feature_cols)` — Return a dict of admission features for one episode, defaulting NaN to None.
- L106 `patient_timeline(long_df, key_record)` — Return one row per timepoint in :data:`PATIENT_TIMELINE` for the episode.
- L122 `patient_meta(ep, key_record)` — Demographics + admission injury summary for the meta strip.
- L155 `cohort_percentile_bands(long_df, ep, value_col, group_keys, min_n, timeline)` — Per-(timepoint × group) percentile bands for ``value_col``.

### loader.py (220 lines)
ALL_SCIDATA.csv loader + cleaner. Patient data is held in-memory only — NEVER pe…
- L14 `RAW_PATH_DEFAULT` (const)
- L26 `cord_level_to_int(level)`
- L32 `ais_to_int(grade)`
- L38 `_coerce_numeric(s, allow_bool)` — Coerce a column to numeric. If ``allow_bool``, FALSE/TRUE/NT are mapped to 0/1/N…
- L51 `_split_mfrankel(val)` — Split 'X/Y' modified-Frankel / Frankel pair into ordinal codes.
- L73 `load_raw(path)` — Load raw CSV with cp932 encoding. Patient data stays in-memory only.
- L89 `normalize(df, schema)` — Apply schema-driven cleaning: dtypes, level normalization, derived columns.
- L145 `add_isncsci_summaries(df, schema)` — Compute UEMS / LEMS / total motor / per-modality sensory totals per row.
- L197 `add_scim_subscales(df, schema)` — Compute SCIM-III sub-scale and total scores.
- L213 `load_clean(path, schema)` — Public entrypoint: load → normalize → add ISNCSCI summaries → add SCIM subscales…

### phenotypes.py (609 lines)
Multivariate growth mixture model (GMM) for observed-trajectory phenotyping (G3)…
- L46 `MEASURES` (const)
- L51 `WINDOW` (const)
- L52 `WINDOW_DAYS` (const)
- L56 `TIME_SCALE` (const)
- L58 `N_RANDOM` (const)
- L62 `scaled_time(timepoint)` — Scaled time in [0, 1] for a window timepoint slot.
- L71 `_poly_basis(t, degree)` — Vandermonde basis ``[1, t, t^2, ..., t^degree]`` (shape ``(n, degree+1)``).
- L77 `class GMMData` — Pre-built per-individual design matrices for the mixture of linear mixed models.
    methods: N, p_fixed, p_random
- L105 `build_individual_design(times, meas, degree, n_measures)` — Build ``(Phi, Z)`` for one individual from per-cell scaled times + measure indic…
- L124 `build_gmm_data(long_df, cohort_keys, degree, measures)` — Assemble :class:`GMMData` from the longitudinal frame for the given cohort.
- L172 `class GMMParams` — Fitted growth-mixture-model parameters (class-invariant G & sigma2).
    methods: n_free_params
- L193 `_block_diag_project(G, n_measures)` — Zero the cross-measure blocks so random effects are independent across measures.
- L207 `_cov_and_inv(Z, meas, G, sigma2)` — Marginal covariance ``V = Z G Z' + diag(sigma2[meas])`` with its inverse + logde…
- L223 `_e_step(data, p)` — Posterior class responsibilities + observed-data log-likelihood.
- L252 `_m_step(data, resp, p, Vinv_list)` — ECM update: pi, then GLS beta, then random-effect covariance G + residual sigma2…
- L310 `_init_params(data, K, resp0)` — Seed parameters from an initial (hard or soft) responsibility matrix.
- L344 `_crude_features(data)` — Per-individual summary (per-measure mean + OLS slope) for k-means initialization…
- L365 `fit_once(data, K, resp0, *, max_iter, tol)` — Run EM to convergence from one initialization.  Returns ``(params, resp, loglik)…
- L383 `fit(data, K, *, n_restarts, seed, max_iter, tol)` — Fit ``K``-class GMM with multiple restarts; keep the highest-likelihood solution…
- L423 `predict_proba(data, p)` — Posterior phenotype membership for (possibly partially observed) individuals.
- L433 `bic(loglik, n_free, N)`
- L437 `diagnostics(resp)` — GMM separation diagnostics: relative entropy + per-class APPA + min class share.
- L458 `class_means(p, timepoints)` — Fitted class mean trajectories, shape ``(K, n_measures, len(timepoints))``.
- L470 `class_support(long_df, assignments, k, measures, *, min_coverage)` — Last window index per (class, measure) where >= ``min_coverage`` of the class is…
- L510 `order_by_discharge(p, resp, support)` — Relabel classes by ascending SCIM recovery (class 0 = lowest recovery).
- L530 `select(data_by_degree, k_range, degrees, *, n_restarts, seed, min_class_share, progress)` — Sweep ``K x degree`` by BIC.  Returns ``(best_key, fits, table)``.
- L576 `phenotype_summary(ep_eligible, assignments, k)` — Per-phenotype demographics + conditioned outcomes (mirrors archetype_summary).

### quality.py (647 lines)
Data-quality / clinical-consistency report over the SCI dataset.
- L56 `MODELS_DIR` (const)
- L57 `SUMMARY_PATH` (const)
- L58 `DETAIL_PATH` (const)
- L61 `SENTINELS` (const)
- L63 `OPEN_ENDED_LEVELS` (const)
- L66 `PACKED_COLUMNS` (const)
- L69 `MFRANKEL_TO_AIS_SEV` (const)
- L74 `CERVICAL_MAX_ORD` (const)
- L75 `SCIM_DROP_MIN` (const)
- L76 `NLI_DRIFT_SEGMENTS` (const)
- L77 `MFRANKEL_AIS_GAP` (const)
- L79 `SEV_ERROR` (const)
- L80 `SEV_WARN` (const)
- L81 `SEV_INFO` (const)
- L85 `class Violation`
- L97 `class Rule`
- L105 `RULES` (const)
- L108 `rule(rid, category, severity, description)` — Register a rule function ``(ctx) -> list[Violation]``.
- L119 `class Context` — Loaded data + precomputed lookups shared across rules.
    methods: build, col, rows, at
- L169 `_eq(s, value)`
- L173 `_is_sentinel(value)` — True for tokens meaning 'missing / not tested', including paired or packed
- L187 `_scalar(v)` — JSON-safe scalar (numpy → python, NaN/NA → None).
- L204 `_num_range(ctx)`
- L227 `_num_parse(ctx)`
- L250 `_cat_level(ctx)`
- L274 `_sacral_signals(ctx)`
- L285 `_sacral_ais_a(ctx)`
- L302 `_sacral_ais_inc(ctx)`
- L319 `_vac_ais(ctx)`
- L332 `_comp_ais(ctx)`
- L347 `_ais_e_max(ctx)`
- L366 `_para_nli(ctx)`
- L383 `_nli_levels(ctx)`
- L406 `_mfrankel_ais(ctx)`
- L424 `_auto_manual(ctx)`
- L440 `_ordered_episode_series(ctx, value_col)` — Yield (KeyRecordNumber, frame) per episode, rows sorted chronologically,
- L457 `_ais_deterioration(ctx)`
- L474 `_scim_drop(ctx)`
- L492 `_nli_drift(ctx)`
- L510 `class QualityReport`
    methods: summary, detail
- L570 `run_quality_checks(path, schema)`
- L587 `_print_summary(summary)`
- L619 `main(argv)`

### similarity.py (151 lines)
Patient similarity via Gower distance on admission features.
- L18 `gower_distance_one_vs_all(query, candidates, numeric_cols, categorical_cols, ranges)` — Gower distance from a single query to every row in *candidates*.
- L67 `MIN_FEATURE_OVERLAP` (const)
- L70 `find_nearest(ep, key_record, feature_cols, numeric_cols, categorical_cols, ranges, k)` — Return the *k* nearest episodes to *key_record* by Gower distance.

## src/rehab_sci/models

### __init__.py (0 lines)
- (no top-level symbols)

### archetypes.py (143 lines)
Compute recovery archetypes and persist artifacts.
- L30 `ROOT` (const)
- L31 `MODELS_DIR` (const)
- L34 `main()`

### conformal.py (188 lines)
Split-conformal & APS prediction-set helpers (Mondrian per-AIS / per-paralysis).
- L14 `AIS_ORD_COL` (const)
- L15 `PARALYSIS_COL` (const)
- L16 `MONDRIAN_MIN_N` (const)
- L19 `_conformal_q(residuals, alpha)`
- L25 `_compute_mondrian_q(residuals_t, X_cal, alpha)` — Per-AIS-grade and per-paralysis-class conformal quantiles.
- L55 `_resolve_mondrian_q_array(marginal_q, q_by_group, X)` — Per-row conformal q: AIS group -> paralysis group -> marginal.
- L81 `_mondrian_test_coverage(y_raw, lo, hi, X)` — Per-group coverage on the test set using Mondrian PI bounds.
- L111 `_aps_scores(proba, y_true)` — APS nonconformity scores for conformal classification sets.
- L130 `_aps_prediction_set(proba_row, q_hat)` — Class indices in the APS prediction set for one sample.
- L143 `_aps_test_metrics(proba, y_true, q_arr, X)` — Coverage and avg set size on test set using per-row Mondrian APS q.

### conversion.py (440 lines)
AIS-grade conversion modeling (G4) — predict the admission->discharge AIS *trans…
- L75 `ROOT` (const)
- L76 `OUT` (const)
- L78 `ALPHA` (const)
- L79 `N_SPLITS` (const)
- L80 `N_CAL_BINS` (const)
- L82 `ADM_COL` (const)
- L83 `DIS_COL` (const)
- L87 `ENDPOINTS` (const)
- L93 `MAG_CAP` (const)
- L94 `MAG_ADM_GRADES` (const)
- L99 `_typed_X(used, af)` — Admission feature matrix with the schema's categorical / numeric dtypes applied.
- L111 `_cohort(ep, adm_grades)` — Episodes admitted at one of ``adm_grades`` with a discharge AIS and a real IDNum…
- L120 `_params_binary()` — Binary conversion params — no ``class_weight`` (endpoints are near-balanced; wei…
- L139 `_fit_binary(params, X_tr, y_tr, X_val, y_val, cat_cols)`
- L150 `_refit(params, X, y, cat_cols, best_iter)` — Refit a classifier on the full cohort at a fixed iteration count (no eval split)…
- L161 `_logit(p)`
- L166 `_fit_platt(prob, y)` — Fit a 1-feature logistic recalibration over the LightGBM logit (Platt scaling).
- L176 `_apply_platt(cal, prob)`
- L182 `_oof_binary(X, y, groups, cat_cols)` — Grouped-CV out-of-fold positive-class probabilities + median best-iteration.
- L201 `_oof_multiclass(X, y_codes, groups, cat_cols, n_classes)` — Grouped-CV out-of-fold class-probability matrix + median best-iteration.
- L224 `_calibration_curve(prob, y, n_bins)` — Reliability curve over quantile bins (avoids empty bins on small cohorts).
- L239 `_shap_top(model, X, top_n)` — Descriptive global driver ranking: top features by mean |SHAP| on the full cohor…
- L256 `_run_endpoint(spec, ep, af)` — Fit + score one binary conversion endpoint; return (metrics, persisted-model).
- L303 `_run_magnitude(ep, af)` — Fit + score the ordinal improvement-magnitude head; return (metrics, persisted-m…
- L357 `_landscape(ep)` — Descriptive conversion landscape over every episode with both admission + discha…
- L381 `main()`

### landmark.py (458 lines)
Landmark (dynamic) prediction — sharpen the discharge prognosis as early recover…
- L70 `ROOT` (const)
- L71 `OUT` (const)
- L73 `ALPHA` (const)
- L76 `LANDMARKS` (const)
- L77 `LANDMARK_DAYS` (const)
- L81 `LANDMARK_COLS` (const)
- L93 `LM_PREFIX` (const)
- L96 `TIMEPOINT_ORDER` (const)
- L103 `MIN_COHORT` (const)
- L108 `_latest_intermediate_oidx(long)` — Per-episode index of the latest intermediate timepoint carrying any tracked obse…
- L123 `_locf_block(long, landmark)` — LOCF landmark block: last non-null value at or before ``landmark`` per episode.
- L142 `_prep_landmark(af, target_col, eligible, lm_block)` — Build paired (X_base, X_landmark) matrices + target/groups for one (outcome, lan…
- L178 `_refit_all(params, X, y, cat_cols, best_iter, *, clf)`
- L187 `_eval_regression(X, y_t, y_raw, cat_cols, tr, cal, te, transform, clip_min, clip_max, *, persist)` — Fit a regression head on the train fold, conformalise on the calibration fold, s…
- L228 `_eval_multiclass(X, y_codes, groups, cat_cols, class_codes, tr, cal, te, *, persist)` — Fit the AIS multiclass head, calibrate APS sets on the calibration fold, score o…
- L278 `_eval_cell(spec, X, y_t, y_raw, y_codes, groups, cat_cols, tr, cal, te)` — Eval + persist one head on matrix ``X`` for ``spec`` (dispatches by task).
- L295 `_run_outcome(spec, af, lm_blocks, max_oi)` — Fit every landmark (paired baseline + landmark model) for one outcome.
- L371 `main()`

### multistate.py (457 lines)
AIS multi-state recovery modeling (G6) — neurological-grade *dynamics* over earl…
- L88 `ROOT` (const)
- L89 `OUT` (const)
- L91 `ADM_COL` (const)
- L94 `STATES` (const)
- L95 `WINDOW` (const)
- L96 `WINDOW_DAYS` (const)
- L102 `CONV_THRESHOLDS` (const)
- L103 `THRESH_LABEL` (const)
- L106 `IMPROVE_ADM_GRADES` (const)
- L107 `MIN_WINDOW_OBS` (const)
- L112 `_ais_grid(long)` — Wide AIS-grade grid: index=KeyRecordNumber, columns=WINDOW slots, values=AIS ord…
- L125 `_transition_matrices(grid)` — Empirical per-step transition probability + count matrices on the WINDOW grid.
- L153 `_occupancy(probs, pi0)` — Forward-propagate an initial distribution over the grid -> (K, S) occupancy (row…
- L162 `_absorbing_above(probs, thresh)` — Copy of the per-step matrices with states >= ``thresh`` made absorbing (row -> i…
- L176 `_conversion_curve(probs, g0, thresh)` — First-passage P(reached grade >= ``thresh`` by slot k) from admission grade ``g0…
- L187 `_median_day_to_event(curve)` — Interpolated day at which a first-passage curve first crosses 0.5 (None if it ne…
- L201 `_expected_days_in_state(occ)` — Trapezoidal expected days spent in each state over the window, given an occupanc…
- L207 `_population_dynamics(grid)` — Assemble per-admission-grade occupancy / conversion / sojourn from the empirical…
- L260 `_landscape(grid, ep)` — Descriptive within-window AIS dynamics summary (improve/stable/decline by admiss…
- L295 `_improve_cohort(ep, grid)` — Episodes admitted A-D with a real IDNumber and >= MIN_WINDOW_OBS in-window AIS o…
- L306 `_run_improve_head(ep, grid, af)` — Fit + score the binary 'improves >=1 grade within the window' head; return (metr…
- L354 `_curves_to_lists(conv)` — JSON-able conversion curves keyed by AIS letter -> label -> list.
- L362 `_by_letter(d)`
- L368 `main()`

### outcomes.py (109 lines)
Outcome registry — the source of truth for what `train.py` predicts.
- L33 `class OutcomeSpec`
- L46 `OUTCOMES` (const)
- L105 `get(key)`

### phenotypes.py (210 lines)
Observed-trajectory phenotyping (G3) — fit + persist a growth mixture model.
- L47 `ROOT` (const)
- L48 `MODELS_DIR` (const)
- L50 `MIN_SCIM_OBS` (const)
- L51 `K_RANGE` (const)
- L52 `DEGREES` (const)
- L53 `N_RESTARTS` (const)
- L54 `SEED` (const)
- L57 `_cohort_keys(ep, long)` — Episodes with >= MIN_SCIM_OBS observed SCIM points in the window and a real IDNu…
- L71 `main()`

### shap_utils.py (56 lines)
TreeSHAP interaction-value encoding + top feature-pair ranking helpers.
- L9 `_encode_cats_for_shap(X)` — Encode category-dtype columns to integer codes for shap_interaction_values.
- L19 `_top_interactions(shap_interaction, feature_names, top_n)` — Rank feature pairs by mean |SHAP interaction| (regression: 3-D input).
- L39 `_top_interactions_multiclass(shap_interaction, feature_names, top_n)` — Rank feature pairs by mean |SHAP interaction| (multiclass: 4-D input).

### subgroups.py (245 lines)
Subgroup discovery + effect sizes for all prediction outcomes.
- L30 `ROOT` (const)
- L31 `OUT` (const)
- L34 `cliffs_delta(a, b)` — Cliff's δ: P(X>Y) - P(X<Y). Bounded in [-1, 1].
- L47 `cohens_d(a, b)`
- L60 `kruskal_eta_squared(h, k, n)` — Effect size for Kruskal–Wallis (Tomczak & Tomczak 2014).
- L67 `_adjust_p(p)` — Return (Holm, BH) adjusted p-values matching the input order.
- L92 `_summary(g)`
- L106 `run_one(df, feature, outcome, kind)` — Run one feature-outcome comparison; ``kind`` ∈ {"categorical","numeric_quartile"…
- L168 `run_all_subgroups(df, outcome, categorical_features, numeric_features)`
- L191 `_console_summary(key, out)`
- L217 `main()`

### temporal.py (358 lines)
Temporal (out-of-time) validation via rolling-origin expanding-window backtest.
- L66 `ROOT` (const)
- L67 `OUT` (const)
- L69 `YEAR_COL` (const)
- L70 `ALPHA` (const)
- L73 `TEST_YEARS` (const)
- L74 `MIN_DEV` (const)
- L75 `MIN_TEST` (const)
- L80 `_prep_with_year(af, spec)` — ``train._prep`` plus the per-row BusinessYear (aligned to X's index).
- L91 `_origin_masks(groups, year, test_year)` — Boolean dev/test masks for one origin, group-safe by patient.
- L110 `_eval_regression_origin(X, y_raw, groups, cat_cols, year, spec, test_year)`
- L160 `_eval_multiclass_origin(X, y_raw, groups, cat_cols, year, spec, test_year)`
- L227 `_load_baselines()`
- L235 `_baseline_for(spec, metrics)`
- L256 `_summarize(origins, task, baseline)`
- L297 `main()`

### train.py (886 lines)
Train one model per outcome spec + split-conformal PI + SHAP cache.
- L75 `ROOT` (const)
- L76 `OUT` (const)
- L79 `RANDOM_STATE` (const)
- L82 `TRAJECTORY_TIMEPOINTS` (const)
- L87 `_prep(ep, feature_cols, numeric_cols, categorical_cols, target_col)`
- L108 `_apply_transform(y, transform)`
- L115 `_inverse_transform(y, transform)`
- L121 `_clip(arr, lo, hi)`
- L132 `_params_lgbm_reg()`
- L150 `_params_quantile(alpha)`
- L157 `_params_lgbm_clf(n_classes)`
- L176 `_fit_reg(params, X_tr, y_tr, X_val, y_val, cat_cols)`
- L188 `_fit_clf(params, X_tr, y_tr, X_val, y_val, cat_cols)`
- L200 `_grouped_holdout(X, y, groups, test_size)`
- L210 `_cv_score_reg(X, y_raw, groups, cat_cols, transform, clip_min, clip_max, n_splits)` — CV metrics reported on the *raw* (back-transformed, clipped) scale.
- L252 `_cv_score_multiclass(X, y_codes, groups, cat_cols, class_codes, n_splits)`
- L292 `_train_regression(spec, af, out_root)`
- L476 `_train_multiclass(spec, af, out_root)`
- L664 `_train_trajectory(af, out_root)` — Train per-timepoint SCIM-total models for recovery trajectory forecasting.
- L787 `_simulator_defaults(af)` — Return (defaults, ranges_and_categories) over the full episode frame.
- L831 `main()`
