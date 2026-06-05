# MAP.md — generated code map (do not edit by hand)

Regenerate after structural changes: `uv run python scripts/gen_map.py`.
Line numbers are 1-indexed — slice with `Read(path, offset, limit)` instead of
reading whole files.  Sources: src/rehab_sci, scripts.
Index: 43 files, 10134 source lines.

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

### compute.py (392 lines)
Pure computation helpers for model inference, conformal PI, and SHAP.
- L26 `resolve_group_q(q_by_group, marginal, X)` — Resolve Mondrian q for a single-row input.
- L49 `resolve_conformal_q(fspec, X)`
- L57 `resolve_aps_q(fspec, X)`
- L66 `predict_trajectory(X)` — Predict SCIM-total at each trajectory timepoint for a single-row input.
- L103 `aps_prediction_set(proba_row, q_hat)`
- L116 `inv_transform_scalar(x, transform)`
- L122 `clip_scalar(x, lo, hi)`
- L130 `format_value(col, value)`
- L139 `compute_ref_predictions(X)` — Compute predictions for all outcomes on a single-row X.
- L178 `collect_sim_inputs(num_vals, num_ids, cat_vals, cat_ids)`
- L197 `shap_for_row_regression(X, model)`
- L210 `shap_for_row_class(X, clf, class_idx, n_classes)`
- L239 `episode_row_for_model(key_record)` — Build a one-row model input from an episode's admission features.
- L255 `episode_has_admission(key_record)`
- L260 `get_observed_for_outcome(key_record, spec)`
- L273 `_landmark_input(X_base, observed, feature_cols)` — Build a one-row model input over ``feature_cols`` from base features + observed …
- L296 `_predict_landmark_head(head, X, task, transform, cmin, cmax)`
- L323 `predict_landmark(outcome_key, landmark, X_base, observed)` — Paired admission-only baseline vs landmark prediction for one outcome at landmar…
- L350 `_episode_timepoint_oidx(key_record)` — Episode rows restricted to the landmark measures, plus the timepoint order-index…
- L360 `landmark_observed_for_episode(key_record, landmark)` — Real LOCF observed block for one episode: last non-null value of each landmark m…
- L379 `episode_landmark_eligibility(key_record)` — Per-landmark still-admitted eligibility: True when the episode has a tracked obs…

### i18n.py (38 lines)
Bilingual translation helpers used by every dashboard component.
- L10 `t(schema, key, lang)`
- L14 `col_label(schema, raw, lang)`
- L21 `level_label(schema, level_key, raw_value, lang)`
- L25 `all_levels_in_order(schema, level_key, lang)` — Return (display, ja-or-en label) pairs in their YAML declaration order.
- L36 `level_key_for_column(schema, raw)`

### layout.py (323 lines)
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

### state.py (109 lines)
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
- L108 `PATIENT_OPTIONS` (const)
- L109 `PATIENT_OPTIONS_BY_ID` (const)

### theme.py (104 lines)
Plotly theme + palettes used everywhere on the dashboard.
- L9 `PALETTE_CATEGORICAL` (const)
- L20 `PALETTE_DIVERGING` (const)
- L31 `PALETTE_AIS` (const)
- L40 `PALETTE_PARA` (const)
- L46 `INK` (const)
- L58 `apply_template()` — Register and activate the medical-grade plotly template.

## src/rehab_sci/dashboard/figures

### __init__.py (77 lines)
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

### methods.py (417 lines)
Plotly figures for the Methods tab — calibration and performance visualizations.
- L12 `fig_pred_vs_observed(shap_pack, schema, lang, *, clip_min, clip_max, axis_label)`
- L79 `fig_residual_hist(shap_pack, schema, lang, *, axis_label)`
- L120 `fig_confusion_matrix(shap_pack, schema, lang)`
- L164 `fig_calibration_curve(shap_pack, schema, lang, *, n_bins)`
- L225 `fig_dataquality_overview(summary, lang)` — Stacked bar of finding counts per category, split by severity.
- L269 `fig_temporal_drift(t_outcome, lang)` — Out-of-time drift across rolling-origin test years (F24).
- L345 `fig_landmark_value(lm_outcome, landmark_days, lang)` — Value of observation: discharge-outcome accuracy + PI sharpening vs landmark tim…

### overview.py (412 lines)
Plotly figures for the Overview tab — cohort demographics, injury, recovery curv…
- L15 `fig_age_distribution(ep, schema, lang)`
- L34 `fig_sex_donut(ep, schema, lang)`
- L53 `fig_mechanism(ep, schema, lang)`
- L75 `fig_discharge_scim(ep, schema, lang)`
- L102 `fig_injury_treemap(ep, schema, lang)`
- L176 `fig_ais_admit_discharge_sankey(ep, schema, lang)`
- L221 `fig_recovery_curves(long_df, schema, lang)`
- L292 `PALETTE_ARCHETYPE` (const)
- L301 `ARCHETYPE_NAMES_JA` (const)
- L304 `ARCHETYPE_NAMES_EN` (const)
- L307 `fig_archetype_curves(centroids, timepoint_labels, summaries, schema, lang)` — Archetype recovery trajectory curves with centroid lines and member count annota…
- L377 `fig_archetype_demographics(summaries, schema, lang)` — Stacked bar chart showing AIS grade distribution per archetype.

### patient.py (508 lines)
Plotly figures for the Patient explorer tab — SCIM timeline, prediction, similar…
- L23 `_subscale_label(key, lang)`
- L33 `fig_patient_scim_timeline(long_df, ep, key_record, strata, schema, lang, trajectory)` — SCIM-III timeline for a single episode against cohort percentile bands.
- L267 `fig_patient_prediction(pred, lo, hi, observed, schema, lang, clip_min, clip_max, axis_label)` — Predicted discharge outcome with 80% PI and the observed value (if any).
- L345 `fig_neighbor_outcomes(neighbors, pred, lo, hi, observed, schema, lang, *, clip_min, clip_max, axis_label)` — Strip chart of K nearest neighbors' actual outcomes on the prediction scale.
- L448 `fig_neighbor_ais_distribution(neighbors, pred_proba, observed_ais, schema, lang)` — Bar chart comparing neighbor AIS grade distribution to the model's predicted pro…

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

### methods.py (367 lines)
Methods tab — model documentation + per-outcome performance visualizations.
- L21 `_perf_block_regression(spec, info, lang)`
- L83 `_perf_block_multiclass(spec, info, lang)`
- L152 `_perf_block_trajectory(lang)`
- L179 `_temporal_block(lang)` — F24 — out-of-time rolling-origin drift, one card per outcome.
- L229 `_landmark_block(lang)` — G1 — landmark (dynamic) prediction: value-of-observation curve, one card per out…
- L278 `_dataquality_block(lang)`
- L326 `render_methods(lang)`

### overview.py (257 lines)
Overview tab — cohort KPIs, demographic charts, archetype curves with interactiv…
- L27 `render_overview(lang)` — Return filter bar + empty content div (populated by callback).
- L78 `_apply_filters(ais, para, age_range, arch)` — AND-combine all active filters on the global EP/LONG frames.
- L102 `_filtered_archetype_summaries(ep_f)` — Rebuild per-archetype summaries on the filtered episode subset.
- L140 `update_overview_content(ais, para, age_range, arch, lang)` [callback]

### patient.py (750 lines)
Patient explorer tab — real-patient predictions, similarity, PDF report.
- L64 `_patient_picker_options(lang)`
- L87 `_episode_options_for_patient(id_number, lang)`
- L97 `_meta_strip(meta, lang)`
- L150 `_isncsci_table(long_df, key_record, lang)`
- L193 `_landmark_obs_note(observed, landmark, lang)` — One-line summary of the real early-recovery scores feeding the landmark predicti…
- L208 `_patient_landmark_card(lang)` — Real-data dynamic-prediction card: at a chosen landmark the patient's own observ…
- L237 `render_patient(lang)`
- L344 `_patient_regression(bundle, X, key_record, lang)`
- L400 `_patient_multiclass(bundle, X, key_record, lang)`
- L442 `_build_similarity_section(key_record, bundle, X, lang)`
- L531 `_compute_patient_tab(key_record, strata, outcome_key, lang)`
- L598 `update_patient_picker(id_number, lang)` [callback]
- L607 `reset_episode_on_patient_change(id_number, current)` [callback]
- L631 `update_patient_tab(key_record, strata, outcome_key, lang)` [callback]
- L641 `update_patient_landmark_options(key_record)` [callback] — Offer only the landmarks this episode is still-admitted-eligible for; default to…
- L659 `update_patient_landmark(landmark, key_record, outcome_key, lang)` [callback]
- L685 `download_report(n_clicks, key_record, id_number, strata, lang)` [callback]

### simulator.py (554 lines)
Simulator tab — hypothetical patient prediction + What-if counterfactual.
- L55 `render_simulator(lang, ref_data)`
- L161 `_lm_obs_input(measure, lang)`
- L174 `_landmark_card(lang)` — Hypothetical dynamic-prediction card: pick a landmark, enter observed scores, se…
- L211 `_simulate_regression(bundle, X, lang)`
- L246 `_simulate_multiclass(bundle, X, lang)`
- L275 `_reliability_badge(a, lang)`
- L338 `simulate(num_vals, cat_vals, num_ids, cat_ids, outcome_key, lang, ref_data)` [callback]
- L426 `launch_whatif(n_clicks, key_record, id_number)` [callback]
- L475 `update_whatif_banner(ref_data, lang)` [callback]
- L499 `clear_whatif(_n)` [callback]
- L513 `fill_or_clear(_fill, _clear, num_ids, cat_ids)` [callback] — Fill every field with the cohort default, or clear all to blank (NaN).
- L537 `simulate_landmark(landmark, obs_vals, num_vals, cat_vals, outcome_key, lang, obs_ids, num_ids, cat_ids)` [callback]

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

### landmark.py (395 lines)
Landmark (dynamic) prediction — sharpen the discharge prognosis as early recover…
- L63 `ROOT` (const)
- L64 `OUT` (const)
- L66 `ALPHA` (const)
- L69 `LANDMARKS` (const)
- L70 `LANDMARK_DAYS` (const)
- L74 `LANDMARK_COLS` (const)
- L86 `LM_PREFIX` (const)
- L89 `TIMEPOINT_ORDER` (const)
- L96 `MIN_COHORT` (const)
- L101 `_latest_intermediate_oidx(long)` — Per-episode index of the latest intermediate timepoint carrying any tracked obse…
- L116 `_locf_block(long, landmark)` — LOCF landmark block: last non-null value at or before ``landmark`` per episode.
- L135 `_prep_landmark(af, target_col, eligible, lm_block)` — Build paired (X_base, X_landmark) matrices + target/groups for one (outcome, lan…
- L171 `_refit_all(params, X, y, cat_cols, best_iter, *, clf)`
- L180 `_eval_regression(X, y_t, y_raw, cat_cols, tr, cal, te, transform, clip_min, clip_max, *, persist)` — Fit a regression head on the train fold, conformalise on the calibration fold, s…
- L221 `_eval_multiclass(X, y_codes, groups, cat_cols, class_codes, tr, cal, te, *, persist)` — Fit the AIS multiclass head, calibrate APS sets on the calibration fold, score o…
- L271 `_run_outcome(spec, af, lm_blocks, max_oi)` — Fit every landmark (paired baseline + landmark model) for one outcome.
- L335 `main()`

### outcomes.py (109 lines)
Outcome registry — the source of truth for what `train.py` predicts.
- L33 `class OutcomeSpec`
- L46 `OUTCOMES` (const)
- L105 `get(key)`

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
