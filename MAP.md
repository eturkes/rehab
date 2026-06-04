# MAP.md — generated code map (do not edit by hand)

Regenerate after structural changes: `uv run python scripts/gen_map.py`.
Line numbers are 1-indexed — slice with `Read(path, offset, limit)` instead of
reading whole files.  Sources: src/rehab_sci, scripts.
Index: 36 files, 7661 source lines.

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

### compute.py (266 lines)
Pure computation helpers for model inference, conformal PI, and SHAP.
- L25 `resolve_group_q(q_by_group, marginal, X)` — Resolve Mondrian q for a single-row input.
- L49 `resolve_conformal_q(fspec, X)`
- L57 `resolve_aps_q(fspec, X)`
- L66 `predict_trajectory(X)` — Predict SCIM-total at each trajectory timepoint for a single-row input.
- L103 `aps_prediction_set(proba_row, q_hat)`
- L116 `inv_transform_scalar(x, transform)`
- L122 `clip_scalar(x, lo, hi)`
- L130 `format_value(col, value)`
- L139 `compute_ref_predictions(X)` — Compute predictions for all outcomes on a single-row X.
- L178 `collect_sim_inputs(num_vals, num_ids, cat_vals, cat_ids)`
- L198 `shap_for_row_regression(X, model)`
- L211 `shap_for_row_class(X, clf, class_idx, n_classes)`
- L240 `episode_row_for_model(key_record)` — Build a one-row model input from an episode's admission features.
- L256 `episode_has_admission(key_record)`
- L261 `get_observed_for_outcome(key_record, spec)`

### i18n.py (38 lines)
Bilingual translation helpers used by every dashboard component.
- L10 `t(schema, key, lang)`
- L14 `col_label(schema, raw, lang)`
- L21 `level_label(schema, level_key, raw_value, lang)`
- L25 `all_levels_in_order(schema, level_key, lang)` — Return (display, ja-or-en label) pairs in their YAML declaration order.
- L36 `level_key_for_column(schema, raw)`

### layout.py (216 lines)
Shared layout components: topbar, cards, sliders, prediction figures.
- L20 `topbar(lang)`
- L48 `kpi_card(label, value, sub)`
- L55 `chart_card(title, content)`
- L60 `input_id(prefix, col)`
- L64 `slider_for(feature, lang, defaults)`
- L92 `dropdown_for(feature, lang, defaults)`
- L120 `fig_shap_local(values, X, base, lang)`
- L149 `fig_prediction_interval(pred, lo, hi, spec, lang)`
- L183 `fig_class_probabilities(proba, class_labels, spec, lang, conformal_set)`

### report.py (329 lines)
PDF patient report generator.
- L23 `AIS_ORD_TO_LETTER` (const)
- L67 `_t(key, lang)`
- L77 `class _ReportPDF`
    methods: __init__, _font, header, footer, section_heading, kv_pair
- L122 `_fig_to_png(fig, width, height)`
- L126 `_shap_fig_for_pdf(fig)` — Return a copy of the SHAP figure with margins adjusted for PDF rendering.
- L138 `_safe(v, na, fmt)`
- L147 `generate_patient_report(meta, predictions, trajectory_fig, shap_fig, outcome_labels, lang)` — Build a 2-page PDF report for one patient episode.

### state.py (77 lines)
Startup data loading and global state for the dashboard.
- L21 `ROOT` (const)
- L22 `MODELS_DIR` (const)
- L26 `SCHEMA` (const)
- L27 `AF` (const)
- L28 `EP` (const)
- L29 `LONG` (const)
- L35 `FEATURE_SPEC` (const)
- L37 `AIS_ORD_TO_LETTER` (const)
- L38 `DEFAULT_OUTCOME` (const)
- L41 `_load_outcome_bundle(spec)`
- L60 `OUTCOME_BUNDLES` (const)
- L61 `SCIM_TOTAL_BUNDLE` (const)
- L71 `TRAJECTORY_BUNDLE` (const)
- L74 `ARCHETYPE_DATA` (const)
- L76 `PATIENT_OPTIONS` (const)
- L77 `PATIENT_OPTIONS_BY_ID` (const)

### theme.py (104 lines)
Plotly theme + palettes used everywhere on the dashboard.
- L9 `PALETTE_CATEGORICAL` (const)
- L20 `PALETTE_DIVERGING` (const)
- L31 `PALETTE_AIS` (const)
- L40 `PALETTE_PARA` (const)
- L46 `INK` (const)
- L58 `apply_template()` — Register and activate the medical-grade plotly template.

## src/rehab_sci/dashboard/figures

### __init__.py (71 lines)
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

### methods.py (217 lines)
Plotly figures for the Methods tab — calibration and performance visualizations.
- L12 `fig_pred_vs_observed(shap_pack, schema, lang, *, clip_min, clip_max, axis_label)`
- L79 `fig_residual_hist(shap_pack, schema, lang, *, axis_label)`
- L120 `fig_confusion_matrix(shap_pack, schema, lang)`
- L164 `fig_calibration_curve(shap_pack, schema, lang, *, n_bins)`

### overview.py (416 lines)
Plotly figures for the Overview tab — cohort demographics, injury, recovery curv…
- L15 `fig_age_distribution(ep, schema, lang)`
- L34 `fig_sex_donut(ep, schema, lang)`
- L53 `fig_mechanism(ep, schema, lang)`
- L75 `fig_discharge_scim(ep, schema, lang)`
- L102 `fig_injury_treemap(ep, schema, lang)`
- L176 `fig_ais_admit_discharge_sankey(ep, schema, lang)`
- L223 `fig_recovery_curves(long_df, schema, lang)`
- L296 `PALETTE_ARCHETYPE` (const)
- L305 `ARCHETYPE_NAMES_JA` (const)
- L308 `ARCHETYPE_NAMES_EN` (const)
- L311 `fig_archetype_curves(centroids, timepoint_labels, summaries, schema, lang)` — Archetype recovery trajectory curves with centroid lines and member count annota…
- L381 `fig_archetype_demographics(summaries, schema, lang)` — Stacked bar chart showing AIS grade distribution per archetype.

### patient.py (512 lines)
Plotly figures for the Patient explorer tab — SCIM timeline, prediction, similar…
- L22 `_subscale_label(key, lang)`
- L32 `fig_patient_scim_timeline(long_df, ep, key_record, strata, schema, lang, trajectory)` — SCIM-III timeline for a single episode against cohort percentile bands.
- L267 `fig_patient_prediction(pred, lo, hi, observed, schema, lang, clip_min, clip_max, axis_label)` — Predicted discharge outcome with 80% PI and the observed value (if any).
- L345 `AIS_ORD_TO_LETTER` (const)
- L348 `fig_neighbor_outcomes(neighbors, pred, lo, hi, observed, schema, lang, *, clip_min, clip_max, axis_label)` — Strip chart of K nearest neighbors' actual outcomes on the prediction scale.
- L452 `fig_neighbor_ais_distribution(neighbors, pred_proba, observed_ais, schema, lang)` — Bar chart comparing neighbor AIS grade distribution to the model's predicted pro…

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

### methods.py (204 lines)
Methods tab — model documentation + per-outcome performance visualizations.
- L14 `_perf_block_regression(spec, info, lang)`
- L76 `_perf_block_multiclass(spec, info, lang)`
- L145 `_perf_block_trajectory(lang)`
- L172 `render_methods(lang)`

### overview.py (257 lines)
Overview tab — cohort KPIs, demographic charts, archetype curves with interactiv…
- L27 `render_overview(lang)` — Return filter bar + empty content div (populated by callback).
- L78 `_apply_filters(ais, para, age_range, arch)` — AND-combine all active filters on the global EP/LONG frames.
- L102 `_filtered_archetype_summaries(ep_f)` — Rebuild per-archetype summaries on the filtered episode subset.
- L140 `update_overview_content(ais, para, age_range, arch, lang)` [callback]

### patient.py (657 lines)
Patient explorer tab — real-patient predictions, similarity, PDF report.
- L59 `_patient_picker_options(lang)`
- L82 `_episode_options_for_patient(id_number, lang)`
- L92 `_meta_strip(meta, lang)`
- L145 `_isncsci_table(long_df, key_record, lang)`
- L189 `render_patient(lang)`
- L291 `_patient_regression(bundle, X, key_record, lang)`
- L347 `_patient_multiclass(bundle, X, key_record, lang)`
- L389 `_build_similarity_section(key_record, bundle, X, lang)`
- L478 `_compute_patient_tab(key_record, strata, outcome_key, lang)`
- L545 `update_patient_picker(id_number, lang)` [callback]
- L554 `reset_episode_on_patient_change(id_number, current)` [callback]
- L578 `update_patient_tab(key_record, strata, outcome_key, lang)` [callback]
- L592 `download_report(n_clicks, key_record, id_number, strata, lang)` [callback]

### simulator.py (373 lines)
Simulator tab — hypothetical patient prediction + What-if counterfactual.
- L49 `render_simulator(lang, ref_data)`
- L132 `_simulate_regression(bundle, X, lang)`
- L167 `_simulate_multiclass(bundle, X, lang)`
- L209 `simulate(num_vals, cat_vals, num_ids, cat_ids, outcome_key, lang, ref_data)` [callback]
- L299 `launch_whatif(n_clicks, key_record, id_number)` [callback]
- L348 `update_whatif_banner(ref_data, lang)` [callback]
- L372 `clear_whatif(_n)` [callback]

## src/rehab_sci/data

### __init__.py (0 lines)
- (no top-level symbols)

### archetypes.py (208 lines)
Recovery archetype discovery via k-means clustering on predicted trajectories.
- L27 `RANDOM_STATE` (const)
- L30 `build_trajectory_matrix(ep, trajectory_bundle, discharge_model, feature_cols, categorical_cols, numeric_cols)` — Predict 10-point recovery trajectory (9 intermediate + discharge) for all eligib…
- L84 `find_best_k(traj_matrix, k_range)` — Evaluate k-means for each k in range; return best k by silhouette score.
- L104 `cluster_trajectories(traj_matrix, k)` — Run k-means on standardized trajectory matrix.
- L124 `order_archetypes_by_discharge(labels, centroids)` — Re-label archetypes so archetype 0 has the lowest discharge SCIM (last column).
- L141 `archetype_summary(ep_eligible, labels)` — Compute per-archetype demographics and outcome summary.
- L185 `assign_single(X_row, trajectory_bundle, discharge_model, scaler, centroids_std)` — Assign a single patient (one-row DataFrame) to the nearest archetype.

### dataset.py (234 lines)
Construct the analysis-ready frame: one row per patient-episode.
- L29 `ADMISSION_FALLBACK` (const)
- L32 `ADMISSION_FEATURES` (const)
- L70 `NUMERIC_FEATURES` (const)
- L90 `CATEGORICAL_FEATURES` (const)
- L107 `class AnalysisFrame`
- L117 `_first_non_null(group, col, order)`
- L126 `build_episode_frame(longitudinal)` — Collapse the long longitudinal frame to one row per episode (KeyRecordNumber).
- L192 `_identify_ghost_episodes(ep, admission_features)` — Return KeyRecordNumbers of pure placeholder episodes.
- L209 `build_analysis_dataset()`
- L231 `_replace_nan_to_none(o)`

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
- L13 `RAW_PATH_DEFAULT` (const)
- L26 `cord_level_to_int(level)`
- L32 `ais_to_int(grade)`
- L38 `_coerce_numeric(s, allow_bool)` — Coerce a column to numeric. If ``allow_bool``, FALSE/TRUE/NT are mapped to 0/1/N…
- L51 `_split_mfrankel(val)` — Split 'X/Y' modified-Frankel / Frankel pair into ordinal codes.
- L73 `load_raw(path)` — Load raw CSV with cp932 encoding. Patient data stays in-memory only.
- L89 `normalize(df, schema)` — Apply schema-driven cleaning: dtypes, level normalization, derived columns.
- L145 `add_isncsci_summaries(df, schema)` — Compute UEMS / LEMS / total motor / per-modality sensory totals per row.
- L197 `add_scim_subscales(df, schema)` — Compute SCIM-III sub-scale and total scores.
- L213 `load_clean(path, schema)` — Public entrypoint: load → normalize → add ISNCSCI summaries → add SCIM subscales…

### similarity.py (151 lines)
Patient similarity via Gower distance on admission features.
- L18 `gower_distance_one_vs_all(query, candidates, numeric_cols, categorical_cols, ranges)` — Gower distance from a single query to every row in *candidates*.
- L67 `MIN_FEATURE_OVERLAP` (const)
- L70 `find_nearest(ep, key_record, feature_cols, numeric_cols, categorical_cols, ranges, k)` — Return the *k* nearest episodes to *key_record* by Gower distance.

## src/rehab_sci/models

### __init__.py (0 lines)
- (no top-level symbols)

### archetypes.py (144 lines)
Compute recovery archetypes and persist artifacts.
- L31 `ROOT` (const)
- L32 `MODELS_DIR` (const)
- L35 `main()`

### outcomes.py (109 lines)
Outcome registry — the source of truth for what `train.py` predicts.
- L33 `class OutcomeSpec`
- L46 `OUTCOMES` (const)
- L105 `get(key)`

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

### train.py (1108 lines)
Train one model per outcome spec + split-conformal PI + SHAP cache.
- L61 `ROOT` (const)
- L62 `OUT` (const)
- L65 `RANDOM_STATE` (const)
- L67 `AIS_ORD_TO_LETTER` (const)
- L68 `AIS_ORD_COL` (const)
- L69 `PARALYSIS_COL` (const)
- L70 `MONDRIAN_MIN_N` (const)
- L72 `TRAJECTORY_TIMEPOINTS` (const)
- L77 `_prep(ep, feature_cols, numeric_cols, categorical_cols, target_col)`
- L98 `_apply_transform(y, transform)`
- L105 `_inverse_transform(y, transform)`
- L111 `_clip(arr, lo, hi)`
- L122 `_params_lgbm_reg()`
- L140 `_params_quantile(alpha)`
- L147 `_params_lgbm_clf(n_classes)`
- L166 `_fit_reg(params, X_tr, y_tr, X_val, y_val, cat_cols)`
- L178 `_fit_clf(params, X_tr, y_tr, X_val, y_val, cat_cols)`
- L190 `_grouped_holdout(X, y, groups, test_size)`
- L200 `_cv_score_reg(X, y_raw, groups, cat_cols, transform, clip_min, clip_max, n_splits)` — CV metrics reported on the *raw* (back-transformed, clipped) scale.
- L242 `_cv_score_multiclass(X, y_codes, groups, cat_cols, class_codes, n_splits)`
- L283 `_conformal_q(residuals, alpha)`
- L289 `_compute_mondrian_q(residuals_t, X_cal, alpha)` — Per-AIS-grade and per-paralysis-class conformal quantiles.
- L319 `_resolve_mondrian_q_array(marginal_q, q_by_group, X)` — Per-row conformal q: AIS group -> paralysis group -> marginal.
- L345 `_mondrian_test_coverage(y_raw, lo, hi, X)` — Per-group coverage on the test set using Mondrian PI bounds.
- L378 `_aps_scores(proba, y_true)` — APS nonconformity scores for conformal classification sets.
- L397 `_aps_prediction_set(proba_row, q_hat)` — Class indices in the APS prediction set for one sample.
- L410 `_aps_test_metrics(proba, y_true, q_arr, X)` — Coverage and avg set size on test set using per-row Mondrian APS q.
- L460 `_encode_cats_for_shap(X)` — Encode category-dtype columns to integer codes for shap_interaction_values.
- L472 `_top_interactions(shap_interaction, feature_names, top_n)` — Rank feature pairs by mean |SHAP interaction| (regression: 3-D input).
- L492 `_top_interactions_multiclass(shap_interaction, feature_names, top_n)` — Rank feature pairs by mean |SHAP interaction| (multiclass: 4-D input).
- L514 `_train_regression(spec, af, out_root)`
- L698 `_train_multiclass(spec, af, out_root)`
- L886 `_train_trajectory(af, out_root)` — Train per-timepoint SCIM-total models for recovery trajectory forecasting.
- L1009 `_simulator_defaults(af)` — Return (defaults, ranges_and_categories) over the full episode frame.
- L1053 `main()`
