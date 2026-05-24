# AGENT_NOTES.md — sticky knowledge for future sessions

Optimized for LLM ingestion: short bullets, no prose padding.  Append at the
bottom of each section as new lessons land; do **not** delete prior entries.

---

## 0. Read-first

* `CLAUDE.md` is the user's policy file.  Treat as authoritative.  You may
  modify it freely when content becomes obsolete or can be improved.
* `SESSION_PROMPT.md` contains the reusable prompt the user pastes to start
  new sessions.  Update it if the bootstrapping workflow changes.
* `README.md` is the human-facing entry point.  Keep in sync with code.
* This file is the agent-facing scratchpad.  Always read it before planning.
  Keep it accurate: update sections after every session; prune obsolete
  entries when they conflict with current state.
* **Default-work pool for fresh sessions: §8 Feature backlog.**
  All features F1–F6 are shipped as of session 12.  Propose new feature
  candidates or maintenance work unless the user redirects.  Historical
  "Open items rolled forward" lists inside prior §7 session entries are
  **superseded** by user decision in session 4 — treat them as history.

## 0b. Lessons & mistakes (append here; prune when superseded)

* **Session 1: Holm step-down implemented backwards** — used running
  `min` instead of running `max` over sorted p×(n−k+1).  Produced
  nonsensical adjusted-p values (~10⁻⁵⁵).  Fix: verified the Holm
  formula against a stats reference before committing.  *Takeaway:*
  always validate statistical procedures against a reference
  implementation or textbook definition, especially for correction
  methods where the direction of an operator is easy to confuse.
* **Session 2: Plotly silent failure on sunburst** — `branchvalues=
  "total"` with zero-valued parents renders blank with no error.
  Spent time debugging before realizing Plotly silently swallows the
  misconfiguration.  *Takeaway:* Plotly has several silent-failure
  modes; when a chart renders blank, check the `branchvalues` /
  parent-child value contract before assuming a data issue.
* **Session 1–3: `_apply_missing_sentinels` phantom function** — early
  notes referenced a helper that was never written; sentinels are
  handled by two different mechanisms in the loader.  Corrected in
  session 6.  *Takeaway:* verify that referenced functions actually
  exist in the codebase before documenting them; grep for the name.
* **Session 5: Dropdown option labels too long for narrow viewports** —
  concatenating full Japanese demographic strings produced multi-line
  wrapping in the Dash Dropdown that blended adjacent options together.
  *Takeaway:* always test UI components at minimum viewport width;
  Dash Dropdown `optionHeight` must match the actual rendered height.
* **General: ghost-episode discovery delayed** — 301 placeholder
  episodes inflated cohort counts for 5 sessions before investigation.
  The loader was correct but the anomaly was visible in session 1.
  *Takeaway:* when encountering unexpected NaN patterns or row counts,
  investigate immediately rather than deferring.

## 1. Data invariants (do not rediscover)

* **Raw file** — `ALL_SCIDATA.csv` at repo root.  Never commit; gitignored.
* **Encoding** — `cp932` (Shift-JIS superset).  UTF-8 will silently mangle
  half the column names.
* **Missing sentinels** in raw file: `""`, `"_"`, `"NA"` are parsed to NaN
  directly via `pd.read_csv(na_values=...)` in `loader.py::load_raw`.
  `"NT"` and `"ND"` are *not* listed there but are NaN'd as a side effect
  of `pd.to_numeric(errors="coerce")` for numeric/ordinal columns and of
  `schema.normalize_level()` returning `pd.NA` for unknown categorical
  levels.  (Earlier sessions referred to a `_apply_missing_sentinels`
  helper — no such function exists; the effect is the same but the path
  is the two mechanisms just described.)
* **Excel booleans** — many bool-like columns arrive as the literal strings
  `"FALSE"` / `"TRUE"` (note: uppercase).  Coerced to `0`/`1` then to `Y`/`N`
  via the schema's level mapping.
* **mFrankel/Frankel** — single combined raw column; split on slash into
  `mFrankel_ord` (5-grade A–E with substages) and `Frankel_ord` (5-grade).
* **Raw shape** — long format: 31 200 rows × 219 cols, 1 200
  `KeyRecordNumber`s × 26 timepoint slots (`0day`, `72h`, `2w`, `4w`,
  `6w`, `2m..11m`, `1y..10y`, `discharge`).  **The grid is perfectly
  rectangular** — every episode has a row at every timepoint slot.
* **Ghost-episode filter (session 6, 2026-05-18)** — 301 of the 1 200
  raw episodes are pure placeholder rows: `IDNumber` is null AND every
  admission feature is null AND every outcome is null.  Across their
  7 826 long-frame rows, only `BusinessYear`, `AnualCaseNumber`, and
  `mFrankel_Frankel` (= `_/_`) are populated; everything else is null.
  `build_analysis_dataset()` filters them out via
  `_identify_ghost_episodes(ep, ADMISSION_FEATURES)`, dropping the
  matching `KeyRecordNumber`s from both the episode frame and the long
  frame.  **Post-filter universe: 899 episodes / 866 unique patients.**
  The long frame is 23 374 rows (= 899 × 26).
* **Partial-id orphans (27 episodes)** — have admission features but
  null `IDNumber`.  They survive the ghost filter (they have data) but
  are excluded from training by `dropna(subset=["IDNumber", outcome])`
  in `_prep()` and from the patient-explorer picker by
  `list_patient_options(ep)`'s `dropna(subset=["IDNumber"])`.  They
  contribute to cohort-level aggregates.  Among these 27: 9 have a
  discharge SCIM, 10 have a discharge AIS, 14 have a `LOS_days`.
* **IDNumber 1-off (raw 867 → clean 866)** — `KeyRecord 446` has the
  literal string `'6641/10/15'` (a malformed date in the ID field) as
  its IDNumber in the raw CSV; the schema declares `IDNumber: numeric`,
  so `pd.to_numeric(errors="coerce")` correctly NaN's it.
* **Outcome cardinality (post-filter, 899-episode universe)** —
  `y_discharge_scim`: 507; `y_discharge_ais`: 638; `y_discharge_wisci`:
  **50 only — too sparse for F2 regression**; `LOS_days`: 682.
* **Subscale outcomes share SCIM-total's universe** — the three
  subscale outcomes `y_discharge_scim_{self_care,resp_sphincter,mobility}`
  are pulled from the same `discharge` rows as `y_discharge_scim`, so
  they have identical n=507.  Training n=498 after dropping
  IDNumber-null partial-orphans.  Effective ranges: self-care 0–20,
  resp/sphincter 0–40, mobility 0–40, total 0–100.
* **AIS class imbalance (n=638)** — D=377 (59 %), C=105, A=63, E=62,
  B=31.  After training-time `dropna(IDNumber, target)` this becomes
  D=371, C=103, A=63, E=61, B=30 (n=628).  We use LightGBM
  `multiclass` with `class_weight="balanced"` because B and E are
  ~5 % each — without weighting they would never be predicted.
* **LOS distribution** — n=682, min=1, median=139.5, max=788; heavy
  right tail.  Modelled on `log1p` scale; conformal q computed in
  log-space and back-transformed.

## 2. Schema (`schema/*.yaml`) — the source of truth

* Every column the dashboard renders must have a `columns.yaml` entry.  If
  it doesn't, you'll see the raw Japanese in the UI.
* Every categorical raw value should resolve through `categorical_levels.yaml`
  via either the canonical `display` or a `raw_aliases` entry.
* UI strings live in `ui_strings.yaml` only.  No inline literals in
  dashboard code — use `t(schema, "key", lang)`.
* `columns.yaml` uses `families:` to template the ISNCSCI dermatomes
  (56 light-touch + 56 pin-prick + 20 key-muscle + 20 non-key-muscle
  columns are *expanded by `schema.py` at load time*, not literally in the
  YAML).  When adding a new dermatome family, extend the family block; do
  not paste 56 entries.

## 3. Model conventions

* **Random state:** `20260518`.  Embedded in `models/training_metrics.json`.
* **Group split** by `IDNumber` (patient ID) — never by row — to prevent
  same-patient leakage.
* **Outcome registry** — `src/rehab_sci/models/outcomes.py` defines the
  6 outcomes the pipeline trains.  `OUTCOMES` is an ordered tuple of
  `OutcomeSpec` records; `train.py` iterates it; the dashboard imports
  the same list so the simulator selector and the Methods tab stay in
  lockstep with the training side.  To add an outcome: extend
  `OUTCOMES`, ensure the target column is on the episode frame, add a
  `ui_strings.yaml` entry under `outcome_{key}`.
* **Per-outcome artifact layout** — `models/{spec.key}/` holds
  `lgbm_median.joblib` + `lgbm_p10.joblib` + `lgbm_p90.joblib` for
  regression heads (or `lgbm_multiclass.joblib` for AIS), plus a
  `feature_spec.joblib` (with `conformal_q_transformed`, `transform`,
  `clip_min`, `clip_max`) and `shap_test.joblib`.  The top-level
  `models/feature_spec.joblib` is the *shared* feature universe
  (feature_cols, ranges, categories) — no model-specific fields.
  `models/training_metrics.json` is `{"outcomes": {key: …}, "outcome_keys": […]}`.
* **80% conformal interval** = (1−α)-quantile of `|y − ŷ|` on a held-out
  calibration fold.  Computed on the *transformed* scale (identity for
  SCIM/AIS, log1p for LOS) so bounds remain symmetric on the modelling
  scale; back-transformed and then clipped to `[clip_min, clip_max]`.
  Coverage on n≈100 test is 81–83 % for all four regression heads.
  LightGBM quantile heads alone give ~0.41 coverage on SCIM total —
  *do not remove the conformal layer*.  At inference, the PI is the
  *union* of the conformal interval and the raw quantile interval
  (`lo = min(lo_conf, lo_q10)`, `hi = max(hi_conf, hi_q90)`) so the
  user always sees the more conservative bound.
* **Mondrian conformal (F3, session 9)** — per-AIS-grade and
  per-paralysis-class conformal quantiles replace the single marginal q.
  Stored in `feature_spec.joblib["conformal_q_by_group"]` as
  `{"ais": {letter: q}, "paralysis": {label: q}, "marginal": q,
  "min_n": 8}`.  Groups with fewer than `MONDRIAN_MIN_N=8` calibration
  samples are omitted; inference falls back AIS → paralysis → marginal.
  AIS-C consistently gets the widest PI (highest outcome variance);
  AIS-D gets the tightest for SCIM outcomes.  Per-group test coverage
  for the dominant group (AIS-D, n≈48) hits ~83 %; smaller groups
  (A, B, E) show higher variance due to small test-set n.
* **AIS (multiclass) head** — LightGBM multiclass with
  `class_weight="balanced"`.  Classes are encoded by severity
  (A=index 0 … E=index 4), so the `predict_proba` columns and the
  cached SHAP last axis are ordinally sorted.  Reported metrics:
  accuracy, quadratic-weighted Cohen κ, MAE-on-ordinal-code (1–5).
  **APS conformal classification sets (F5, session 11):** a calibration
  fold is carved from dev (20%); APS nonconformity scores (cumulative
  probability mass to include the true class) are computed and
  thresholded at ⌈(n+1)·0.8⌉ quantile to produce `q_hat`.  Mondrian
  per-AIS/per-paralysis variants stored in `feature_spec.joblib` under
  `aps_q_hat` + `aps_q_by_group`.  At inference, class probabilities
  are sorted descending and accumulated until cumsum ≥ resolved q_hat;
  the resulting set is displayed with solid/muted bar coloring in the
  dashboard.  Coverage is 99% (conservative for K=5); avg set size 2.77.
* **LOS (log1p) head** — same LightGBM regression machinery as SCIM,
  but `transform="log1p"` is applied to `y` before fitting and the
  conformal q is computed in log-space.  Predictions, PI bounds, and
  raw quantile heads are all back-transformed via `expm1` and clipped
  to `[0, ∞)` before display.  CV/test metrics are reported in days
  (raw scale) so they are human-interpretable.
* **TreeSHAP** is run on the held-out test set only, never the training
  set (would be optimistic).  Cached in `shap_test.joblib`.  For
  multiclass AIS the SHAP cache is a 3-D `(n, p, K=5)` tensor with the
  AIS axis last; at inference the simulator shows SHAP for the
  predicted class.  The insight engine's SHAP dependence panel (F6)
  slices this tensor by a user-selected class via `class_idx` kwarg to
  `fig_dependence`.
* **Holm correction**: running **max** over sorted p × (n−k+1), not
  running min.  (Fixed 2026-05-18; previous values were ~10⁻⁵⁵ for every
  test.)

## 4. Dashboard conventions

* `dcc.Store("lang-store")` holds `"ja"` / `"en"`.  Every callback that
  renders text takes it as `Input` so swaps are instant.
* Pattern-matched simulator inputs use IDs `{"type": "num"/"cat", "col": <raw>}`
  with `dash.ALL` in the consumer.  Order of the input list is fixed by
  `feature_spec.joblib['feature_cols']`.
* Plotly template name: `"medical"`.  Registered in `dashboard/theme.py`.
* Palettes: `PALETTE_CATEGORICAL`, `PALETTE_AIS` (A→E cool→warm),
  `PALETTE_PARA` (TETRA / PARA / NONE).  Use them — do not hand-pick
  colors per chart.
* Japanese rendering needs the font stack `"Hiragino Sans", "Noto Sans
  JP", "Yu Gothic UI"` in both Plotly and CSS.

## 5. Known gotchas

* **`IntCastingNaNError` on `IDNumber`** — patients with no admission row
  at all produce NaN IDs in the episode frame.  Cast via
  `dropna(subset=[outcome, "IDNumber"])` then `float64 → int64`.
* **Stale dashboard process** — `kill <PID>` only stops the `uv run`
  wrapper; the Python child keeps serving old code.  Use
  `pkill -f 'rehab_sci.dashboard.app'`.  The shell may report exit
  code 144 (signal 16) for unrelated reasons — verify with `pgrep -af`.
* **pandas fragmentation warning** when adding many columns serially —
  batch via `pd.concat([df, new_cols_df], axis=1)`.  Loader does this.
* **`@dataclass(frozen=True)` + dict fields** breaks under `@lru_cache`
  on instance methods (unhashable type).  `Schema` uses plain class with
  `__slots__` for this reason; do not "modernize" it.
* **`kaleido<1`** has no Linux x86_64 wheel under current resolver.  Keep
  `kaleido>=1.0,<2`.
* **`python -m rehab_sci.*` needs `PYTHONPATH=src`** — **RESOLVED
  2026-05-18 (session 3)**: project is now a real packaged uv project
  (hatchling build-system + `[tool.hatch.build.targets.wheel] packages =
  ["src/rehab_sci"]`).  `uv sync` installs `rehab-sci` editable into the
  venv, so `uv run python -m rehab_sci.*` works without any
  `PYTHONPATH` prefix.  Historical context preserved: previously
  `pyproject.toml` declared `[tool.uv] package = false`, which made the
  `src/rehab_sci/` layout invisible to the venv and forced every launch
  command to be prefixed with `PYTHONPATH=src`.
* **Background dashboard from inside a bash one-liner** — `nohup … &`
  inside the harness's wrapper sometimes does not survive the wrapper's
  exit (parent shell exit code 144 = SIGTERM bookkeeping).  Use the Bash
  tool's `run_in_background: true` flag, or run the command as the
  *last* statement of the bash command so it inherits the wrapper's
  lifetime.
* **Plotly `Sunburst(branchvalues="total")` requires parent value = sum
  of children**.  Setting parent values to `0` silently renders a blank
  chart (no JS error, no log line).  Either accumulate leaf counts into
  every ancestor (see `fig_injury_sunburst`) or switch to
  `branchvalues="remainder"`.  The accumulate-into-ancestor pattern is
  preferred — Plotly hover then shows the true subtotal at each ring.

## 6. Commands cheat sheet

```bash
uv sync                                          # install deps
uv run python scripts/01_profile_raw.py          # refresh schema profile
uv run python -m rehab_sci.models.train          # train + conformal + SHAP
uv run python -m rehab_sci.models.subgroups      # subgroup discovery
uv run python -m rehab_sci.dashboard.app         # serve at :8050
pkill -f 'rehab_sci.dashboard.app'               # stop stale dashboard
```

## 7. Session log (most recent first)

### 2026-05-24 (session 12, F6 SHAP class selector for AIS shipped)

* Shipped **F6 SHAP class selector for AIS multiclass dependence**.
* **Problem:** The insight engine's SHAP dependence panel showed a
  "regression only" note for the AIS discharge outcome because the
  multiclass SHAP tensor is 3-D `(n_samples=126, n_features=30,
  n_classes=5)` and `fig_dependence` assumed a 2-D array.
* **`dashboard/figures.py`** — `fig_dependence()` gains optional
  keyword `class_idx: int | None`.  When provided and `shap_values` is
  3-D, slices `sv[:, feature_idx, class_idx]` to produce a 1-D SHAP
  vector.  Y-axis label becomes `"SHAP (AIS {letter})"`.  Hovertemplate
  also updated.  Backward-compatible: regression callers pass no
  `class_idx`.
* **`dashboard/app.py`** layout — `render_insights()` adds a class
  selector dropdown (`ins-dep-class`) alongside the existing feature
  dropdown in a flex row.  Wrapped in `ins-dep-class-wrap` div whose
  `display` is toggled by a new callback.
* **`dashboard/app.py`** callbacks:
  - New `update_dep_class_options`: `ins-outcome` → sets class dropdown
    visibility (`display:none` for regression, `flex:1` for multiclass),
    options (A/B/C/D/E), and default value (0=A).
  - Modified `update_dependence`: now takes `ins-dep-class` as an Input.
    Removed the `bundle["task"] != "regression"` short-circuit.  Passes
    `class_idx` to `fig_dependence` for multiclass; `None` for
    regression.  Defaults `class_idx=0` if the class dropdown hasn't
    fired yet (guard against init timing).
* New UI string: `insight_dep_class_label` (ja: "対象クラス",
  en: "Target class").
* **Per-class SHAP observations (test set, AIS, feature=年齢/age):**
  mean |SHAP| varies meaningfully: C=0.154, D=0.107, B=0.064, E=0.062,
  A=0.041.  Each class shows genuinely different dependence patterns —
  the class selector is informative, not just cosmetic.
* Verification: dashboard HTTP 200 on `/`, `/_dash-layout`,
  `/_dash-dependencies`.  Callback graph verified: 3 insight-dep
  callbacks with correct dependency ordering (outcome → class/feature
  options → figure).  `fig_dependence` tested for all 5 class indices
  on both numeric and categorical features.

### 2026-05-24 (session 11, F5 APS conformal classification sets shipped)

* Shipped **F5 APS conformal classification sets for AIS**.
* **Training side** (`models/train.py`):
  - `_train_multiclass()` now splits dev into train (80%) + calibration
    (20%), matching the regression path.  Calibration fold is used for
    APS nonconformity score computation; early-stopping val is split from
    the training portion (10% of the 80%).
  - New helpers: `_aps_scores(proba, y_true)` — cumulative probability
    mass until the true class is included; `_aps_prediction_set(proba, q)`
    — inference-time set computation; `_aps_test_metrics(proba, y, q, X)`
    — per-row coverage and set size on test data using Mondrian q values.
  - APS threshold `q_hat` + Mondrian per-AIS/per-paralysis variants
    stored in `feature_spec.joblib` under `aps_q_hat` and
    `aps_q_by_group` (same fallback structure as regression Mondrian:
    AIS grade → paralysis → marginal, groups with <8 cal samples
    omitted).
  - Existing `_conformal_q` and `_compute_mondrian_q` helpers reused
    directly — APS scores are just another nonconformity measure.
  - Test-set APS metrics reported in `training_metrics.json`:
    `aps_q_hat`, `aps_coverage_80`, `aps_avg_set_size`,
    `aps_mondrian_coverage` (per-group breakdown).
* **Dashboard side** (`dashboard/app.py`):
  - `_resolve_conformal_q()` refactored: common Mondrian resolution
    logic extracted into `_resolve_group_q(q_by_group, marginal, X)`,
    called by both `_resolve_conformal_q` (regression) and new
    `_resolve_aps_q` (classification).  `_aps_prediction_set()` also
    added to app.py for inference.
  - `_fig_class_probabilities()` gains optional `conformal_set`
    parameter: bars in the set get solid accent color; bars outside get
    muted `rgba(17,122,139,0.18)`.  Backward-compatible (default=None).
  - `_simulate_multiclass()` and `_patient_multiclass()` both compute
    the APS conformal set using Mondrian q resolution and display it in
    the readout as "80% prediction set (APS) : {C, D}" and in the bar
    chart via the muted/solid visual distinction.
  - `_perf_block_multiclass()` extended: shows APS coverage, avg set
    size, and per-group (AIS + paralysis) Mondrian breakdown in the
    Methods tab.  Backward-compatible: checks for `aps_q_hat` presence
    in metrics before rendering.
* New UI string: `sim_conformal_set` (ja: "80%予測集合 (APS)",
  en: "80% prediction set (APS)").
* **APS metrics (SCIM test set, n=126):**
  - Marginal q_hat=0.917, coverage=99% (exceeds 80% — conservative for
    K=5 due to discrete APS score distribution), avg set size=2.77.
  - Per-AIS Mondrian q: A=0.944, C=0.865, D=0.917.  AIS-C gets the
    tightest sets (motor-incomplete outcomes most predictable); AIS-A
    gets the widest (complete injuries most uncertain).
  - Per-AIS test set sizes: A=3.9, B=3.5, C=2.8, D=2.3, E=2.0.
    AIS-D patients typically get 2-class sets; AIS-A patients typically
    get 4-class sets.
  - Point accuracy slightly lower (0.683 vs 0.714 previously) due to
    smaller dev training set (calibration carved out); CV unchanged
    (0.669).  Shipped model is refitted on all data — test metrics
    reflect the weaker dev model, not the shipped model.
* Verification: full pipeline `uv run python -m rehab_sci.models.train`
  + `subgroups`; dashboard HTTP 200 on `/`, `/_dash-layout`,
  `/_dash-dependencies`.  Smoke tests confirm: (1) APS set displayed
  in simulator + patient explorer readouts, (2) bar chart solid/muted
  distinction correct, (3) Mondrian fallback B→TETRA works, (4)
  Methods tab shows per-group APS metrics, (5) bilingual labels correct.

### 2026-05-24 (session 10, F4 multi-outcome insight + patient explorer shipped)

* Shipped **F4 Multi-outcome insight engine + patient explorer**.
* **Insight engine** gains an outcome selector (`ins-outcome` dropdown)
  controlling all three panels:
  - **Global SHAP importance**: moved from inline static rendering to a
    callback (`update_importance`); now shows the top-15 features for the
    selected outcome.
  - **Subgroup comparison**: `update_subgroup` now dispatches to the
    selected outcome's subgroups data and target column; box-plot y-axis
    shows the outcome's display label.
  - **SHAP dependence**: `update_dependence` uses the selected outcome's
    SHAP bundle; feature dropdown updates per outcome via
    `update_dep_feature_options`.  For multiclass (AIS), the dependence
    panel shows a "regression only" note instead of a broken plot.
* **Patient explorer** gains an outcome selector (`patient-outcome`
  dropdown) in the picker card.  `_compute_patient_tab` accepts
  `outcome_key` and branches:
  - **Regression outcomes**: prediction readout shows value ± PI,
    observed value, residual; `fig_patient_prediction` uses the outcome's
    `clip_min`/`clip_max`/label for the x-axis range.
  - **Multiclass (AIS)**: shows class-probability bar chart and SHAP for
    the predicted class; observed value mapped back to letter via
    `AIS_ORD_TO_LETTER`.
  - Timeline chart remains SCIM-based (the only longitudinal measure).
* **`models/subgroups.py`** — `main()` now iterates over all six
  `OUTCOMES` and writes a multi-keyed `subgroups.json`:
  `{"scim_total": {...}, "ais_discharge": {...}, ...}`.
  `run_all_subgroups` is unchanged; the loop happens in `main()`.
  Backward-compatible loading in `app.py` handles both old flat and new
  keyed formats.
* **`dashboard/figures.py`** — `fig_subgroup_box` gains `outcome_col`
  and `outcome_label` params (defaults preserve backward compat).
  `fig_patient_prediction` gains `clip_min`, `clip_max`, `axis_label`
  params (defaults preserve backward compat).
* New UI strings: `insight_outcome_label`, `insight_dependence_regression_only`,
  `patient_outcome_label`.
* New helpers in `app.py`: `_insight_outcome_options(lang)`,
  `_get_observed_for_outcome(key_record, spec)`,
  `_patient_regression(bundle, X, key_record, lang)`,
  `_patient_multiclass(bundle, X, key_record, lang)`.
* New callbacks: `update_insight_outcome_options`, `update_importance`,
  `update_dep_feature_options`.  Modified callbacks: `update_subgroup`,
  `update_dependence`, `update_patient_tab`.
* Verification: `uv run python -m rehab_sci.models.subgroups` produces
  all six outcome blocks; dashboard HTTP 200 on `/`, `/_dash-layout`,
  `/_dash-dependencies`.  Smoke test confirms all six outcomes produce
  correct predictions in both patient explorer and insight engine.

### 2026-05-24 (session 9, F3 Mondrian conformal shipped)

* Shipped **F3 Mondrian per-AIS / per-paralysis conformal calibration**
  for all five regression heads.
* New helpers in `models/train.py`: `_conformal_q()` (refactored from
  inline), `_compute_mondrian_q()`, `_resolve_mondrian_q_array()`,
  `_mondrian_test_coverage()`.  Constants: `AIS_ORD_TO_LETTER`,
  `MONDRIAN_MIN_N=8`.
* `_train_regression()` now: (1) computes per-group conformal q on the
  calibration fold (AIS grades + paralysis classes, each with ≥8
  samples), (2) stores the dict in `feature_spec.joblib` under
  `conformal_q_by_group`, (3) evaluates test-set coverage using the
  Mondrian q's (each test point gets its group-specific q), (4) reports
  per-group coverage in `training_metrics.json`.
* `dashboard/app.py` gains `_resolve_conformal_q(fspec, X)` — priority:
  AIS group → paralysis group → marginal.  Both the simulator and
  patient-explorer inference paths now use the resolved q instead of
  the marginal.
* Methods tab `_perf_block_regression` extended with per-group Mondrian
  coverage line (bilingual).
* **Per-outcome Mondrian q values (SCIM total):** A=17.8, C=35.7,
  D=18.0; Paralysis: TETRA=25.5, PARA=24.6; marginal=24.6.  AIS-C
  (motor-incomplete) gets a 2× wider PI than AIS-D — clinically
  correct.  AIS-B and E omitted (≤4 cal samples, fall back to
  paralysis or marginal).
* Marginal test coverage preserved identically (83 % for SCIM total).
  Mondrian overall coverage = 80 % (slightly lower because per-group
  q's redistribute width).  Per-group test coverage: D=83 %(n=48),
  C=91 %(n=23), A=67 %(n=18), B=71 %(n=7).  Small-group (A, B)
  undercoverage is a finite-sample artifact of ~8-cal-sample groups.
* Verification: full `uv run python -m rehab_sci.models.train`
  reproduces all metrics; dashboard HTTP 200 on `/`, `/_dash-layout`,
  `/_dash-dependencies`.  Smoke test confirms AIS-D PI width=36 vs
  AIS-C PI width=71 (was 49 for both with marginal q).

### 2026-05-24 (session 8, CLAUDE.md update response)

* User updated `CLAUDE.md` with several new/expanded directives.  Key
  additions: (a) memory/scratchpad system mandate, (b) reusable session
  prompt requirement, (c) directory-scoping constraint, (d) CLAUDE.md
  modification freedom (previously required user approval), (e) testing
  philosophy, (f) KISS/UNIX/overengineering guidance, (g) expanded
  security audit + reasoning methodology directives.
* Updated §0 to remove the outdated "requires user approval to modify"
  note about CLAUDE.md, and added a reminder to keep AGENT_NOTES
  accurate and pruned.
* Added §0b "Lessons & mistakes" section, consolidating past mistakes
  from session logs into a dedicated section for quick reference.
* Emitted the reusable session prompt for future sessions.
* No code changes this session — pure documentation/process.

### 2026-05-18 (session 7, F2 multi-outcome prediction shipped)

* Shipped **F2 multi-outcome prediction** — `train.py` now trains six
  heads in a single run; the simulator gets an outcome selector; the
  Methods tab reports per-outcome metrics.  WISCI stays dropped
  (n=50 too sparse, decided session 6).
* Six heads, in registry order:
  - **SCIM-III total** (regression, 0–100) — R²=0.696, RMSE=18.92,
    conformal80=83 % (identical to the pre-refactor single-outcome
    run — confirms the refactor preserves invariants).
  - **SCIM self-care** (0–20) — R²=0.666, RMSE=4.08, conformal80=77 %.
  - **SCIM resp/sphincter** (0–40) — R²=0.618, RMSE=8.10, conformal80=81 %.
  - **SCIM mobility** (0–40) — R²=0.695, RMSE=8.39, conformal80=82 %.
  - **AIS at discharge** (multiclass A→E, n=628) — accuracy=0.714,
    quadratic-weighted κ=0.772 ("substantial" by Landis-Koch),
    ordinal-MAE=0.365 (most errors within ±0 or ±1 grade).
  - **LOS in days** (regression, log1p transform, n=668) — R²=0.215,
    RMSE=110 d, conformal80=81 %.  Genuinely hard outcome at admission;
    the calibrated PI is the operational deliverable, not the point R².
* New module `src/rehab_sci/models/outcomes.py` with `OutcomeSpec`
  dataclass + `OUTCOMES` ordered tuple.  Single source of truth: both
  `train.py` and `dashboard/app.py` import this.
* `train.py` rewritten as `_train_regression(spec, …)` and
  `_train_multiclass(spec, …)` over a shared `_prep` + helper
  surface.  Regression path: CV → grouped holdout → calibration
  fold → split conformal → final refit on all data → TreeSHAP on
  test set.  Multiclass path: CV → grouped holdout (no separate
  calibration — no conformal sets this session) → final refit →
  multiclass TreeSHAP (3-D tensor `(n, p, K=5)`).
* `data/dataset.py::build_episode_frame()` now pulls SCIM subscales
  at the `discharge` timepoint (`y_discharge_scim_{self_care,
  resp_sphincter, mobility}`) the same way it pulls `y_discharge_scim`.
* Artifacts re-organized to per-outcome subdirectories
  (`models/scim_total/…`, `models/ais_discharge/…`, etc.).  The
  top-level `models/feature_spec.joblib` is now the *shared* feature
  universe only (no model-specific fields).  `training_metrics.json`
  is `{"outcomes": {key: {cv, test, …}}, "outcome_keys": […]}`.
* Dashboard simulator gets a `dcc.Dropdown(id="sim-outcome")` above
  the readout.  The callback dispatches on `bundle["task"]` — the
  regression branch renders the existing horizontal PI bar
  (re-parametrized with the outcome's display label / units / clip
  range / transform); the multiclass branch renders a 5-bar
  class-probability chart and a SHAP plot for the predicted class.
* Methods tab now loops over outcomes and renders a per-outcome
  performance card.  Six new `outcome_*` keys + `sim_outcome_label`
  + `sim_class_probabilities` + `sim_predicted_class_label` +
  `methods_per_outcome_heading` in `ui_strings.yaml`.  Both JA and
  EN labels.
* Patient explorer continues to predict SCIM-III total only.
  Extending it to all outcomes is a clean follow-up (see new §8 F4
  candidate) but out of F2 scope.
* Insight engine (global SHAP, dependence) is anchored on
  SCIM-total.  Per-outcome variant also deferred to F4 candidate.
* CSS additions to `assets/style.css`: `.sim-outcome-selector`
  (dropdown above readout with hairline divider) and
  `.methods-perf-card` (per-outcome perf block).
* `subgroups.py` was *not* touched this session — it still compares
  every feature against `y_discharge_scim`.  Extending it to all
  outcomes is also out of F2 scope; subgroup discovery against AIS
  or LOS would be a useful F4-tier follow-up.
* Verification: full pipeline `uv run python -m rehab_sci.models.train`
  produced metrics in the table above; dashboard `uv run python -m
  rehab_sci.dashboard.app` serves HTTP 200 on `/`, `/_dash-layout`,
  and `/_dash-dependencies`.  Python-level dispatch smoke test
  confirms all six heads return plausible predictions on the
  default-feature row (e.g., LOS=208 d with PI 82–468, AIS=D 71 %).
* Open items rolled forward:
  - **F3 Mondrian per-AIS / per-paralysis conformal calibration** —
    now top of the §8 default-work pool.
  - **F4 candidate: extend insight engine + patient explorer to
    multi-outcome.**  Insight global SHAP + subgroup box + SHAP
    dependence should accept an outcome selector; patient explorer
    should let the clinician pick which outcome to predict for the
    chosen episode.  Estimated medium effort (UI surface).
  - **F5 candidate: APS / RAPS conformal classification sets for
    AIS.**  Multiclass head currently returns probabilities only.
  - Pytest smoke suite (still un-done, still low priority).

### 2026-05-18 (session 6, loader sanity check + ghost-episode filter)

* User redirected from F2 to first investigate the two §1 anomalies
  flagged at end-of-session-5: the 328 NaN-`IDNumber` episodes and the
  301 all-NaN-admission episodes.  Goal: decide whether they are a
  loader bug or a data-quality artefact, then act accordingly.
* Probe ran in two passes (no on-disk artefacts; everything via
  `uv run python -` heredoc).  Findings:
  - **Loader is correct.**  The raw CSV ships 8 502 rows with
    `IDNumber = '_'` (correctly parsed to NaN by the loader's
    `na_values=["", "_", "NA"]`).  8 502 / 26 = 327 episodes; the +1
    drift to 328 traces to `KeyRecord 446` whose raw `IDNumber` is the
    literal `'6641/10/15'` — coerced to NaN by
    `pd.to_numeric(errors="coerce")` because the schema declares
    `IDNumber: numeric`.
  - **The 301 all-NaN-admission episodes are pure placeholder rows.**
    Strict subset of the 328 NaN-id set.  Across their 7 826
    long-frame rows, only 25 of 234 non-structural columns ever hold a
    value, and 22 of those 25 have a total of 26 non-null cells across
    7 826 rows (rounding error).  The only consistently populated
    columns are `BusinessYear`, `AnualCaseNumber`, `mFrankel_Frankel`
    (= `_/_`).  Zero of these 301 have a discharge SCIM / AIS / WISCI
    or a `LOS_days`.  Unusable for any task.
  - **27 partial-id orphans** have admission features but no IDNumber.
    Training already drops them via `dropna(IDNumber)`; the patient
    explorer's picker already drops them via the same.
  - **`LOS_days` is real and well-covered**: 682 / 1 200 episodes
    (median 139 d, range 1–788), already pulled directly from
    `入院期間`.  F2 LOS outcome viable.
  - **WISCI is too sparse for F2**: only 50 episodes have a discharge
    WISCI (`y_discharge_wisci`).  F2 should drop or de-prioritize it.
* User chose "filter at loader, document".  Implemented as
  `_identify_ghost_episodes(ep, ADMISSION_FEATURES)` inside
  `data/dataset.py`; called by `build_analysis_dataset()` after
  `build_episode_frame()` to drop matching `KeyRecordNumber`s from
  both `ep` and `long_df` (`reset_index(drop=True)` on both).  Module
  docstring updated to describe the rule.
* Re-ran `models.subgroups` and `models.train` on the filtered
  dataset.  **Training metrics identical** (training rows = 498,
  R²=0.696, RMSE=18.92, MAE=13.70, conformal-80 = 83 %, top SHAP
  features unchanged) — confirming the ghosts were already excluded
  via `dropna(IDNumber, outcome)` and the filter only fixes the
  cohort-level n.  `n_episodes_total` in `training_metrics.json` will
  now read 899 instead of 1 200.
* Dashboard served HTTP 200 on `:8050` post-filter; `_dash-layout` and
  `_dash-dependencies` endpoints both 200; no tracebacks in the
  server log.  Cohort `episodes_n` value sourced from `len(EP)` will
  now display 899.
* §1 invariants rewritten to reflect the post-filter universe and to
  correct the prior session's bogus `_apply_missing_sentinels`
  reference (no such function — sentinels are NaN'd via two real
  mechanisms documented inline).
* README §2 episode-frame paragraph rewritten to explain the
  ghost-filter rule and the 899/866 post-filter counts.  README
  trained-model n (498) unchanged.
* Open items rolled forward:
  - **F2 multi-outcome prediction** — still next default work.
    **Caveat from this session:** WISCI's n=50 is below any reasonable
    regression power; F2 should drop WISCI from the outcome set, or
    treat it as a classification-of-walker-status proxy if we want it
    at all.  AIS (n=638), per-subscale SCIM (≈507), and LOS (n=682)
    remain viable.
  - **F3 Mondrian per-AIS / per-paralysis conformal** — third in §8.
  - Pytest smoke suite (still low priority per session-5 user
    direction).

### 2026-05-18 (session 5, F1 patient explorer shipped)

* Shipped **F1 Patient explorer tab** end-to-end.  New tab between
  the simulator and insight engine; bilingual; user-driven picker.
* New module `src/rehab_sci/data/episodes.py` with:
  - `PATIENT_TIMELINE` (the 11-point sequence `0day → discharge`,
    matching `fig_recovery_curves`; later timepoints are too sparse).
  - `list_patient_options(ep)` → ordered patient picker rows
    (`PatientOption` with id_number, n_episodes, age, sex, paralysis,
    ais_admit, key_records).
  - `patient_timeline(long_df, key_record)` → per-timepoint frame
    re-indexed to `PATIENT_TIMELINE` (so gaps render as gaps).
  - `patient_meta(ep, key_record)` → demographics + admission + outcome
    summary dict used by the meta-chip strip.
  - `episode_admission_features(ep, key_record, feature_cols)` →
    dict-of-features for building the model input row.
  - `cohort_percentile_bands(long_df, ep, value_col, group_keys, min_n=5)`
    → per-(timepoint × admission-strata) p10/p25/p50/p75/p90 + n.
    **Key gotcha:** the long frame carries per-row copies of demographics
    /injury fields, so the inner-join is renamed to `_band_<key>` before
    merging to avoid `_x`/`_y` suffix collisions.
* New figure factories in `dashboard/figures.py`:
  - `fig_patient_scim_timeline(long_df, ep, key_record, strata, schema, lang)`
    — SCIM-III total + subscale lines (subscales `visible="legendonly"`
    by default so they don't clutter), overlaid on cohort `p10–p90` +
    `p25–p75` ribbons and a dashed cohort median.  Falls back from
    `para_ais` → `para` if the patient's admission AIS is null.
  - `fig_patient_prediction(pred, lo, hi, observed, ...)` — PI bar +
    predicted-median diamond + observed crimson X.
  - `_hex_to_rgba(hex_color, alpha)` helper (was inlined elsewhere).
* `dashboard/app.py`:
  - New tab `dcc.Tab(value="patient", ...)` and `render_patient(lang)`.
  - `_compute_patient_tab(key_record, strata, lang)` is a plain
    function the `@callback` delegates to — keeps the business logic
    directly callable for tests/probes.  Handy because Dash's
    `@callback`-wrapped functions can't be invoked from Python without
    a Dash request context.
  - Callbacks: `update_patient_picker` (refreshes options on lang
    change), `reset_episode_on_patient_change` (auto-picks the first
    episode of the chosen patient), `update_patient_tab` (renders the
    7 outputs: meta strip + timeline + ISNCSCI table + prediction
    readout + prediction figure + SHAP figure + footer note).
* New UI strings in `schema/ui_strings.yaml` (`tab_patient`,
  `patient_*`).  AIS in the picker label is shown as the letter only
  (e.g. `AIS D`) — using the level_label gives `AIS D (運動不全)`
  which double-prefixes when concatenated.  Picker labels were also
  shortened (TETRA→四麻/Tetra, age suffix dropped to 歳/y) so all 866
  entries fit on one line at 320 px sidebar width.
* CSS additions to `assets/style.css`: `.patient-grid` (sticky picker
  card + content column), `.patient-meta-chip`, `.patient-isncsci-table`,
  `.patient-pred-readout/.patient-pred-empty`, dropdown menu
  hardening (`white-space:nowrap`, hairline row separator, focused-row
  accent-soft background, `optionHeight=36` on the Dropdown component).
* Browser verification: user drove the tab in a real browser; **single
  defect surfaced** — clicking the patient ID dropdown produced
  multi-line option wraps that blended into each other.  Root cause:
  default Dash Dropdown row height (35 px) is shorter than a wrapped
  long Japanese+ASCII label, so adjacent rows overlapped.  Fix: short
  one-line labels + `optionHeight=36` + the CSS hardening above.
* Behavioural observations recorded in §1 invariants (866 patients,
  328 NaN-id episodes, 301 all-NaN-admission episodes).
* Open items rolled forward:
  - **F2 Multi-outcome prediction** (subscales + AIS + WISCI + LOS) —
    next default-work item.
  - **F3 Mondrian per-AIS / per-paralysis conformal calibration** —
    third in the §8 backlog.
  - Investigate the 328 NaN-id and 301 all-NaN-admission episodes —
    are they truly feature-less or is the loader dropping signal?
  - Pytest smoke suite (still un-done; the QA loop continues to catch
    real bugs that tests would not have caught, so this remains low
    priority for now per user direction).

### 2026-05-18 (session 4, pivot to feature backlog)

* User redirected the project's default-work pool away from maintenance
  tasks (pytest suite, CI, second exhaustive browser-QA pass) and toward
  features.  Per user decision, the historical "Open items rolled
  forward" lists from sessions 1–3 are now **superseded**; future fresh
  sessions must propose work from §8 unless the user redirects.
* Added §8 "Feature backlog" with three Tier-A candidates, in priority
  order:
  - **F1** Patient explorer tab.
  - **F2** Multi-outcome prediction (subscales + AIS + WISCI + LOS).
  - **F3** Mondrian per-AIS / per-paralysis conformal calibration.
* Added a §0 "Read-first" pointer naming §8 as the default-work pool and
  explicitly marking the older rolled-forward open items as history.
* No code changes this session — pure roadmap edit.

### 2026-05-18 (session 3, packaging refactor)

* Converted the project to a real packaged uv project, eliminating the
  `PYTHONPATH=src` launch quirk that had been carried since session 1.
* `pyproject.toml` changes:
  - Added `[build-system] requires = ["hatchling"]` /
    `build-backend = "hatchling.build"`.
  - Replaced `[tool.uv] package = false` with
    `[tool.hatch.build.targets.wheel] packages = ["src/rehab_sci"]`
    (explicit src-layout target — hatchling auto-detection would also
    work for `rehab-sci` → `rehab_sci`, but explicit is safer).
* `uv.lock` flipped `rehab-sci`'s source from `virtual = "."` to
  `editable = "."`.  No transitive dep churn.
* Verified all modules import without `PYTHONPATH=src`
  (`rehab_sci.schema`, `data.loader`, `data.dataset`, `models.train`,
  `models.subgroups`, `dashboard.{app,figures,theme,i18n}`); dashboard
  serves HTTP 200 on `:8050` via plain
  `uv run python -m rehab_sci.dashboard.app`.
* §5 gotcha marked **RESOLVED**, historical wording preserved per the
  "append; never delete" policy.
* Open items rolled forward (unchanged from session 2):
  - Pytest smoke + invariants suite (schema round-trip, ISNCSCI sums,
    episode-frame shape 1200×N + 867 patients, `build_analysis_dataset()`
    end-to-end, loadability of `models/*.joblib`, shape of
    `subgroups.json`).  Now the highest-priority deferred work.
  - CI (after tests exist).
  - Mondrian per-AIS conformal for per-subgroup coverage.
  - Second browser-QA pass for other Plotly silent-failure traps and
    JA/EN parity bugs.

### 2026-05-18 (session 2, first browser QA pass)

* User drove the dashboard in a real browser (JA + EN).  Single defect
  surfaced: **Cohort overview → Injury hierarchy** sunburst rendered as
  blank white space in both languages.
* Root cause: `fig_injury_sunburst` used `branchvalues="total"` but
  assigned `value=0` to every parent ring (para, AIS) and only the leaf
  count to the NLI ring.  Plotly's "total" mode requires parent =
  Σchildren, so it silently refused to draw.  Fix accumulates each
  leaf count into all three ancestors via an `_upsert` helper; verified
  by hand: 122 nodes, top-ring totals 699/139/4 = 842 (= `sub.shape[0]`
  after `dropna`), zero parent/child mismatches.
* Documented two new recurring gotchas in §5: the `PYTHONPATH=src`
  launch requirement (because `pyproject.toml` has
  `[tool.uv] package = false`) and the Plotly sunburst `branchvalues`
  trap.
* Open items rolled forward:
  - Convert the project to a real packaged uv project so
    `PYTHONPATH=src` is no longer needed.  Deferred — touches build
    system, low urgency.
  - Pytest smoke + invariants (schema round-trip, ISNCSCI sums,
    episode-frame shape 1200×N + 867 patients, `build_analysis_dataset()`
    end-to-end).  Still un-done; the QA session uncovered a defect a
    test would not have caught (Plotly visual), so the test suite is
    still worth doing as a separate session.
  - CI (after tests exist).
  - Mondrian per-AIS conformal for per-subgroup coverage.

### 2026-05-18 (session 1, initial build)

* Built phases 1–3 end-to-end.
* Initial commit `6ffab8a`.
* Fixed Holm step-down bug (`min` → `max`); regenerated `subgroups.json`.
* Open items:
  - Visual QA of the dashboard in a real browser (probes return HTTP 200
    but no human has eyeballed it yet).
  - No tests yet.  Consider a minimal `pytest` covering schema round-trip,
    ISNCSCI sums, episode frame shape, and an end-to-end smoke of
    `build_analysis_dataset()`.
  - No CI yet.
  - Conformal calibration on n=80 is small; a larger calibration set or
    Mondrian per-AIS conformal could give per-subgroup coverage.

## 8. Feature backlog

Default-work pool for fresh sessions.  Propose from this list unless the
user redirects.  Each entry: **what / why / effort / files / data
dependency**.  Ordered by recommended start order (F1 first).

### F1. Patient explorer tab — **STATUS: shipped (session 5, 2026-05-18)**

* **What:** New dashboard tab.  Pick a `KeyRecordNumber` (or `IDNumber`
  for multi-episode patients) and see that patient's observed timeline —
  SCIM total + subscales, AIS, ISNCSCI summaries at every timepoint —
  overlaid on cohort percentile bands stratified by paralysis / AIS.
  For episodes with an admission row, also overlay the model's predicted
  discharge SCIM ± 80 % PI and the local SHAP attribution for this
  patient.
* **Why:** The dashboard today is cohort + hypothetical-patient.
  Clinicians' first question is *"what does this tool say about my real
  patient X?"*  Largest UX gap.
* **Effort:** medium (~1 session).
* **Files:** `dashboard/app.py` (new tab + callbacks),
  `dashboard/figures.py` (timeline + percentile bands), possibly new
  `data/episodes.py::patient_view()` to aggregate one patient's rows.
* **Data dependency:** existing long + episode frames; no new ingestion.

### F2. Multi-outcome prediction — **STATUS: shipped (session 7, 2026-05-18)**

* **What was built:** Six prediction heads driven by a single outcome
  registry (`models/outcomes.py::OUTCOMES`).  SCIM-III total + the three
  subscales (self-care 0–20, resp/sphincter 0–40, mobility 0–40) +
  AIS at discharge (multiclass A→E with `class_weight="balanced"`) +
  LOS in days (regression with `log1p` transform).  `train.py`
  iterates the registry; each head writes its own subdirectory under
  `models/{spec.key}/` (median + p10/p90 + feature_spec + shap_test,
  or the multiclass equivalent).  Dashboard simulator gains an
  outcome dropdown; multiclass renders a class-probability bar chart
  instead of a PI bar.  Methods tab loops over outcomes.
* **Metrics summary** (test split):
  - scim_total: R²=0.696, RMSE=18.92, conformal80=83 %
  - scim_self_care: R²=0.666, RMSE=4.08, conformal80=77 %
  - scim_resp_sphincter: R²=0.618, RMSE=8.10, conformal80=81 %
  - scim_mobility: R²=0.695, RMSE=8.39, conformal80=82 %
  - ais_discharge: accuracy=0.714, κ_quad=0.772, ordMAE=0.365
  - los_days: R²=0.215, RMSE=110 d, conformal80=81 %
* **WISCI** stayed dropped (50 episodes — below regression power).
* **Out of scope for F2 (rolled forward as F4 / F5):** patient
  explorer multi-outcome support, insight engine multi-outcome
  support, conformal classification sets for AIS, subgroup discovery
  for non-SCIM outcomes.

### F3. Mondrian per-AIS / per-paralysis conformal — **STATUS: shipped (session 9, 2026-05-24)**

* **What was built:** Per-AIS-grade and per-paralysis-class conformal
  quantiles for all five regression heads.  `feature_spec.joblib` now
  carries `conformal_q_by_group` with AIS and paralysis sub-dicts plus
  a marginal fallback.  Groups with <8 calibration samples are omitted;
  inference resolves AIS → paralysis → marginal.  Dashboard simulator
  and patient explorer both use the resolved q.  Methods tab reports
  per-group test-set coverage.
* **Key finding:** AIS-C (motor-incomplete) gets ~2× wider PI than
  AIS-D across all SCIM outcomes — clinically correct and previously
  hidden by the marginal.  LOS q values are similar across AIS groups
  on the log scale.
* **Metrics (SCIM total):** AIS-D q=18.0 (PI width=36), AIS-C q=35.7
  (PI width=71), marginal q=24.6 (was PI width=49 for everyone).
  Mondrian overall test coverage=80 %; per-group: D=83 %(n=48),
  C=91 %(n=23), A=67 %(n=18).  Small-group undercoverage is a known
  artifact of n_cal≈8; acceptable given the dominant-group improvement.

### F4. Multi-outcome insight engine + patient explorer — **STATUS: shipped (session 10, 2026-05-24)**

* **What was built:** Outcome selector added to both the insight engine
  and the patient explorer tabs.  Insight engine's global SHAP
  importance, subgroup box, and SHAP dependence all respond to the
  selected outcome.  Patient explorer predicts whichever outcome the
  clinician picks — regression outcomes show PI bar + residual;
  multiclass (AIS) shows class-probability chart.  `subgroups.py` now
  runs discovery for all six outcomes, producing a multi-keyed
  `subgroups.json`.  SHAP dependence is disabled for multiclass (3-D
  SHAP tensor needs a class selector — deferred).
* **Files changed:** `dashboard/app.py` (selectors, callbacks,
  helpers), `dashboard/figures.py` (`fig_subgroup_box`,
  `fig_patient_prediction` parameterized), `models/subgroups.py`
  (multi-outcome loop), `schema/ui_strings.yaml` (3 new keys).

### F5. APS conformal classification sets for AIS — **STATUS: shipped (session 11, 2026-05-24)**

* **What was built:** APS (Adaptive Prediction Sets) conformal
  classification sets for the AIS multiclass head.  A calibration fold
  is carved from the dev set (same as regression); APS nonconformity
  scores are computed on it; the ⌈(n+1)(1-α)⌉-th quantile becomes
  `q_hat`.  Mondrian per-AIS/per-paralysis `q_hat` variants give
  tighter sets for well-predicted groups (C) and wider sets for
  uncertain groups (A).  Dashboard renders the prediction set in both
  the simulator and patient explorer, with solid/muted bar coloring.
  Methods tab reports coverage + avg set size per group.
* **Key finding:** APS coverage is 99% (vs 80% target) because K=5
  produces discrete APS scores; the ⌈(n+1)·0.8⌉ quantile overshoots.
  This is the expected conservative behavior for small K.  Average set
  size is 2.77, with AIS-D patients getting ~2-class sets and AIS-A
  patients getting ~4-class sets.
* **Metrics:** q_hat=0.917; per-AIS q: A=0.944, C=0.865, D=0.917.
  Test accuracy=0.683 (dev model on smaller training set), CV=0.669
  (unchanged).
* **Files changed:** `models/train.py` (APS helpers + calibration
  fold), `dashboard/app.py` (q resolution refactor + set rendering),
  `schema/ui_strings.yaml` (1 new key).

### F6. SHAP class selector for AIS multiclass dependence — **STATUS: shipped (session 12, 2026-05-24)**

* **What was built:** A class selector dropdown (A/B/C/D/E) added to
  the insight engine's SHAP dependence panel.  When the user selects
  the AIS discharge outcome, the class dropdown appears alongside the
  feature dropdown; selecting a class slices the 3-D SHAP tensor
  `(n, p, K=5)` at the chosen class axis and renders the standard
  dependence scatter/box plot.  For regression outcomes the class
  dropdown is hidden.  `fig_dependence` now supports an optional
  `class_idx` keyword parameter.
* **Key finding:** Per-class SHAP dependence patterns differ
  meaningfully — e.g., age has 3.7× higher mean |SHAP| for AIS-C vs
  AIS-A.  The selector is informative, not cosmetic.
* **Files changed:** `dashboard/figures.py` (`fig_dependence`
  signature), `dashboard/app.py` (layout + 1 new callback + 1 modified
  callback), `schema/ui_strings.yaml` (1 new key).
