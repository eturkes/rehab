# AGENT_NOTES.md — sticky knowledge for future sessions

Optimized for LLM ingestion: short bullets, no prose padding.

**Inclusion rule (CLAUDE.md policy):** every entry must provide value *beyond*
what `CLAUDE.md`, the codebase, `models/training_metrics.json`, and Git history
already capture.  Durable invariants, design contracts, anti-mistake lessons,
and non-obvious gotchas belong here.  Metric tables, package versions, exact
line counts, and verbatim file-change lists do **not** — they live in the
artifacts above and only invite drift.  Prune freely when an entry is
superseded, duplicated elsewhere, or has gone stale.

---

## 0. Read-first

* `CLAUDE.md` — user policy file.  Authoritative.  Edit freely when content
  becomes obsolete or improvable.
* `SESSION_PROMPT.md` — reusable bootstrap prompt the user pastes to start a
  session.  Update if the workflow changes.
* `README.md` — human-facing entry point.  Keep in sync with code.
* `MAP.md` — **generated** symbol index (per file: purpose, line count, every
  top-level symbol with its 1-indexed line number; Dash callbacks tagged).  Read
  it to locate code, then `Read(path, offset, limit)` only the slice — this
  replaces whole-file reads.  Regenerate after structural edits:
  `uv run python scripts/gen_map.py` (idempotent + timestamp-free, so an
  unchanged tree yields no diff).
* `./compaction.sh` — manual context gauge (`pct used/window`); run via Bash to
  read own usage from the transcript.  The user's global statusline
  (`$HOME/.claude/compaction.sh`, byte-identical) already covers this repo, so
  `.claude/settings.json` stays empty (resist re-adding `statusLine`); keep the
  two copies in sync.  Dual-mode keyed on `CLAUDE_CODE_SESSION_ID` (set ⇒
  manual/transcript; unset ⇒ statusline/stdin) — any edit must keep both.  Per
  CLAUDE.md, wrap to a clean boundary at ≥80 % for a manual `/compact`.
* This file — agent-facing scratchpad.  Read before planning; update after each
  session; prune duplication per the inclusion rule above.
* **Default-work pool: §8 backlog.**  F1–F25 shipped + G1 landmark (dynamic)
  prediction fully shipped (training + Methods curve s29; interactive simulator +
  patient surfaces s30; see §7).  Open: G2 value-of-information · G3
  observed-trajectory phenotyping · G4 AIS conversion · F26 test harness · F27
  dep refresh.  The user steers toward *insightful* (clinical/scientific)
  features over infra/maintenance — propose those (G-series) first unless
  redirected.

## 0b. Lessons & mistakes (append; prune when superseded)

* **Stats procedures: validate against a reference before committing.**  Holm
  step-down was first implemented with a running `min` instead of running
  `max` over sorted p×(n−k+1), yielding nonsensical ~10⁻⁵⁵ adjusted-p.  The
  direction of such operators is easy to flip.
* **Plotly fails silently — check the data contract first.**  `Sunburst(
  branchvalues="total")` with zero-valued parents renders blank with no error
  or log.  When a chart renders empty, suspect the parent/child value contract
  (or `branchvalues`) before assuming a data bug.  See §5.
* **Verify a referenced symbol exists before documenting it (grep first).**
  Early notes cited an `_apply_missing_sentinels` helper that was never
  written; sentinels are actually handled by two other mechanisms (see §1).
* **Test UI components at minimum viewport width.**  Concatenated full Japanese
  demographic strings wrapped to multiple lines in the Dash Dropdown and
  blended adjacent options.  Dropdown `optionHeight` must match the actual
  rendered row height.
* **Investigate unexpected NaN / row-count anomalies immediately.**  301
  placeholder "ghost" episodes inflated cohort counts for ~5 sessions before
  investigation, though visible from session 1.  (Filter now in loader, §1.)
* **uv resolves unconstrained transitive deps to their minimum-compatible
  version.**  shap declares `numba` unconstrained for non-macOS; `uv lock`
  (even `--resolution highest`) pulled 2021-era numba/llvmlite with no Py3.13
  support.  Fix: add explicit lower bounds in `pyproject.toml` and verify the
  resolved versions in `uv.lock` after any such upgrade.
* **`shap_interaction_values()` has a stricter input contract than
  `shap_values()`** — it rejects category-dtype columns.  Encode categoricals
  to integer codes (`cat.codes.astype(float)`) before calling.  TreeSHAP's
  `shap_values()` handles LightGBM categoricals internally; the interaction
  variant does not.
* **Relocating the project root breaks `.venv` wholesale — `uv run` included.**
  A moved checkout leaves stale absolute paths in every `.venv/bin` console
  shebang, in `activate`'s `VIRTUAL_ENV=`, and in the editable `rehab_sci`
  install (its import finder pins the old `src/`), so even
  `uv run --no-sync python -c "import rehab_sci"` fails — not just console
  scripts.  Fix is `rm -rf .venv && uv sync`; the warm uv cache makes it ~1 min
  and offline.  `uv.lock` and the tracked tree hold no absolute paths, so
  nothing in the repo needs editing — the fix is purely venv recreation.
* **`grep` in this container is `ugrep`, not GNU grep** (`grep --version` to
  confirm).  BRE alternation `\|` is matched *literally*, so `grep "a\|b"`
  silently finds nothing — use ERE `grep -E "a|b"`.  Re-check surprising empty
  sweeps before trusting them.
* **Split a megafile by carving line-ranges + an AST-equivalence assertion, not
  by retyping bodies.**  A one-shot script slices each top-level symbol by its
  def-to-def line range and asserts every original `FunctionDef`/const survives
  byte-identical (`ast.dump`) before deleting the monolith — token-cheap and
  transcription-safe.  Give each new submodule the full import header, then
  `ruff check --fix` prunes the unused ones (F401); `ruff check --select F`
  (F401 + F821 undefined-name) is the gate that proves no symbol was misplaced.
* **Confirm dead code before deleting via a repo-wide grep of every call form.**
  `figures.kpi_card` was a stale dict-returning dup of `layout.kpi_card` (the
  `html.Div` version every caller imports); nothing referenced the figures one.
* **`ruff` ambiguous-unicode rules (RUF001/002/003) fire on intentional text —
  ignore them in config, never "fix" the strings.**  The flagged glyphs are
  deliberate: full-width Japanese punctuation in bilingual UI strings and
  scientific typography (en-dash ranges `0–100`, `×` for interactions, `σ`,
  thin spaces).  Rewriting them to ASCII look-alikes corrupts the UI/maths.
  They sit in `[tool.ruff.lint] ignore` beside `E501`/`B008`.
* **LightGBM training is byte-reproducible here; k-means archetypes are not.**
  A full retrain reproduces every LightGBM metric in `training_metrics.json`
  exactly (fixed `random_state` suffices — no `deterministic=`/`num_threads=1`
  needed), so a diff in the `outcomes` block is a *real* change worth
  investigating.  The lone retrain churn is the `archetypes.centroids` array:
  ~1e-15 last-bit noise from thread-order in numpy/sklearn BLAS.  `git checkout
  models/training_metrics.json` drops that meaningless diff.
* **Two writers touch `training_metrics.json`; run the full pipeline to restore
  its committed form.**  `train.py` writes it first, then `archetypes.py`
  rewrites it last (adding the `archetypes` key).  All metric writers now use
  `ensure_ascii=False` (raw UTF-8 Japanese); a bare `train.py` run alone leaves
  the file without the `archetypes` section, so run `train` → `archetypes` (the
  full sequence in §6) before trusting or diffing it.
* **Data-quality validation must read the RAW frame (or a raw→clean diff), not
  the cleaned frame — the loader is defensive and hides anomalies.**  Out-of-
  range numeric/ordinal values are coerced to NaN (re-parse raw vs the schema
  `range` to recover them); unmapped categorical values are *kept as the raw
  token* — `schema.normalize_level` falls back to the stripped raw string and
  never returns NA — so detect those by testing the cleaned value against the
  level set's `display` set, NOT by looking for NaN.  Pattern lives in
  `data/quality.py`.
* **The admission-feature count is 30, not 32 — docs had drifted; cite "the
  admission features" without a number.**  `af.feature_cols` (== `len(
  ADMISSION_FEATURES)`) is **30**: 2 demographics + 9 injury/admin + 15
  ISNCSCI/AIS + 4 SCIM-ADL.  Stale "32" copies were corrected in `report.py`
  (PDF methods text), `train.py` (trajectory docstring), and §3 here.  When a
  count is load-bearing, derive it from `len(af.feature_cols)` at runtime
  rather than re-typing a literal that silently drifts.

## 1. Data invariants (do not rediscover)

* **Raw file** — `ALL_SCIDATA.csv` at repo root.  Never commit; gitignored.
* **Encoding** — `cp932` (Shift-JIS superset).  UTF-8 silently mangles half the
  column names.
* **Missing sentinels** in raw file: `""`, `"_"`, `"NA"` → NaN via
  `pd.read_csv(na_values=...)` in `loader.py::load_raw`.  `"NT"` / `"ND"` are
  *not* listed there but become NaN as a side effect of
  `pd.to_numeric(errors="coerce")` (numeric/ordinal cols) and
  `schema.normalize_level()` returning `pd.NA` for unknown categorical levels.
  (No `_apply_missing_sentinels` helper exists — that was a phantom; the two
  mechanisms just described produce the effect.)
* **Feature missingness is high and preserved into the model.**
  `train.py::_prep` drops only rows missing the *target* or `IDNumber`; feature
  NaNs flow into LightGBM unchanged (native missing handling).  Admission
  missingness is the cohort norm — e.g. 糖尿病 ~79 %, Frankel ~74 %,
  損傷部位/保険 ~73 %, ALLEN ~66 %, motor/sensory levels ~30 %, SCIM ~22 %,
  AIS ~12 %.  So passing NaN at inference (F25 partial input) matches training,
  and the conformal calibration set was itself built under this missingness —
  the 80 % PI stays approximately valid for partial input.  The conformal
  half-width `q` is a fixed calibrated scalar (Mondrian-by-AIS → marginal when
  `AIS_ord` is blank), so it does **not** widen as input grows sparser; the
  simulator's reliability badge is the separate completeness/OOD cue.
* **Excel booleans** — many bool-like columns arrive as the literal strings
  `"FALSE"` / `"TRUE"` (uppercase).  Coerced to `0`/`1` then to `Y`/`N` via the
  schema level mapping.
* **mFrankel/Frankel** — single combined raw column; split on slash into
  `mFrankel_ord` (5-grade A–E with substages) and `Frankel_ord` (5-grade).
* **Raw shape** — long format: 31 200 rows × 219 cols, 1 200 `KeyRecordNumber`s
  × 26 timepoint slots (`0day`, `72h`, `2w`, `4w`, `6w`, `2m..11m`, `1y..10y`,
  `discharge`).  The grid is perfectly rectangular — every episode has a row at
  every slot.
* **Ghost-episode filter** — 301 of the 1 200 raw episodes are pure placeholder
  rows: `IDNumber` null AND every admission feature null AND every outcome
  null.  `build_analysis_dataset()` drops them via
  `_identify_ghost_episodes(ep, ADMISSION_FEATURES)`.  **Post-filter universe:
  899 episodes / 866 unique patients; long frame = 23 374 rows (899 × 26).**
* **Partial-id orphans (27 episodes)** — have admission features but null
  `IDNumber`.  Survive the ghost filter (they have data) but are excluded from
  training by `dropna(subset=["IDNumber", outcome])` in `_prep()` and from the
  patient picker by `list_patient_options(ep)`.  They still feed cohort
  aggregates.  (Of the 27: 9 have discharge SCIM, 10 discharge AIS, 14
  `LOS_days`.)
* **IDNumber 1-off (raw 867 → clean 866)** — `KeyRecord 446` has literal
  `'6641/10/15'` (a malformed date) in the ID field; schema declares
  `IDNumber: numeric`, so `to_numeric(errors="coerce")` correctly NaN's it.
* **Outcome cardinality (899-episode universe)** — `y_discharge_scim`: 507;
  `y_discharge_ais`: 638; `y_discharge_wisci`: **50 — too sparse to model,
  stays dropped**; `LOS_days`: 682.
* **Subscale outcomes share SCIM-total's universe** — `y_discharge_scim_{
  self_care,resp_sphincter,mobility}` come from the same `discharge` rows as
  `y_discharge_scim` (n=507; training n=498 after dropping IDNumber-null
  orphans).  Effective ranges: self-care 0–20, resp/sphincter 0–40,
  mobility 0–40, total 0–100.
* **AIS class imbalance** — D dominates (~59 %); B and E are ~5 % each.  Use
  LightGBM `multiclass` with `class_weight="balanced"`, else B/E are never
  predicted.
* **LOS distribution** — heavy right tail (median ≈140 d, max ≈790 d).
  Modelled on `log1p` scale; conformal q computed in log-space and
  back-transformed.
* **`LOS_days` is right-censored by recency** — `入院期間` is recorded only at
  discharge, so recent business years are incomplete: complete through 2022,
  ~28/78 in 2023, and **zero for `BusinessYear` ≥ 2024** (whereas SCIM/AIS
  discharge outcomes extend to 2025).  Any time-stratified LOS analysis must
  expect missing recent years; F24's temporal backtest therefore covers only
  2020–2023 origins for LOS.
* **`BusinessYear` is an episode-frame passport, never a feature** — added to
  `build_episode_frame` as first-non-null per episode (100 % populated,
  2014–2025).  Absent from `ADMISSION_FEATURES`, so it never enters
  `feature_cols`; it exists only as the temporal-split key for F24.
* **`mFrankel_Frankel` is a packed pair column** — missing marker `_/_`, and
  valid values include mismatched pairs (`C1/C2` = mFrankel C1 / Frankel C2).
  Validate it through the split `mFrankel_ord` / `Frankel_ord`, never the pair
  string (which matches no `mfrankel_pair` display).
* **`ALLEN分類` raw values are messy** — full-width Roman-numeral stages
  (`DE-Ⅱ` vs the schema's ASCII `DE-2`), out-of-enum stages (`VC-3`), and
  misplaced bony/level tokens — so ~442 rows fail to normalize via the `allen`
  level set.  It is a model feature, so this is a real coverage gap (add
  Roman-numeral `raw_aliases`, or treat the column as noisy).
* **PinPrick sensory cells carry asterisk-annotated scores (`1*`, `0*`)** — the
  ISNCSCI "non-SCI cause" convention.  `to_numeric` drops them to NaN, so the
  annotation is silently lost on load.

## 2. Schema (`schema/*.yaml`) — source of truth

* Every column the dashboard renders must have a `columns.yaml` entry, else the
  raw Japanese leaks into the UI.
* Every categorical raw value should resolve through `categorical_levels.yaml`
  via the canonical `display` or a `raw_aliases` entry.
* UI strings live in `ui_strings.yaml` only.  No inline literals in dashboard
  code — use `t(schema, "key", lang)`.
* `columns.yaml` uses `families:` to template the ISNCSCI dermatomes (light
  touch, pin prick, key / non-key muscles), expanded by `schema.py` at load
  time — not pasted literally.  Extend the family block to add a dermatome
  family.

## 3. Model conventions (design contracts; metrics live in `training_metrics.json`)

* **Random state:** `20260518` (also embedded in `training_metrics.json`).
* **Group split** by `IDNumber` (patient), never by row — prevents same-patient
  leakage.
* **Outcome registry** — `models/outcomes.py::OUTCOMES` is the ordered tuple of
  `OutcomeSpec` records (6 outcomes).  `train.py` iterates it; the dashboard
  imports the same list so simulator/Methods stay in lockstep.  To add an
  outcome: extend `OUTCOMES`, ensure its target column is on the episode frame,
  add a `ui_strings.yaml` `outcome_{key}` entry.
* **Per-outcome artifact layout** — `models/{spec.key}/` holds
  `lgbm_median/p10/p90.joblib` (regression) or `lgbm_multiclass.joblib` (AIS),
  plus `feature_spec.joblib` (with `conformal_q_*`, `transform`, `clip_min/max`)
  and `shap_test.joblib`.  Top-level `models/feature_spec.joblib` is the
  *shared* feature universe (feature_cols, ranges, categories) only.
  `training_metrics.json` = `{"outcomes": {...}, "outcome_keys": [...]}`.
  (Any top-level `lgbm_*`/`shap_*.joblib` are stale single-outcome-era debris —
  not loaded by any code.)
* **Conformal PI (regression)** — (1−α)-quantile of `|y−ŷ|` on a held-out
  calibration fold, computed on the *transformed* scale (identity for SCIM/AIS,
  log1p for LOS), back-transformed, then clipped to `[clip_min, clip_max]`.
  **Required — LightGBM quantile heads alone give ~0.41 coverage on SCIM total;
  do not remove the conformal layer.**  At inference the PI is the *union* of
  the conformal interval and the raw quantile interval
  (`lo=min(lo_conf,lo_q10)`, `hi=max(hi_conf,hi_q90)`) — user sees the more
  conservative bound.
* **Mondrian conformal** — per-AIS-grade and per-paralysis-class q replace the
  single marginal, stored in `feature_spec.joblib["conformal_q_by_group"]` =
  `{"ais": {...}, "paralysis": {...}, "marginal": q, "min_n": 8}`.  Groups with
  < `MONDRIAN_MIN_N=8` calibration samples are omitted; inference falls back
  AIS → paralysis → marginal.  Qualitatively AIS-C (motor-incomplete) gets the
  widest PI, AIS-D the tightest for SCIM — clinically correct, previously
  hidden by the marginal.
* **AIS multiclass head** — classes encoded by severity (A=0 … E=4 — the model's
  0-based class index; note the `AIS_ord` *data* column is 1–5 via
  `constants.AIS_LETTER_TO_ORD`), so
  `predict_proba` columns and the SHAP last axis are ordinally sorted.  Metrics:
  accuracy, quadratic-weighted κ, MAE-on-ordinal-code.
* **APS conformal classification sets (AIS)** — calibration fold carved from
  dev; APS nonconformity = cumulative prob mass to include the true class;
  threshold at ⌈(n+1)·0.8⌉ quantile → `q_hat`.  Mondrian per-AIS/per-paralysis
  variants in `feature_spec.joblib` under `aps_q_hat` + `aps_q_by_group`.  At
  inference, sort class probs descending, accumulate until cumsum ≥ resolved
  q_hat.  Coverage runs **conservative (~99 % vs 80 % target) because K=5 makes
  APS scores discrete — expected, not a bug** (avg set size ≈2.8).
* **LOS (log1p) head** — same regression machinery; `y` log1p'd before fit,
  conformal q in log-space; predictions/PI/quantiles back-transformed via
  `expm1` and clipped to `[0, ∞)`.  Metrics reported in raw days.
* **TreeSHAP** runs on the held-out **test set only** (training-set SHAP would
  be optimistic).  Cached in `shap_test.joblib`.  Multiclass cache is a 3-D
  `(n, p, K=5)` tensor, AIS axis last; the insight engine slices it by
  user-selected class via `class_idx`.
* **SHAP interaction values** — `shap_test.joblib` also persists
  `shap_interaction`: regression `(n, p, p)`, multiclass `(n, p, p, K)`.
  Diagonal `[i,i]` = main effect, off-diagonal `[i,j]` = symmetric pairwise
  interaction.  Computed with category cols pre-encoded to int codes
  (`_encode_cats_for_shap`; see §0b).  `training_metrics.json` stores
  `global_interaction_top25` per outcome.  Dominant interaction across
  functional outcomes is age × motor score.
* **Test-set predictions** — `shap_test.joblib` also persists `y_test`,
  `y_pred` (and `y_pred_proba` for multiclass), powering the Methods tab's
  calibration visuals.
* **Holm correction** — running **max** over sorted p×(n−k+1), not running min
  (see §0b).
* **Trajectory models** — 9 independent LightGBM regressions predicting
  SCIM-total at intermediate timepoints (72h…6m) from the same admission
  features.  `models/trajectory/bundle.joblib` = dict (`timepoints`, `models`,
  `conformal`, `clip_min`, `clip_max`); each timepoint has median/p10/p90 +
  Mondrian q.  No SHAP (redundant with per-outcome SHAP).  Admission features
  are most predictive of ~1-month outcomes, least of ~5-month; conformal q
  widens monotonically with horizon.  Clinical value is the trajectory *shape*
  (when recovery plateaus), not per-timepoint point accuracy.
* **Recovery archetypes** — k-means (k=3, chosen by silhouette) on predicted
  10-point trajectories (9 trajectory timepoints + discharge), z-scored per
  timepoint.  Ordered by discharge SCIM: 0 limited / 1 gradual / 2 rapid.
  `data/archetypes.py::assign_single()` assigns new/hypothetical patients at
  runtime.  Persisted in `models/archetypes/archetypes.joblib`.  Primary
  separators are AIS grade and age — consistent with the age × motor SHAP
  interaction.
* **Temporal validation (`models/temporal.py`, F24)** — out-of-time
  rolling-origin backtest, **diagnostic only**: it writes its own
  `models/temporal_metrics.json` and never touches `train.py`'s artifacts, so
  byte-repro is preserved.  Expanding window — test years 2020–2025, train =
  `BusinessYear < T`, group-safe by patient (test episodes whose patient is in
  the past are dropped + counted as `n_dropped_overlap`).  Reuses `train.py`'s
  prep/fit/transform helpers + `conformal.py`'s `_conformal_q`/`_aps_*` so the
  methodology matches production, with two deltas: the dev/test cut is temporal,
  and conformal/APS q is **marginal** (per-origin per-group folds are too small
  for Mondrian).  The reported coverage is conformal-only (comparable to
  `test.conformal_coverage_80`), not the union PI used at inference.  Baselines
  are echoed from `training_metrics.json` (CV point + holdout coverage).
  Finding: SCIM heads lose only ΔR²≈−0.04…−0.08 out-of-time with coverage
  ≈nominal; AIS robust (APS stays conservative ~0.99); LOS is hard both in- and
  out-of-time (R²≈0.2) and censored (§1).
* **Landmark (dynamic) prediction (`models/landmark.py`, G1)** — diagnostic +
  inference layer built like `temporal.py`: writes its own tracked
  `models/landmark_metrics.json` + a git-ignored `models/landmark/bundle.joblib`
  and never touches `train.py`'s artifacts (byte-repro preserved).  Per
  landmark L ∈ {72h, 2w, 4w, 6w, 2m, 3m}, refits the **discharge** outcome on
  admission features **plus a LOCF observation block** — the 10 early-recovery
  measures in `LANDMARK_COLS` (SCIM total + 3 subscales, AIS_ord, UEMS, LEMS,
  TotalMotor, LightTouch, PinPrick), each carried forward as its last non-null
  value at an intermediate timepoint ≤ L, prefixed `L_`.  **Eligibility = the
  still-admitted risk set**: an episode enters L only if it has a tracked
  observation at an intermediate timepoint with order-index ≥ L (the
  `discharge` slot is excluded from `TIMEPOINT_ORDER`), which avoids
  immortal-time/leakage.  Each landmark model is paired with an **admission-only
  baseline refit on the identical eligible cohort + split** (split computed once
  on `X_base`, reused for `X_lm`) so the reported delta isolates the *value of
  observation* and is not confounded by the shrinking risk set.  Conformal/APS
  is **marginal** (per-landmark folds too small for Mondrian); regression
  reports R²/RMSE/MAE + `pi_halfwidth_raw` (mean conformal half-width), AIS
  reports κ_quadratic + `aps_avg_set_size`.  `MIN_COHORT=120` skips landmarks
  whose eligible n is too small.  Findings: SCIM-total ΔR² climbs ≈+0.04 (72h)
  → +0.30 (3m) with PI half-width roughly halving (≈23→13 pts); AIS κ rises
  toward ≈0.88; LOS stays hard but improves (R²≈0.15→0.37).  Bundle shape and
  metrics keys are documented inline at the top of `landmark.py`.  For the
  interactive surfaces the bundle persists **both heads per (outcome, L) cell** —
  the admission-only baseline and the landmark head — under
  `outcomes[key]["by_landmark"][L] = {"baseline", "landmark"}` (regression head
  `{median, conformal_q, feature_cols}`, multiclass `{clf, aps_q_hat,
  class_codes, class_labels, feature_cols}`; baseline `feature_cols`=30, landmark
  =40), plus `timepoint_order` so the dashboard reconstructs real-patient LOCF
  without importing the training module.  Persisting both heads keeps the live
  baseline-vs-landmark comparison method-matched (same cohort/split/conformal);
  p10/p90 quantile heads are **not** persisted (the live PI is conformal-only).

## 4. Dashboard conventions

* **Module layout** (`src/rehab_sci/dashboard/`) — `MAP.md` is the authoritative
  per-file symbol inventory; only the non-obvious contracts live here:
  - `state.py` holds all startup globals, loaded once; depends only on theme +
    data/model layers (no other dashboard module imports it transitively).
  - `compute.py` is pure (model inference, conformal-q, APS, SHAP, row-prep) —
    **no Dash/Plotly**.  Landmark inference lives here too: `predict_landmark`
    (paired baseline+landmark heads from `LANDMARK_BUNDLE`),
    `landmark_observed_for_episode` (real LOCF block on the still-admitted risk
    set), `episode_landmark_eligibility` (per-L still-admitted gate).
  - `reliability.py` is pure (no Dash) — `assess_input(X, bundle, feature_spec)`
    returns importance-weighted completeness + OOD (range violations /
    atypicality from `feature_spec['ranges']` q05–q95) for the simulator's
    partial-input badge; gain importances cached per outcome key.
  - `layout.py` owns the shared *prediction* figures (`fig_shap_local`,
    `fig_prediction_interval`, `fig_class_probabilities`) plus the landmark
    baseline-vs-observation comparison (`fig_landmark_compare` + the
    `landmark_readout` text block, shared by the simulator and patient dynamic
    cards) — these are **not** in the figures package.
  - `figures/` is a **package** split by tab (`overview`, `insights`, `patient`,
    `methods`, `simulator`) plus `_common` (`_hex_to_rgba`).  Every public name +
    `ARCHETYPE_NAMES_*` / `PALETTE_ARCHETYPE` is re-exported via
    `figures/__init__.py`'s `__all__`, so `from rehab_sci.dashboard import
    figures as fg` and every `fg.fig_*` call are unchanged.  To add a figure:
    put it in its tab's submodule and append the name to `__all__`.
  - `tabs/*` = per-tab layout + `@callback`s; `app.py` = entry + chrome callbacks.
* **Dependency graph (acyclic):** `state` → data/model; `compute` → `state`;
  `layout` → `state`, `compute`; `tabs/*` → `state`, `compute`, `layout`,
  `figures`; `app` → `tabs/*` (imports trigger `@callback` registration —
  Dash function-level `@callback` registers globally, no `@app.callback`).
* `dcc.Store("lang-store")` holds `"ja"`/`"en"`.  Most text callbacks take it
  as `Input` for instant swaps.  **Exception:** `update_overview_content` takes
  it as `State` to avoid a race with `update_tab` (both fire on lang change;
  `State` ensures overview fires *after* `update_tab` rebuilds the filter
  components with new lang labels).
* Pattern-matched simulator inputs use IDs `{"type":"num"/"cat","col":<raw>}`
  with `dash.ALL`.  Input order is fixed by `feature_spec['feature_cols']`.
  Numeric fields are **clearable `dcc.Input(number)`** (not sliders) and the
  form opens **blank** (F25) — `collect_sim_inputs` leaves blanks as NaN (no
  imputation); *Fill cohort defaults* / *Clear all* set every value via a single
  pattern-matching value-Output callback; What-if mode prefills from the
  reference episode.  (`simulate` returns 5 outputs incl. `sim-reliability`.)
* Plotly template name: `"medical"` (registered in `theme.py`).  Palettes:
  `PALETTE_CATEGORICAL`, `PALETTE_AIS` (A→E cool→warm), `PALETTE_PARA`,
  `PALETTE_ARCHETYPE`.  Use them — do not hand-pick per-chart colors.
* Japanese rendering needs the font stack `"Hiragino Sans","Noto Sans
  JP","Yu Gothic UI"` in both Plotly and CSS.
* `dcc.Store("patient-ref")` (session-scoped) carries the What-if
  counterfactual reference `{id_number, key_record, features, outcomes,
  trajectory}`.  `update_tab()` reads it as State to pre-fill simulator
  defaults; `simulate()` reads it as State for the reference overlay.

## 5. Known gotchas

* **`IntCastingNaNError` on `IDNumber`** — patients with no admission row
  produce NaN IDs.  `dropna(subset=[outcome,"IDNumber"])` then cast
  `float64 → int64`.
* **Stale dashboard process** — `kill <PID>` stops only the `uv run` wrapper;
  the Python child keeps serving old code.  Root cause of the exit-144 (signal
  16) report: `pkill -f 'rehab_sci.dashboard.app'` **self-matches the calling
  shell** (the pattern string is in its own argv), so the shell is signalled
  too.  Two consequences you must design around: (1) it still kills the
  dashboard, so it is fine as the *last* statement of a cleanup-only command —
  but (2) **never put it before anything you need to run after** (e.g. a launch),
  or the shell dies at signal 16 before reaching it (empty logs, nothing
  listening — looks like a boot failure but is not).  Prefer: launch with the
  Bash tool's background mode (no pkill), poll readiness in a *separate* command
  that contains no matching literal (curl only), and stop by port —
  `fuser -k 8050/tcp` — which never self-matches.  Verified boot = `curl -s -o
  /dev/null -w '%{http_code}' http://127.0.0.1:8050/` → 200 plus
  `/_dash-dependencies` for the callback graph.
* **pandas fragmentation warning** when adding many columns serially — batch via
  `pd.concat([df, new_cols_df], axis=1)`.  Loader already does this.
* **`@dataclass(frozen=True)` + dict fields** breaks under `@lru_cache` on
  instance methods (unhashable).  `Schema` is a plain `__slots__` class for this
  reason — do not "modernize" it.
* **`kaleido<1`** has no Linux x86_64 wheel under the current resolver.  Keep
  `kaleido>=1.0,<2`.  Plotly→PNG export needs Chrome (`kaleido.get_chrome_sync()`
  at first use).
* **Packaging** — project is a real hatchling uv package
  (`[tool.hatch.build.targets.wheel] packages=["src/rehab_sci"]`); `uv sync`
  installs it editable, so `uv run python -m rehab_sci.*` works with no
  `PYTHONPATH` prefix.  (Historical: an earlier `[tool.uv] package=false` forced
  `PYTHONPATH=src` on every launch.)
* **Background dashboard from a bash one-liner** — `nohup … &` inside the
  harness wrapper may not survive wrapper exit (exit 144 = SIGTERM
  bookkeeping).  Use the Bash tool's `run_in_background: true`, or make the
  launch the *last* statement so it inherits the wrapper lifetime.
* **Plotly `Sunburst(branchvalues="total")` requires parent value = sum of
  children.**  Parent value `0` renders blank silently.  Prefer accumulating
  leaf counts into every ancestor (so hover shows true subtotals) over
  `branchvalues="remainder"`.  (The injury chart later moved to a treemap.)

## 6. Commands cheat sheet

```bash
uv sync                                          # install deps
uv run python scripts/01_profile_raw.py          # refresh schema profile
uv run python -m rehab_sci.models.train          # train + conformal + SHAP
uv run python -m rehab_sci.models.subgroups      # subgroup discovery
uv run python -m rehab_sci.models.archetypes     # recovery archetype clustering
uv run python -m rehab_sci.data.quality          # data-quality / clinical-consistency report
uv run python -m rehab_sci.models.temporal       # out-of-time temporal validation (F24)
uv run python -m rehab_sci.models.landmark       # landmark (dynamic) prediction — value of observation (G1)
uv run python -m rehab_sci.dashboard.app         # serve at :8050
pkill -f 'rehab_sci.dashboard.app'               # stop stale dashboard
uv cache prune                                   # reclaim uv cache space
uv run pip-audit                                 # dependency vuln scan (dev dep)
uv run python scripts/gen_map.py                 # refresh MAP.md code index (idempotent)
uv run ruff check src/ scripts/                  # lint (file:line); --select F = regression gate; --fix = safe autofix
./compaction.sh                                  # context-usage gauge (manual; statusline is global)
```

Persistent REPL — load data/models once, query across separate Bash tool calls:

```bash
export BGCMDDIR=/tmp/bgpy BGCMDPROMPT='>>> '
bgcmd START .venv/bin/python -i -q               # filesystem-backed; survives across Bash calls
bgcmd 'from rehab_sci.dashboard import state as S, compute as C; import rehab_sci.dashboard.figures as fg'
bgcmd 'S.EP.shape'                               # single-line sends; reuses the loaded objects
bgcmd 'exit()'; rm -rf "$BGCMDDIR"               # stop + clean
```

## 7. Session index (most recent first)

One line per session; full detail is in Git history (`git log`, diffs).

* **s30** — G1 landmark (dynamic) prediction, part 2 (interactive surfaces; user
  chose "both surfaces").  Re-persisted `landmark/bundle.joblib` with **paired
  baseline+landmark heads** per (outcome, L) cell (dropped unused p10/p90; added
  `timepoint_order`); `landmark_metrics.json` byte-identical, production
  artifacts untouched.  Pure inference helpers in `compute.py`
  (`predict_landmark`, `landmark_observed_for_episode`,
  `episode_landmark_eligibility`); shared `fig_landmark_compare` +
  `landmark_readout` in `layout.py`.  **Simulator** gains a hypothetical card
  (landmark dropdown + 10 observed-score inputs); **patient** gains a real-data
  card (eligibility-gated dropdown → the patient's own LOCF observations sharpen
  the admission-only prognosis).  Bilingual (reuses `lm_*` keys); `.lm-*` CSS.
  Lint clean; dashboard boots 200; both surfaces smoke-tested (SCIM PI half-width
  ±26→±14 at 3m for ep 1).
* **s29** — G1 landmark (dynamic) prediction, part 1 (training + Methods curve).
  New `models/landmark.py`: per landmark L∈{72h,2w,4w,6w,2m,3m}, refits each
  discharge outcome on admission features + a LOCF block of 10 early-recovery
  measures (`L_`-prefixed), conditioned on the still-admitted risk set, each
  paired with an admission-only baseline on the identical cohort+split to
  isolate the *value of observation*; marginal conformal/APS.  Writes tracked
  `landmark_metrics.json` + git-ignored `landmark/bundle.joblib`; production
  artifacts untouched (byte-repro verified).  Bilingual Methods-tab
  value-of-observation curve (`fig_landmark_value`, dual-axis: point accuracy +
  PI-width/APS-set-size vs landmark, baseline dashed).  Findings: SCIM ΔR²
  +0.04→+0.30 (72h→3m), PI half-width ~halves; AIS κ→≈0.88; LOS improves
  (R²≈0.15→0.37).  Also corrected the long-standing "32 admission features"
  doc-drift to the actual 30 (see §0b).  Lint clean; dashboard boots 200.
  G1-pt2 (interactive simulator + patient surfaces) shipped in s30.
* **s28** — F25 partial-input prediction.  Simulator opens **blank**; numeric
  sliders → clearable `dcc.Input(number)`; blanks pass through as NaN
  (`collect_sim_inputs` no longer imputes `SIM_DEFAULTS`).  New pure
  `dashboard/reliability.py` (`assess_input`) → importance-weighted completeness
  + OOD (range/atypicality from `feature_spec` ranges); rendered as a reliability
  badge (`sim-reliability`, 5th `simulate` output) with a conformal-missingness
  caveat.  *Fill cohort defaults* / *Clear all* buttons.  No retrain, no new
  artifact, production model untouched.  Files: `reliability.py`, `compute.py`,
  `layout.py`, `tabs/simulator.py`, `ui_strings.yaml`, `assets/style.css`.
* **s27** — F24 temporal (out-of-time) validation.  New `models/temporal.py`:
  expanding-window rolling-origin backtest (test years 2020–2025; train =
  earlier years; group-safe by patient) for all 6 outcomes; marginal
  conformal/APS coverage measured out-of-time; writes tracked
  `models/temporal_metrics.json`.  `BusinessYear` added as an episode-frame
  passport (never a feature).  Bilingual Methods-tab drift curves
  (`fig_temporal_drift`, dual-axis: point accuracy + coverage vs test year,
  baseline + nominal-80 % reference lines).  Findings: SCIM generalizes
  temporally (small ΔR²; coverage ≈nominal), AIS robust, LOS hard +
  right-censored (no labels ≥2024 → 4 origins).  Lint clean; dashboard boots
  200; `training_metrics.json` + model joblibs untouched.
* **s26** — F23 data-quality / clinical-consistency report.  New
  `data/quality.py`: declarative rule engine (15 rules — domain / cross-field /
  longitudinal) over the raw + cleaned frame; writes a tracked aggregate
  `models/dataquality_summary.json` (counts only) + a git-ignored detailed
  `models/dataquality_report.json` (carries IDs + offending values).  Bilingual
  Methods-tab panel surfaces the aggregate.  Real issues surfaced: out-of-range
  SCIM items, the malformed `IDNumber`, asterisk PinPrick scores, the ALLEN
  Roman-numeral coverage gap, tetra↔thoracic-NLI and AIS-deterioration flags.
  Findings reproduced independently; lint clean; dashboard boots 200.
* **s25** — s24 optional cleanups, all three landed.  Ruff debt cleared (config
  ignores `E501`/`B008`/`RUF001-003`; ~85 safe autofixes + 22 manual; repo lints
  clean).  `AIS_ORD_TO_LETTER` deduped (was 4×) into new `constants.py` (bottom
  of the import graph — imports nothing from the project).  `train.py` (1108 ln)
  carved → core + new `models/conformal.py` (Mondrian/APS) + `models/shap_utils.py`
  (SHAP-interaction helpers) via the AST-equivalence method.  Also fixed
  `train.py`'s metrics write to `ensure_ascii=False` (was the lone escaped
  writer).  Verified: full lint clean, all modules import, `train`+`archetypes`
  retrain end-to-end (LightGBM metrics byte-identical), dashboard boots 200.
* **s24** — token-efficiency pass: `figures.py` (1573 ln) → `figures/` package
  (6 tab modules; public surface preserved; carved via one-shot script + an
  AST-equivalence assertion, then ruff-pruned imports).  Added
  `scripts/gen_map.py` → `MAP.md` code index (read-first navigation).  Verified
  + documented the ruff and `bgcmd` persistent-REPL feedback loops.  Trimmed
  §0/§4.  Dropped dead `figures.kpi_card` (stale dup of `layout.kpi_card`).
* **s23** — post-relocation repair: root moved `Documents/pro/rehab` →
  `Projects/rehab`; only `.venv` broke (stale abs paths in 60 console shebangs,
  `activate`, editable `rehab_sci`).  Fixed by `rm -rf .venv && uv sync` from
  the clean lockfile; verified state/app/train load (899 ep, 6 outcomes).
* **s22** — F22 overview cohort filtering (AIS/paralysis/age/archetype filter
  bar drives all overview KPIs + charts; `update_overview_content` callback).
* **s21** — F20 refactor: `dashboard/app.py` monolith → 9 files
  (state/compute/layout + 5 tab modules); acyclic deps; zero behavior change.
* **s20** — F18 recovery archetype clustering (k-means on predicted
  trajectories; 3 archetypes; overview curves + patient chip).
* **s19** — F16 patient similarity explorer (Gower-distance KNN over the 30
  admission features; neighbor outcome strip / AIS-distribution charts).
* **s18** — F13 SHAP interaction explorer (interaction values at train time;
  heatmap + 2-feature dependence in insight engine).
* **s17** — F14 dependency audit + security update (pyarrow CVE; pandas 3 /
  Dash 4 majors; transitive numba/llvmlite lower-bound pins; pip-audit added).
* **s16** — F10 PDF patient report (fpdf2 + IPAexGothic + kaleido; bilingual
  2-page report).
* **s15** — F9 What-if counterfactual explorer (patient → simulator pre-fill +
  reference overlays via `patient-ref` store).
* **s14** — F8 calibration & performance visuals (Methods tab: pred-vs-obs,
  residual hist, confusion matrix, reliability curve; persists y_test/y_pred).
* **s13** — F7 recovery trajectory forecasting (9 intermediate-timepoint models
  + Mondrian PIs; trajectory overlay on patient timeline + simulator).
* **s12** — F6 SHAP class selector for AIS multiclass dependence (`class_idx`).
* **s11** — F5 APS conformal classification sets for AIS (Mondrian q_hat).
* **s10** — F4 multi-outcome insight engine + patient explorer (outcome
  selector everywhere; subgroups for all 6 outcomes).
* **s9** — F3 Mondrian per-AIS / per-paralysis conformal.
* **s8** — CLAUDE.md update response (no feature).
* **s7** — F2 multi-outcome prediction (6 heads via `OUTCOMES` registry;
  per-outcome artifact dirs; simulator outcome dropdown).
* **s6** — loader sanity check + ghost-episode filter (1 200 → 899 episodes).
* **s5** — F1 patient explorer tab.
* **s4** — pivot to feature backlog as default-work pool (supersedes earlier
  "open items rolled forward" lists — treat those as history).
* **s3** — packaging refactor: real hatchling uv package (drops `PYTHONPATH=src`).
* **s2** — first browser QA pass; Plotly sunburst silent-failure fix.
* **s1** — initial build: loader (cp932, sentinels, mFrankel split), schema
  YAMLs, SCIM-total LightGBM + conformal + SHAP, Dash dashboard scaffold.

## 8. Feature backlog (default-work pool)

Propose from here unless the user redirects.  **Items F1–F25 + G1 (part 1) are
shipped** — see §7 for the session each landed in, and Git history for
implementation detail.  The user steers toward *insightful* (clinical/
scientific) features over infra/maintenance — lead with the **G-series**.
Shipped ledger (terse, by feature number):

* F1 patient explorer · F2 multi-outcome prediction · F3 Mondrian conformal ·
  F4 multi-outcome insight engine + explorer · F5 APS classification sets ·
  F6 SHAP class selector (AIS) · F7 recovery trajectory forecasting ·
  F8 calibration & performance visuals · F9 What-if counterfactual explorer ·
  F10 PDF patient report · F13 SHAP interaction explorer · F14 dependency
  audit · F16 patient similarity explorer · F18 recovery archetype clustering ·
  F20 app.py refactor · F22 overview cohort filtering · F23 data-quality /
  clinical-consistency report · F24 temporal (out-of-time) validation ·
  F25 partial-input prediction (clearable inputs + reliability/OOD badge).
* (F11/F12/F15/F17/F19/F21 were never opened — numbering gaps only.)
* **G1 landmark (dynamic) prediction** (s29 training + Methods curve; s30
  interactive simulator + patient surfaces): `models/landmark.py` +
  value-of-observation curve + paired-head bundle driving the live
  baseline-vs-observation cards.  See §3 for the contract and §7 for the
  sessions.  Fully shipped.

**F23 (shipped s26): data-quality / clinical-consistency report** — see §7 and
`data/quality.py`; durable data facts it surfaced live in §0b/§1, and the
regenerated `models/dataquality_summary.json` holds the per-rule scorecard.

**Ready candidates (pick the next unless redirected; lead with G-series):**
* **G2 value-of-information** — *what:* per patient, rank which *next*
  observation (which measure × which next landmark) most tightens the PI /
  reduces expected loss.  *why:* prescribes what to measure next, not just what
  observing helps on average.  *effort:* M–L.  *files:* landmark bundle + new
  compute helper + patient/Methods surface.  *data dep:* landmark bundle.
* **G3 observed-trajectory phenotyping** — *what:* cluster patients by their
  *observed* early-recovery trajectory (contrast §3 archetypes, which cluster
  *predicted* curves) and surface phenotype-conditioned prognosis.  *why:*
  data-driven recovery shapes independent of model bias.  *effort:* L.  *files:*
  new model module + overview/patient surface.  *data dep:* longitudinal frame.
* **G4 AIS-conversion modeling** — *what:* model AIS-grade *conversion* (Δ from
  admission to discharge, or across landmarks) as its own outcome, not just the
  absolute discharge grade.  *why:* conversion is the clinically salient
  endpoint.  *effort:* M.  *files:* new outcome/model + Methods.  *data dep:*
  `AIS_ord` at admission + discharge (present).
* **F26 invariant test harness** — narrow pytest enforcing §1 data + model
  invariants + a smoke test; skip-if-CSV-absent.  M.  files: `tests/`, pyproject.
* **F27 dependency refresh** — minor/patch bumps + raise the `shap<0.52` cap;
  retrain to verify byte-repro.  S, low value now (no CVEs, lint clean).

Propose new feature candidates or maintenance and add them here with **what /
why / effort / files / data dependency** before starting.
