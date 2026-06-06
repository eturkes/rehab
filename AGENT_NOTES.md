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
* **Default-work pool: §8 backlog.**  F1–F25 + G1 (s29/s30) + G2 (s31) + G3
  observed-trajectory phenotyping (s32/s33) + G4 AIS-grade conversion (s34/s35) + G6 AIS
  multi-state recovery (s36/s37) fully shipped (see §7).  **G7 functional-independence profile is
  MID-FLIGHT: Part 1 (model + metrics) shipped s38; Part 2 (dashboard surfaces) is the immediate
  next task** (see §8 G7 bullet + §3 contract).  Other open: F26 test harness · F27 dep refresh.
  The user steers toward *insightful* (clinical/scientific) features over infra/maintenance, so
  lead with those.

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
* **`class_weight="balanced"` trades probability calibration for recall — never surface its
  raw class probabilities as calibrated.**  G4's magnitude head (balanced, for the imbalanced
  improvement-size classes) emits *inflated* minority-class probs; for an event it shares with a
  calibrated, unweighted binary head the two disagree (adm A: binary P(≥C)=0.51 vs magnitude
  P(≥+2)=0.76).  Rule: for a near-balanced target omit weighting and Platt/isotonic-calibrate;
  for an imbalanced one that needs weighting, surface the *APS set / argmax class*, not the raw
  probability.  (Also: `shap.TreeExplainer(lgbm_binary).shap_values(X)` now returns a *list of
  ndarray* — take `[-1]` for the positive class; the 3-D `(n,p,2)` form takes `[:,:,-1]`.)
* **"Dashboard boots 200" does NOT exercise tab content — render each touched tab in both langs
  before claiming it works.**  An `INK["600"]` typo (no such key; INK has 900/700/500/300/…) in the
  Methods VOI paragraph (added s31) crashed the entire Methods-tab render for *four sessions* yet
  every "boots 200" check passed, because the tab bodies are built lazily by the `update_tab`
  callback, not at boot.  The cheap guard is a headless `render_methods('ja'/'en')` /
  `render_patient` / `render_simulator` call (plus invoking the new callbacks directly) — it caught
  this immediately.  Add such a render to any verification that touches a tab.
* **The admission-feature count is 30, not 32 — docs had drifted; cite "the
  admission features" without a number.**  `af.feature_cols` (== `len(
  ADMISSION_FEATURES)`) is **30**: 2 demographics + 9 injury/admin + 15
  ISNCSCI/AIS + 4 SCIM-ADL.  Stale "32" copies were corrected in `report.py`
  (PDF methods text), `train.py` (trajectory docstring), and §3 here.  When a
  count is load-bearing, derive it from `len(af.feature_cols)` at runtime
  rather than re-typing a literal that silently drifts.
* **A conformal *prediction set* over a BINARY head degenerates — do not surface
  it; use the calibrated probability + reliability curve instead.**  G7 first put
  an APS set over {dependent, independent}; coverage came out 100 % with ~90–100 %
  *abstain* (`{both}`) on every one of the 18 heads.  Root cause: the APS
  nonconformity score is the cumulative mass to reach the true class, so every
  misranked episode scores **exactly 1.0**; accuracy ~0.85 pins ~15 % of scores at
  1.0, dragging the 80 %-quantile q to ~0.98–1.0, which then forces `{both}`
  whenever `p_top < q` (i.e. almost always).  For K=2 the calibrated probability
  already *is* the honest uncertainty (a singleton `{argmax}` covers ≈accuracy ≥
  target anyway), so a set adds nothing.  Reserve APS/abstention sets for K≥3
  (AIS, magnitude).  Verified q_hat=0.98–1.00 before removing the layer.

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
* **Value-of-information (`models/landmark.py` + `compute.landmark_voi`, G2)** —
  extends G1: per (outcome, L) the trainer additionally fits the 10 **single-add
  heads** — admission features **plus exactly one** `L_<measure>` (31 features),
  each on the *same* eligible cohort/split as the paired baseline, each with its
  **own** marginal conformal q (regression) / APS q_hat (multiclass).  This makes
  "how much does measure m alone sharpen the prognosis at L" a literal,
  measure-specific quantity instead of an attribution.  Metrics land under
  `outcomes[key]["by_landmark"][L]["single"][measure]`; the fitted heads under
  the bundle's matching `["single"]` dict (same head shape as the landmark head,
  31 feats).  The bundle also persists `lm_value_summary[L][measure] = {median,
  q25, q75, n}` over the still-admitted eligible cohort, so the dashboard can
  impute an unobserved measure at its cohort median.  `compute.landmark_voi(
  outcome_key, landmark, X_base, observed)` ranks the measures for one patient:
  each is scored at the patient's **real** value when present (`which="observed"`,
  the gain is realized) else at the cohort median (`which="prescriptive"`, the
  gain is hypothetical — what obtaining it *would* buy); regression ranks by
  PI-halfwidth reduction vs baseline (`d_halfwidth`), multiclass by APS-set
  shrink (`d_setsize`).  **Crux invariant:** because conformal q is *marginal*
  per (outcome, L), within a landmark the PI half-width of a single-add head is a
  per-**measure** constant, *not* per-patient — for identity-transform outcomes
  (SCIM/AIS) `d_halfwidth` is patient-independent and only the point estimate
  personalizes; the lone exception is **log1p LOS**, whose back-transformed
  half-width is patient-specific (the q is constant in log-space but `expm1`
  scales it by the prediction).  So the Patient VOI bars rank by a population
  quantity for SCIM/AIS and a genuinely personal one for LOS — by design, not a
  bug.  Two surfaces (user chose both): **Methods** `fig_voi_scorecard` =
  measure×landmark heatmap of baseline−single PI-halfwidth (or APS set-size)
  improvement, rows ordered by mean gain; **Patient** `fig_voi_patient` +
  `voi_readout` = per-patient horizontal bars (teal = prescriptive / still
  obtainable, grey = already observed) with a "next best to obtain" + "most
  informative observed" readout.

* **Observed-trajectory phenotyping (`data/phenotypes.py` + `models/phenotypes.py`, G3)** —
  a **multivariate growth mixture model** (mixture of linear mixed models) over the *observed*
  early-recovery trajectories of **SCIM-total + total motor**, jointly.  Contrast §3 archetypes
  (k-means on model-*predicted* curves): this clusters the *raw* curves, soft-assigns, and is
  likelihood-based.  Per individual `y_{i,m}(t) = beta_{k,m}·phi(t) + b_{i,m}·psi(t) + eps`;
  `phi` = polynomial basis (degree D, BIC-chosen ∈{1,2}), `psi`=[1,t] random intercept+slope
  per measure.  **Class-invariant `G` (block-diagonal across measures) and `sigma2` (per-measure
  residual)** — the robust spec that avoids variance-collapse non-identifiability, so the
  marginal covariance `V_i = Z_i G Z_i' + R_i` is class-independent and built once per individual
  per EM iter; only the mean `Phi_i beta_k` is class-specific (distinct from LCGA, which sets
  G=0).  **Missingness is native** (each individual contributes only its observed (measure,
  timepoint) cells) — the methodological win over k-means-on-imputed-curves.  Fit = EM
  (log-sum-exp E-step; ECM M-step: resp-weighted GLS beta + Laird-Ware variance components) with
  multiple restarts (k-means init + Dirichlet perturbation); **K×degree chosen by BIC** subject to
  a min-class-share floor; separation reported by relative entropy / per-class APPA / min share.
  Fit landed **K=5, degree=2** (cohort: 590 episodes with ≥3 observed SCIM points in the 0day–6m
  window + real IDNumber).  **CRUX (two coupled extrapolation traps a degree-2 basis creates):**
  the fitted means are only valid over each class's *observed support* — an early-discharge class
  (e.g. mild/rapid recoverers, LOS≈51d) whose members exit by ~6w has a quadratic mean that dives
  to absurd values out-of-range (SCIM −1277 at 6m).  `class_support(long, assignments, k,
  min_coverage=0.20)` → `(K, M)` last window index where ≥20 % of the class is still observed; the
  trainer stores it and **(a)** the Overview figure blanks (NaN) each line past its support so a
  curve is drawn only where observed (the figure also clips to [0,100] for the at-ceiling
  overshoot), and **(b)** `order_by_discharge(p, resp, support)` ranks phenotypes (class 0 =
  lowest recovery) by clipped SCIM at the *latest universally-supported* timepoint — never by the
  raw 6m value, which would mislabel the best early-discharge recoverer as the worst.  Persists a
  tracked identifier-free `models/phenotype_metrics.json` (k, degree, selection table, **raw**
  class_means [no NaN tokens], class_support, min_coverage, summaries, diagnostics) and a
  git-ignored `models/phenotypes/phenotypes.joblib` (GMMParams + per-episode hard assignments +
  posterior + class_means + class_support + summaries) for the dashboard / part-2 patient surface.
  Diagnostic/inference module, NOT a production training step — never touches `train.py` artifacts.
  Part 1 (s32): model + **Overview** cohort surface (`fig_phenotype_curves` truncated per-measure
  stacked panels with conditioned-prognosis hover + `fig_phenotype_demographics` AIS bars, reusing
  the archetype `_ais_distribution_bars` helper).  **Part 2 (s33, shipped): patient-level phenotype
  prognosis** — pure `compute.predict_phenotype_membership(key_record, cutoff)` rebuilds the
  one-episode `GMMData` from `LONG` over the observed (SCIM, motor) cells on/before an
  observation-cutoff and applies `predict_proba`, returning soft membership + the
  membership-weighted conditioned-outcome mix (renormalized per-stat over the per-phenotype
  `summaries`).  `compute.phenotype_cutoff_options` gates the interactive cutoff dropdown
  (window timepoints where cumulative observed cells reach ≥2, new-obs only; last entry = full
  window).  Patient-tab card: `fig_phenotype_membership` bar + the `fig_phenotype_curves` overlay
  (new optional `patient_obs` arg draws the patient's own points over the support-truncated
  phenotype means) + a conditioned-prognosis readout.  **Invariant: per-individual responsibilities
  are independent and the full-window cutoff equals an episode's latest observation (= the exact
  set used at fit time), so full-window membership reproduces the bundle's stored `posterior` row
  byte-for-byte for an in-cohort episode (verified diff 0.0).**  The surface also generalizes to
  pickable episodes outside the fit cohort (fresh `predict_proba`, no stored row).

* **AIS-grade conversion (`models/conversion.py`, G4)** — models the admission→discharge AIS
  *transition* (not the absolute discharge grade `y_discharge_ais`).  Three heads, each on its
  at-risk cohort with **admission grade kept as a feature** (conversion is admission-gated:
  ceiling at AIS D, no room at E): **clinical endpoint panel** — two *calibrated binary*
  probabilities, `motor_incomplete` (motor-complete A/B → motor-incomplete discharge ≥C) and
  `ambulatory` (non-ambulatory A–C → ambulatory-capable ≥D); plus an **ordinal magnitude** head —
  multiclass over improvement size `{0, +1, ≥+2}` (`MAG_CAP=2`) on the room-to-improve cohort
  (A–D), deterioration (~1.5 %) folding into class 0.  **Methodology (small cohorts, few heads):**
  grouped 5-fold CV by IDNumber → **out-of-fold (OOF) predictions** drive every reported metric,
  the Platt (sigmoid) calibrator, AND the APS q (a **cross-conformal** pool of per-fold
  nonconformity scores — the valid small-n analogue of production's single split-conformal fold);
  final heads **refit on the full cohort** reusing the OOF calibrator / APS q (conservative,
  mirroring `landmark.py`).  SHAP drivers are **descriptive in-sample** global importances (rank
  drivers, not an OOS claim).  Diagnostic + inference layer like `landmark`/`temporal`: writes
  tracked identifier-free `models/conversion_metrics.json` + git-ignored
  `models/conversion/bundle.joblib` (auto-ignored by `models/*/`); **production `train.py`
  artifacts untouched (byte-repro verified)**.  Bundle: `endpoints[key]={clf, calibrator (1-feat
  LogisticRegression on the LightGBM logit), adm_grades, discharge_min, feature_cols, base_rate}`,
  `magnitude={clf, aps_q_hat, class_codes, mag_cap, adm_grades, feature_cols}`; `_apply_platt`
  (logit→`calibrator.predict_proba`) is **mirrored in compute.py** so the dashboard never imports
  `models.conversion` (which pulls shap via `train.py`).  **CRUX — the two head families are NOT
  numerically comparable:** binary heads use *no* `class_weight` (near-balanced → calibrated
  probabilities), the magnitude head uses `class_weight="balanced"` (imbalanced) so its class
  probs are *uncalibrated and inflated* for the minority improvement classes; for an overlapping
  event they disagree (adm A: binary P(≥C)=0.51 vs magnitude P(≥+2)=0.76).  So the UI must surface
  the **binary heads as the calibrated conversion probabilities** and the **magnitude head as its
  APS set / most-likely class** (not as a competing probability).  Inference **requires `AIS_ord`
  present** (cohort membership is otherwise undefined) and gates by it: an endpoint applies only if
  admission grade ∈ its `adm_grades`; magnitude only for A–D.  Findings: `ambulatory` AUC ≈0.87
  (admission features predict ambulatory conversion well); `motor_incomplete` AUC ≈0.62 (admission
  data poorly predicts complete→incomplete — a real clinical reality, an insightful contrast, not
  a bug); magnitude κ_quadratic ≈0.49, APS conservative (~99 %, set ≈2.4, as documented for
  discrete K).  **Fully shipped: model + tracked metrics (s34) + dashboard surfaces (s35).**
  **Dashboard contract (s35):** pure `compute.predict_conversion(X_row)` (inline `_apply_platt`
  mirror — never import `models.conversion`, it pulls shap) returns per-endpoint calibrated prob +
  applicability flag (gated by admission grade ∈ `adm_grades`) and the magnitude class-probs / APS
  set / argmax (A–D only); `ais_ord=None` ⇒ an all-N/A result the UI renders as a "needs admission
  grade" prompt.  Shared inference figs live in `layout.py` (`fig_conversion_endpoints` calibrated
  horizontal bars + cohort base-rate diamonds; `fig_conversion_magnitude` ordinal set/argmax bars;
  `conversion_readout`), Methods-only metric figs in `figures/methods.py` (`fig_conversion_landscape`,
  `_delta`, `_reliability` calibrated-vs-raw, `_shap`, `_confusion`).  Surfaces: Methods landscape +
  per-endpoint calibration/driver cards + magnitude confusion; Simulator hypothetical card (driven by
  the admission inputs); Patient card.  **Patient-card gating invariant:** `episode_row_for_model`
  imputes cohort defaults for missing features, so the patient callback **overrides `AIS_ord` with
  the episode's raw admission grade** before inference — conversion cohort membership is undefined
  when the grade is unrecorded (~12 % of episodes), so a missing real grade shows the prompt rather
  than a transition *from* an imputed grade (stricter than the production AIS head, by design).
  The Simulator leaves blanks as NaN (no imputation), so its prompt is natural.

* **AIS multi-state recovery (`models/multistate.py`, G6)** — models the AIS-grade *trajectory*
  across the dense early-recovery grid (`0day`..`6m`, the G3 window), the complement to G4's
  admission→discharge *endpoint*.  Two layers (user-chosen design A).  **(1) Population multi-state
  Markov** — a time-inhomogeneous discrete-time Markov chain over the 5 AIS states (A=1..E=5).
  Per grid step `k`, the empirical transition matrix `P_k` is estimated from the **pairwise-complete**
  episodes (observed grade at *both* `WINDOW[k]` and `WINDOW[k+1]`; MAR within the observed-at-both
  subset); a row with zero observed departures **falls back to identity** (assume stable) so the
  chain conserves mass.  Forward-multiplying from an admission-grade point mass yields, **stratified
  by admission grade**: state-occupancy curves `π_k`, first-passage **conversion-to-≥X** curves
  (states ≥X made *absorbing* ⇒ curves monotone non-decreasing; labels `improve`=≥adm+1, `ge_C`,
  `ge_D`), median day to first improvement (0.5-crossing), and expected days per state (trapezoid;
  **`np.trapezoid`**, not the numpy-1 `np.trapz`, which is removed under numpy 2).  **CRUX —
  apparent regressions are real, not a bug:** the empirical chain faithfully reproduces the
  bidirectional transitions in the data (B→A, D→C, E→D — AIS re-assessment / inter-rater noise), so
  occupancy curves carry some backward mass; surface it honestly (the Methods per-step counts expose
  the small-n cells that drive it — e.g. the B→≥C 0.5-crossing at 72h rides on n≈44).  **(2)
  Covariate improve-by-6m head** — one LightGBM *binary* head, P(≥1-grade improvement anywhere in
  the window) on the room-to-improve cohort (admission A–D, ≥`MIN_WINDOW_OBS`=2 in-window AIS obs;
  n=690, ~49 % pos).  No `class_weight` (near-balanced ⇒ calibrated), grouped-CV OOF → metrics +
  global Platt; refit on full cohort reusing the calibrator; descriptive in-sample SHAP drivers
  (admission grade + LEMS/TotalMotor/UEMS + age).  AUC ≈0.90, Brier ≈0.12 (base 0.25);
  **per-grade calibration holds despite a single global Platt** (mean pred tracks obs at every
  grade).  **Clinical finding — the improve base rate is non-monotone in admission grade: B highest
  (~85 %), C ~74 %, A ~64 %, D lowest (~15 %)** because "improvement" for a D admission means
  reaching **E** (full-normal — a high bar); the UI must frame D's `improve` head as P(D→E), not
  generic recovery.  Binary plumbing (typed X, params, OOF, Platt `_apply_platt`, calibration curve,
  SHAP) is **reused verbatim from `conversion.py`**; `_apply_platt` is mirrored in `compute.py` so
  the dashboard never imports this module (it pulls shap via `train.py`).  Diagnostic + inference
  layer like landmark/conversion/temporal: tracked identifier-free `models/multistate_metrics.json`
  + git-ignored `models/multistate/bundle.joblib`; **production `train.py` artifacts untouched
  (byte-repro verified)**.  Bundle shape documented inline at the top of `multistate.py`.  **Fully
  shipped: model + tracked metrics (s36) + dashboard surfaces (s37).**
  **Dashboard contract (s37):** pure `compute.predict_multistate(X_row)` (inline `_apply_platt`
  mirror — never import `models.multistate`, it pulls shap) is admission-grade gated (`AIS_ord`
  required; `None` ⇒ `applicable=False` "needs-grade" prompt) and returns the per-admission-grade
  cohort Markov curves (occupancy, the available first-passage `conversion` labels, `sojourn`,
  `median_day_to_improve`) **looked up by admission grade alone — NOT personalized** + the one
  feature-driven quantity, the calibrated `improve` prob (A–D only; `to_e=True` for a D admission ⇒
  the UI frames it P(D→E)).  An AIS-E admission is flagged `at_ceiling` (empty `conversion`, no
  improve head) but still carries occupancy (the honest downward re-assessment drift).
  `multistate_observed_grades(key_record)` reads `LONG` for the patient's own observed AIS-ordinal
  trajectory (the Patient overlay).  **CRUX — the personalized cards are mostly a COHORT object:**
  occupancy/conversion/sojourn depend only on admission grade, so for two patients of the same
  grade only the `improve` prob and the own-trajectory overlay differ — surface the curves as
  "cohort for this admission grade" (the `ms_cohort_caption` string), not as individualized
  prediction.  **First-passage dedup invariant:** `layout._ms_target_curves` maps the raw
  `improve`/`ge_C`/`ge_D` curves to DISTINCT target grades, because `improve` (=reach adm+1)
  coincides with `≥C`/`≥D` for a B/C admission (those curves are then identical) — so the personal
  conversion fig shows one line per reachable grade, not redundant duplicates.  Shared inference
  figs live in `layout.py` (`fig_multistate_trajectory` cohort expected-grade IQR band + optional
  own-observed overlay; `fig_multistate_conversion_personal` deduped first-passage curves;
  `multistate_readout`), Methods-only cohort-dynamics figs in `figures/methods.py`
  (`fig_multistate_{occupancy,conversion,transition,sojourn,improve_base}`).  The improve head's
  **calibration + SHAP reuse `fig_conversion_{reliability,shap}`** verbatim (same `em`-dict shape —
  `calibration`/`calibration_raw`/`shap_top`).  Surfaces: Methods cohort-dynamics centerpiece (all
  4 figs + per-grade base-rate + calibration/drivers); Patient personalized card (real-grade
  override before inference, mirroring the conversion-card invariant, + own-trajectory overlay);
  Simulator hypothetical card (admission inputs, blanks stay NaN ⇒ natural prompt).  Bilingual
  `ms_*`/`methods_ms_*` strings; the surfaces reuse the `.lm-card`/`.conv-readout`/`.sim-*` CSS
  (`.ms-card` is an unstyled hook like `.conv-card`).

* **Functional-independence profile (`models/independence.py`, G7)** — predicts, per individual
  SCIM-III ADL item, the calibrated P(**functional independence** at discharge) from admission
  features, reframing the aggregate SCIM heads into the per-function "will I feed myself / manage
  my bladder / walk?" question.  **18 calibrated binary heads** (one per modelable item;
  Respiration excluded — 95 % independent at discharge, too imbalanced; its rate is noted in the
  metrics' `excluded`).  **Label = functional independence (aids/devices allowed, no human
  assistance)** via a per-item SCIM-rubric threshold, mapped to the rubric and validated against
  the observed score distributions — the canonical table (also in `independence.py::ITEMS`):
  feeding/bathing/dressing/grooming **≥2** (independent with devices/simple clothes);
  bladder **≥9** / bowel **≥8** (self-manages, no caregiver); toilet **≥3**; bed-mobility =6 and
  the 0–2 transfers (bed↔WC, WC↔toilet, WC↔car) + ground↔WC at their **top score** (the scale
  lumps human-assist with device-use ⇒ no clean "with-aids" middle); the three **walking** items
  (indoors/moderate/outdoors) **≥4** = ambulates independently *with or without aids* —
  wheelchair-independence does NOT count, so they read as distinct ambulation milestones (the
  user's explicit framing).  Domains (display grouping): self_care(6) / sphincter(3) /
  mobility(5, transfers+bed) / ambulation(4, walking+stair).  **Methodology — binary plumbing
  reused verbatim from `conversion.py`** (`_typed_X`/`_params_binary`/`_oof_binary`/`_fit_platt`/
  `_apply_platt`/`_refit`/`_calibration_curve`/`_shap_top`, imported): grouped-5fold-CV OOF →
  AUC/Brier/calibration + Platt calibrator; refit on full cohort; descriptive in-sample SHAP
  drivers.  **CRUX — the uncertainty surface is the calibrated probability + reliability curve,
  NOT a conformal set: an APS set over a binary head degenerates (§0b) and was removed.**  No
  `class_weight` (items near-balanced ⇒ calibratable).  Findings: mean AUC ≈0.905 across 18 items
  (range 0.88–0.93), every Brier well below its base-rate baseline; the **expected-independent
  count** (Σ of the 18 calibrated probs, a clean profile summary) rises monotonically with
  admission AIS (A≈0.9 → B≈6 → C≈13 → D/E≈17 of 18), and **outdoor walking is the hardest
  milestone** (P≈0.36 even at AIS-C) — the differentiated walking signal.  Diagnostic + inference
  layer like conversion/multistate/landmark: tracked identifier-free
  `models/independence_metrics.json` + git-ignored `models/independence/bundle.joblib`;
  **production `train.py` artifacts untouched (empty `training_metrics.json` diff)**.  Bundle
  shape (consumed by the pending `compute.predict_independence`): `items` registry +
  `heads[key]={clf, calibrator, thr, col, domain, feature_cols, base_rate}`; mirror `_apply_platt`
  inline in compute.py (never import this module — it pulls shap via conversion→train).  **Part 1
  shipped (s38): model + tracked metrics; dashboard surfaces PENDING (backlog G7 Part 2).**

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
uv run python -m rehab_sci.models.phenotypes     # observed-trajectory phenotyping — growth mixture model (G3); ~5 min
uv run python -m rehab_sci.models.conversion     # AIS-grade conversion modeling (G4); ~15 s
uv run python -m rehab_sci.models.multistate     # AIS multi-state recovery — transition Markov + improve head (G6); ~5 s
uv run python -m rehab_sci.models.independence   # functional-independence profile — 18 per-SCIM-item calibrated heads (G7); ~50 s
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

* **s38** — G7 functional-independence profile, **Part 1** (model + tracked metrics; user chose
  the *functional* independence definition — aids/devices allowed — and the *ambulation* framing
  for the 3 walking items).  New `models/independence.py`: 18 calibrated binary heads (one per
  modelable SCIM-III ADL item; Respiration excluded as near-universal), label = per-item
  rubric-mapped functional-independence threshold (validated against the observed discharge score
  distributions; full table in §3).  Reuses `conversion.py` binary plumbing verbatim
  (grouped-CV OOF → Platt + AUC/Brier/calibration; refit full cohort; in-sample SHAP).  **Caught +
  removed a degenerate binary APS abstention set** (100 % coverage / ~99 % abstain because the APS
  q pins at ~1.0 for K=2 — new §0b lesson); the calibrated probability + reliability curve is the
  uncertainty surface.  Tracked identifier-free `independence_metrics.json` + git-ignored bundle;
  production byte-repro verified (empty `training_metrics.json` diff).  Findings: mean AUC ≈0.905,
  expected-independent-count rises monotonically A→E (0.9→6→13→17 of 18), outdoor walking the
  hardest milestone.  Validated bundle inference + clinical monotonicity per AIS grade; lint +
  F-gate clean; MAP regenerated.  Dashboard surfaces (Part 2) deferred to the backlog.
* **s37** — G6 AIS multi-state recovery, **Part 2** (dashboard surfaces across Methods + Patient +
  Simulator; user chose all 4 cohort-dynamics visuals + all-three-surfaces with a Methods
  checkpoint commit).  Pure `compute.predict_multistate` (inline `_apply_platt` mirror;
  admission-grade gated; E→`at_ceiling`; D-improve→`to_e`/P(D→E)) + `multistate_observed_grades`
  (LONG overlay) + `state.MULTISTATE`/`MULTISTATE_BUNDLE` loaders.  Methods cohort-dynamics
  centerpiece — `fig_multistate_{occupancy,conversion,transition,sojourn,improve_base}` (occupancy
  stacked-area per grade, deduped first-passage curves, pooled transition heatmap, sojourn bars,
  non-monotone improve base rate) + the improve head's calibration/drivers **reusing
  `fig_conversion_{reliability,shap}`**.  Shared `layout` figs `fig_multistate_trajectory`
  (cohort expected-grade IQR band + own-observed overlay) / `fig_multistate_conversion_personal`
  (target-grade-deduped) / `multistate_readout`; Patient card (real-grade override) + Simulator
  card (blanks-NaN) + callbacks.  Bilingual `ms_*`/`methods_ms_*`; CSS reused (`.ms-card` hook).
  Documented the cohort-not-personalized CRUX + the first-passage dedup invariant in §3.  Lint +
  F-gate clean; all 3 tabs render both langs; dashboard boots 200 with the 2 new callbacks
  registered; production + multistate artifacts untouched (no retrain).
* **s36** — G6 AIS multi-state recovery, **Part 1** (model + tracked metrics; user chose approach A
  empirical Markov + covariate head · Methods+Patient+Simulator surfaces).  New
  `models/multistate.py`: a time-inhomogeneous discrete-time Markov chain over AIS A–E on the
  0day–6m grid (pairwise-complete `P_k`, identity-fallback for empty rows, absorbing-above
  first-passage) → per-admission-grade occupancy / conversion-to-≥X / median-day-to-improve /
  sojourn; plus a binary improve-by-6m head (A–D, n=690, AUC ≈0.90, Platt-calibrated, SHAP
  drivers).  Reuses `conversion.py` binary plumbing.  Tracked `multistate_metrics.json` +
  git-ignored bundle; production byte-repro verified (empty `training_metrics.json` diff).
  Validated: occupancy mass-conserving, conversion monotone, transitions row-stochastic, per-grade
  calibration holds.  Surfaced the non-monotone improve base rate (B>C>A>>D, D="reach E") and the
  faithful apparent-regression caveat.  Lint + F-gate clean; MAP regenerated.  Dashboard surfaces
  shipped in s37.
* **s35** — G4 AIS-grade conversion, **Part 2** (dashboard surfaces across Methods + Patient +
  Simulator).  Pure `compute.predict_conversion` (inline `_apply_platt` mirror; admission-grade
  gating; real-grade override on the Patient card) + `state.CONVERSION`/`CONVERSION_BUNDLE`
  loaders.  Shared `layout` figs (`fig_conversion_endpoints` calibrated bars + base-rate diamonds,
  `fig_conversion_magnitude` ordinal set/argmax, `conversion_readout`) and 5 Methods metric figs
  (`fig_conversion_landscape`/`_delta`/`_reliability`/`_shap`/`_confusion`).  Methods landscape +
  per-endpoint calibration/driver cards + magnitude confusion; Simulator + Patient inference cards.
  Bilingual `conv_*` strings + `.conv-*` CSS.  Honored the §3 CRUX (binary = calibrated probs,
  magnitude = APS set / argmax, never competing numbers).  **Caught + fixed a latent s31 bug:
  `INK["600"]` (nonexistent key) in the Methods VOI paragraph had been crashing the whole Methods
  tab render since G2 — invisible to "boots 200" checks (boot doesn't render tab content).**  Lint
  + F-gate clean; final code boots 200 with both conversion callbacks registered; functional tests
  pass for AIS A/C/D + missing-grade + blank-sim, both langs.  No retrain — production + conversion
  artifacts untouched.
* **s34** — G4 AIS-grade conversion modeling, **Part 1** (model + tracked metrics; user chose
  panel+magnitude · diagnostic module · Methods+Patient+Simulator surfaces).  New
  `models/conversion.py`: two *calibrated binary* endpoints (`motor_incomplete` A/B→≥C,
  `ambulatory` A–C→≥D) + an *ordinal magnitude* head `{0,+1,≥+2}` on A–D, each on its at-risk
  admission cohort.  Grouped 5-fold OOF drives metrics + Platt calibration + cross-conformal APS
  q; final heads refit on full cohort; descriptive in-sample SHAP drivers.  Diagnostic layer:
  tracked `conversion_metrics.json` + git-ignored `conversion/bundle.joblib`; production
  byte-repro verified (empty `training_metrics.json` diff).  Validated bundle inference +
  applicability gating on one episode per grade.  Findings: ambulatory AUC ≈0.87,
  motor_incomplete AUC ≈0.62 (real clinical contrast), magnitude κ ≈0.49.  Documented the
  binary-calibrated vs magnitude-balanced non-comparability (§3 CRUX, §0b lesson).  **Dashboard
  surfaces (compute inference + figures + 3 tabs + bilingual strings + CSS) PENDING — resume at
  backlog tasks 4–7.**  Lint + F-gate clean; MAP.md regenerated.
* **s33** — G3 observed-trajectory phenotyping, **Part 2** (patient-level phenotype
  prognosis; user chose readout+overlay with an interactive observation-cutoff).
  Pure `compute.predict_phenotype_membership` / `phenotype_cutoff_options` (single-
  episode `GMMData` rebuild → `predict_proba` soft membership + membership-weighted
  conditioned-outcome mix); new `fig_phenotype_membership` bar + a `patient_obs`
  overlay arg on `fig_phenotype_curves` (draws the patient's own points over the
  support-truncated phenotype means); eligibility-gated cutoff dropdown + two
  callbacks on the Patient tab.  Bilingual `pheno_*` strings + `.pheno-*` CSS.
  Verified: full-window membership reproduces the stored posterior (diff 0.0) and
  sharpens as the cutoff advances (entropy 1.01→0.00); lint + F-gate clean;
  dashboard boots 200 with the 4 phenotype callbacks registered; Overview call
  backward-compatible.  No retrain — production + phenotype artifacts untouched.
* **s32** — G3 observed-trajectory phenotyping, **Part 1** (user fork: GMM with
  random effects · multivariate SCIM+motor · model+cohort surface).  New
  `data/phenotypes.py` (multivariate growth mixture model — mixture of linear
  mixed models, class-invariant block-diagonal G + per-measure sigma2, native
  missingness, EM w/ k-means+Dirichlet restarts, BIC over K×degree) + trainer
  `models/phenotypes.py`.  Validated the EM on synthetic data first (ARI 0.95,
  params + BIC-K recovered).  Fit → **K=5, degree=2** on 590 episodes.  **Caught
  two coupled degree-2 extrapolation traps in the sanity gate:** (1) early-
  discharge classes' quadratic means dive out-of-support (SCIM −1277 @6m) →
  added `class_support` (per-class/measure last window ≥20 % observed); the
  figure blanks each line past its support (+[0,100] clip for ceiling overshoot);
  (2) `order_by_discharge` sorted on the raw 6m value → mislabeled the best
  early-discharge recoverer as worst → now ranks by clipped SCIM at the latest
  *universally-supported* timepoint, giving a clean severe→full ordering.
  Overview gains the phenotype surface (`fig_phenotype_curves` support-truncated
  stacked panels w/ conditioned-prognosis hover + `fig_phenotype_demographics`,
  reusing `_ais_distribution_bars`); bilingual (`chart_phenotype_*`,
  `phenotype_caption`, `pheno_measure_*`); `.ov-section-note` CSS.  Tracked
  identifier-free `phenotype_metrics.json` (raw class_means, no NaN tokens) +
  git-ignored `phenotypes/phenotypes.joblib`; production artifacts untouched.
  Lint clean; dashboard boots 200; phenotype surface renders both langs.  Part 2
  (patient-level phenotype prognosis) deferred to the backlog.
* **s31** — G2 value-of-information.  Per (outcome, L) the landmark trainer now
  also fits 10 **single-add heads** (admission + exactly one `L_<measure>`, 31
  feats) on the same eligible cohort/split, each with its own marginal conformal
  q / APS q_hat; metrics under `by_landmark[L]["single"][measure]`, heads in the
  bundle's `["single"]`, plus `lm_value_summary[L][measure]={median,q25,q75,n}`
  for median-imputing unobserved measures.  Pure `compute.landmark_voi` ranks a
  patient's measures by PI-halfwidth (regression) / APS-set (multiclass)
  reduction, scoring each at its real value (observed/realized) or cohort median
  (prescriptive/hypothetical).  **Both surfaces:** Methods `fig_voi_scorecard`
  (measure×landmark improvement heatmap) and Patient `fig_voi_patient` +
  `voi_readout` ("next best to obtain" / "most informative observed").  Bilingual
  (`voi_*` + reuse `lm_measure_*`).  Production byte-repro preserved
  (`landmark_metrics.json` + git-ignored bundle only); lint clean; dashboard
  boots 200; VOI surfaces smoke-tested (regression + multiclass).  Documented the
  marginal-conformal ⇒ per-measure-not-per-patient half-width invariant in §3.
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

Propose from here unless the user redirects.  **Items F1–F25 + G1–G4 + G6 are
shipped; G7 Part 1 is shipped (s38) and G7 Part 2 is the immediate next task** —
see §7 for the session each landed in, and Git history for implementation detail.
The user steers toward *insightful* (clinical/scientific) features over
infra/maintenance.
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
* **G2 value-of-information** (s31): single-add landmark heads (one per measure,
  own conformal q) + `compute.landmark_voi` + `lm_value_summary`; ranks the next
  best measure to obtain per patient.  Methods VOI scorecard heatmap + Patient
  VOI bars/readout.  See §3 for the contract and §7 for the session.  Fully
  shipped.
* **G3 observed-trajectory phenotyping** (s32, Part 1): multivariate growth
  mixture model (`data/phenotypes.py` + `models/phenotypes.py`) over observed
  SCIM+motor early-recovery trajectories → K=5 phenotypes; Overview cohort
  surface (`fig_phenotype_curves` support-truncated + `fig_phenotype_demographics`).
  Part 2 (s33): patient-level phenotype prognosis — `compute.predict_phenotype_membership`
  + interactive observation-cutoff + membership bar / curve overlay / conditioned-outcome
  readout on the Patient tab.  See §3 for the contract (incl. the support-truncation /
  ordering extrapolation traps + the reproduce-stored-posterior invariant) and §7.  Fully
  shipped.
* **G4 AIS-grade conversion** (s34 model + tracked metrics; s35 dashboard surfaces):
  `models/conversion.py` (two calibrated binary endpoints + ordinal magnitude head) +
  `compute.predict_conversion` + shared `layout` inference figs + 5 Methods metric figs +
  Methods/Patient/Simulator surfaces (`conv_*` strings, `.conv-*` CSS).  See §3 for the model
  + dashboard contract (incl. the binary-vs-magnitude non-comparability CRUX and the patient
  real-grade gating invariant) and §7.  Fully shipped.
* **G6 AIS multi-state recovery** (s36 model + s37 dashboard): `models/multistate.py` — population
  multi-state Markov (per-admission-grade state-occupancy / first-passage conversion-to-≥X /
  median-day-to-improve / sojourn on the 0day–6m grid) + a binary improve-by-6m covariate head with
  SHAP drivers.  Complement to G4 (trajectory *between*, not the admission→discharge endpoint).
  `compute.predict_multistate` + Methods cohort-dynamics centerpiece (4 figs + calibration/drivers)
  + Patient personalized (own-trajectory overlay) + Simulator hypothetical cards (`ms_*` strings).
  See §3 for the model + dashboard contract (pairwise-complete/identity-fallback estimation,
  absorbing-above monotone first-passage, the faithful apparent-regression caveat, the non-monotone
  improve base rate, the cohort-not-personalized CRUX + first-passage dedup invariant) and §7.
  Fully shipped.
* **G7 functional-independence profile** (s38 Part 1 — model + tracked metrics; Part 2 dashboard
  pending): `models/independence.py` — 18 per-SCIM-III-item calibrated binary heads predicting
  P(functional independence at discharge) per ADL item, reframing the aggregate SCIM total as
  concrete acts ("will I feed myself / manage my bladder / walk?").  Independence = item score ≥ its
  functional-with-aids rubric threshold (aids/devices count); the 3 walking-mobility items use the
  ambulation (walking, ≥4) framing so wheelchair-independence does **not** count.  Borrows the
  conversion.py binary plumbing verbatim (`_oof_binary`/`_fit_platt`/`_apply_platt`/`_shap_top`,
  grouped-CV by IDNumber, Platt calibration).  Respiration excluded (near-universal).  No APS layer
  (binary APS is degenerate — see §0b).  See §3 for the contract and §7.  Part 1 shipped; Part 2
  (Methods/Patient/Simulator surfaces) is the immediate next task.

**F23 (shipped s26): data-quality / clinical-consistency report** — see §7 and
`data/quality.py`; durable data facts it surfaced live in §0b/§1, and the
regenerated `models/dataquality_summary.json` holds the per-rule scorecard.

**Ready candidates (pick the next unless redirected; the user prefers a new insightful G-series
feature over the infra items below):**
* **G7 Part 2 — functional-independence dashboard surfaces (IMMEDIATE NEXT):** pure
  `compute.predict_independence` (inline `_apply_platt` mirror, no model-module import — keeps
  compute.py shap-free), a shared layout profile figure (horizontal P(independence) bars grouped by
  domain), Methods metric figs (per-item AUC/Brier scorecard, calibration, SHAP driver heatmap,
  cohort base-rate landscape), and Methods+Patient+Simulator surfaces with bilingual `ind_*` /
  `methods_ind_*` strings + `.ind-*` CSS.  Mirror the G4/G6 dashboard pattern.  See §3 for the
  contract.
* **F26 invariant test harness** — narrow pytest enforcing §1 data + model
  invariants + a smoke test (incl. a headless `render_{methods,patient,simulator}` per the §0b
  lesson, which would have caught the s31 `INK["600"]` crash).  skip-if-CSV-absent.  M.  files:
  `tests/`, pyproject.
* **F27 dependency refresh** — minor/patch bumps + raise the `shap<0.52` cap;
  retrain to verify byte-repro.  S, low value now (no CVEs, lint clean).
* **New G-series ideas (propose to the user; none scoped yet):** e.g. competing-risks /
  time-to-event discharge modeling; a counterfactual "treatment lever" explorer; calibration
  drift monitoring.  Add any candidate here with **what / why / effort / files / data
  dependency** before starting.
