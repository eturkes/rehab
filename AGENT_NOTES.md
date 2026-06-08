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
  `.claude/settings.json` needs **no** `statusLine` (resist re-adding it; the
  file now holds only `permissions.deny` Read rules — see next bullet); keep the
  two `compaction.sh` copies in sync.  Dual-mode keyed on `CLAUDE_CODE_SESSION_ID` (set ⇒
  manual/transcript; unset ⇒ statusline/stdin) — any edit must keep both.  Per
  CLAUDE.md, wrap to a clean boundary at ≥80 % for a manual `/compact`.
* `.claude/settings.json` — `permissions.deny` `Read()` rules (CLAUDE.md policy)
  that hide low-benefit paths from the Read tool, Grep/Glob, and `cat`/`head`/
  `sed`: the venv, `.git`, caches, raw `ALL_SCIDATA.csv`, `uv.lock`, every
  `*.joblib` (+ `*.pkl/.npy/.npz/.parquet`), and the 3 oversized generated dumps
  `models/subgroups.json`, `models/dataquality_report.json`,
  `schema/raw_profile.json`.  Deny does **not** touch `python`/`jq`/`ls`/`git`,
  so query a denied JSON with `jq` and confirm artifacts via `ls`/`git
  ls-files`.  The canonical `models/*_metrics.json` stay readable on purpose
  (model-performance source of truth).  Rules reload live; inspect with
  `/permissions`; grow the list as new low-value paths appear.
* This file — agent-facing scratchpad.  Read before planning; update after each
  session; prune duplication per the inclusion rule above.
* **Default-work pool: §8 backlog.**  F1–F25 + G1 (s29/s30) + G2 (s31) + G3
  observed-trajectory phenotyping (s32/s33) + G4 AIS-grade conversion (s34/s35) + G6 AIS
  multi-state recovery (s36/s37) + G7 functional-independence profile (s38/s39) + G8 recovery
  topography map (s40 model + s41 body-map dashboard) + G9 Δ score-recovery prediction (s42)
  **all fully shipped** (see §7); **G10 neurological-level descent — Part 1 model shipped (s43);
  Part 2 dashboard USER-APPROVED for all 5 levels (build spec in §8) — the next pick** (deferred
  from s43 to a fresh session per the G-series rhythm).
  Open: F26 test harness · F27 dep refresh · **G10 Part 2 (next)** · a new G-series idea (data is exhausted of
  NEW field families — see §8; any new G must reuse the existing ISNCSCI/SCIM/AIS signal — e.g.
  ZPP descent or calibration-drift monitoring).
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
* **`build_episode_frame` ends with `feat.reset_index()` ⇒ `ep` (= `af.df`) has a POSITIONAL
  index (0..N−1) and `KeyRecordNumber` is a COLUMN — so align any *separately-built* per-timepoint
  matrix to `ep` by `ep["KeyRecordNumber"]`, never by `.reindex(ep.index)`.**  G8 first reindexed
  the per-segment ISNCSCI discharge/admission matrices (which are `set_index("KeyRecordNumber")`)
  onto `ep.index`, silently pulling KeyRecordNumbers 0..898 (the real values are the non-contiguous
  1..1169) — a wholesale row scramble.  **The trap: per-segment AUC stayed HIGH (~0.87) so a naive
  metrics check passed** — because the label `y` and the segment's own admission grade `adm_self`
  were pulled by the *same* wrong reindex (mutually consistent), the head leaned on `adm_self` and
  the misaligned 30-feature context merely added noise.  Fix:
  `mtx.reindex(ep["KeyRecordNumber"].to_numpy()).set_axis(ep.index)` (lifted AUC to ~0.93, and the
  30 features then contribute correctly).  **Catch a co-misaligned feature+label with a BEHAVIORAL
  sanity check, not AUC**: a max-admission-motor patient must predict P(antigravity)≈1 and a
  zero-motor complete injury ≈0 (mean |ΔP|≈0.9); the scrambled model gave a flat ~0.68/0.96.
* **Claude Code deny-`Read()` rules are gitignore-style, and a `dir/*/**` pattern
  over-denies — verify every rule empirically.**  `Read(/models/*/**)` (meant for
  the binary per-outcome subdirs) silently *also* denied the direct-child
  `models/*_metrics.json` files, because the matcher lets `*` match a filename and
  `/**` match zero trailing segments.  The failure mode is silent in *both*
  directions — a mistyped pattern can protect nothing, or over-protect a file you
  need.  After any edit (rules reload live), attempt a `Read` on a path that MUST
  be blocked AND one that MUST stay readable.  Prefer extension globs
  (`**/*.joblib`) for binary trees + exact file rules for individual dumps; a
  single leading `/` anchors to the project root (`//` = filesystem-absolute), a
  bare name matches at any depth, and deny still leaves `python`/`jq`/`ls`/`git`
  working (so query a denied JSON with `jq`).
* **A feature that nails a *state* need not help a *change / threshold-crossing* target — test the
  transfer, never assume it.**  G8 added a segment's own granular admission grade and segment-*state*
  AUC jumped 0.5→0.93 (most segments don't change between admission and discharge ⇒ high
  autocorrelation).  Replaying that recipe for G10 neurological-*level descent* (concat the
  modality-matched 20/112/132 per-segment admission grades onto the 30 aggregates) moved descent AUC
  by ≤0.02 on all five heads — within OOF noise, sometimes worse — so it was reverted.  A level
  descends only when its *most rostral impaired segment* crosses a grade threshold, a recovery-
  dynamics event the static admission grades don't encode; the admission level ordinal already
  locates the boundary, so the surrounding per-segment grades add only noise.  Lesson: enrichment
  that wins on a state/autocorrelated target can be inert on a change/event target — measure it and
  keep the simpler model when it doesn't pay.

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
  `OutcomeSpec` records — the 6 production heads (4 SCIM + AIS + LOS) plus the 5
  G9 Δ score-recovery heads (see the G9 bullet below).  `train.py` iterates it;
  the dashboard imports the same list so simulator/Methods stay in lockstep.  To
  add an outcome: extend `OUTCOMES`, ensure its target column is on the episode
  frame, add a `ui_strings.yaml` `outcome_{key}` entry — **everything else
  (training, conformal, SHAP, the simulator/patient/insights/Methods cards)
  extends automatically; no new code or callbacks** (G9 added zero callbacks).
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
  shape: `items` registry + `heads[key]={clf, calibrator, thr, col, domain, feature_cols,
  base_rate}` + `discharge_timepoint` (added s39 for the patient overlay); mirror `_apply_platt`
  inline in compute.py (never import this module — it pulls shap via conversion→train).  **Fully
  shipped: model + tracked metrics (s38) + dashboard surfaces (s39).**
  **Dashboard contract (s39):** pure `compute.predict_independence(X)` (inline `_apply_platt`
  mirror) runs all 18 heads and returns, in display order, each item's calibrated `prob` +
  `base_rate` grouped by `domain` + the **expected-independent count** (Σ probs); it is **NOT
  admission-grade gated** (independence is predicted for everyone) — so unlike the G4/G6 patient
  cards it needs **no real-grade override** (`episode_row_for_model`'s cohort-default imputation is
  used as-is, matching the production SCIM patient cards).  `independence_observed_for_episode`
  reads `LONG`'s discharge slot vs each head's `thr` for the patient achieved/not-achieved overlay.
  Shared `layout` figs: `fig_independence_profile` (domain-grouped horizontal P(indep) bars +
  cohort base-rate diamonds + optional realized-independence markers) + `independence_readout`
  (expected count + per-domain breakdown + most/least-likely item).  Methods-only figs in
  `figures/methods.py`: `fig_independence_{scorecard,calibration,landscape,shap_heatmap}` (per-item
  AUC/Brier-skill; all-18-heads reliability overlay; item×admission-AIS rate landscape = the
  monotone-A→E centerpiece; item×driver heatmap).  The **per-item drilldown reuses
  `fig_conversion_{reliability,shap}` verbatim** (a head's metrics entry shares the
  `calibration`/`calibration_raw`/`shap_top` shape) and is driven by the **first-ever Methods
  `@callback`** (`update_methods_independence_item`; the dropdown rebuilds + resets to the first
  item on lang/tab change — acceptable).  **CRUX — `AIS_ord` alone is a weak driver
  (TotalMotor/UEMS dominate the SHAP), so a Simulator profile with ONLY an admission grade set is
  flat (~8.5/18), NOT grade-differentiated; the A→E spread (real-cohort patient-card means ≈
  4.8→6.5→8.0→15.3→17.2 of 18) emerges only when the correlated motor/sensory features move with
  the grade.**  This is correct (honest under missingness, consistent with F25), not a bug.  Item
  display labels reuse `columns.yaml` via `col_label` (no per-item string keys); domain palette in
  `theme.PALETTE_INDEPENDENCE_DOMAIN`; bilingual `ind_*`/`methods_ind_*`; surfaces reuse the
  `.lm-card`/`.conv-readout`/`.sim-*`/`.chart-card`/`.pheno-subtitle` CSS (`.ind-card` is an
  unstyled hook like `.ms-card`).  Surfaces: Methods cohort centerpiece (4 figs + drilldown);
  Patient card (profile + realized overlay); Simulator hypothetical card (blanks→NaN, no overlay).

* **Recovery topography map (`models/topography.py`, G8)** — per-ISNCSCI-segment functional-recovery
  atlas: for **every one of the 132 segments** (20 motor key muscles + 56 light-touch + 56
  pin-prick dermatomes, L/R), the calibrated P(*functional milestone* at discharge).  Milestone =
  motor **P(grade ≥3)** = antigravity; sensory **P(grade ≥1)** = protective/preserved sensation
  (the closest analogue to motor antigravity).  The impairment/neurology complement to G7's ADL
  profile; mines the largest *unused* signal (the 132 per-segment columns the 30 features collapse
  into 5 totals).  **Method = 132 INDEPENDENT calibrated binary heads (NOT the structured low-rank
  model first chosen).  CRUX — a diagnostic settled the architecture: the 30 aggregate admission
  features predict a *specific* segment's discharge state barely above chance (per-segment OOF AUC
  0.43–0.63), because the dominant predictor is the segment's OWN admission grade — a
  per-segment/diagonal relationship, not the cross-segment correlation a low-rank model exploits
  (the discharge matrix is also not low-rank).  The low-rank attempt collapsed to base-rate
  (AUC ≈0.5, flat-by-AIS).  Adding the segment's own admission grade lifts AUC to ~0.93.**  So each
  head = LightGBM on `[the 30 admission features] + [adm_self = that segment's own admission grade,
  LOCF over the admission-fallback timepoints]` + Platt calibration — the `conversion.py` binary
  plumbing imported verbatim (`_typed_X`/`_oof_binary`/`_fit_platt`/`_apply_platt`/`_refit`/
  `_calibration_curve`/`_shap_top`).  `adm_self` is still admission-only (no leakage); it is the
  per-segment ISNCSCI the curated 30 omit.  Grouped-5fold-CV-by-IDNumber OOF → metrics + Platt;
  refit full cohort; descriptive in-sample SHAP.  **No conformal/APS** (binary APS degenerates,
  §0b) — calibrated prob + reliability curve is the uncertainty surface (as in G7).  **Degenerate
  heads (12: the high-cervical C2/C3 LT+PP ceiling dermatomes, < `MIN_MINORITY`=12 minority cases)
  carry an honest constant base-rate prob, no fitted model** (mirrors G7 excluding Respiration).
  **Alignment invariant: align the per-segment matrices to `ep` by `ep["KeyRecordNumber"]`, NOT
  `ep.index` (§0b lesson) — a co-misalignment keeps AUC high while scrambling the 30-feature context
  + CV groups; verify with the behavioral personalization check.**  Findings: mean OOF AUC motor
  **0.94** / LT **0.93** / PP **0.91**; calibration MAE ~0.01; the model strongly personalizes
  (own admission grade dominant, then LEMS/TotalMotor/mFrankel global severity).  **The cohort
  expected-antigravity count (~14.7/20) is driven by injury LEVEL, not AIS grade — the raw
  antigravity count is NON-monotone in admission AIS (A 12.7, B 16.3, C 14.5, D 15.2, E 15.8) — so a
  naive "rises with AIS" sanity check is WRONG here** (unlike the SCIM/independence counts).
  Diagnostic + inference layer like conversion/independence/multistate: tracked identifier-free
  `models/topography_metrics.json` + git-ignored `models/topography/bundle.joblib`; **production
  `train.py` artifacts untouched (empty `training_metrics.json` diff)**.  Bundle/metrics shape
  documented inline at the top of `topography.py`; mirror `_apply_platt` inline in compute.py for
  Part 2 (never import this module — it pulls shap via conversion→train).  **Fully shipped: model +
  tracked metrics (s40) + dashboard surfaces (s41).**
  **Dashboard contract (s41):** the user chose the *richest* atlas (anatomical dermatome **silhouette + motor
  myotome ladder**) and a *seeded Simulator worksheet*.  Pure `compute.predict_topography(X, adm_grades)` (inline
  `_apply_platt` mirror over the 132 heads; degenerate→`base_rate`; a missing `adm_self`→NaN, LightGBM-native, matching
  training) + `topography_admission_grades(key_record)` (the patient's real per-segment admission exam via the SAME
  ADMISSION_FALLBACK LOCF the trainer used — aligns by KeyRecordNumber) + `topography_observed_discharge` (realized
  discharge milestones for the patient overlay) + `topography_cohort_atlas` (per-segment base-rate map for Methods, no
  inference) + `state.TOPOGRAPHY`/`TOPOGRAPHY_BUNDLE` loaders.  Shared
  `layout.fig_topography_bodymap(result, lang, sensory_modality, observed, title)` = a `make_subplots` composite — a
  stylised front-view humanoid (`_topo_body_shapes` SVG paths) with dermatome markers at hand-placed `_DERMATOME_XY`
  (28 levels × L/R, the toggled LT **or** PP modality) + a motor myotome ladder (10 levels × L/R squares), shaded by
  `theme.COLORSCALE_TOPOGRAPHY` (crimson→teal); `observed` rings achieved/not-achieved segments green/crimson.
  `topography_readout` = expected antigravity / LT / PP counts + strongest/weakest motor segment.  Methods-only figs in
  `figures/methods.py`: `fig_topography_{calibration` (pooled per-modality reliability), `scorecard` (per-segment
  AUC/Brier-skill box per modality), `drivers` (per-modality SHAP grouped bars, `adm_self`-dominant)`}`; the cohort
  atlas reuses `fig_topography_bodymap`(`topography_cohort_atlas`) and the per-segment drilldown reuses
  `fig_conversion_{reliability,shap}` **verbatim** (each modelable segment's metrics entry carries
  `calibration`/`calibration_raw`/`shap_top`).  Surfaces: Methods cohort atlas (LT/PP radio) + calibration + scorecard
  + drivers + per-segment-dropdown drilldown (**2 Methods `@callback`s** — the 2nd/3rd ever); Patient card (real exam →
  predicted map + achieved-vs-predicted overlay, LT/PP toggle, **NOT grade-gated ⇒ no real-grade override**, like G7);
  Simulator card with an editable **132-cell ISNCSCI worksheet** (`{type:topo-seg,seg:<key>}` debounced number inputs,
  motor 0–5 / sensory 0–2, absent cells blanked) **seeded from the What-if `patient-ref`** (its real admission exam,
  via `render_simulator(ref_data)` + a Seed/Clear value-Output callback), the worksheet `adm_self` + the 30-field form
  driving the body map (blanks→NaN, no overlay).  Bilingual `topo_*`/`methods_topo_*`; new
  `theme.{COLORSCALE_TOPOGRAPHY,PALETTE_TOPOGRAPHY_MODALITY}`; CSS reuses `.lm-card`/`.sim-*`/`.pheno-subtitle` + new
  `.topo-worksheet*` (sticky-header scroll table); `.topo-card` unstyled hook.  +5 callbacks (34→39); production
  byte-repro preserved (dashboard-only diff).  **CRUX — the cohort atlas (base rate) is mostly teal for light touch
  (LT preserved ~87% cohort-wide, near ceiling) and the antigravity count is injury-LEVEL- not AIS-driven (G8 Part 1);
  the personalization shows on the *patient/sim* map, where the segment's own admission grade dominates — a severe
  injury reds out the affected region (real AIS-A mean motor P≈0.25 vs AIS-D ≈0.95).**

* **Δ score-recovery prediction (G9 — `outcomes.py` registry, NOT a standalone module)** —
  predicts the admission→discharge *change* in each ISNCSCI summary score: **ΔUEMS, ΔLEMS,
  Δtotal-motor, Δlight-touch, Δpin-prick** (keys `delta_{uems,lems,totalmotor,lighttouch,pinprick}`),
  the canonical SCI-trial primary endpoint.  These five scores already feed the model as admission
  *inputs* (they are in `ADMISSION_FEATURES`); G9 re-targets their *recovery*.  **Unlike G1-G8
  (standalone diagnostic modules with their own metrics files), G9 is a pure PRODUCTION-registry
  extension** — only two code edits: (1) five `OutcomeSpec`s appended to `OUTCOMES`; (2) five Δ
  targets in `build_episode_frame` — `y_delta_*` = discharge-slot score − the same first-non-null
  admission feature `feat[col]` the model already sees (NaN on either side ⇒ NaN target, dropped by
  `train._prep`'s `dropna`).  `train.py` is **untouched** (fully generic over `OutcomeSpec`); the Δ
  heads reuse the SCIM regression machinery verbatim — LightGBM median/p10/p90 + Mondrian
  split-conformal 80 % PI + TreeSHAP, identity `transform=None`.  **No leakage:** only the *delta* is
  the label, never the discharge score itself.  **Negative-range invariant:** `clip_min < 0` (±50
  motor / ±100 total-motor / ±112 sensory) so a predicted/observed **deterioration** is representable
  (re-assessment noise + genuine decline); `np.clip` already supports it, so the production clip path
  needed no change.  `clip_min < 0` is also the UI's **unique flag for a Δ head** (no absolute-score
  outcome has a negative floor): the simulator shows a **signed** point/PI (`+11`, `−3`) and **drops
  the "/ max" suffix** (a recovery has no ceiling reading), and `layout.fig_prediction_interval`
  draws a dotted **"no change" zero-line** with the axis spanning the full ±range so 0 is always
  visible.  **Auto-extension:** the registry drives the simulator/patient/insights/Methods
  outcome-selector with **zero new callbacks**.  **Scope boundary — the heavier diagnostic Methods
  surfaces (temporal F24, landmark/VOI G1/G2) do NOT cover the Δ heads:** those modules carry
  pre-computed bundles keyed to the original 6 outcomes and their loops **gracefully skip** an absent
  outcome (`if not info: continue` / `SUBGROUPS.get(key, {})`), so a Δ selection simply shows no
  temporal/landmark panel — an accepted boundary, not a bug (re-running those for Δ was out of scope).
  **Findings (qualitative — numbers live in `training_metrics.json`): motor recovery is more
  predictable than sensory** (ΔUEMS/ΔLEMS/Δtotal-motor R² > Δlight-touch > Δpin-prick), and the
  dominant SHAP interaction shifts from the SCIM heads' *age × motor* to **`mFrankel_ord × <the
  matching baseline score>`** (admission severity × the baseline of the very score whose change is
  predicted).  **Behavioral gate (the right check, not R²):** a max-admission-motor patient (at the
  neurological ceiling) must predict Δ≈0 and a high-headroom episode a large +Δ — verified (ΔUEMS
  at-ceiling +1.2 vs room +29.1; Δtotal-motor +2.2 vs +56.3), confirming the model learned the
  ceiling/room structure rather than a flat cohort mean.

* **Neurological-level descent (`models/level_descent.py`, G10)** — predicts the admission→discharge
  change in the five ISNCSCI neurological *levels* (NLI + bilateral motor + bilateral sensory) — the
  anatomical complement to G9's Δ summary *scores* (a level can descend without the score rising and
  vice-versa, so SCI trials report level conversion and score recovery separately).  Cord-level
  ordinal C1=0..S45=28 (smaller = more rostral = more severe); **Δ>0 = caudal descent = improvement.**
  **INT→`INT_ORD`=29 ceiling:** full recovery (raw `INT`, which the loader's `*_ord` maps to NaN) is
  lifted to ord 29 at discharge so a cure becomes the Δ *ceiling*, not a dropped row; the admission
  baseline reuses the loader `*_ord` (INT/missing→NaN), so already-intact admissions (no room to
  descend) drop from the cohort — only ord-0..28 admissions enter.  **Two heads per level (10 total):**
  a *calibrated binary* descent head P(Δ≥1) (no `class_weight`, Platt — cohort near-balanced ~57–61 %)
  + an *ordinal magnitude* head {0,+1,≥+2} (`MAG_CAP`=2, `class_weight="balanced"`, APS) — the same
  calibrated-binary-vs-balanced-magnitude non-comparability CRUX as G4 (surface the binary as the
  probability, the magnitude as APS set / argmax, never a competing probability).  Methodology + all
  binary plumbing reused **verbatim from `conversion.py`** (grouped-5fold-CV by IDNumber → OOF metrics
  + Platt + cross-conformal APS q; refit full cohort; descriptive in-sample SHAP); `MIN_COHORT`=120.
  Diagnostic + inference layer like conversion/topography: tracked identifier-free
  `models/level_descent_metrics.json` + git-ignored `models/level_descent/bundle.joblib`; **production
  `train.py` artifacts untouched (byte-repro verified)**.  Bundle shape documented inline at the top of
  `level_descent.py`.  **Findings — only the NLI is well-predicted from admission (AUC ≈0.73,
  calibration well below base Brier); bilateral motor & sensory levels sit at AUC ≈0.62–0.63, a
  genuine admission-signal ceiling** (a single level boundary's threshold crossing is intrinsically
  noisy, and sensory levels carry re-assessment noise), parallel to G8's "the aggregates forecast a
  *specific* target near-chance" finding; the magnitude heads are near-degenerate (APS set ≈3.0 — the
  conservative-discrete-K behaviour).  **INT→29 inflates the mean Δ (NLI mean +3.1 vs median +1) —
  report the median.**  **Enrichment rejected:** a G8-style concat of the modality-matched per-segment
  admission grades (20 motor / 112 sensory / 132 for NLI) left descent AUC flat-to-worse on every head
  (within OOF noise), so the clean 30-feature module ships (see §0b for the state-vs-threshold-crossing
  lesson).  **Part 1 shipped (s43): model + tracked metrics.  Part 2 (dashboard surfaces, all 5
  levels) USER-APPROVED, not yet built — see the §8 build spec.**

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
uv run python -m rehab_sci.models.topography      # recovery topography — 132 per-ISNCSCI-segment calibrated heads (G8); ~4 min
uv run python -m rehab_sci.models.level_descent   # neurological-level descent — 5 levels × (descent + magnitude) heads (G10); ~1 min
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

* **s43** — G10 neurological-level descent, **Part 1** (model + tracked metrics; user chose a
  standalone diagnostic module + the full 5-level profile + the INT→29 ceiling).  New
  `models/level_descent.py`: per ISNCSCI level (NLI + bilateral motor/sensory) a calibrated binary
  descent head P(Δ≥1) + a balanced ordinal magnitude head {0,+1,≥+2} + APS, all reusing
  `conversion.py`'s binary plumbing; INT→29 so full recoveries become the Δ ceiling not dropped
  rows; admission-INT (no room) excluded from each cohort.  Tracked `level_descent_metrics.json` +
  git-ignored bundle; production byte-repro verified (empty `training_metrics.json` diff).
  Findings: only NLI is well-predicted from admission (AUC ≈0.73); bilateral motor/sensory hit a
  genuine admission-signal ceiling (≈0.62–0.63); magnitude near-degenerate (APS≈3.0).  **Enrichment
  experiment (user: "enrich, then Part 2") — a G8-style per-segment admission-grade concat did NOT
  lift descent AUC (flat-to-worse, within noise) and was reverted** (the §0b state-vs-threshold-
  crossing lesson).  Lint + F-gate clean; MAP regenerated.  **Part 2 (dashboard surfaces)
  USER-APPROVED for all 5 levels** (build spec in §8); deferred to a fresh session per the G-series
  rhythm (60 % context at decision time).
* **s42** — G9 Δ score-recovery prediction (admission→discharge change in each ISNCSCI summary
  score: ΔUEMS/ΔLEMS/Δtotal-motor/Δlight-touch/Δpin-prick — the canonical SCI-trial primary
  endpoint; user chose the **production OUTCOMES-registry** integration over a standalone module).
  Two-line code surface: 5 `OutcomeSpec`s appended to `OUTCOMES` + 5 `y_delta_*` targets in
  `build_episode_frame` (discharge slot − the first-non-null admission feature the model already
  sees; no leakage).  `train.py` untouched (generic over `OutcomeSpec`; the Δ heads reuse the SCIM
  regression + Mondrian-conformal + SHAP machinery verbatim).  Δ-specific UI keyed on `clip_min<0`
  (signed point/PI, dropped "/ max" suffix, dotted "no change" zero-line on
  `fig_prediction_interval`).  Registry auto-extends the simulator/patient/insights/Methods selector
  with **zero new callbacks** (boot count unchanged at 39).  Findings: motor recovery more
  predictable than sensory; dominant SHAP interaction shifts to mFrankel × the matching baseline
  score; behavioral ceiling/room gate passes (at-ceiling Δ≈0 vs large-headroom large +Δ).  Full
  retrain: existing 6 outcomes byte-identical (`training_metrics.json` diff = the 5 added Δ blocks;
  `subgroups.json` additions-only; archetypes-centroid churn reverted).  Lint + F-gate clean; MAP
  regenerated; all 3 tabs render both langs; boots 200.  Scope boundary documented: temporal/landmark
  Methods panels gracefully skip the Δ heads (separate diagnostic bundles).
* **s41** — G8 recovery topography map, **Part 2** (dashboard surfaces across Methods + Patient +
  Simulator; user chose the *richest* atlas — anatomical dermatome **silhouette + motor myotome ladder** — and a
  *seeded Simulator worksheet*).  New `compute.predict_topography` (inline `_apply_platt` mirror over the 132 heads;
  degenerate→base_rate; missing adm_self→NaN) + `topography_admission_grades` (real per-segment admission exam via
  ADMISSION_FALLBACK LOCF) + `topography_observed_discharge` (patient overlay) + `topography_cohort_atlas` (Methods
  base-rate map) + `state.TOPOGRAPHY`/`TOPOGRAPHY_BUNDLE` loaders.  Shared `layout.fig_topography_bodymap` (humanoid
  silhouette from `_topo_body_shapes` SVG paths + `_DERMATOME_XY` markers + motor ladder, `theme.COLORSCALE_TOPOGRAPHY`,
  optional achieved/not-achieved rings) + `topography_readout`.  Methods figs `fig_topography_{calibration,scorecard,
  drivers}` + cohort-atlas reuse + per-segment drilldown reusing `fig_conversion_{reliability,shap}` via 2 new Methods
  `@callback`s.  Patient card (real exam→map+overlay, LT/PP toggle, not grade-gated) + Simulator card (editable 132-cell
  ISNCSCI worksheet seeded from the What-if `patient-ref` + Seed/Clear, worksheet adm_self + 30-field form → map).
  Visually iterated the body silhouette to a clean humanoid (PNG inspection via kaleido/Chrome).  Bilingual
  `topo_*`/`methods_topo_*`; `theme.{COLORSCALE_TOPOGRAPHY,PALETTE_TOPOGRAPHY_MODALITY}`; `.topo-worksheet*` CSS.
  Verified: behavioral personalization (real AIS-A motor P≈0.25 vs AIS-D ≈0.95); all 3 tabs render both langs; all 5
  new callbacks invoked directly; boots 200 with 39 callbacks (34→39); full lint clean; MAP regenerated; production
  artifacts untouched (dashboard-only diff).
* **s40** — G8 recovery topography map, **Part 1** (model + tracked metrics; user chose all-132
  segments, the P(≥3 motor antigravity)/P(≥1 sensory protective) milestones, and — after a
  diagnostic invalidated the originally-chosen joint low-rank model — **independent per-segment
  heads**).  New `models/topography.py`: 132 calibrated binary heads (20 motor + 56 LT + 56 PP),
  each LightGBM on the 30 admission features + the segment's OWN admission grade (`adm_self`, LOCF
  over the admission-fallback timepoints) + Platt calibration, reusing `conversion.py` plumbing
  verbatim.  **Diagnosed + abandoned the structured low-rank approach** (the 30 aggregates predict a
  *specific* segment near-chance, OOF AUC 0.43–0.63; the segment's own admission grade dominates →
  ~0.93; the discharge matrix is not low-rank) and **caught a co-misalignment bug** (per-segment
  matrices reindexed on `ep.index` instead of `ep["KeyRecordNumber"]` — kept AUC ~0.87 while
  scrambling the 30-feature context + CV groups; new §0b lesson + a behavioral personalization
  check that AUC alone misses).  12 high-cervical C2/C3 ceiling dermatomes flagged degenerate
  (honest constant base-rate, no model).  Tracked identifier-free `topography_metrics.json` +
  git-ignored bundle; production byte-repro verified (empty `training_metrics.json` diff).
  Findings: mean OOF AUC 0.94/0.93/0.91, calibration MAE ~0.01, strong personalization (own grade +
  LEMS/TotalMotor severity); the cohort antigravity count is injury-LEVEL- not AIS-driven
  (non-monotone in AIS).  Lint + F-gate clean; MAP regenerated.  Part 2 (body-map dashboard
  surfaces; needs per-segment admission inputs) deferred to the backlog.
* **s39** — G7 functional-independence profile, **Part 2** (dashboard surfaces across Methods +
  Patient + Simulator; user chose the Methods "Both" calibration layout — static all-18-heads
  overlay + the first interactive per-item drilldown — and the Patient achieved-vs-predicted
  overlay).  Pure `compute.predict_independence` (inline `_apply_platt` mirror, all 18 heads, NO
  admission-grade gating ⇒ no real-grade override on the patient card) + `independence_observed_for_episode`
  (realized discharge independence for the overlay) + `state.INDEPENDENCE`/`INDEPENDENCE_BUNDLE`
  loaders; bundle gained `discharge_timepoint`.  Shared `layout.fig_independence_profile`
  (domain-grouped P(indep) bars + base-rate diamonds + achieved/not-achieved markers) +
  `independence_readout` (expected-independent count + per-domain + most/least-likely).  Methods
  cohort figs `fig_independence_{scorecard,calibration,landscape,shap_heatmap}` + the per-item
  drilldown reusing `fig_conversion_{reliability,shap}` via the **first-ever Methods `@callback`**.
  `theme.PALETTE_INDEPENDENCE_DOMAIN`; item labels reuse `columns.yaml` (`col_label`); bilingual
  `ind_*`/`methods_ind_*`; `.ind-card` hook.  Documented the **AIS-alone-is-flat** invariant (§3)
  — TotalMotor dominates, so a grade-only Simulator profile is flat; the A→E rise
  (≈4.8→17.2 of 18) needs the correlated motor/sensory features.  Lint + F-gate clean; all 3 tabs
  render both langs; the 3 new callbacks invoked directly; boots 200 (34 callbacks); MAP
  regenerated.  No retrain — production + independence metrics byte-identical.
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

Propose from here unless the user redirects.  **Items F1–F25 + G1–G4 + G6 + G7 + G8 (recovery
topography map) + G9 (Δ score-recovery prediction, s42) are all fully shipped** — see §7 for the
session each landed in, and Git history for implementation detail.  **G10 neurological-level descent
Part 1 (model + metrics) shipped s43; its Part 2 (dashboard) is USER-APPROVED for all 5 levels (build
spec below) and is the next pick** (deferred from s43 to a fresh session).  Then F26 / F27 / a new G.
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
* **G7 functional-independence profile** (s38 model + tracked metrics; s39 dashboard surfaces):
  `models/independence.py` — 18 per-SCIM-III-item calibrated binary heads predicting
  P(functional independence at discharge) per ADL item, reframing the aggregate SCIM total as
  concrete acts ("will I feed myself / manage my bladder / walk?").  Independence = item score ≥ its
  functional-with-aids rubric threshold (aids/devices count); the 3 walking-mobility items use the
  ambulation (walking, ≥4) framing so wheelchair-independence does **not** count.  Borrows the
  conversion.py binary plumbing verbatim (`_oof_binary`/`_fit_platt`/`_apply_platt`/`_shap_top`,
  grouped-CV by IDNumber, Platt calibration).  Respiration excluded (near-universal).  No APS layer
  (binary APS is degenerate — see §0b).  `compute.predict_independence` (no admission-grade gating)
  + shared `layout` profile fig/readout + 4 Methods cohort figs + first-ever Methods drilldown
  callback + Methods/Patient/Simulator surfaces (`ind_*`/`methods_ind_*` strings).  See §3 for the
  model + dashboard contract (incl. the AIS-alone-is-flat invariant) and §7.  Fully shipped.
* **G8 recovery topography map** (s40 model + tracked metrics; s41 body-map dashboard): `models/topography.py` —
  132 independent calibrated per-ISNCSCI-segment binary heads predicting P(functional milestone at discharge): motor
  P(≥3 antigravity), sensory P(≥1 protective sensation).  The impairment/neurology complement to G7; mines the largest
  unused signal (the 132 per-segment columns).  **The user's first-choice joint low-rank structured model was
  diagnosed mismatched and abandoned** in favour of independent heads each fed the segment's OWN admission grade
  (`adm_self`) — the dominant predictor (the 30 aggregates predict a specific segment near-chance).  Reuses
  conversion.py binary plumbing; no APS (binary degenerate).  mean OOF AUC 0.94/0.93/0.91, calibration MAE ~0.01.
  Dashboard (s41): `compute.predict_topography` + an anatomical **dermatome silhouette + motor myotome ladder** body
  map (`layout.fig_topography_bodymap`), cohort base-rate atlas + per-modality calibration/scorecard/drivers + a
  per-segment drilldown (reusing `fig_conversion_{reliability,shap}`) on Methods, a real-exam patient card with an
  achieved-vs-predicted overlay, and a **seeded 132-cell ISNCSCI worksheet** on the Simulator (seeded from the What-if
  reference patient).  See §3 for the full model + dashboard contract (incl. the architecture CRUX, the
  level-not-AIS count caveat, the cohort-teal-vs-personalized CRUX) + the §0b KeyRecordNumber alignment lesson, and §7.
  Fully shipped.
* **G9 Δ score-recovery prediction** (s42 — pure production-registry extension, no new module or
  dashboard code): five `OutcomeSpec` regression heads predicting the admission→discharge change in
  each ISNCSCI summary score (ΔUEMS/ΔLEMS/Δtotal-motor/Δlight-touch/Δpin-prick), the canonical
  SCI-trial primary endpoint.  The five scores feed the model as admission inputs; G9 re-targets
  their *recovery*.  Only edits: 5 specs in `OUTCOMES` + 5 `y_delta_*` columns in
  `build_episode_frame`; `train.py` and every dashboard tab extend automatically via the registry
  (identity transform, `clip_min<0` allows deterioration + drives the signed/zero-line Δ UI, zero
  new callbacks).  See §3 for the full contract (registry-not-module, no-leakage Δ construction,
  negative-clip invariant, the temporal/landmark scope boundary, the
  motor-more-predictable-than-sensory + ceiling/room behavioral findings) and §7.  Fully shipped.
* **G10 neurological-level descent** (s43 Part 1: model + tracked metrics): `models/level_descent.py`
  — per ISNCSCI level (NLI + bilateral motor/sensory) a calibrated binary descent head P(Δ≥1) + a
  balanced ordinal magnitude head {0,+1,≥+2}+APS, reusing conversion.py's binary plumbing; INT→29
  ceiling; the anatomical complement to G9's Δ-scores.  Only NLI predicts well from admission (AUC
  ≈0.73); motor/sensory hit a genuine ≈0.63 ceiling.  A G8-style per-segment-grade enrichment was
  tried and **rejected** (no AUC lift — §0b).  See §3 for the contract.  **Part 2 (dashboard
  surfaces) USER-APPROVED for all 5 levels (s43) — the next pick.**
  **Part 2 build spec (all 5 levels, user-approved):** pure `compute.predict_level_descent(X_row)` —
  inline `_apply_platt` mirror over the bundle's 10 heads (never import `models.level_descent`, it
  pulls shap via conversion→train); per level return the calibrated descent prob + cohort `base_rate`
  and the magnitude class-probs / APS set / argmax; **gate per level on its admission `*_ord` being
  present** (admission-INT or missing ⇒ a "no room / needs level" prompt, mirroring the conversion-
  card invariant — and on the Patient tab override each `*_ord` with the episode's real admission
  value before inference, since `episode_row_for_model` imputes).  Reconstruct the INT-aware
  level→ord from `bundle.int_ord` + the loader cord order.  Shared `layout` figs: a 5-level
  calibrated descent-prob bar (+ base-rate diamonds), a magnitude set/argmax fig, a readout; reuse
  `fig_conversion_{reliability,shap,confusion}` for the per-level Methods drilldown.  Surfaces:
  Methods (per-level Δ-landscape + calibration/drivers + magnitude confusion), Patient (real-grade-
  gated card + own admission→discharge level overlay), Simulator (hypothetical from admission inputs,
  blanks stay NaN ⇒ natural prompt).  Bilingual `ld_*`/`methods_ld_*` strings; reuse `.conv-*` CSS.
  **No retrain** — the bundle already persists feature_cols/numeric/categorical/int_ord/mag_cap/
  levels/level_meta/heads.  Frame motor/sensory honestly (calibrated but low-discrimination, like
  G4's 0.62 `motor_incomplete`): surface the binary as the probability, the magnitude as the APS
  set, never as competing probabilities (the G4 CRUX).

**F23 (shipped s26): data-quality / clinical-consistency report** — see §7 and
`data/quality.py`; durable data facts it surfaced live in §0b/§1, and the
regenerated `models/dataquality_summary.json` holds the per-rule scorecard.

**Ready candidates (pick the next unless redirected):**
* **F26 invariant test harness** — narrow pytest enforcing §1 data + model
  invariants + a smoke test (incl. a headless `render_{methods,patient,simulator}` per the §0b
  lesson, which would have caught the s31 `INK["600"]` crash; and a topography-style row-alignment +
  behavioral-personalization check per the new §0b alignment lesson).  skip-if-CSV-absent.  M.
  files: `tests/`, pyproject.
* **F27 dependency refresh** — minor/patch bumps + raise the `shap<0.52` cap;
  retrain to verify byte-repro.  S, low value now (no CVEs, lint clean).
* **New G-series ideas — DATA IS EXHAUSTED OF NEW FIELD FAMILIES (s40 audit).**  An exhaustive
  audit (all 219 raw cols diffed vs the schema; the loader ingests only `ALL_SCIDATA.csv`, no
  external joins) confirmed the dataset is a pure SCI impairment+function registry: demographics +
  injury context + ISNCSCI/AIS + SCIM, plus thin extras (WISCI [sparse], ZPP, COMP_INCOMP [≈AIS],
  NonKeyMuscle [low-info]).  **Infeasible — the fields simply do not exist: disposition / mortality
  / competing-risks; calendar time-to-event / survival (no admission/discharge/injury/birth dates —
  only `LOS_days` duration + `BusinessYear`); complications / comorbidities; treatment-lever
  counterfactuals (zero clinician-modifiable variables); pain / QoL / psychosocial.**  Any new G
  must reuse the existing ISNCSCI/SCIM/AIS signal.  **The standard neurological-level endpoints are
  now covered: Δ-score recovery = G9, NLI/motor/sensory-level descent = G10** (admission predicts
  only the NLI level well — motor/sensory-level descent sits at a ≈0.63 ceiling).  Remaining untried
  reuse-ideas: ZPP (zone-of-partial-preservation) descent; calibration-drift monitoring (infra).
  Scope **what / why / effort / files / data dependency** before starting.
