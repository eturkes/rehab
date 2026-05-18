# AGENT_NOTES.md — sticky knowledge for future sessions

Optimized for LLM ingestion: short bullets, no prose padding.  Append at the
bottom of each section as new lessons land; do **not** delete prior entries.

---

## 0. Read-first

* `CLAUDE.md` is the user's policy file.  Treat as authoritative.  Requires
  user approval to modify.
* `README.md` is the human-facing entry point.  Keep in sync with code.
* This file is the agent-facing scratchpad.  Always read it before planning.
* **Default-work pool for fresh sessions: §8 Feature backlog (F1–F3).**
  Propose work from §8 unless the user redirects.  Historical "Open items
  rolled forward" lists inside prior §7 session entries are **superseded**
  by user decision in session 4 — treat them as history, not as a to-do.

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
* **80% conformal interval** = (1−α)-quantile of `|y − ŷ|` on a held-out
  calibration fold, then clipped to `[0, 100]`.  Coverage on n=100 test
  is currently 0.83.  LightGBM quantile heads alone give ~0.41 coverage
  on this dataset — *do not remove the conformal layer*.
* **TreeSHAP** is run on the held-out test set only, never the training
  set (would be optimistic).  Cached in `shap_test.joblib`.
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

### F2. Multi-outcome prediction (subscales + AIS + LOS; WISCI dropped)

* **What:** Today `train.py` trains only on `y_discharge_scim`.  Extend
  the outcome set to:
  - Per-subscale SCIM: self-care (0–20), respiration / sphincter (0–40),
    mobility (0–40).
  - `y_discharge_ais` — ordinal classification (LightGBM `multiclass`
    or an ordinal-logit head).
  - `LOS_days` — regression; the column is already on the episode
    frame (sourced from `入院期間`), n=682 / 899 episodes post-ghost-filter.
  - ~~`y_discharge_wisci` — regression.~~ **DROPPED (session 6):**
    only 50 episodes have a discharge WISCI, below any reasonable
    regression-power threshold.  Reconsider as a walker-vs-non-walker
    classification proxy if requested.
* **Why:** Each subscale is independently actionable for clinical goal
  setting; AIS is the headline ordinal outcome in SCI literature.
* **Effort:** medium.  Mostly a refactor of `train.py` to loop over a
  list of outcome specs and persist a dict-of-models.  Dashboard
  simulator gains an outcome selector.
* **Files:** `models/train.py`, `models/__init__.py`, `dashboard/app.py`
  (simulator outcome selector + PI/SHAP rendering), `dashboard/figures.py`.
* **Data dependency:** all four remaining outcomes (`y_discharge_scim`,
  3 subscales already computable via `SCIM_self_care` /
  `SCIM_respiration_sphincter` / `SCIM_mobility` at the discharge
  timepoint — needs the same `discharge.set_index(KR)` trick used for
  the total — plus `y_discharge_ais` and `LOS_days`) are on the
  episode frame already.

### F3. Mondrian per-AIS / per-paralysis conformal calibration

* **What:** Replace the single marginal conformal quantile with a
  Mondrian taxonomy — separate calibration per AIS grade (A/B/C/D/E)
  and per paralysis class (TETRA / PARA / NONE).  PI half-width is
  chosen by the test patient's group at inference time.
* **Why:** AIS-A and AIS-D residual distributions differ substantially;
  the marginal 83 % coverage hides per-grade under/over-coverage.
  Per-group conformal restores a per-group guarantee.
* **Effort:** small (~½ session).
* **Files:** `models/train.py` (calibration block + persisted artifact —
  store a dict of half-widths keyed by `(ais, paralysis)` with a
  marginal fallback for empty cells), `dashboard/app.py` (simulator
  PI lookup uses the simulated patient's group),
  `models/training_metrics.json` (per-group coverage report for the
  Methods tab).
* **Data dependency:** AIS / paralysis columns already in episode frame.
