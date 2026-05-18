# AGENT_NOTES.md — sticky knowledge for future sessions

Optimized for LLM ingestion: short bullets, no prose padding.  Append at the
bottom of each section as new lessons land; do **not** delete prior entries.

---

## 0. Read-first

* `CLAUDE.md` is the user's policy file.  Treat as authoritative.  Requires
  user approval to modify.
* `README.md` is the human-facing entry point.  Keep in sync with code.
* This file is the agent-facing scratchpad.  Always read it before planning.

## 1. Data invariants (do not rediscover)

* **Raw file** — `ALL_SCIDATA.csv` at repo root.  Never commit; gitignored.
* **Encoding** — `cp932` (Shift-JIS superset).  UTF-8 will silently mangle
  half the column names.
* **Missing sentinels** in raw file: `""`, `"_"`, `"NT"`, `"NA"`, `"ND"`.
  All mapped to `pd.NA` in `loader.py::_apply_missing_sentinels`.
* **Excel booleans** — many bool-like columns arrive as the literal strings
  `"FALSE"` / `"TRUE"` (note: uppercase).  Coerced to `0`/`1` then to `Y`/`N`
  via the schema's level mapping.
* **mFrankel/Frankel** — single combined raw column; split on slash into
  `mFrankel_ord` (5-grade A–E with substages) and `Frankel_ord` (5-grade).
* **Shape after clean** — long format: 31 200 rows × 219 cols, 1 200
  episodes (`KeyRecordNumber`), 867 patients (`IDNumber`).
* **Outcome cardinality** — only 498 episodes have a `discharge` SCIM total.
  The remaining 702 episodes are still useful for cohort visuals.

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
