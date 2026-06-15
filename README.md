# Rehabilitation Analytics & Prediction Suite (SCI)

Bilingual (JA / EN) analytics + prediction pipeline for a spinal-cord-injury (SCI)
rehabilitation center.  All clinical content (column labels, factor levels, UI
strings) is data-driven from `schema/*.yaml`, so the same code base renders
identical results in Japanese and English.

> **Data privacy.**  `ALL_SCIDATA.csv` and every derived artifact (parquet,
> joblib, figures, etc.) are explicitly gitignored.  The repository only ships
> code, schema, and aggregated descriptive statistics — *no* row-level patient
> data.

---

## Repository layout

```
ALL_SCIDATA.csv                   # raw input (NOT committed; provided locally)
pyproject.toml                    # uv project, pinned deps
uv.lock                           # reproducible resolution

schema/                           # bilingual schema (committed)
  columns.yaml                    # 219 columns × {ja,en,group,role,dtype,range,levels}
  categorical_levels.yaml         # factor levels (raw -> display + ja/en labels)
  isncsci.yaml                    # ISNCSCI scoring metadata
  scim_iii.yaml                   # SCIM-III subscale composition
  ui_strings.yaml                 # all dashboard strings (ja/en)
  raw_profile.json                # column-level descriptive stats; git-ignored, regen via scripts/01_profile_raw.py

scripts/
  01_profile_raw.py               # rebuild schema/raw_profile.json from the CSV

src/rehab_sci/
  schema.py                       # Schema loader (by_raw, level_label, t, ...)
  data/loader.py                  # raw → normalized → ISNCSCI/SCIM summaries
  data/dataset.py                 # long → episode frame (1 row per KeyRecordNumber)
  models/train.py                 # LightGBM + split conformal PI + TreeSHAP
  models/temporal.py              # out-of-time rolling-origin validation (drift)
  models/landmark.py              # landmark (dynamic) prediction — value of observation
  models/subgroups.py             # Mann-Whitney / KW + Cliff's δ / d / η²
  data/quality.py                 # data-quality / clinical-consistency report
  dashboard/                      # Plotly Dash app (JA default, EN toggle)
    app.py state.py compute.py layout.py figures/ tabs/ theme.py i18n.py
    reliability.py                # partial-input completeness + OOD assessment

models/                           # trained artifacts (gitignored)
reports/                          # exported figures (gitignored)
MAP.md                            # generated code index — uv run python scripts/gen_map.py
```

## Single-command setup

```bash
# 1. install deps in a local venv (pinned by uv.lock)
uv sync

# 2. drop the raw CSV at the repo root
cp /path/to/ALL_SCIDATA.csv .

# 3. (optional) regenerate descriptive schema profile
uv run python scripts/01_profile_raw.py

# 4. train the prediction model (writes models/*.joblib + metrics.json)
uv run python -m rehab_sci.models.train

# 5. run subgroup discovery (writes models/subgroups.json)
uv run python -m rehab_sci.models.subgroups

# 6. (optional) run the data-quality / clinical-consistency report
#    writes models/dataquality_summary.json (tracked) + dataquality_report.json (git-ignored)
uv run python -m rehab_sci.data.quality

# 7. (optional) run out-of-time temporal validation (writes models/temporal_metrics.json)
uv run python -m rehab_sci.models.temporal

# 8. (optional) run landmark (dynamic) prediction — value of observed early recovery
#    writes models/landmark_metrics.json (tracked) + models/landmark/bundle.joblib (git-ignored)
uv run python -m rehab_sci.models.landmark

# 9. launch the dashboard at http://127.0.0.1:8050/
uv run python -m rehab_sci.dashboard.app
```

## Pipeline overview

### 1. Schema-driven cleaning (`data/loader.py`)

- Reads the raw file as **cp932** (Shift-JIS superset).
- Normalizes the `_` / `NT` / `NA` / `ND` missing sentinels.
- Coerces Excel-mangled booleans (`"FALSE"`, `"TRUE"`) and Japanese yes/no
  (`有`/`無`) to canonical Y/N.
- Splits the combined `mFrankel_Frankel` column into `mFrankel_ord` +
  `Frankel_ord`.
- Adds the ISNCSCI summaries — `UEMS`, `LEMS`, `TotalMotor`,
  `LightTouchTotal`, `PinPrickTotal` — by summing the 5×5×2 motor cells
  and 28×2×2 sensory cells.
- Adds the SCIM-III subscale totals — `SCIM_self_care` (0–20),
  `SCIM_respiration_sphincter` (0–40), `SCIM_mobility` (0–40),
  `SCIM_total` (0–100).

### 2. Episode frame (`data/dataset.py`)

The unit of analysis is `KeyRecordNumber`.  The raw CSV ships a perfect
1 200 × 26 grid (1 200 episodes × 26 timepoint slots), but 301 of those
episodes are pure placeholder rows — `IDNumber` is null *and* every
admission feature is null *and* every outcome is null — so
`build_analysis_dataset()` filters them at load time.  The analysis
universe is therefore **899 episodes / 866 unique patients** (with 27
partial-data orphans that have admission features but no `IDNumber`, so
they cannot enter a group-by-patient training split but still contribute
to cohort-level visuals).

Admission features are taken from the `0day` timepoint with backfill from
`72h → 2w → 4w` (first-non-null wins) so that patients with a missing
day-0 baseline are not silently dropped.

Outcomes (predicted heads in **bold**; auxiliary columns below):
| column | meaning | n | task |
|---|---|---|---|
| **`y_discharge_scim`** | SCIM-III total at discharge | 507 | regression |
| **`y_discharge_scim_self_care`** | SCIM-III self-care subscale (0–20) | 507 | regression |
| **`y_discharge_scim_resp_sphincter`** | SCIM-III respiration / sphincter (0–40) | 507 | regression |
| **`y_discharge_scim_mobility`** | SCIM-III mobility subscale (0–40) | 507 | regression |
| **`y_discharge_ais`**  | AIS grade at discharge (A → E) | 638 | multiclass |
| **`LOS_days`** | Length of stay in days (log1p-modelled) | 682 | regression |
| `y_discharge_wisci`| WISCI II at discharge | 50 | *dropped — too sparse* |
| `y_max_scim`       | best SCIM during the stay | — | auxiliary |
| `y_last_scim` / `y_last_timepoint` | last observed SCIM + which timepoint | — | auxiliary |

### 3. Prediction models (`models/train.py` + `models/outcomes.py`)

The outcome registry in `models/outcomes.py` declares the six prediction
heads.  `train.py` iterates that registry; each head writes its own
bundle under `models/{spec.key}/` (median model + p10 / p90 quantile
heads + `feature_spec.joblib` + `shap_test.joblib` for regression heads,
or `lgbm_multiclass.joblib` + the same support files for the AIS head).

- **Estimator:** LightGBM regression (median + 10 % / 90 % quantile
  heads) for the five regression heads; LightGBM multiclass with
  `class_weight="balanced"` for AIS.
- **Group split by patient** (`IDNumber`) prevents leakage between
  multiple episodes of the same patient.
- **5-fold GroupKFold CV** for every head, reporting metrics on the
  human-interpretable scale (back-transformed for LOS).
- **Split conformal prediction** (regression only): 20 % of the dev
  set is held out as a calibration fold; the 80 % PI half-width is the
  0.80-quantile of `|y − ŷ|` on the modelling scale (log1p for LOS,
  identity otherwise).  Bounds are back-transformed and clipped to the
  outcome's range.  This yields a *marginal* 80 % coverage guarantee
  even when the LightGBM quantile heads overfit on small calibration
  folds.
- **LOS log-transform:** length of stay is fitted on `log1p(y_days)` so
  the conformal residuals are symmetric on the modelling scale; the
  reported R² / RMSE / MAE / PI are all in raw days.
- **AIS ordinal-aware metrics:** alongside accuracy we report
  quadratic-weighted Cohen κ and MAE on the ordinal code 1–5 so we
  monitor *how far off* misclassifications are — not just how often.
  AIS also gets **APS conformal classification sets** (Mondrian per-AIS /
  per-paralysis `q_hat`), so the simulator can show a calibrated set of
  plausible grades rather than a single point.
- **TreeSHAP** values are cached on the held-out test set.  Multiclass
  AIS stores a 3-D `(n, p, K=5)` tensor; the simulator surfaces SHAP
  for the *predicted* class.

Per-head test-split metrics — `n_train` / `n_calib` / `n_test`, R² / RMSE /
MAE for the regression heads, accuracy + quadratic-weighted κ for AIS, and
conformal coverage — are written to `models/training_metrics.json` at train
time (the dashboard reads the same file).  That JSON is the single source of
truth for scores, so they cannot drift from this README.

### 4. Subgroup discovery (`models/subgroups.py`)

For every admission feature, the dataset is split into 2 (binary) or 4
(quartile / multi-level) subgroups and the discharge SCIM is compared
across them:

- Mann-Whitney U + Cliff's δ + Cohen's d (binary)
- Kruskal-Wallis H + η² (multi-level)
- Holm and Benjamini-Hochberg p-adjustment

Results land in `models/subgroups.json`, which the dashboard reads
on startup.

### 5. Data-quality report (`data/quality.py`)

The loader is deliberately defensive — it coerces out-of-range numbers and
unparseable tokens to `NaN` and retains unmapped categorical levels — which
keeps the dashboard robust but hides data-entry errors.  This report re-reads
the **raw** frame and runs a declarative rule engine that surfaces them:

- **Domain** — values the loader dropped: numbers outside the schema `range`
  (e.g. a SCIM item above its max), non-numeric tokens in numeric fields (e.g.
  the asterisk-annotated `1*` ISNCSCI scores, a malformed `IDNumber`), and
  categorical values matching no canonical level / `raw_alias` (a
  schema-coverage check — e.g. `ALLEN分類` full-width Roman numerals).
- **Cross-field** — per-assessment clinical consistency grounded in the
  ISNCSCI / AIS / SCIM definitions: sacral sparing ↔ AIS completeness,
  VAC ↔ AIS, complete/incomplete ↔ AIS, AIS-E ↔ maximal scores,
  paraplegia/tetraplegia ↔ NLI region, NLI ↔ sensory/motor levels,
  mFrankel ↔ AIS.
- **Longitudinal** — per episode across its timepoints: AIS deterioration,
  large SCIM-total drops, implausible NLI drift.

Each rule fires only when every field it needs is present and the values
contradict, so missing data is never a violation.  Two artifacts are written:
an aggregate `models/dataquality_summary.json` (per-rule counts, no
identifiers — **tracked**) and a detailed `models/dataquality_report.json`
(row-level findings with `KeyRecordNumber` + offending values — **git-ignored**,
never committed).  Findings are for data review only and never feed model
training; the Methods tab surfaces the aggregate scorecard.

### 6. Temporal validation (`models/temporal.py`)

The production heads are scored with a random GroupKFold split, which is blind
to calendar time.  This optional out-of-time backtest measures the drift that
split hides: for each test year `T` in 2020–2025 it trains on every episode with
`BusinessYear < T` and tests on year `T` (expanding-window rolling origin,
group-safe by patient).  It mirrors the production methodology, except the
dev/test cut is temporal and the conformal / APS calibration is marginal.  Per
outcome and origin it records point accuracy (R²/RMSE/MAE, or accuracy /
quadratic-κ / ordinal-MAE for AIS) and out-of-time 80 % PI / APS coverage,
alongside the in-time baseline echoed from `training_metrics.json`.  Results land
in the tracked `models/temporal_metrics.json`; the Methods tab plots each
outcome's drift curve.  It is a diagnostic — no artifact the dashboard loads for
prediction is modified.  (`LOS_days` is right-censored for recent years —
unlabelled from 2024 — so its backtest covers fewer origins.)

### 7. Landmark (dynamic) prediction (`models/landmark.py`)

The production heads predict the discharge outcome from admission data alone.
This optional layer asks a complementary question: **as a patient's early
recovery is actually observed, how much does the discharge prognosis sharpen?**
For each landmark time `L` ∈ {72 h, 2 w, 4 w, 6 w, 2 m, 3 m} it fits one model
per outcome from the admission features **plus** the last value observed on or
before `L` (LOCF) for ten recovery-tracking measures (SCIM total + subscales,
AIS grade, motor / sensory totals).  Eligibility is the **still-admitted risk
set** at `L` (an intermediate observation at or after `L`), so the model never
"predicts" an already-discharged patient (the immortal-time / leakage trap).
Because that risk set shrinks as `L` advances, a paired **admission-only
baseline** is fit on the same cohort and split — the landmark-minus-baseline gap
is the net *value of observation*, controlling for the cohort shift.  Reusing the
production prep / conformal / APS helpers (marginal calibration), it records per
outcome and landmark the point accuracy and the prediction-interval half-width /
APS set size for both models.  Results land in the tracked
`models/landmark_metrics.json` (curves) and `models/landmark/bundle.joblib`
(paired baseline + landmark heads per outcome × landmark, git-ignored).  Three
surfaces consume it: the **Methods** tab plots each outcome's
value-of-observation curve; the **Simulator** offers a what-if card (pick a
landmark, type hypothetical early-recovery scores, watch the interval tighten);
and the **Patient** tab makes it concrete — at any landmark the patient is still
admitted for, their own observed scores sharpen the admission-only prognosis
live.  It is a diagnostic + inference layer — no artifact the production pipeline
writes is modified.  Finding: for SCIM the gain grows
from ΔR² ≈ +0.04 (72 h) to **+0.30 (3 m)** while the interval half-width nearly
halves; AIS quadratic-κ climbs to ≈0.88; LOS stays hard but still improves.

### 8. Dashboard (`dashboard/app.py`)

Tabs:

1. **Cohort overview** — KPIs (n, age, AIS-A%, tetra%, mean discharge
   SCIM); age / sex / mechanism distributions; injury sunburst
   (paralysis → AIS → NLI); AIS admit→discharge Sankey; SCIM recovery
   curves with IQR ribbons by paralysis type.
2. **Patient simulator** — every admission feature is exposed as a
   clearable input; an outcome dropdown selects which of the six
   prediction heads to render.  Regression heads (SCIM total / 3
   subscales / LOS) show a live point prediction + 80 % conformal
   interval; the AIS head shows the predicted class + a
   class-probability bar chart.  A local SHAP bar (top contributors
   for the current input) accompanies whichever head is selected.
   **Partial input (F25):** the form opens blank — supply only what
   is known and blanks pass through as NaN (LightGBM's native missing
   handling, matching training).  A reliability badge reports
   importance-weighted input completeness plus an out-of-distribution
   flag (values outside the observed range / atypical), since the
   conformal half-width is fixed and will not itself widen as input
   grows sparser.  *Fill cohort defaults* / *Clear all* buttons toggle
   between a representative patient and a blank form.
   **Dynamic prediction (G1):** a landmark card lets you pick an
   observation time (72 h … 3 m) and type hypothetical early-recovery
   scores, then watch the admission-only interval sharpen against the
   landmark prediction — the value-of-observation curve, made live.
3. **Patient explorer** — pick a real patient by IDNumber and see
   their observed SCIM-III trajectory (total + subscales) against
   cohort 10–90 / 25–75 percentile bands stratified by paralysis ±
   AIS; a longitudinal ISNCSCI / AIS table; the model's predicted
   discharge SCIM with 80 % PI, observed value (if recorded), and
   the local SHAP contributions for this episode.
   **Dynamic prediction (G1):** for any landmark this episode is still
   admitted for, the patient's own observed early-recovery scores
   refine the admission-only prognosis, shown side by side.
4. **Insight engine** — global SHAP importance; per-feature subgroup
   box plot with effect-size annotation; SHAP dependence plot.
5. **Methods** — model card with population, target, training protocol,
   metrics, limitations; per-outcome temporal-drift curves (out-of-time
   accuracy + interval coverage vs the random-split baseline);
   per-outcome landmark value-of-observation curves (accuracy + interval
   half-width vs landmark time, landmark model vs admission-only baseline);
   plus a data-quality scorecard (counts of flagged anomalies by category
   and severity).

The top-right `日本語 / English` toggle drives a single `dcc.Store`;
every label, axis, hover, tab title, and option list rerenders from the
schema's `ja` / `en` fields.

## Reproducibility

- **Python:** `>=3.12,<3.14` (CI tested on 3.13).
- **uv** resolves and locks all dependencies; `uv sync` is the single
  entry point and is deterministic given `uv.lock`.
- **Random state:** `20260518`, embedded in `training_metrics.json`.
- All artifacts under `models/` are gitignored but reproducible from
  the committed code + the raw CSV.

## License

Internal research code; no public license at this time.
