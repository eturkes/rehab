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
  raw_profile.json                # column-level descriptive stats (no rows)

scripts/
  01_profile_raw.py               # rebuild schema/raw_profile.json from the CSV

src/rehab_sci/
  schema.py                       # Schema loader (by_raw, level_label, t, ...)
  data/loader.py                  # raw → normalized → ISNCSCI/SCIM summaries
  data/dataset.py                 # long → episode frame (1 row per KeyRecordNumber)
  models/train.py                 # LightGBM + split conformal PI + TreeSHAP
  models/subgroups.py             # Mann-Whitney / KW + Cliff's δ / d / η²
  dashboard/                      # Plotly Dash app (JA default, EN toggle)
    theme.py figures.py i18n.py app.py assets/style.css

models/                           # trained artifacts (gitignored)
reports/                          # exported figures (gitignored)
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

# 6. launch the dashboard at http://127.0.0.1:8050/
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

Outcomes:
| column | meaning |
|---|---|
| `y_discharge_scim` | SCIM-III total at the `discharge` timepoint (primary) |
| `y_discharge_ais`  | AIS ordinal at discharge |
| `y_discharge_wisci`| WISCI II at discharge |
| `y_max_scim`       | best SCIM during the stay |
| `y_last_scim` / `y_last_timepoint` | last observed SCIM + which timepoint |

### 3. Prediction model (`models/train.py`)

- **Estimator:** LightGBM (median + 10% / 90% quantile heads).
- **Group split by patient** (`IDNumber`) prevents leakage between
  multiple episodes of the same patient.
- **5-fold GroupKFold** CV for the median model with early stopping on
  an inner validation slice.
- **Split conformal prediction**: 20% of the dev set is held out as a
  calibration fold; the 80% PI half-width is the 0.80-quantile of the
  absolute residuals on that fold, then clipped to `[0, 100]`.  This
  produces a *marginal* 80%-coverage guarantee even when the quantile
  heads themselves overfit.
- **TreeSHAP** values are cached on the held-out test set for the
  dashboard's Insight Engine.

Current scores on this dataset (n_episodes = 498, n_patients = 498,
random_state = 20260518):

| split | R² | RMSE | MAE | coverage(80%) |
|---|---|---|---|---|
| 5-fold CV (median) | 0.652 ± 0.054 | 20.79 ± 1.86 | 14.93 ± 1.97 | — |
| held-out test      | 0.696         | 18.92         | 13.70         | 83% (conformal) |

### 4. Subgroup discovery (`models/subgroups.py`)

For every admission feature, the dataset is split into 2 (binary) or 4
(quartile / multi-level) subgroups and the discharge SCIM is compared
across them:

- Mann-Whitney U + Cliff's δ + Cohen's d (binary)
- Kruskal-Wallis H + η² (multi-level)
- Holm and Benjamini-Hochberg p-adjustment

Results land in `models/subgroups.json`, which the dashboard reads
on startup.

### 5. Dashboard (`dashboard/app.py`)

Tabs:

1. **Cohort overview** — KPIs (n, age, AIS-A%, tetra%, mean discharge
   SCIM); age / sex / mechanism distributions; injury sunburst
   (paralysis → AIS → NLI); AIS admit→discharge Sankey; SCIM recovery
   curves with IQR ribbons by paralysis type.
2. **Patient simulator** — every admission feature is exposed as an
   input.  Live point prediction + 80% conformal interval + local
   SHAP bar chart (top-10 contributors for the current input).
3. **Patient explorer** — pick a real patient by IDNumber and see
   their observed SCIM-III trajectory (total + subscales) against
   cohort 10–90 / 25–75 percentile bands stratified by paralysis ±
   AIS; a longitudinal ISNCSCI / AIS table; the model's predicted
   discharge SCIM with 80 % PI, observed value (if recorded), and
   the local SHAP contributions for this episode.
4. **Insight engine** — global SHAP importance; per-feature subgroup
   box plot with effect-size annotation; SHAP dependence plot.
5. **Methods** — model card with population, target, training protocol,
   metrics, limitations.

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
