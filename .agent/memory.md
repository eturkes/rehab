# Project memory

Cross-session context beyond code, `docs/`, and `git log`. Durable operating facts only.

## Gate honesty

- `uv run pytest` → **39 passed, ~22 s**. `uv run ruff check .` → clean under the full `pyproject.toml` rule set (`E,F,I,B,UP,SIM,RUF`), so the narrower `--select F` regression gate is a subset, not the ceiling.
- `tests/conftest.py` skips every data- and model-dependent test when `data/raw/ALL_SCIDATA.csv` or the `models/<head>/` joblib bundles are absent; only the pure-registry tests then run, and pytest still exits **0**. Read the counts, not the exit code — **39 passed** means the gate held, a green run dominated by skips proves nothing.
- Raw CSV + trained bundles are both present on this machine → gates are live here.

## Read cost

- Tracked metric JSONs are large: `models/subgroups.json` 624K, `models/topography_metrics.json` 316K, `models/landmark_metrics.json` 264K, `models/training_metrics.json` 100K. Pull keys with `jq`; a whole-file `Read` costs a large fraction of a context window.
- They are deliberately **absent** from `.claude/settings.json` `permissions.deny`: a `Read()` deny also blocks `jq`/`grep` naming the same path, which would remove the cheap access route along with the expensive one.

## Environments

- `.venv` = uv-managed CPython 3.13.5, the interpreter `uv run` resolves. `.venv-host` = separate CPython 3.12.13 install, unused by `uv run`. Both gitignored; keep both.
