# Project memory

Cross-session context beyond code, `docs/`, and `git log`. Durable operating facts only.

## Gate honesty

- `uv run pytest` → **56 passed, ~19 s**. `uv run ruff check .` → clean under the full `pyproject.toml` rule set (`E,F,I,B,UP,SIM,RUF`), so the narrower `--select F` regression gate is a subset, not the ceiling.
- `tests/conftest.py` skips every data- and model-dependent test when `data/raw/ALL_SCIDATA.csv` or the `models/<head>/` joblib bundles are absent; only the pure-registry tests then run, and pytest still exits **0**. Read the counts, not the exit code — **56 passed** means the gate held, a green run dominated by skips proves nothing.
- Raw CSV + trained bundles are both present on this machine → gates are live here.

## Worktree gates

Mechanics = global `CLAUDE.md` `Subagents`; `/session-roadmap` binds the path (`.scratch/`, gitignored). Project delta: a worktree carries tracked content only — no `data/raw`, no `models/<head>/` bundles — so its suite skips green (see Gate honesty). Link the gitignored artifacts, then gate off the primary environment:

```sh
P=<primary root>; cd "$P/.scratch/worktrees/<name>"
mkdir -p data && ln -sfn "$P/data/raw" data/raw
for d in "$P"/models/*/; do ln -sfn "$d" "models/$(basename "$d")"; done
ln -sfn "$P/models/feature_spec.joblib" models/feature_spec.joblib
PYTHONPATH="$PWD/src" "$P/.venv/bin/python" -m pytest    # 56 passed ⇒ links live
"$P/.venv/bin/ruff" check .
```

- `PYTHONPATH="$PWD/src"` is load-bearing: the primary `.venv` holds an editable `.pth` appending the **primary** `src`, and `PYTHONPATH` precedes it. Without it a worktree run gates primary code. Assert per run that `rehab_sci.__file__` resolves under the worktree.
- Artifact paths track the imported package (`RAW_PATH_DEFAULT` = `parents[3]/data/raw/…`), so the links belong in the worktree.
- Absolute primary-interpreter calls only. `uv run` inside a worktree builds a second environment; under this recipe the primary environment stays read-only.
- `.pytest_cache` + `.ruff_cache` land worktree-private (rootdir = worktree) → no shared-cache holder needed.
- Baseline restore strips the links, silently returning the gate to skips → an inherited or restored worktree re-runs the link block verbatim (idempotent) before gating.
- Links are read paths **into** the primary tree: a write landing on one (`models/<head>/bundle.joblib`, `models/feature_spec.joblib`, `data/raw/*`) mutates the primary tree, so any retrain/regeneration unit takes real copies or a private output dir. Tracked `models/*.json` are real worktree files and stay isolated.
- Teardown (Close order's worktree-removal step): `git worktree remove --force` — plain `remove` aborts rc=128 (`contains modified or untracked files`) because the links + private caches are untracked; `--force` unlinks the links without traversing them, so primary `data/raw` + `models/` survive intact. Then `git branch -D wt/<name>`: `-d` refuses rc=1 `not fully merged` for every checkpointed branch, `prod` squash-harvests included, since a squash leaves no ancestry link. `git worktree list` + `git branch --list 'wt/*'` printing no `wt/` row = teardown proven.

## Denominator semantics

- **Every headline-finding cohort requires a real `IDNumber`** — `independence.py` (`score.notna() & IDNumber.notna()`), `multistate.py` improve head, `landmark.py` (`dropna([target, IDNumber])`). The 27 partial-id orphans are therefore **absent** from the ladder / improve / measure / certainty denominators, and post-merge those counts are exactly distinct patients. **`subgroups.py` does NOT filter `IDNumber`** → the gradient panel (n=482) carries 9 id-less rows and is the one place the id-less caveat belongs. Bind any such caveat to a live count, never a literal: the universe-wide 27 is the wrong number for every individual finding.
- `data/quality.py` reads the raw CSV through `Context.build`, **upstream of both the ghost filter and the duplicate-registration merge** → its `n_episodes` is the 1,200-record register, not the 893-episode universe. Its labels say *records* for that reason; the count disagreeing with the dashboard is correct, not drift.
- Full artifact regeneration = the §6 cheat-sheet order run start to finish (`train → subgroups → archetypes → quality → temporal → landmark → phenotypes → conversion → multistate → independence → topography → level_descent → dissociation`), ~75 min on a loaded box, `landmark` alone ~35 min. `archetypes` must follow `train` (it rewrites `training_metrics.json` last).
- **New dashboard copy can break the font gate.** `tests/test_font_coverage.py` fails on any character no shipped subset draws (two new JA glyphs did). Fix = `uv run python scripts/02_build_fonts.py`, which rewrites the four `ibm-plex-sans-jp-*.woff2`; rerun pytest.

## Read cost

- Tracked metric JSONs are large: `models/subgroups.json` 624K, `models/topography_metrics.json` 316K, `models/landmark_metrics.json` 264K, `models/training_metrics.json` 100K. Pull keys with `jq`; a whole-file `Read` costs a large fraction of a context window.
- They are deliberately **absent** from `.claude/settings.json` `permissions.deny`: a `Read()` deny also blocks `jq`/`grep` naming the same path, which would remove the cheap access route along with the expensive one.

## Environments

- `.venv` = uv-managed CPython 3.13.5, the interpreter `uv run` resolves. `.venv-host` = separate CPython 3.12.13 install, unused by `uv run`. Both gitignored; keep both.

## AIS improvement semantics

- **Two cohorts, both live, easily confused.** `_landscape` (n=714) takes any admission grade, no `IDNumber` requirement. `_improve_cohort` (n=686) = admitted A–D + real `IDNumber` + ≥2 in-window obs — **this is finding 02's denominator and the improve head's**. Both persist to `multistate_metrics.json`; building a finding off `landscape.by_admission_grade` silently mismatches the block metric by ~10 episodes + adds an E row.
- Improvement target = `wmax > adm` — window **maximum**, so ≥1 grade (not exactly 1), transient crossings count permanently, and `improved`/`declined` are computed independently against admission (a fluctuating episode is in both; only `stable` is the complement of both).
- `flow` block = `best`/`last`/`peak_to_last`/`adm_peak_last` matrices + per-grade `rate_best`/`rate_last`/`reverted_of_peaked`. Peak→final is **non-ascending by construction** (max includes the last obs) → decline surfaces in `last` alone, never in `best`.
- Reversion is wildly asymmetric: adm A 34.1% of crossings do not hold vs D 2.3%; 13 episodes (1.9%) end below admission. **Never gloss this as lost recovery** — the A→B boundary turns on one sacral finding, the measurement most exposed to inter-rater disagreement, so instability and exam disagreement are not separable here. `basis` carries that caveat.

## Visual QA

- Dashboard UI claims need a rendered check — the Dash tree renders per-callback, so a Python-side assert says nothing about type scale, wrap or resolved face. Harness = Playwright driving the chromiumfish binary (`$(chromiumfish path)`, `--no-sandbox --disable-gpu`), no browser download: `uv run --no-project --with playwright python .scratch/uiqa/shot.py <outdir> <tag> [--lang en|ja] [--tab <t>] [--w --h] [--full] [--wait ms] [--sel CSS ...]`. Scratch-local + gitignored → port pending (`.agent/polish.md`).
- Tab switch = `.dash-tab` locator by label text; Dash tabs carry no ARIA `tab` role, so `get_by_role("tab")` never matches.
- Callback-rendered tabs (patient, simulator) need `--wait 15000`+: at 3.5 s they screenshot as blank axes and read exactly like a broken figure. The dashboard log shows `POST /_dash-update-component 200` either way.
- Resolved-face proof (never eyeball it): CDP `CSS.getPlatformFontsForNode` returns the real family + glyph count + `isCustomFont` — `IBM Plex Sans SmBld` vs `Liberation Serif` distinguishes a loaded webfont from a fontconfig substitution, and the returned weight name proves no synthetic bolding. Sweep `body *` per tab × lang and bucket on `isCustomFont`, never a hand-picked selector list: the two surviving fallbacks (Dash stepper buttons, our own `.sim-action-btn`/`.whatif-btn`) were on elements no selector list would have thought to include. Zero-host-face run ≈ 37k glyphs across 5 tabs × 2 langs.
- Static font coverage is a *test* (`tests/test_font_coverage.py`), not a QA step — it needs no data, runs on a bare checkout, and fails on new copy whose characters no shipped face draws. A render pass cannot replace it: substitution is silent and per-host.
- `pgrep -f 'rehab_sci[.]dashboard' | while read -r p; do kill "$p"; done` stops the server (bracketed pattern avoids self-match); the call can still exit 144 under the harness while the kill lands — verify with `curl -s -o /dev/null -w '%{http_code}'` rather than trusting rc.
- JA line breaking: `word-break: auto-phrase` (`assets/style.css`, `.finding-evidence__copy h3`) is inert unless the content language is declared, so `lang=lang` on the rendered container is load-bearing (`tabs/overview.py::_finding_block`); `<html lang>` stays empty because the toggle is client-side. Chromium 149 supports it (`CSS.supports('word-break','auto-phrase')`); without it JA wraps per character and splits katakana + numeral/counter pairs (セルフ|ケア, 1|段階). Assets are fingerprinted at app start → restart the server before re-shooting a CSS change, or the capture shows the old sheet. Only the four overview finding blocks + the glance band carry the pair so far.
