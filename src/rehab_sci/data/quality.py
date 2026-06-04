"""Data-quality / clinical-consistency report over the SCI dataset.

The loader (``data/loader.py``) is intentionally *defensive*: it coerces every
out-of-range number and unparseable token to ``NaN`` and silently retains
unmapped categorical levels.  That keeps the dashboard robust, but it also hides
data-entry errors from view.  This module surfaces them, running three classes
of check over the raw + cleaned long frame:

* **DOMAIN** — re-parse each raw cell against the schema and report values the
  loader dropped: numbers outside the declared ``range`` (``D-NUM-RANGE``),
  non-numeric tokens in numeric columns (``D-NUM-PARSE``), and categorical
  values that match no canonical level / ``raw_alias`` (``D-CAT-LEVEL`` — this
  doubles as a schema-coverage check).
* **CROSS-FIELD** — per-assessment clinical consistency grounded in the ISNCSCI
  / AIS / SCIM definitions (sacral sparing ↔ AIS completeness, VAC ↔ AIS,
  complete/incomplete ↔ AIS, AIS-E ↔ maximal scores, paraplegia/tetraplegia ↔
  NLI region, NLI ↔ sensory/motor levels, mFrankel ↔ AIS, auto ↔ manual).
* **LONGITUDINAL** — per episode (``KeyRecordNumber``) across its timepoints:
  AIS deterioration, large SCIM drops, NLI drift.

Guiding contract: **a rule fires only when every field it needs is present and
the values logically contradict.**  Missing data is never a violation — that is
a completeness concern, reported separately as coverage, not a consistency
error.  This keeps false positives near zero.

Privacy: ``run_quality_checks`` returns row-level findings (carrying
``KeyRecordNumber`` and the offending value) that must never be committed; the
CLI writes them to a git-ignored detail file.  Only the aggregate summary
(counts, no identifiers, no raw values) is safe to track.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from rehab_sci.constants import AIS_ORD_TO_LETTER
from rehab_sci.data.loader import (
    RAW_PATH_DEFAULT,
    add_isncsci_summaries,
    add_scim_subscales,
    load_raw,
    normalize,
)
from rehab_sci.schema import Schema, load_schema

# Artifact paths (mirror the models/ aggregate-vs-detail split used elsewhere).
MODELS_DIR: Final[Path] = RAW_PATH_DEFAULT.parent / "models"
SUMMARY_PATH: Final[Path] = MODELS_DIR / "dataquality_summary.json"   # tracked: counts only
DETAIL_PATH: Final[Path] = MODELS_DIR / "dataquality_report.json"     # git-ignored: row-level

# Tokens that legitimately mean "missing / not tested" — never flagged as bad data.
SENTINELS: Final[frozenset[str]] = frozenset({"", "_", "NA", "NT", "ND", "NAN", "NONE"})
# Categorical level sets that are deliberately open-ended (raw token kept as-is).
OPEN_ENDED_LEVELS: Final[frozenset[str]] = frozenset({"bony_level"})
# Packed columns the loader splits into derived ordinals — validate those, not the
# raw pair string (e.g. "C1/C2" is a valid mFrankel/Frankel mismatch, "_/_" a sentinel).
PACKED_COLUMNS: Final[frozenset[str]] = frozenset({"mFrankel_Frankel"})

# mFrankel ordinal (1..11: A,B1,B2,B3,C1,C2,D0,D1,D2,D3,E) → AIS severity (A=1..E=5).
MFRANKEL_TO_AIS_SEV: Final[dict[int, int]] = {
    1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 4, 10: 4, 11: 5,
}

# Review heuristics (NOT clinical constants — surfaced magnitudes let a reviewer judge).
CERVICAL_MAX_ORD: Final[int] = 7        # C1..C8 occupy cord ordinals 0..7; T1+ ≥ 8.
SCIM_DROP_MIN: Final[float] = 10.0      # flag SCIM-total drops larger than this between visits.
NLI_DRIFT_SEGMENTS: Final[int] = 6      # NLI ascends a little with recovery; flag only larger swings.
MFRANKEL_AIS_GAP: Final[int] = 2        # flag mFrankel-vs-AIS disagreement of ≥ this many grades.

SEV_ERROR: Final[str] = "error"
SEV_WARN: Final[str] = "warn"
SEV_INFO: Final[str] = "info"


@dataclass(frozen=True)
class Violation:
    rule: str
    severity: str
    category: str
    detail: str
    key_record: int | None = None
    timepoint: str | None = None
    column: str | None = None
    value: Any = None


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    severity: str
    description: str
    fn: Callable[[Context], list[Violation]]


RULES: list[Rule] = []


def rule(rid: str, category: str, severity: str, description: str) -> Callable:
    """Register a rule function ``(ctx) -> list[Violation]``."""

    def deco(fn: Callable[[Context], list[Violation]]) -> Callable:
        RULES.append(Rule(rid, category, severity, description, fn))
        return fn

    return deco


@dataclass
class Context:
    """Loaded data + precomputed lookups shared across rules."""

    raw: pd.DataFrame
    clean: pd.DataFrame
    schema: Schema
    cord_inv: dict[int, str] = field(default_factory=dict)   # ord → cord level label
    time_order: dict[str, int] = field(default_factory=dict)  # TIME_Name → chronological index

    @classmethod
    def build(cls, path: Path | str | None = None, schema: Schema | None = None) -> Context:
        schema = schema or load_schema()
        raw = load_raw(path)
        clean = normalize(raw, schema=schema)          # normalize() copies — raw stays pristine
        clean = add_isncsci_summaries(clean, schema=schema)
        clean = add_scim_subscales(clean, schema=schema)
        cord = schema.isncsci["cord_levels_ordered"]
        return cls(
            raw=raw,
            clean=clean,
            schema=schema,
            cord_inv=dict(enumerate(cord)),
            time_order={lv.display: i for i, lv in enumerate(schema.level_sets["time_name"])},
        )

    # ---- per-row helpers ----
    def col(self, name: str) -> pd.Series:
        """Cleaned column, or an all-NA series when absent (rules then no-op)."""
        if name in self.clean.columns:
            return self.clean[name]
        return pd.Series(pd.NA, index=self.clean.index)

    def rows(self, mask: pd.Series) -> Iterator[tuple[int, int | None, str | None]]:
        """Yield (index, KeyRecordNumber, TIME_Name) for each flagged row."""
        m = mask.fillna(False).to_numpy(dtype=bool)
        kr = self.clean.get("KeyRecordNumber")
        tp = self.clean.get("TIME_Name")
        for idx in self.clean.index[m]:
            k = kr.at[idx] if kr is not None else None
            yield (
                idx,
                int(k) if k is not None and pd.notna(k) else None,
                (str(tp.at[idx]) if tp is not None and pd.notna(tp.at[idx]) else None),
            )

    def at(self, idx: int, name: str) -> Any:
        v = self.clean.at[idx, name] if name in self.clean.columns else None
        return None if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA else v


def _eq(s: pd.Series, value: str) -> pd.Series:
    return s.astype("string").str.strip() == value


def _is_sentinel(value: Any) -> bool:
    """True for tokens meaning 'missing / not tested', including paired or packed
    forms like '_/_', 'NT/NT', '__'.  Such tokens are never flagged as bad data."""
    if value is None or value is pd.NA:
        return True
    s = str(value).strip()
    if not s or s.upper() in SENTINELS:
        return True
    if set(s) <= set("_-/ ."):                      # punctuation-only blanks ('__', '-/-')
        return True
    parts = [p.strip().upper() for p in s.split("/")]
    return len(parts) > 1 and all(p in SENTINELS or not p for p in parts)


def _scalar(v: Any) -> Any:
    """JSON-safe scalar (numpy → python, NaN/NA → None)."""
    if v is None or v is pd.NA:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return str(v)


# ======================================================================== DOMAIN

@rule("D-NUM-RANGE", "domain", SEV_ERROR,
      "Numeric value outside the schema-declared range (loader coerced it to NaN).")
def _num_range(ctx: Context) -> list[Violation]:
    out: list[Violation] = []
    kr, tp = ctx.raw.get("KeyRecordNumber"), ctx.raw.get("TIME_Name")
    for col in ctx.raw.columns:
        spec = ctx.schema.by_raw(col)
        if spec is None or spec.dtype not in ("numeric", "ordinal") or spec.range is None:
            continue
        lo, hi = spec.range
        num = pd.to_numeric(ctx.raw[col], errors="coerce")
        bad = num.notna() & ((num < lo) | (num > hi))
        for idx in ctx.raw.index[bad.to_numpy(dtype=bool)]:
            out.append(Violation(
                "D-NUM-RANGE", SEV_ERROR, "domain",
                f"{col}={_scalar(num.at[idx])} outside [{lo:g}, {hi:g}]",
                int(kr.at[idx]) if kr is not None and pd.notna(kr.at[idx]) else None,
                str(tp.at[idx]) if tp is not None and pd.notna(tp.at[idx]) else None,
                col, _scalar(num.at[idx]),
            ))
    return out


@rule("D-NUM-PARSE", "domain", SEV_INFO,
      "Non-numeric, non-sentinel token in a numeric column (undocumented missing-marker or typo).")
def _num_parse(ctx: Context) -> list[Violation]:
    out: list[Violation] = []
    kr, tp = ctx.raw.get("KeyRecordNumber"), ctx.raw.get("TIME_Name")
    for col in ctx.raw.columns:
        spec = ctx.schema.by_raw(col)
        if spec is None or spec.dtype not in ("numeric", "ordinal"):
            continue
        raw = ctx.raw[col]
        num = pd.to_numeric(raw, errors="coerce")
        bad = raw.notna() & num.isna() & ~raw.map(_is_sentinel)
        for idx in ctx.raw.index[bad.fillna(False).to_numpy(dtype=bool)]:
            out.append(Violation(
                "D-NUM-PARSE", SEV_INFO, "domain",
                f"{col}={raw.at[idx]!r} is not numeric",
                int(kr.at[idx]) if kr is not None and pd.notna(kr.at[idx]) else None,
                str(tp.at[idx]) if tp is not None and pd.notna(tp.at[idx]) else None,
                col, _scalar(raw.at[idx]),
            ))
    return out


@rule("D-CAT-LEVEL", "domain", SEV_WARN,
      "Categorical value matches no canonical level or raw_alias (schema-coverage gap / typo).")
def _cat_level(ctx: Context) -> list[Violation]:
    out: list[Violation] = []
    kr, tp = ctx.clean.get("KeyRecordNumber"), ctx.clean.get("TIME_Name")
    for col in ctx.clean.columns:
        spec = ctx.schema.by_raw(col)
        if (spec is None or spec.dtype != "categorical" or not spec.levels
                or spec.levels in OPEN_ENDED_LEVELS or col in PACKED_COLUMNS):
            continue
        displays = {lv.display for lv in ctx.schema.level_sets.get(spec.levels, ())}
        ser = ctx.clean[col].astype("string").str.strip()
        bad = ser.notna() & ~ser.isin(displays) & ~ser.map(_is_sentinel)
        for idx in ctx.clean.index[bad.fillna(False).to_numpy(dtype=bool)]:
            out.append(Violation(
                "D-CAT-LEVEL", SEV_WARN, "domain",
                f"{col}={ser.at[idx]!r} not in level set '{spec.levels}'",
                int(kr.at[idx]) if kr is not None and pd.notna(kr.at[idx]) else None,
                str(tp.at[idx]) if tp is not None and pd.notna(tp.at[idx]) else None,
                col, _scalar(ser.at[idx]),
            ))
    return out


# =================================================================== CROSS-FIELD

def _sacral_signals(ctx: Context) -> dict[str, pd.Series]:
    vac, dap, star = ctx.col("VAC"), ctx.col("DAP"), ctx.col("STAR")
    s45 = [ctx.col(f"{side}{mod}S45")
           for side in ("Right", "Left") for mod in ("LightTouch", "PinPrick")]
    s45_pos = pd.concat([pd.to_numeric(s, errors="coerce") > 0 for s in s45], axis=1).any(axis=1)
    any_pos = _eq(vac, "Y") | _eq(dap, "Y") | _eq(star, "Y") | s45_pos
    return {"vac": vac, "dap": dap, "star": star, "any_pos": any_pos}


@rule("X-SACRAL-AIS-A", "cross_field", SEV_WARN,
      "AIS A (complete) recorded alongside sacral sparing (VAC/DAP/STAR/S4-5 sensory present).")
def _sacral_ais_a(ctx: Context) -> list[Violation]:
    sig = _sacral_signals(ctx)
    ais = ctx.col("AIS")
    mask = _eq(ais, "A") & sig["any_pos"]
    out = []
    for idx, kr, tp in ctx.rows(mask):
        out.append(Violation(
            "X-SACRAL-AIS-A", SEV_WARN, "cross_field",
            f"AIS=A but sacral sparing present (VAC={ctx.at(idx, 'VAC')}, "
            f"DAP={ctx.at(idx, 'DAP')}, STAR={ctx.at(idx, 'STAR')})",
            kr, tp, "AIS", "A",
        ))
    return out


@rule("X-SACRAL-AIS-INC", "cross_field", SEV_WARN,
      "AIS incomplete/normal (B-E) yet VAC and DAP both explicitly absent and no S4-5 sparing.")
def _sacral_ais_inc(ctx: Context) -> list[Violation]:
    sig = _sacral_signals(ctx)
    ais = ctx.col("AIS")
    no_sparing = _eq(sig["vac"], "N") & _eq(sig["dap"], "N") & ~sig["any_pos"]
    mask = ais.astype("string").str.strip().isin(["B", "C", "D", "E"]) & no_sparing
    out = []
    for idx, kr, tp in ctx.rows(mask):
        out.append(Violation(
            "X-SACRAL-AIS-INC", SEV_WARN, "cross_field",
            f"AIS={ctx.at(idx, 'AIS')} (incomplete) but VAC=N, DAP=N and no S4-5 sparing",
            kr, tp, "AIS", ctx.at(idx, "AIS"),
        ))
    return out


@rule("X-VAC-AIS", "cross_field", SEV_WARN,
      "Voluntary anal contraction (VAC=Y) implies motor-incomplete (AIS C/D), but AIS is A/B.")
def _vac_ais(ctx: Context) -> list[Violation]:
    mask = _eq(ctx.col("VAC"), "Y") & ctx.col("AIS").astype("string").str.strip().isin(["A", "B"])
    out = []
    for idx, kr, tp in ctx.rows(mask):
        out.append(Violation(
            "X-VAC-AIS", SEV_WARN, "cross_field",
            f"VAC=Y (motor function at S4-5) but AIS={ctx.at(idx, 'AIS')}", kr, tp, "VAC", "Y",
        ))
    return out


@rule("X-COMP-AIS", "cross_field", SEV_WARN,
      "Complete/incomplete flag contradicts AIS (complete ⇔ AIS A).")
def _comp_ais(ctx: Context) -> list[Violation]:
    ci, ais = ctx.col("COMP_INCOMP"), ctx.col("AIS").astype("string").str.strip()
    mask = (_eq(ci, "C") & ais.notna() & (ais != "A") & (ais != "ND")) | (_eq(ci, "I") & (ais == "A"))
    out = []
    for idx, kr, tp in ctx.rows(mask):
        out.append(Violation(
            "X-COMP-AIS", SEV_WARN, "cross_field",
            f"COMP_INCOMP={ctx.at(idx, 'COMP_INCOMP')} but AIS={ctx.at(idx, 'AIS')}",
            kr, tp, "COMP_INCOMP", ctx.at(idx, "COMP_INCOMP"),
        ))
    return out


@rule("X-AIS-E-MAX", "cross_field", SEV_WARN,
      "AIS E (normal) but motor/sensory totals are sub-maximal.")
def _ais_e_max(ctx: Context) -> list[Violation]:
    tm = pd.to_numeric(ctx.col("TotalMotor"), errors="coerce")
    lt = pd.to_numeric(ctx.col("LightTouchTotal"), errors="coerce")
    pp = pd.to_numeric(ctx.col("PinPrickTotal"), errors="coerce")
    sub = ((tm < 100) | (lt < 112) | (pp < 112)) & (tm.notna() | lt.notna() | pp.notna())
    mask = _eq(ctx.col("AIS"), "E") & sub
    out = []
    for idx, kr, tp in ctx.rows(mask):
        out.append(Violation(
            "X-AIS-E-MAX", SEV_WARN, "cross_field",
            f"AIS=E but TotalMotor={_scalar(tm.at[idx])}/100, "
            f"LightTouch={_scalar(lt.at[idx])}/112, PinPrick={_scalar(pp.at[idx])}/112",
            kr, tp, "AIS", "E",
        ))
    return out


@rule("X-PARA-NLI", "cross_field", SEV_WARN,
      "Paraplegia/tetraplegia inconsistent with NLI region (tetra⇔cervical C1-C8).")
def _para_nli(ctx: Context) -> list[Violation]:
    para = ctx.col("対麻痺_四肢麻痺").astype("string").str.strip()
    nli = pd.to_numeric(ctx.col("NLI_ord"), errors="coerce")
    tetra_bad = (para == "TETRA") & nli.notna() & (nli > CERVICAL_MAX_ORD)
    para_bad = (para == "PARA") & nli.notna() & (nli <= CERVICAL_MAX_ORD)
    out = []
    for idx, kr, tp in ctx.rows(tetra_bad | para_bad):
        lvl = ctx.cord_inv.get(int(nli.at[idx]), "?")
        out.append(Violation(
            "X-PARA-NLI", SEV_WARN, "cross_field",
            f"{ctx.at(idx, '対麻痺_四肢麻痺')} but NLI={lvl}", kr, tp, "NLI", lvl,
        ))
    return out


@rule("X-NLI-LEVELS", "cross_field", SEV_WARN,
      "NLI is caudal to a determined sensory/motor level (NLI must be the most rostral).")
def _nli_levels(ctx: Context) -> list[Violation]:
    nli = pd.to_numeric(ctx.col("NLI_ord"), errors="coerce")
    levels = pd.concat(
        [pd.to_numeric(ctx.col(c), errors="coerce") for c in (
            "RightSensoryLevel_ord", "LeftSensoryLevel_ord",
            "RightMotorLevel_ord", "LeftMotorLevel_ord")],
        axis=1,
    )
    min_lvl = levels.min(axis=1)
    mask = nli.notna() & min_lvl.notna() & (nli > min_lvl)
    out = []
    for idx, kr, tp in ctx.rows(mask):
        out.append(Violation(
            "X-NLI-LEVELS", SEV_WARN, "cross_field",
            f"NLI={ctx.cord_inv.get(int(nli.at[idx]), '?')} caudal to most-rostral "
            f"level {ctx.cord_inv.get(int(min_lvl.at[idx]), '?')}",
            kr, tp, "NLI", ctx.cord_inv.get(int(nli.at[idx]), "?"),
        ))
    return out


@rule("X-MFRANKEL-AIS", "cross_field", SEV_WARN,
      "mFrankel grade and AIS disagree by ≥2 grades.")
def _mfrankel_ais(ctx: Context) -> list[Violation]:
    mf = pd.to_numeric(ctx.col("mFrankel_ord"), errors="coerce")
    ais = pd.to_numeric(ctx.col("AIS_ord"), errors="coerce")
    fam = mf.map(lambda v: MFRANKEL_TO_AIS_SEV.get(int(v)) if pd.notna(v) else np.nan)
    mask = fam.notna() & ais.notna() & ((fam - ais).abs() >= MFRANKEL_AIS_GAP)
    out = []
    for idx, kr, tp in ctx.rows(mask):
        out.append(Violation(
            "X-MFRANKEL-AIS", SEV_WARN, "cross_field",
            f"mFrankel≈AIS {AIS_ORD_TO_LETTER.get(int(fam.at[idx]), '?')} "
            f"but AIS={AIS_ORD_TO_LETTER.get(int(ais.at[idx]), '?')}",
            kr, tp, "AIS", AIS_ORD_TO_LETTER.get(int(ais.at[idx]), "?"),
        ))
    return out


@rule("X-AUTO-MANUAL", "cross_field", SEV_INFO,
      "Auto-computed AIS/NLI disagrees with the manually corrected value.")
def _auto_manual(ctx: Context) -> list[Violation]:
    out = []
    for auto, manual, label in (("AIS", "AIS_manual", "AIS"), ("NLI", "NLI_manual", "NLI")):
        a = ctx.col(auto).astype("string").str.strip()
        m = ctx.col(manual).astype("string").str.strip()
        mask = a.notna() & m.notna() & (a != m)
        for idx, kr, tp in ctx.rows(mask):
            out.append(Violation(
                "X-AUTO-MANUAL", SEV_INFO, "cross_field",
                f"{label}: auto={a.at[idx]} vs manual={m.at[idx]}", kr, tp, auto, a.at[idx],
            ))
    return out


# ================================================================== LONGITUDINAL

def _ordered_episode_series(ctx: Context, value_col: str) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield (KeyRecordNumber, frame) per episode, rows sorted chronologically,
    restricted to timepoints where ``value_col`` is observed."""
    if value_col not in ctx.clean.columns:
        return
    df = ctx.clean[["KeyRecordNumber", "TIME_Name", value_col]].copy()
    df["_v"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df[df["_v"].notna()].copy()
    df["_ord"] = df["TIME_Name"].map(ctx.time_order)
    df = df[df["_ord"].notna()].sort_values(["KeyRecordNumber", "_ord"])
    for kr, g in df.groupby("KeyRecordNumber", sort=True):
        if len(g) >= 2:
            yield int(kr), g


@rule("L-AIS-DETERIORATION", "longitudinal", SEV_WARN,
      "AIS grade worsened (moved toward A) between consecutive timepoints.")
def _ais_deterioration(ctx: Context) -> list[Violation]:
    out = []
    for kr, g in _ordered_episode_series(ctx, "AIS_ord"):
        v, names = g["_v"].to_numpy(), g["TIME_Name"].to_numpy()
        for i in range(1, len(v)):
            if v[i] < v[i - 1]:
                out.append(Violation(
                    "L-AIS-DETERIORATION", SEV_WARN, "longitudinal",
                    f"AIS {AIS_ORD_TO_LETTER.get(int(v[i - 1]), '?')}→"
                    f"{AIS_ORD_TO_LETTER.get(int(v[i]), '?')} ({names[i - 1]}→{names[i]})",
                    kr, str(names[i]), "AIS_ord", AIS_ORD_TO_LETTER.get(int(v[i]), "?"),
                ))
    return out


@rule("L-SCIM-DROP", "longitudinal", SEV_WARN,
      f"SCIM-III total fell by more than {SCIM_DROP_MIN:g} points between consecutive timepoints.")
def _scim_drop(ctx: Context) -> list[Violation]:
    out = []
    for kr, g in _ordered_episode_series(ctx, "SCIM_total"):
        v, names = g["_v"].to_numpy(), g["TIME_Name"].to_numpy()
        for i in range(1, len(v)):
            d = v[i - 1] - v[i]
            if d > SCIM_DROP_MIN:
                out.append(Violation(
                    "L-SCIM-DROP", SEV_WARN, "longitudinal",
                    f"SCIM total {v[i - 1]:g}→{v[i]:g} (−{d:g}) ({names[i - 1]}→{names[i]})",
                    kr, str(names[i]), "SCIM_total", float(-d),
                ))
    return out


@rule("L-NLI-DRIFT", "longitudinal", SEV_WARN,
      f"NLI varies implausibly across an episode (> {NLI_DRIFT_SEGMENTS} segments — beyond "
      "normal recovery-related ascension).")
def _nli_drift(ctx: Context) -> list[Violation]:
    out = []
    for kr, g in _ordered_episode_series(ctx, "NLI_ord"):
        v = g["_v"].to_numpy()
        span = int(v.max() - v.min())
        if span > NLI_DRIFT_SEGMENTS:
            levels = [ctx.cord_inv.get(int(x), "?") for x in sorted(set(v))]
            out.append(Violation(
                "L-NLI-DRIFT", SEV_WARN, "longitudinal",
                f"NLI spans {span} segments across visits: {', '.join(levels)}",
                kr, None, "NLI_ord", span,
            ))
    return out


# ========================================================================= ENGINE

@dataclass
class QualityReport:
    violations: list[Violation]
    n_rows: int
    n_cols: int
    n_episodes: int
    n_patients: int
    source_file: str

    def summary(self) -> dict[str, Any]:
        """Aggregate, identifier-free summary safe to commit."""
        by_sev: Counter = Counter()
        by_cat: Counter = Counter()
        per_rule_col: dict[str, Counter] = defaultdict(Counter)
        per_rule_count: Counter = Counter()
        per_rule_eps: dict[str, set[int]] = defaultdict(set)
        episodes_flagged: set[int] = set()
        for v in self.violations:
            by_sev[v.severity] += 1
            by_cat[v.category] += 1
            per_rule_count[v.rule] += 1
            if v.category == "domain" and v.column:
                per_rule_col[v.rule][v.column] += 1
            if v.key_record is not None:
                episodes_flagged.add(v.key_record)
                per_rule_eps[v.rule].add(v.key_record)

        rules_out = []
        for r in RULES:
            entry: dict[str, Any] = {
                "id": r.id, "category": r.category, "severity": r.severity,
                "description": r.description, "count": int(per_rule_count.get(r.id, 0)),
                "episodes": len(per_rule_eps.get(r.id, ())),
            }
            if per_rule_col.get(r.id):
                entry["by_column"] = dict(sorted(per_rule_col[r.id].items(),
                                                  key=lambda kv: (-kv[1], kv[0])))
            rules_out.append(entry)

        return {
            "source": {
                "file": self.source_file, "n_rows": self.n_rows, "n_cols": self.n_cols,
                "n_episodes": self.n_episodes, "n_patients": self.n_patients,
            },
            "totals": {
                "violations": len(self.violations),
                "by_severity": dict(sorted(by_sev.items())),
                "by_category": dict(sorted(by_cat.items())),
                "episodes_flagged": len(episodes_flagged),
            },
            "rules": rules_out,
        }

    def detail(self) -> list[dict[str, Any]]:
        """Row-level findings (carries identifiers + offending values — never commit)."""
        recs = [asdict(v) for v in self.violations]
        recs.sort(key=lambda r: (r["rule"], r["key_record"] or -1,
                                 r["timepoint"] or "", r["column"] or ""))
        return recs


def run_quality_checks(path: Path | str | None = None, schema: Schema | None = None) -> QualityReport:
    ctx = Context.build(path, schema)
    violations: list[Violation] = []
    for r in RULES:
        violations.extend(r.fn(ctx))
    n_episodes = int(ctx.clean["KeyRecordNumber"].nunique()) if "KeyRecordNumber" in ctx.clean else 0
    n_patients = (int(ctx.clean["IDNumber"].nunique()) if "IDNumber" in ctx.clean else 0)
    return QualityReport(
        violations=violations,
        n_rows=len(ctx.raw),
        n_cols=int(ctx.raw.shape[1]),
        n_episodes=n_episodes,
        n_patients=n_patients,
        source_file=(Path(path).name if path else RAW_PATH_DEFAULT.name),
    )


def _print_summary(summary: dict[str, Any]) -> None:
    from rich.console import Console
    from rich.table import Table

    con = Console()
    src = summary["source"]
    tot = summary["totals"]
    con.print(
        f"\n[bold]Data-quality report[/bold] — {src['file']}  "
        f"({src['n_rows']:,} rows × {src['n_cols']} cols, "
        f"{src['n_episodes']} episodes, {src['n_patients']} patients)"
    )
    con.print(
        f"[bold]{tot['violations']:,}[/bold] findings across "
        f"{tot['episodes_flagged']} episodes  |  "
        + "  ".join(f"{k}={v}" for k, v in tot["by_severity"].items())
        + "  |  " + "  ".join(f"{k}={v}" for k, v in tot["by_category"].items())
    )
    table = Table(show_lines=False)
    for c, j in (("Rule", "left"), ("Cat", "left"), ("Sev", "left"),
                 ("Rows", "right"), ("Ep", "right"), ("Description", "left")):
        table.add_column(c, justify=j, overflow="fold")
    palette = {SEV_ERROR: "red", SEV_WARN: "yellow", SEV_INFO: "cyan"}
    for r in summary["rules"]:
        sev = r["severity"]
        table.add_row(
            r["id"], r["category"], f"[{palette.get(sev, 'white')}]{sev}[/]",
            f"{r['count']:,}", f"{r['episodes']:,}", r["description"],
        )
    con.print(table)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="SCI data-quality / clinical-consistency report.")
    ap.add_argument("--csv", default=None, help="Path to ALL_SCIDATA.csv (default: repo root).")
    ap.add_argument("--no-write", action="store_true", help="Print only; do not write artifacts.")
    args = ap.parse_args(argv)

    report = run_quality_checks(args.csv)
    summary = report.summary()
    _print_summary(summary)

    if not args.no_write:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        DETAIL_PATH.write_text(
            json.dumps(report.detail(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        from rich.console import Console
        Console().print(
            f"[green]wrote[/green] {SUMMARY_PATH.relative_to(MODELS_DIR.parent)} "
            f"(aggregate) + {DETAIL_PATH.relative_to(MODELS_DIR.parent)} (detail, git-ignored)"
        )


if __name__ == "__main__":
    main()
