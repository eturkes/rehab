"""Schema / bilingual translation registry.

Single source of truth for:
  * raw → canonical column metadata (group, role, dtype, range, …)
  * raw value → canonical level + ja/en display (categoricals)
  * UI string ja/en
  * SCIM-III subscales / ISNCSCI scoring metadata

No patient data flows through this module; it only reads ``schema/*.yaml`` files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "schema"


def _load_yaml(name: str) -> dict[str, Any]:
    with (SCHEMA_DIR / name).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class ColumnSpec:
    raw: str
    ja: str
    en: str
    group: str
    role: str
    dtype: str
    unit: str | None = None
    range: tuple[float, float] | None = None
    levels: str | None = None
    description: str | None = None
    family: str | None = None  # set when expanded from a family template
    family_side: str | None = None
    family_level: str | None = None

    def label(self, lang: str = "ja") -> str:
        return self.ja if lang == "ja" else self.en


@dataclass(frozen=True)
class LevelSpec:
    display: str
    ja: str
    en: str
    raw_aliases: tuple[str, ...] = field(default_factory=tuple)

    def label(self, lang: str = "ja") -> str:
        return self.ja if lang == "ja" else self.en


class Schema:
    """Bilingual schema registry. Constructed once via :func:`load_schema`."""

    __slots__ = (
        "columns",
        "level_sets",
        "ui",
        "scim",
        "isncsci",
        "_raw_index",
        "_alias_index",
        "_canon_index",
    )

    def __init__(
        self,
        columns: tuple[ColumnSpec, ...],
        level_sets: dict[str, tuple[LevelSpec, ...]],
        ui: dict[str, dict[str, str]],
        scim: dict[str, Any],
        isncsci: dict[str, Any],
    ) -> None:
        self.columns = columns
        self.level_sets = level_sets
        self.ui = ui
        self.scim = scim
        self.isncsci = isncsci
        self._raw_index: dict[str, ColumnSpec] = {c.raw: c for c in columns}
        self._alias_index: dict[str, dict[str, str]] = {}
        self._canon_index: dict[str, dict[str, LevelSpec]] = {}
        for k, lvs in level_sets.items():
            self._canon_index[k] = {lv.display: lv for lv in lvs}
            a: dict[str, str] = {}
            for lv in lvs:
                a[lv.display] = lv.display
                for alias in lv.raw_aliases:
                    a[alias] = lv.display
            self._alias_index[k] = a

    # ---- lookups ----
    def by_raw(self, raw: str) -> ColumnSpec | None:
        return self._raw_index.get(raw)

    def all_raw(self) -> list[str]:
        return [c.raw for c in self.columns]

    def columns_in_group(self, group: str) -> list[ColumnSpec]:
        return [c for c in self.columns if c.group == group]

    def columns_by_role(self, role: str) -> list[ColumnSpec]:
        return [c for c in self.columns if c.role == role]

    def level_label(self, level_key: str, raw_value: str, lang: str = "ja") -> str:
        """Translate a raw value of a categorical column to its display label.

        Falls back to the raw value if nothing matches (so unknown codes are still surfaced).
        """
        canonical = self.normalize_level(level_key, raw_value)
        spec = self._canon_index.get(level_key, {}).get(canonical)
        if spec is None:
            return raw_value
        return spec.label(lang)

    def normalize_level(self, level_key: str, raw_value: str) -> str:
        return self._alias_index.get(level_key, {}).get(
            str(raw_value).strip(), str(raw_value).strip()
        )

    def ui_str(self, key: str, lang: str = "ja") -> str:
        v = self.ui.get(key, {})
        return v.get(lang) or v.get("ja") or v.get("en") or key


def _expand_families(raw_columns: list[dict], families: list[dict]) -> list[ColumnSpec]:
    cols: list[ColumnSpec] = [
        ColumnSpec(
            raw=c["raw"],
            ja=c["ja"],
            en=c["en"],
            group=c["group"],
            role=c["role"],
            dtype=c["dtype"],
            unit=c.get("unit"),
            range=tuple(c["range"]) if c.get("range") else None,
            levels=c.get("levels"),
            description=c.get("description"),
        )
        for c in raw_columns
    ]
    for fam in families:
        for side, side_meta in fam["sides"].items():
            for level in fam["levels"]:
                raw = fam["template_raw"].format(side=side, level=level)
                ja = fam["ja_template"].format(level=level, **side_meta)
                en = fam["en_template"].format(level=level, **side_meta)
                cols.append(
                    ColumnSpec(
                        raw=raw,
                        ja=ja,
                        en=en,
                        group=fam["group"],
                        role="feature",
                        dtype=fam["dtype"],
                        range=tuple(fam["range"]) if fam.get("range") else None,
                        levels=fam.get("levels_ref"),
                        family=fam["id"],
                        family_side=side,
                        family_level=level,
                    )
                )
    return cols


def _load_levels() -> dict[str, tuple[LevelSpec, ...]]:
    data = _load_yaml("categorical_levels.yaml")
    out: dict[str, tuple[LevelSpec, ...]] = {}
    for key, levels in data.items():
        if key == "version":
            continue
        specs = []
        for lv in levels:
            specs.append(
                LevelSpec(
                    display=lv["display"],
                    ja=lv["ja"],
                    en=lv["en"],
                    raw_aliases=tuple(lv.get("raw_aliases", []) or []),
                )
            )
        out[key] = tuple(specs)

    # cord_level_zpp also reuses cord_level entries (avoid duplication in YAML).
    if "cord_level" in out and "cord_level_zpp" in out:
        seen = {lv.display for lv in out["cord_level_zpp"]}
        extra = tuple(lv for lv in out["cord_level"] if lv.display not in seen)
        out["cord_level_zpp"] = out["cord_level_zpp"] + extra
    return out


def load_schema() -> Schema:
    cols_yaml = _load_yaml("columns.yaml")
    levels = _load_levels()
    ui = _load_yaml("ui_strings.yaml")
    scim = _load_yaml("scim_iii.yaml")
    isncsci = _load_yaml("isncsci.yaml")

    columns = tuple(_expand_families(cols_yaml["columns"], cols_yaml.get("families", [])))
    return Schema(columns=columns, level_sets=levels, ui=ui, scim=scim, isncsci=isncsci)
