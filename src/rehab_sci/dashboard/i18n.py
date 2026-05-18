"""Bilingual translation helpers used by every dashboard component."""

from __future__ import annotations

from rehab_sci.schema import Schema

Lang = str  # "ja" or "en"


def t(schema: Schema, key: str, lang: Lang = "ja") -> str:
    return schema.ui_str(key, lang)


def col_label(schema: Schema, raw: str, lang: Lang = "ja") -> str:
    spec = schema.by_raw(raw)
    if spec is None:
        return raw
    return spec.label(lang)


def level_label(schema: Schema, level_key: str, raw_value: str, lang: Lang = "ja") -> str:
    return schema.level_label(level_key, raw_value, lang)


def all_levels_in_order(
    schema: Schema, level_key: str, lang: Lang = "ja"
) -> list[tuple[str, str]]:
    """Return (display, ja-or-en label) pairs in their YAML declaration order."""
    return [
        (lv.display, lv.label(lang))
        for lv in schema.level_sets.get(level_key, ())
    ]


# Map a raw admission column to its level_key (for the simulator dropdowns).
def level_key_for_column(schema: Schema, raw: str) -> str | None:
    spec = schema.by_raw(raw)
    return spec.levels if spec else None
