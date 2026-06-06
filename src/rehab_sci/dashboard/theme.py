"""Plotly theme + palettes used everywhere on the dashboard."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# Calm, professional, color-blind-aware palette.
PALETTE_CATEGORICAL = [
    "#117a8b",  # deep teal
    "#d4773c",  # warm orange
    "#2c8a6b",  # forest green
    "#5b6f80",  # slate
    "#a3354e",  # muted crimson
    "#9b6a3a",  # bronze
    "#42688f",  # ocean
    "#6d4f78",  # mauve
]

PALETTE_DIVERGING = [
    "#a3354e",
    "#d4773c",
    "#f1cbb1",
    "#f7f9fb",
    "#cfe1e5",
    "#5cabb8",
    "#117a8b",
]

# AIS grade palette: A=most severe → E=normal, color cool→warm
PALETTE_AIS = {
    "A": "#1a3148",
    "B": "#3d5e7f",
    "C": "#7392a8",
    "D": "#b6c3cf",
    "E": "#2c8a6b",
}

# paralysis-type palette
PALETTE_PARA = {
    "TETRA": "#117a8b",
    "PARA": "#d4773c",
    "NONE": "#7d92a8",
}

# G7 functional-independence domains (display grouping for the per-SCIM-item profile);
# one color per SCIM-ADL domain, drawn from the categorical palette.
PALETTE_INDEPENDENCE_DOMAIN = {
    "self_care": "#117a8b",   # deep teal
    "sphincter": "#d4773c",   # warm orange
    "mobility": "#2c8a6b",    # forest green
    "ambulation": "#5b6f80",  # slate
}

# G8 recovery-topography body map: sequential P(functional milestone) scale, 0..1.
# Clinically intuitive: low P (function unlikely to return) = muted crimson, mid = pale,
# high P (milestone likely) = deep teal.  Shared by the dermatome silhouette + the myotome ladder.
COLORSCALE_TOPOGRAPHY = [
    [0.0, "#a3354e"],   # crimson — milestone unlikely
    [0.25, "#d4773c"],  # warm orange
    [0.5, "#f1cbb1"],   # pale
    [0.75, "#5cabb8"],  # light teal
    [1.0, "#117a8b"],   # deep teal — milestone likely
]
# G8 modality accents (legend swatches / readout chips).
PALETTE_TOPOGRAPHY_MODALITY = {
    "motor": "#117a8b",        # deep teal
    "light_touch": "#d4773c",  # warm orange
    "pin_prick": "#6d4f78",    # mauve
}

INK = {
    "900": "#0e1b2a",
    "700": "#1c3147",
    "500": "#3d566e",
    "300": "#7d92a8",
    "200": "#b6c4d2",
    "100": "#d8e0e7",
    "50":  "#eef2f5",
    "paper": "#f7f9fb",
}


def apply_template() -> None:
    """Register and activate the medical-grade plotly template."""
    pio.templates["medical"] = go.layout.Template(
        layout=dict(
            font=dict(
                family='"Inter", "Hiragino Sans", "Noto Sans JP", "Yu Gothic UI", system-ui, sans-serif',
                size=12.5,
                color=INK["900"],
            ),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            colorway=PALETTE_CATEGORICAL,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                ticks="outside",
                tickcolor=INK["200"],
                linecolor=INK["200"],
                title=dict(font=dict(size=12, color=INK["500"])),
                tickfont=dict(color=INK["500"]),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=INK["100"],
                zeroline=False,
                ticks="outside",
                tickcolor=INK["200"],
                linecolor=INK["200"],
                title=dict(font=dict(size=12, color=INK["500"])),
                tickfont=dict(color=INK["500"]),
            ),
            margin=dict(l=50, r=20, t=24, b=44),
            legend=dict(
                bgcolor="rgba(255,255,255,0)",
                font=dict(size=12),
                bordercolor="rgba(0,0,0,0)",
            ),
            hoverlabel=dict(
                bgcolor="#fff",
                bordercolor=INK["200"],
                font=dict(family='"Inter", "Hiragino Sans", system-ui, sans-serif',
                          color=INK["900"]),
            ),
            hovermode="closest",
        )
    )
    pio.templates.default = "medical"
