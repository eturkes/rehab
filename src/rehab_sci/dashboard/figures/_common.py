"""Shared figure helpers (color utilities) used across the figure submodules."""

import plotly.graph_objects as go


def blank_figure() -> go.Figure:
    """Placeholder for an unsatisfied precondition.

    A bare ``go.Figure()`` still draws default axes, so an empty slot renders as
    several hundred pixels of ``-1..6`` grid.  The prompt text lives in an
    adjacent DOM node, so the figure collapses to Plotly's 10px floor.
    """
    fig = go.Figure()
    fig.update_layout(
        height=10,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"
