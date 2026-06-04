"""Plotly figures for the Simulator tab — hypothetical recovery trajectory."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from rehab_sci.dashboard.figures._common import _hex_to_rgba
from rehab_sci.dashboard.i18n import level_label
from rehab_sci.schema import Schema


def fig_sim_trajectory(
    trajectory: dict,
    schema: Schema,
    lang: str,
    *,
    ref_trajectory: dict | None = None,
) -> go.Figure:
    """Predicted SCIM-total recovery trajectory for a hypothetical patient (simulator).

    ``trajectory`` has keys ``timepoints``, ``pred``, ``lo``, ``hi``.
    ``ref_trajectory`` (optional) overlays the reference patient's trajectory as a
    dashed line with a muted PI ribbon for What-if comparison.
    """
    tps = trajectory["timepoints"]
    pred = trajectory["pred"]
    lo = trajectory["lo"]
    hi = trajectory["hi"]

    x_labels = [level_label(schema, "time_name", tp, lang) for tp in tps]
    traj_color = "#5B6CC1"
    ref_color = "#a3354e"

    fig = go.Figure()

    # Reference trajectory (behind current, if provided)
    if ref_trajectory:
        ref_tps = ref_trajectory["timepoints"]
        ref_pred = ref_trajectory["pred"]
        ref_lo = ref_trajectory["lo"]
        ref_hi = ref_trajectory["hi"]
        ref_x = [level_label(schema, "time_name", tp, lang) for tp in ref_tps]
        ref_label = ("参考値" if lang == "ja" else "Reference")
        fig.add_trace(
            go.Scatter(
                x=ref_x + ref_x[::-1],
                y=list(ref_hi) + list(ref_lo)[::-1],
                fill="toself",
                fillcolor=_hex_to_rgba(ref_color, 0.08),
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ref_x,
                y=ref_pred,
                mode="lines+markers",
                line=dict(color=ref_color, width=2, dash="dash"),
                marker=dict(size=6, color=ref_color, symbol="circle",
                            line=dict(color="#fff", width=1)),
                name=ref_label,
                hovertemplate=(
                    f"<b>%{{x}}</b><br>{ref_label}: %{{y:.0f}}<extra></extra>"
                ),
            )
        )

    # PI ribbon
    fig.add_trace(
        go.Scatter(
            x=x_labels + x_labels[::-1],
            y=list(hi) + list(lo)[::-1],
            fill="toself",
            fillcolor=_hex_to_rgba(traj_color, 0.14),
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=True,
            name=("80% 予測区間" if lang == "ja" else "80% prediction interval"),
        )
    )
    # Predicted line
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=pred,
            mode="lines+markers",
            line=dict(color=traj_color, width=2.5),
            marker=dict(size=8, color=traj_color, symbol="diamond",
                        line=dict(color="#fff", width=1.2)),
            name=("予測 SCIM-III" if lang == "ja" else "Predicted SCIM-III"),
            customdata=np.stack([lo, hi], axis=-1),
            hovertemplate=(
                "<b>%{x}</b><br>"
                + ("予測" if lang == "ja" else "Predicted")
                + ": %{y:.0f}<br>"
                + "80% PI: %{customdata[0]:.0f}–%{customdata[1]:.0f}"
                + "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=340,
        xaxis_title=("評価時点" if lang == "ja" else "Timepoint"),
        xaxis_tickangle=-45,
        yaxis_title="SCIM-III (0–100)",
        yaxis=dict(range=[0, 102]),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11),
        ),
        margin=dict(l=56, r=24, t=20, b=80),
    )
    # Merge timepoints from both trajectories for x-axis ordering.
    all_x = list(x_labels)
    if ref_trajectory:
        for tp in ref_trajectory["timepoints"]:
            lbl = level_label(schema, "time_name", tp, lang)
            if lbl not in all_x:
                all_x.append(lbl)
    fig.update_xaxes(categoryorder="array", categoryarray=all_x)
    return fig
