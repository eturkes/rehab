"""Plotly figures for the Patient explorer tab — SCIM timeline, prediction, similarity neighbors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from rehab_sci.constants import AIS_ORD_TO_LETTER
from rehab_sci.dashboard.figures._common import _hex_to_rgba
from rehab_sci.dashboard.figures.overview import (
    PALETTE_PHENOTYPE,
    PHENOTYPE_NAMES_EN,
    PHENOTYPE_NAMES_JA,
)
from rehab_sci.dashboard.i18n import level_label, t
from rehab_sci.dashboard.theme import PALETTE_AIS, PALETTE_CATEGORICAL, PALETTE_PARA
from rehab_sci.data.episodes import PATIENT_TIMELINE, cohort_percentile_bands, patient_timeline
from rehab_sci.schema import Schema

_SUBSCALE_STYLES: dict[str, dict[str, str]] = {
    "SCIM_self_care": {"color": "#2c8a6b", "label_key": "scim_self_care"},
    "SCIM_respiration_sphincter": {"color": "#d4773c", "label_key": "scim_respiration"},
    "SCIM_mobility": {"color": "#6d4f78", "label_key": "scim_mobility"},
}


def _subscale_label(key: str, lang: str) -> str:
    return {
        "SCIM_self_care": ("自己ケア" if lang == "ja" else "Self-care"),
        "SCIM_respiration_sphincter": (
            "呼吸・括約筋" if lang == "ja" else "Respiration / sphincter"
        ),
        "SCIM_mobility": ("移動" if lang == "ja" else "Mobility"),
    }[key]


def fig_patient_scim_timeline(
    long_df: pd.DataFrame,
    ep: pd.DataFrame,
    key_record: int,
    strata: str,
    schema: Schema,
    lang: str,
    trajectory: dict | None = None,
) -> go.Figure:
    """SCIM-III timeline for a single episode against cohort percentile bands.

    ``strata`` is ``"para"`` (paralysis-only) or ``"para_ais"`` (paralysis x AIS).
    Band stratum is determined by the chosen episode's admission attributes;
    if those attributes are missing in the episode row, the figure falls back
    to the wider strata (paralysis-only → no band).

    ``trajectory`` is an optional dict with keys ``timepoints`` (list[str]),
    ``pred`` (list[float]), ``lo`` (list[float]), ``hi`` (list[float])``
    rendered as a dashed predicted-recovery line with a PI ribbon.
    """
    pt = patient_timeline(long_df, key_record)
    pt_total = pt["SCIM_total"]

    x_labels = [level_label(schema, "time_name", tp, lang) for tp in PATIENT_TIMELINE]
    x_pos = {tp: x_labels[i] for i, tp in enumerate(PATIENT_TIMELINE)}

    ep_row = ep.loc[ep["KeyRecordNumber"] == key_record]
    para_val = (
        str(ep_row["対麻痺_四肢麻痺"].iloc[0])
        if not ep_row.empty and pd.notna(ep_row["対麻痺_四肢麻痺"].iloc[0])
        else None
    )
    ais_val = (
        str(ep_row["AIS"].iloc[0])
        if not ep_row.empty and pd.notna(ep_row["AIS"].iloc[0])
        else None
    )

    # Decide stratification keys + a single band row to draw.
    bands = pd.DataFrame()
    band_label = ""
    if strata == "para_ais" and para_val is not None and ais_val is not None:
        all_bands = cohort_percentile_bands(
            long_df, ep, "SCIM_total", ["対麻痺_四肢麻痺", "AIS"]
        )
        bands = all_bands[
            (all_bands["対麻痺_四肢麻痺"] == para_val) & (all_bands["AIS"] == ais_val)
        ]
        band_label = (
            level_label(schema, "para_tetra", para_val, lang)
            + " · AIS "
            + level_label(schema, "ais", ais_val, lang)
        )
    if bands.empty and para_val is not None:
        all_bands = cohort_percentile_bands(long_df, ep, "SCIM_total", ["対麻痺_四肢麻痺"])
        bands = all_bands[all_bands["対麻痺_四肢麻痺"] == para_val]
        band_label = level_label(schema, "para_tetra", para_val, lang)

    band_color = PALETTE_PARA.get(para_val, PALETTE_CATEGORICAL[3]) if para_val else PALETTE_CATEGORICAL[3]

    fig = go.Figure()

    # Cohort bands (drawn first so patient lines render on top)
    if not bands.empty:
        bands = bands.sort_values("TIME_Name")
        bx = [x_pos[str(t)] for t in bands["TIME_Name"]]
        p10 = bands["p10"].astype(float).tolist()
        p25 = bands["p25"].astype(float).tolist()
        p75 = bands["p75"].astype(float).tolist()
        p90 = bands["p90"].astype(float).tolist()
        p50 = bands["p50"].astype(float).tolist()
        n = bands["n"].astype(int).tolist()

        # Outer (10–90) ribbon
        fig.add_trace(
            go.Scatter(
                x=bx + bx[::-1],
                y=p90 + p10[::-1],
                fill="toself",
                fillcolor=_hex_to_rgba(band_color, 0.10),
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=True,
                name=(
                    ("コホート 10–90 パーセンタイル" if lang == "ja" else "Cohort 10–90 pct")
                    + f" ({band_label})"
                ),
            )
        )
        # Inner (25–75) ribbon
        fig.add_trace(
            go.Scatter(
                x=bx + bx[::-1],
                y=p75 + p25[::-1],
                fill="toself",
                fillcolor=_hex_to_rgba(band_color, 0.18),
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=True,
                name=("コホート 25–75 パーセンタイル" if lang == "ja" else "Cohort 25–75 pct"),
            )
        )
        # Median dashed line
        fig.add_trace(
            go.Scatter(
                x=bx,
                y=p50,
                mode="lines",
                line=dict(color=band_color, width=1.5, dash="dash"),
                name=("コホート中央値" if lang == "ja" else "Cohort median"),
                customdata=np.array(n).reshape(-1, 1),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    + ("中央値" if lang == "ja" else "Median")
                    + ": %{y:.0f}<br>"
                    + ("N" if lang == "en" else "症例数")
                    + ": %{customdata[0]}<extra></extra>"
                ),
            )
        )

    # Predicted trajectory (PI ribbon + dashed line), drawn after cohort bands
    # but before patient lines so the prediction sits behind the observed data.
    _TRAJ_COLOR = "#5B6CC1"
    if trajectory is not None and trajectory.get("pred"):
        traj_tps = trajectory["timepoints"]
        traj_pred = trajectory["pred"]
        traj_lo = trajectory["lo"]
        traj_hi = trajectory["hi"]
        traj_x = [x_pos.get(tp, tp) for tp in traj_tps]
        # PI ribbon
        fig.add_trace(
            go.Scatter(
                x=traj_x + traj_x[::-1],
                y=list(traj_hi) + list(traj_lo)[::-1],
                fill="toself",
                fillcolor=_hex_to_rgba(_TRAJ_COLOR, 0.12),
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=True,
                name=(
                    "予測 80% PI" if lang == "ja" else "Predicted 80% PI"
                ),
            )
        )
        # Predicted line
        fig.add_trace(
            go.Scatter(
                x=traj_x,
                y=traj_pred,
                mode="lines+markers",
                line=dict(color=_TRAJ_COLOR, width=2.2, dash="dash"),
                marker=dict(size=7, color=_TRAJ_COLOR, symbol="diamond",
                            line=dict(color="#fff", width=1)),
                name=(
                    "予測回復軌道" if lang == "ja" else "Predicted trajectory"
                ),
                connectgaps=False,
                customdata=np.stack(
                    [traj_lo, traj_hi], axis=-1
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    + ("予測" if lang == "ja" else "Predicted")
                    + ": %{y:.0f}<br>"
                    + "80% PI"
                    + ": %{customdata[0]:.0f}–%{customdata[1]:.0f}"
                    + "<extra></extra>"
                ),
            )
        )

    # Subscale lines first (so the SCIM_total line sits on top)
    for col, style in _SUBSCALE_STYLES.items():
        if col not in pt.columns:
            continue
        y_vals = pt[col].astype(float).tolist()
        if all(pd.isna(v) for v in y_vals):
            continue
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=y_vals,
                mode="lines+markers",
                line=dict(color=style["color"], width=1.6, dash="dot"),
                marker=dict(size=6, color=style["color"], opacity=0.9),
                name=_subscale_label(col, lang),
                connectgaps=False,
                hovertemplate=(
                    f"<b>{_subscale_label(col, lang)}</b><br>"
                    + ("時点" if lang == "ja" else "Timepoint") + ": %{x}<br>"
                    + ("スコア" if lang == "ja" else "Score") + ": %{y:.0f}<extra></extra>"
                ),
                visible="legendonly",
            )
        )

    # SCIM total — the headline line, always visible.
    total_color = PALETTE_AIS["A"] if not para_val else (
        "#0c5a66" if para_val == "TETRA" else "#a35225"
    )
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=pt_total.astype(float).tolist(),
            mode="lines+markers",
            line=dict(color=total_color, width=3),
            marker=dict(size=10, color=total_color, line=dict(color="#fff", width=1.5)),
            name="SCIM-III " + ("合計" if lang == "ja" else "total"),
            connectgaps=False,
            hovertemplate=(
                "<b>SCIM-III " + ("合計" if lang == "ja" else "total") + "</b><br>"
                + ("時点" if lang == "ja" else "Timepoint") + ": %{x}<br>"
                + ("スコア" if lang == "ja" else "Score") + ": %{y:.0f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=420,
        xaxis_title=("評価時点" if lang == "ja" else "Timepoint"),
        xaxis_tickangle=-45,
        yaxis_title="SCIM-III (0–100)",
        yaxis=dict(range=[0, 102]),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11),
        ),
        margin=dict(l=56, r=24, t=20, b=80),
    )
    # Force x-axis to show every canonical timepoint, even those the patient lacks.
    fig.update_xaxes(categoryorder="array", categoryarray=x_labels)
    return fig


def fig_patient_prediction(
    pred: float | None,
    lo: float | None,
    hi: float | None,
    observed: float | None,
    schema: Schema,
    lang: str,
    clip_min: float = 0.0,
    clip_max: float | None = 100.0,
    axis_label: str | None = None,
) -> go.Figure:
    """Predicted discharge outcome with 80% PI and the observed value (if any).

    ``clip_min``/``clip_max`` control the x-axis range.  ``axis_label`` sets the
    x-axis title; defaults to ``SCIM-III (0–100)`` for backward compatibility.
    """
    fig = go.Figure()
    band_label = t(schema, "sim_prediction_interval", lang)
    pred_label = ("予測中央値" if lang == "ja" else "Predicted median")
    obs_label = ("実測値" if lang == "ja" else "Observed")

    if pred is not None and lo is not None and hi is not None:
        fig.add_trace(
            go.Bar(
                x=[hi - lo],
                base=[lo],
                y=[band_label],
                orientation="h",
                marker=dict(color="rgba(17,122,139,0.18)", line=dict(width=0)),
                hovertemplate=f"{lo:.0f}–{hi:.0f}<extra></extra>",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[pred],
                y=[band_label],
                mode="markers",
                marker=dict(color="#0c5a66", size=14, symbol="diamond"),
                hovertemplate=f"{pred_label}: %{{x:.0f}}<extra></extra>",
                name=pred_label,
                showlegend=False,
            )
        )
    if observed is not None:
        fig.add_trace(
            go.Scatter(
                x=[observed],
                y=[band_label],
                mode="markers",
                marker=dict(
                    color="#a3354e",
                    size=16,
                    symbol="x-thin-open",
                    line=dict(color="#a3354e", width=3),
                ),
                hovertemplate=f"{obs_label}: %{{x:.0f}}<extra></extra>",
                name=obs_label,
                showlegend=False,
            )
        )
    x_lo = float(clip_min) if clip_min is not None else min(0.0, lo or 0.0)
    if clip_max is not None:
        x_hi = float(clip_max)
    else:
        x_hi = float(max(hi or 0.0, pred or 0.0, observed or 0.0) * 1.1 + 1.0)
    if axis_label is None:
        axis_label = "SCIM-III (0–100)"
    fig.update_layout(
        height=140,
        margin=dict(l=130, r=20, t=10, b=30),
        xaxis=dict(range=[x_lo, x_hi], title=axis_label),
        yaxis=dict(showticklabels=True, tickfont=dict(size=12), showgrid=False),
        showlegend=False,
    )
    return fig


def fig_neighbor_outcomes(
    neighbors: list[dict],
    pred: float | None,
    lo: float | None,
    hi: float | None,
    observed: float | None,
    schema: Schema,
    lang: str,
    *,
    clip_min: float = 0.0,
    clip_max: float | None = 100.0,
    axis_label: str | None = None,
) -> go.Figure:
    """Strip chart of K nearest neighbors' actual outcomes on the prediction scale.

    Neighbor dots are sized by similarity (larger = more similar) and layered
    over the query patient's prediction interval and observed value.
    """
    fig = go.Figure()
    pred_lbl = "予測中央値" if lang == "ja" else "Predicted median"
    obs_lbl = "実測値" if lang == "ja" else "Observed"
    nbr_lbl = "類似患者" if lang == "ja" else "Similar patients"

    if pred is not None and lo is not None and hi is not None:
        fig.add_trace(go.Bar(
            x=[hi - lo], base=[lo], y=[nbr_lbl],
            orientation="h",
            marker=dict(color="rgba(17,122,139,0.12)", line=dict(width=0)),
            hovertemplate=f"80% PI: {lo:.0f}–{hi:.0f}<extra></extra>",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[pred], y=[nbr_lbl],
            mode="markers",
            marker=dict(color="#0c5a66", size=14, symbol="diamond"),
            hovertemplate=f"{pred_lbl}: %{{x:.0f}}<extra></extra>",
            name=pred_lbl, showlegend=True,
        ))

    vals = [n["y_discharge_scim"] for n in neighbors if n.get("y_discharge_scim") is not None]
    sims = [n["similarity"] for n in neighbors if n.get("y_discharge_scim") is not None]
    ids = [n.get("id_number") for n in neighbors if n.get("y_discharge_scim") is not None]
    if vals:
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        sizes = [8 + 10 * s for s in sims]
        id_txt = [str(i) if i is not None else "?" for i in ids]
        hover = [
            f"ID {i}<br>SCIM: {v:.0f}<br>{'類似度' if lang == 'ja' else 'Similarity'}: {s:.0%}"
            for i, v, s in zip(id_txt, vals, sims, strict=True)
        ]
        fig.add_trace(go.Scatter(
            x=vals,
            y=[jitter[j] for j in range(len(vals))],
            mode="markers",
            marker=dict(
                color=PALETTE_CATEGORICAL[1],
                size=sizes,
                opacity=0.75,
                line=dict(color=PALETTE_CATEGORICAL[1], width=1),
            ),
            yaxis="y2",
            hovertext=hover,
            hoverinfo="text",
            name=nbr_lbl, showlegend=True,
        ))

    if observed is not None:
        fig.add_trace(go.Scatter(
            x=[observed], y=[nbr_lbl],
            mode="markers",
            marker=dict(color="#a3354e", size=16, symbol="x-thin-open",
                        line=dict(color="#a3354e", width=3)),
            hovertemplate=f"{obs_lbl}: %{{x:.0f}}<extra></extra>",
            name=obs_lbl, showlegend=True,
        ))

    x_lo = float(clip_min) if clip_min is not None else 0.0
    if clip_max is not None:
        x_hi = float(clip_max)
    else:
        all_vals = vals + ([pred] if pred else []) + ([observed] if observed else [])
        x_hi = float(max(all_vals) * 1.1 + 1.0) if all_vals else 100.0
    if axis_label is None:
        axis_label = "SCIM-III (0–100)"

    fig.update_layout(
        height=180,
        margin=dict(l=130, r=20, t=10, b=30),
        xaxis=dict(range=[x_lo, x_hi], title=axis_label),
        yaxis=dict(showticklabels=True, tickfont=dict(size=12), showgrid=False,
                   domain=[0.0, 0.4]),
        yaxis2=dict(
            showticklabels=False, showgrid=False, zeroline=False,
            range=[-0.5, 0.5], domain=[0.4, 1.0], fixedrange=True,
        ),
        showlegend=True,
        legend=dict(orientation="h", x=0.0, y=1.18, font=dict(size=11),
                    bgcolor="rgba(255,255,255,0)"),
    )
    return fig


def fig_neighbor_ais_distribution(
    neighbors: list[dict],
    pred_proba: list[float] | None,
    observed_ais: int | None,
    schema: Schema,
    lang: str,
) -> go.Figure:
    """Bar chart comparing neighbor AIS grade distribution to the model's predicted probabilities."""
    labels = list(AIS_ORD_TO_LETTER.values())
    ais_colors = [PALETTE_AIS[lbl] for lbl in labels]

    neighbor_grades = [n["y_discharge_ais"] for n in neighbors if n.get("y_discharge_ais") is not None]
    counts = np.zeros(5)
    for g in neighbor_grades:
        if 1 <= g <= 5:
            counts[g - 1] += 1
    total = counts.sum()
    nbr_pct = counts / total if total > 0 else counts

    fig = go.Figure()
    nbr_name = "類似患者の実績" if lang == "ja" else "Similar patients (actual)"
    model_name = "モデル予測" if lang == "ja" else "Model prediction"

    fig.add_trace(go.Bar(
        x=labels, y=nbr_pct, name=nbr_name,
        marker=dict(color=[_hex_to_rgba(c, 0.6) for c in ais_colors],
                    line=dict(color=ais_colors, width=1.5)),
        hovertemplate="%{x}: %{y:.0%}<extra>" + nbr_name + "</extra>",
    ))

    if pred_proba is not None and len(pred_proba) == 5:
        fig.add_trace(go.Scatter(
            x=labels, y=pred_proba, name=model_name,
            mode="lines+markers",
            marker=dict(color=PALETTE_CATEGORICAL[0], size=8),
            line=dict(color=PALETTE_CATEGORICAL[0], width=2, dash="dot"),
            hovertemplate="%{x}: %{y:.0%}<extra>" + model_name + "</extra>",
        ))

    obs_text = ""
    if observed_ais is not None and 1 <= observed_ais <= 5:
        obs_letter = AIS_ORD_TO_LETTER[observed_ais]
        obs_text = f"{'実測' if lang == 'ja' else 'Observed'}: AIS {obs_letter}"

    y_lbl = "割合" if lang == "ja" else "Proportion"
    fig.update_layout(
        height=240,
        margin=dict(l=50, r=20, t=28, b=44),
        xaxis=dict(title="AIS"),
        yaxis=dict(title=y_lbl, range=[0, 1], tickformat=".0%"),
        barmode="group",
        legend=dict(orientation="h", x=0.0, y=1.15, font=dict(size=11),
                    bgcolor="rgba(255,255,255,0)"),
        annotations=([dict(
            x=0.98, y=0.95, xref="paper", yref="paper",
            text=obs_text, showarrow=False,
            font=dict(size=12, color="#a3354e"),
            xanchor="right",
        )] if obs_text else []),
    )
    return fig


def fig_phenotype_membership(
    membership: list[float],
    summaries: list[dict],
    schema: Schema,
    lang: str,
) -> go.Figure:
    """Soft phenotype membership for one patient — horizontal bars over the K phenotypes
    (ordered 0 = lowest recovery), each annotated with its conditioned cohort prognosis.

    ``membership`` is the posterior from :func:`compute.predict_phenotype_membership`;
    ``summaries`` is the per-phenotype summary list carried in the phenotype bundle.
    """
    names = PHENOTYPE_NAMES_JA if lang == "ja" else PHENOTYPE_NAMES_EN
    K = len(membership)
    pct = [w * 100 for w in membership]
    labels = [names[k] for k in range(K)]
    colors = [PALETTE_PHENOTYPE[k % len(PALETTE_PHENOTYPE)] for k in range(K)]
    nan = float("nan")
    cd = np.array([
        [
            (summaries[k].get("median_discharge_scim") if summaries[k].get("median_discharge_scim") is not None else nan),
            (summaries[k].get("mean_los") if summaries[k].get("mean_los") is not None else nan),
            summaries[k].get("n", 0),
        ]
        for k in range(K)
    ])
    scim_lbl = "退院時SCIM(中央値)" if lang == "ja" else "Discharge SCIM (med)"
    los_lbl = "在院日数(平均)" if lang == "ja" else "LOS (mean d)"
    memb_lbl = "メンバーシップ" if lang == "ja" else "Membership"
    fig = go.Figure(
        go.Bar(
            x=pct,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{p:.0f}%" for p in pct],
            textposition="outside",
            cliponaxis=False,
            customdata=cd,
            hovertemplate=(
                "<b>%{y}</b> (n=%{customdata[2]})<br>"
                + f"{memb_lbl}: " + "%{x:.1f}%<br>"
                + f"{scim_lbl}: " + "%{customdata[0]:.0f}<br>"
                + f"{los_lbl}: " + "%{customdata[1]:.0f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=60 + 36 * K,
        xaxis=dict(title=("割合 (%)" if lang == "ja" else "Membership (%)"), range=[0, 108]),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=20, t=10, b=40),
        showlegend=False,
    )
    return fig
