"""Plotly figures for the Overview tab — cohort demographics, injury, recovery curves, archetypes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from rehab_sci.dashboard.figures._common import _hex_to_rgba
from rehab_sci.dashboard.i18n import col_label, level_label, t
from rehab_sci.dashboard.theme import INK, PALETTE_AIS, PALETTE_CATEGORICAL, PALETTE_PARA
from rehab_sci.schema import Schema


def fig_age_distribution(ep: pd.DataFrame, schema: Schema, lang: str) -> go.Figure:
    s = pd.to_numeric(ep["年齢"], errors="coerce").dropna()
    fig = go.Figure(
        go.Histogram(
            x=s,
            xbins=dict(start=0, end=100, size=5),
            marker=dict(color=PALETTE_CATEGORICAL[0], line=dict(width=0)),
            hovertemplate=("%{y} " + ("名" if lang == "ja" else "patients") + "<extra></extra>"),
        )
    )
    fig.update_layout(
        xaxis_title=col_label(schema, "年齢", lang) + (" (歳)" if lang == "ja" else " (years)"),
        yaxis_title=("患者数" if lang == "ja" else "Patients"),
        bargap=0.05,
        height=300,
    )
    return fig


def fig_sex_donut(ep: pd.DataFrame, schema: Schema, lang: str) -> go.Figure:
    s = ep["性別"].dropna().astype(str)
    counts = s.value_counts()
    labels = [level_label(schema, "sex", v, lang) for v in counts.index]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=counts.values,
            hole=0.55,
            sort=False,
            marker=dict(colors=[PALETTE_CATEGORICAL[0], PALETTE_CATEGORICAL[1]]),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return fig


def fig_mechanism(ep: pd.DataFrame, schema: Schema, lang: str) -> go.Figure:
    s = ep["受傷機転"].dropna().astype(str)
    counts = s.value_counts()
    labels = [level_label(schema, "mechanism", v, lang) for v in counts.index]
    fig = go.Figure(
        go.Bar(
            x=counts.values,
            y=labels,
            orientation="h",
            marker=dict(color=PALETTE_CATEGORICAL[0]),
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(260, 30 * len(counts) + 80),
        xaxis_title=("患者数" if lang == "ja" else "Patients"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=180, r=20, t=10, b=44),
    )
    return fig


def fig_discharge_scim(ep: pd.DataFrame, schema: Schema, lang: str) -> go.Figure:
    s = pd.to_numeric(ep["y_discharge_scim"], errors="coerce").dropna()
    fig = go.Figure(
        go.Histogram(
            x=s,
            xbins=dict(start=0, end=100, size=5),
            marker=dict(color=PALETTE_CATEGORICAL[0], line=dict(width=0)),
            hovertemplate=("%{y} " + ("名" if lang == "ja" else "patients") + "<extra></extra>"),
        )
    )
    if len(s) > 0:
        fig.add_vline(
            x=float(s.median()),
            line=dict(color=INK["700"], width=2, dash="dot"),
            annotation_text=("中央値 " if lang == "ja" else "Median ") + f"{s.median():.0f}",
            annotation_position="top right",
            annotation_font_color=INK["700"],
        )
    fig.update_layout(
        xaxis_title=t(schema, "chart_discharge_scim", lang) + " (0–100)",
        yaxis_title=("患者数" if lang == "ja" else "Patients"),
        bargap=0.05,
        height=320,
    )
    return fig


def fig_injury_treemap(ep: pd.DataFrame, schema: Schema, lang: str) -> go.Figure:
    sub = ep[["対麻痺_四肢麻痺", "AIS", "NLI"]].dropna(subset=["対麻痺_四肢麻痺"]).copy()
    sub["AIS"] = sub["AIS"].fillna("?")
    sub["NLI"] = sub["NLI"].fillna("?")
    sub["para_label"] = sub["対麻痺_四肢麻痺"].map(
        lambda v: level_label(schema, "para_tetra", str(v), lang)
    )
    sub["ais_label"] = sub["AIS"].map(
        lambda v: level_label(schema, "ais", str(v), lang) if v != "?" else ("不明" if lang == "ja" else "Unknown")
    )
    sub["nli_label"] = sub["NLI"].map(
        lambda v: level_label(schema, "cord_level", str(v), lang) if v != "?" else ("不明" if lang == "ja" else "Unknown")
    )
    raw_to_label = dict(zip(sub["対麻痺_四肢麻痺"].astype(str), sub["para_label"]))
    label_to_color = {
        raw_to_label.get(raw, raw): color
        for raw, color in PALETTE_PARA.items()
        if raw_to_label.get(raw) is not None
    }
    fallback_color = PALETTE_CATEGORICAL[0]

    grouped = (
        sub.groupby(["para_label", "ais_label", "nli_label"], observed=True)
        .size()
        .reset_index(name="n")
    )
    ids, labels, parents, values, colors = [], [], [], [], []
    idx_by_id: dict[str, int] = {}

    def _upsert(node_id: str, label: str, parent: str, color: str) -> int:
        i = idx_by_id.get(node_id)
        if i is not None:
            return i
        idx_by_id[node_id] = len(ids)
        ids.append(node_id)
        labels.append(label)
        parents.append(parent)
        values.append(0)
        colors.append(color)
        return idx_by_id[node_id]

    for _, r in grouped.iterrows():
        para = r["para_label"]
        ais = r["ais_label"]
        nli = r["nli_label"]
        n = int(r["n"])
        para_id = para
        ais_id = f"{para}|{ais}"
        nli_id = f"{para}|{ais}|{nli}"
        c = label_to_color.get(para, fallback_color)
        i_para = _upsert(para_id, para, "", c)
        i_ais = _upsert(ais_id, ais, para_id, c)
        i_nli = _upsert(nli_id, nli, ais_id, c)
        values[i_para] += n
        values[i_ais] += n
        values[i_nli] += n

    fig = go.Figure(
        go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(colors=colors, line=dict(color="#fff", width=1.5)),
            hovertemplate="<b>%{label}</b><br>" + ("患者数" if lang == "ja" else "Patients") + ": %{value}<extra></extra>",
            textinfo="label+value",
            pathbar=dict(visible=True),
        )
    )
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def fig_ais_admit_discharge_sankey(ep: pd.DataFrame, schema: Schema, lang: str) -> go.Figure:
    # Admit AIS comes from the dataset (admission features include AIS_ord)
    # but we need the raw AIS letter. Re-derive from AIS_ord.
    ais_letter = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
    sub = ep[["AIS_ord", "y_discharge_ais"]].dropna().copy()
    sub["admit"] = sub["AIS_ord"].astype(int).map(ais_letter)
    sub["dis"] = sub["y_discharge_ais"].astype(int).map(ais_letter)
    grouped = sub.groupby(["admit", "dis"], observed=True).size().reset_index(name="n")

    grades = ["A", "B", "C", "D", "E"]
    admit_nodes = [f"{g} →" for g in grades]
    dis_nodes = [f"→ {g}" for g in grades]
    labels = (
        [level_label(schema, "ais", g, lang) + (" (入院)" if lang == "ja" else " (admit)") for g in grades]
        + [level_label(schema, "ais", g, lang) + (" (退院)" if lang == "ja" else " (discharge)") for g in grades]
    )
    src = [grades.index(r["admit"]) for _, r in grouped.iterrows()]
    tgt = [5 + grades.index(r["dis"]) for _, r in grouped.iterrows()]
    val = grouped["n"].tolist()
    colors = [PALETTE_AIS[g] for g in grades] * 2
    link_colors = [
        "rgba("
        + ",".join(
            str(int(c))
            for c in bytes.fromhex(PALETTE_AIS[r["admit"]].lstrip("#"))
        )
        + ",0.35)"
        for _, r in grouped.iterrows()
    ]

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                color=colors,
                pad=12,
                thickness=18,
                line=dict(color=INK["100"], width=0.5),
            ),
            link=dict(source=src, target=tgt, value=val, color=link_colors),
        )
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def fig_recovery_curves(long_df: pd.DataFrame, schema: Schema, lang: str) -> go.Figure:
    timeline = ["0day", "72h", "2w", "4w", "6w", "2m", "3m", "4m", "5m", "6m", "discharge"]
    sub = long_df[long_df["TIME_Name"].isin(timeline)].copy()
    sub = sub.dropna(subset=["SCIM_total", "対麻痺_四肢麻痺"])
    sub["TIME_Name"] = pd.Categorical(sub["TIME_Name"], categories=timeline, ordered=True)
    sub["para_label"] = sub["対麻痺_四肢麻痺"].map(
        lambda v: level_label(schema, "para_tetra", str(v), lang)
    )

    fig = go.Figure()
    for group, sub_g in sub.groupby("para_label", observed=True):
        agg = (
            sub_g.groupby("TIME_Name", observed=True)["SCIM_total"]
            .agg(["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75), "count"])
            .reset_index()
        )
        agg.columns = ["t", "med", "q25", "q75", "n"]
        agg = agg[agg["n"] >= 5]
        x = [str(v) for v in agg["t"]]
        x_label = [t(schema, "time_name", "ja") for _ in x]
        # use translated time labels
        x_disp = [level_label(schema, "time_name", v, lang) for v in x]
        color = PALETTE_PARA.get(
            ep_str := next(
                (k for k, v in PALETTE_PARA.items() if level_label(schema, "para_tetra", k, lang) == group),
                "TETRA",
            ),
            PALETTE_CATEGORICAL[0],
        )
        fig.add_trace(
            go.Scatter(
                x=x_disp,
                y=agg["med"],
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=7, color=color),
                name=group,
                customdata=np.stack([agg["q25"], agg["q75"], agg["n"]], axis=-1),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    + ("中央値" if lang == "ja" else "Median")
                    + ": %{y:.1f}<br>"
                    + "IQR: %{customdata[0]:.0f}–%{customdata[1]:.0f}<br>"
                    + ("N" if lang == "en" else "症例数")
                    + ": %{customdata[2]}<extra></extra>"
                ),
            )
        )
        # IQR ribbon
        fig.add_trace(
            go.Scatter(
                x=x_disp + x_disp[::-1],
                y=list(agg["q75"]) + list(agg["q25"])[::-1],
                fill="toself",
                fillcolor=color.replace("#", "rgba(") + "55"
                if False
                else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.update_layout(
        height=380,
        yaxis_title=t(schema, "chart_discharge_scim", lang),
        xaxis_title=("評価時点" if lang == "ja" else "Timepoint"),
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=20, b=80),
    )
    return fig


PALETTE_ARCHETYPE = [
    "#a3354e",  # crimson — limited recovery
    "#9b6a3a",  # bronze — gradual recovery
    "#2c8a6b",  # forest green — rapid recovery
    "#42688f",  # ocean — spare slot if k=4
    "#6d4f78",  # mauve — spare slot if k=5
]


ARCHETYPE_NAMES_JA = ["限定的回復", "段階的回復", "急速回復", "タイプ 4", "タイプ 5"]


ARCHETYPE_NAMES_EN = ["Limited recovery", "Gradual recovery", "Rapid recovery", "Type 4", "Type 5"]


def fig_archetype_curves(
    centroids: np.ndarray,
    timepoint_labels: list[str],
    summaries: list[dict],
    schema: Schema,
    lang: str,
) -> go.Figure:
    """Archetype recovery trajectory curves with centroid lines and member count annotations."""
    x_disp = [level_label(schema, "time_name", tp, lang) for tp in timepoint_labels]
    names = ARCHETYPE_NAMES_JA if lang == "ja" else ARCHETYPE_NAMES_EN
    fig = go.Figure()

    for i, (row, s) in enumerate(zip(centroids, summaries)):
        color = PALETTE_ARCHETYPE[i % len(PALETTE_ARCHETYPE)]
        label = f"{names[i]} (n={s['n']})"

        fig.add_trace(
            go.Scatter(
                x=x_disp,
                y=row,
                mode="lines+markers",
                line=dict(color=color, width=3),
                marker=dict(size=7, color=color, symbol="diamond"),
                name=label,
                customdata=np.stack([
                    np.full(len(row), s["n"]),
                    np.full(len(row), s["mean_age"] or 0),
                    np.full(len(row), s["pct_tetra"]),
                ], axis=-1),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    + ("予測 SCIM" if lang == "ja" else "Predicted SCIM")
                    + ": %{y:.1f}<br>"
                    + "n=%{customdata[0]:.0f}<br>"
                    + ("平均年齢" if lang == "ja" else "Mean age")
                    + ": %{customdata[1]:.0f}<br>"
                    + ("四肢麻痺" if lang == "ja" else "Tetraplegia")
                    + ": %{customdata[2]:.0f}%"
                    + "<extra></extra>"
                ),
            )
        )

        fill_color = _hex_to_rgba(color, 0.10)
        upper = np.minimum(row + 10, 100.0)
        lower = np.maximum(row - 10, 0.0)
        fig.add_trace(
            go.Scatter(
                x=x_disp + x_disp[::-1],
                y=list(upper) + list(lower)[::-1],
                fill="toself",
                fillcolor=fill_color,
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        height=400,
        yaxis_title="SCIM-III" + (" (予測)" if lang == "ja" else " (predicted)"),
        yaxis_range=[0, 105],
        xaxis_title=("評価時点" if lang == "ja" else "Timepoint"),
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=20, b=80),
    )
    return fig


def fig_archetype_demographics(
    summaries: list[dict],
    schema: Schema,
    lang: str,
) -> go.Figure:
    """Stacked bar chart showing AIS grade distribution per archetype."""
    names = ARCHETYPE_NAMES_JA if lang == "ja" else ARCHETYPE_NAMES_EN
    arch_labels = [f"{names[s['id']]}" for s in summaries]

    ais_grades = ["A", "B", "C", "D", "E"]
    fig = go.Figure()

    for grade in ais_grades:
        vals = [s["ais_distribution"].get(grade, 0.0) * 100 for s in summaries]
        fig.add_trace(
            go.Bar(
                x=arch_labels,
                y=vals,
                name=f"AIS {grade}",
                marker_color=PALETTE_AIS.get(grade, PALETTE_CATEGORICAL[0]),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    + f"AIS {grade}: " + "%{y:.1f}%"
                    + "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="stack",
        height=320,
        yaxis_title="%" if lang == "en" else "割合 (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=20, b=44),
    )
    return fig
