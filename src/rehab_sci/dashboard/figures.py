"""Plotly figure factories used by the dashboard tabs. All labels respect the lang param."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from rehab_sci.dashboard.i18n import all_levels_in_order, col_label, level_label, t
from rehab_sci.dashboard.theme import INK, PALETTE_AIS, PALETTE_CATEGORICAL, PALETTE_PARA
from rehab_sci.schema import Schema


# ---------- KPI ----------
def kpi_card(title: str, value: str, sub: str | None = None) -> dict:
    return {"title": title, "value": value, "sub": sub}


# ---------- Overview tab ----------
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


def fig_injury_sunburst(ep: pd.DataFrame, schema: Schema, lang: str) -> go.Figure:
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
    grouped = (
        sub.groupby(["para_label", "ais_label", "nli_label"], observed=True)
        .size()
        .reset_index(name="n")
    )
    # Build sunburst nodes; with branchvalues="total" each parent's value MUST
    # equal the sum of its children, so we accumulate leaf counts into every
    # ancestor as we go.
    ids, labels, parents, values, colors = [], [], [], [], []
    para_palette = list(PALETTE_PARA.values())
    para_label_to_color: dict[str, str] = {}
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
        if para_id not in para_label_to_color:
            para_label_to_color[para_id] = para_palette[
                len(para_label_to_color) % len(para_palette)
            ]
        c = para_label_to_color[para_id]
        i_para = _upsert(para_id, para, "", c)
        i_ais = _upsert(ais_id, ais, para_id, c)
        i_nli = _upsert(nli_id, nli, ais_id, c)
        values[i_para] += n
        values[i_ais] += n
        values[i_nli] += n

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(colors=colors, line=dict(color="#fff", width=1.5)),
            hovertemplate="<b>%{label}</b><br>" + ("患者数" if lang == "ja" else "Patients") + ": %{value}<extra></extra>",
        )
    )
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10))
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=20, b=44),
    )
    return fig


# ---------- Insight engine ----------
def fig_global_shap_importance(metrics: dict, schema: Schema, lang: str, top_n: int = 15) -> go.Figure:
    items = metrics["global_importance_top25"][:top_n][::-1]
    names = [col_label(schema, r["feature"], lang) for r in items]
    vals = [r["abs_mean_shap"] for r in items]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker=dict(color=PALETTE_CATEGORICAL[0]),
            hovertemplate="%{y}<br>|SHAP|: %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(360, 22 * len(items) + 80),
        xaxis_title="mean(|SHAP|) " + ("点" if lang == "ja" else "pts"),
        margin=dict(l=240, r=20, t=10, b=44),
    )
    return fig


def fig_subgroup_box(
    ep: pd.DataFrame, feature: str, schema: Schema, lang: str
) -> go.Figure:
    spec = schema.by_raw(feature)
    sub = ep[[feature, "y_discharge_scim"]].dropna().copy()
    if spec and spec.dtype == "categorical" and spec.levels:
        sub["_label"] = sub[feature].astype(str).map(
            lambda v: level_label(schema, spec.levels, v, lang)
        )
        order = [level_label(schema, spec.levels, lv, lang)
                 for lv, _ in all_levels_in_order(schema, spec.levels, lang)]
        # keep only labels present
        present = [o for o in order if o in sub["_label"].unique()]
        sub["_label"] = pd.Categorical(sub["_label"], categories=present, ordered=True)
        x = sub["_label"]
        xtitle = col_label(schema, feature, lang)
    else:
        # numeric → quartile bins
        sub["_q"] = pd.qcut(sub[feature], q=4, duplicates="drop")
        sub["_label"] = sub["_q"].astype(str)
        x = sub["_label"]
        xtitle = col_label(schema, feature, lang) + (" (四分位)" if lang == "ja" else " (quartile)")

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            x=x,
            y=sub["y_discharge_scim"],
            boxpoints="outliers",
            marker=dict(color=PALETTE_CATEGORICAL[0], size=3),
            line=dict(color=PALETTE_CATEGORICAL[0]),
            fillcolor="rgba(17,122,139,0.15)",
            hovertemplate="%{x}<br>%{y:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=400,
        xaxis_title=xtitle,
        yaxis_title=t(schema, "chart_discharge_scim", lang),
        margin=dict(l=60, r=20, t=10, b=80),
        xaxis_tickangle=-25,
    )
    return fig


def fig_dependence(
    shap_pack: dict, X_test: pd.DataFrame, feature: str, schema: Schema, lang: str
) -> go.Figure:
    if feature not in X_test.columns:
        return go.Figure()
    idx = X_test.columns.get_loc(feature)
    shap_vals = shap_pack["shap_values"][:, idx]
    spec = schema.by_raw(feature)

    if spec and spec.dtype == "categorical":
        x = X_test[feature].astype(str).map(
            lambda v: level_label(schema, spec.levels, v, lang) if spec.levels else v
        )
        fig = go.Figure(
            go.Box(
                x=x,
                y=shap_vals,
                boxpoints="outliers",
                marker=dict(color=PALETTE_CATEGORICAL[1], size=3),
                line=dict(color=PALETTE_CATEGORICAL[1]),
                fillcolor="rgba(212,119,60,0.15)",
            )
        )
        fig.update_layout(xaxis_title=col_label(schema, feature, lang), yaxis_title="SHAP")
    else:
        x = pd.to_numeric(X_test[feature], errors="coerce")
        fig = go.Figure(
            go.Scatter(
                x=x,
                y=shap_vals,
                mode="markers",
                marker=dict(
                    color=shap_vals,
                    colorscale="RdBu_r",
                    cmid=0,
                    showscale=True,
                    size=8,
                    opacity=0.7,
                    line=dict(width=0.5, color="#fff"),
                ),
                hovertemplate=(
                    col_label(schema, feature, lang)
                    + ": %{x}<br>SHAP: %{y:.2f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            xaxis_title=col_label(schema, feature, lang),
            yaxis_title="SHAP",
        )
    fig.add_hline(y=0, line=dict(color=INK["300"], dash="dash", width=1))
    fig.update_layout(height=360, margin=dict(l=60, r=20, t=10, b=44))
    return fig
