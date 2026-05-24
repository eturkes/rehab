"""Plotly figure factories used by the dashboard tabs. All labels respect the lang param."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from rehab_sci.dashboard.i18n import all_levels_in_order, col_label, level_label, t
from rehab_sci.dashboard.theme import INK, PALETTE_AIS, PALETTE_CATEGORICAL, PALETTE_PARA
from rehab_sci.data.episodes import (
    PATIENT_TIMELINE,
    cohort_percentile_bands,
    patient_timeline,
)
from rehab_sci.schema import Schema


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


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
    ep: pd.DataFrame,
    feature: str,
    schema: Schema,
    lang: str,
    outcome_col: str = "y_discharge_scim",
    outcome_label: str | None = None,
) -> go.Figure:
    spec = schema.by_raw(feature)
    cols = [feature, outcome_col]
    sub = ep[cols].dropna().copy()
    if spec and spec.dtype == "categorical" and spec.levels:
        sub["_label"] = sub[feature].astype(str).map(
            lambda v: level_label(schema, spec.levels, v, lang)
        )
        order = [level_label(schema, spec.levels, lv, lang)
                 for lv, _ in all_levels_in_order(schema, spec.levels, lang)]
        present = [o for o in order if o in sub["_label"].unique()]
        sub["_label"] = pd.Categorical(sub["_label"], categories=present, ordered=True)
        x = sub["_label"]
        xtitle = col_label(schema, feature, lang)
    else:
        sub["_q"] = pd.qcut(sub[feature], q=4, duplicates="drop")
        sub["_label"] = sub["_q"].astype(str)
        x = sub["_label"]
        xtitle = col_label(schema, feature, lang) + (" (四分位)" if lang == "ja" else " (quartile)")

    ytitle = outcome_label if outcome_label else t(schema, "chart_discharge_scim", lang)

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            x=x,
            y=sub[outcome_col],
            boxpoints="outliers",
            marker=dict(color=PALETTE_CATEGORICAL[0], size=3),
            line=dict(color=PALETTE_CATEGORICAL[0]),
            fillcolor="rgba(17,122,139,0.15)",
            hovertemplate="%{x}<br>%{y:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=400,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        margin=dict(l=60, r=20, t=10, b=80),
        xaxis_tickangle=-25,
    )
    return fig


def fig_dependence(
    shap_pack: dict,
    X_test: pd.DataFrame,
    feature: str,
    schema: Schema,
    lang: str,
    *,
    class_idx: int | None = None,
) -> go.Figure:
    if feature not in X_test.columns:
        return go.Figure()
    idx = X_test.columns.get_loc(feature)
    sv = shap_pack["shap_values"]
    if class_idx is not None and sv.ndim == 3:
        shap_vals = sv[:, idx, class_idx]
        labels = shap_pack.get("class_labels", [])
        cls_lbl = labels[class_idx] if class_idx < len(labels) else str(class_idx)
        y_label = f"SHAP (AIS {cls_lbl})"
    else:
        shap_vals = sv[:, idx]
        y_label = "SHAP"
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
        fig.update_layout(xaxis_title=col_label(schema, feature, lang), yaxis_title=y_label)
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
                    + ": %{x}<br>" + y_label + ": %{y:.2f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            xaxis_title=col_label(schema, feature, lang),
            yaxis_title=y_label,
        )
    fig.add_hline(y=0, line=dict(color=INK["300"], dash="dash", width=1))
    fig.update_layout(height=360, margin=dict(l=60, r=20, t=10, b=44))
    return fig


# ---------- SHAP interaction ----------

def fig_interaction_heatmap(
    metrics: dict,
    schema: Schema,
    lang: str,
    *,
    top_n: int = 10,
) -> go.Figure:
    """Upper-triangle heatmap of top feature-pair interactions by mean |SHAP|."""
    items = metrics.get("global_interaction_top25", [])
    if not items:
        return go.Figure()

    # collect unique features from top_n pairs
    feats_seen: list[str] = []
    for item in items[:top_n]:
        for f in (item["feat_a"], item["feat_b"]):
            if f not in feats_seen:
                feats_seen.append(f)
    n = len(feats_seen)
    labels = [col_label(schema, f, lang) for f in feats_seen]
    feat_to_idx = {f: i for i, f in enumerate(feats_seen)}

    # build matrix (upper triangle only; diagonal = 0)
    mat = np.zeros((n, n))
    for item in items[:top_n]:
        ia = feat_to_idx.get(item["feat_a"])
        ib = feat_to_idx.get(item["feat_b"])
        if ia is not None and ib is not None:
            val = item["abs_mean_interaction"]
            mat[ia, ib] = val
            mat[ib, ia] = val

    # mask lower triangle for visual clarity
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    display = np.where(mask, mat, np.nan)

    hover_text = []
    for i in range(n):
        row = []
        for j in range(n):
            if mask[i, j]:
                row.append(f"{labels[i]} × {labels[j]}<br>mean|φ|: {mat[i, j]:.4f}")
            else:
                row.append("")
        hover_text.append(row)

    fig = go.Figure(
        go.Heatmap(
            z=display,
            x=labels,
            y=labels,
            colorscale=[[0, "rgba(17,122,139,0.05)"], [1, PALETTE_CATEGORICAL[0]]],
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
            showscale=True,
            colorbar=dict(title="mean|φ|"),
        )
    )
    fig.update_layout(
        height=max(360, 28 * n + 80),
        margin=dict(l=200, r=20, t=10, b=120),
        xaxis=dict(tickangle=-45, side="bottom"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def fig_interaction_dependence(
    shap_pack: dict,
    X_test: pd.DataFrame,
    feat_x: str,
    feat_y: str,
    schema: Schema,
    lang: str,
    *,
    class_idx: int | None = None,
) -> go.Figure:
    """Scatter of feature-X value vs SHAP interaction(X,Y), colored by feature-Y value."""
    si = shap_pack.get("shap_interaction")
    if si is None or feat_x not in X_test.columns or feat_y not in X_test.columns:
        return go.Figure()
    ix = X_test.columns.get_loc(feat_x)
    iy = X_test.columns.get_loc(feat_y)

    # slice interaction values
    if class_idx is not None and si.ndim == 4:
        int_vals = si[:, ix, iy, class_idx]
        labels = shap_pack.get("class_labels", [])
        cls_lbl = labels[class_idx] if class_idx < len(labels) else str(class_idx)
        y_label = f"SHAP φ(x,y) (AIS {cls_lbl})"
    else:
        int_vals = si[:, ix, iy]
        y_label = "SHAP φ(x,y)"

    x_label = col_label(schema, feat_x, lang)
    y_feat_label = col_label(schema, feat_y, lang)

    spec_x = schema.by_raw(feat_x)
    spec_y = schema.by_raw(feat_y)

    # x-axis values
    if spec_x and spec_x.dtype == "categorical":
        x_vals = X_test[feat_x].astype(str).map(
            lambda v: level_label(schema, spec_x.levels, v, lang) if spec_x.levels else v
        )
    else:
        x_vals = pd.to_numeric(X_test[feat_x], errors="coerce")

    # color by feat_y
    if spec_y and spec_y.dtype == "categorical":
        color_vals = X_test[feat_y].astype(str).map(
            lambda v: level_label(schema, spec_y.levels, v, lang) if spec_y.levels else v
        )
        # categorical coloring: distinct traces per category
        fig = go.Figure()
        cats = [c for c in color_vals.unique() if pd.notna(c) and str(c) != "nan"]
        for ci, cat in enumerate(sorted(cats, key=str)):
            mask_c = color_vals == cat
            fig.add_trace(
                go.Scatter(
                    x=x_vals[mask_c] if not isinstance(x_vals.iloc[0], str) else x_vals[mask_c],
                    y=int_vals[mask_c],
                    mode="markers",
                    name=str(cat),
                    marker=dict(
                        color=PALETTE_CATEGORICAL[ci % len(PALETTE_CATEGORICAL)],
                        size=8, opacity=0.7,
                        line=dict(width=0.5, color="#fff"),
                    ),
                    hovertemplate=(
                        x_label + ": %{x}<br>"
                        + y_feat_label + ": " + str(cat) + "<br>"
                        + y_label + ": %{y:.4f}<extra></extra>"
                    ),
                )
            )
    else:
        color_num = pd.to_numeric(X_test[feat_y], errors="coerce")
        fig = go.Figure(
            go.Scatter(
                x=x_vals,
                y=int_vals,
                mode="markers",
                marker=dict(
                    color=color_num,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=y_feat_label),
                    size=8, opacity=0.7,
                    line=dict(width=0.5, color="#fff"),
                ),
                hovertemplate=(
                    x_label + ": %{x}<br>"
                    + y_feat_label + ": %{marker.color:.1f}<br>"
                    + y_label + ": %{y:.4f}<extra></extra>"
                ),
            )
        )

    fig.add_hline(y=0, line=dict(color=INK["300"], dash="dash", width=1))
    fig.update_layout(
        height=360,
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=60, r=20, t=10, b=44),
    )
    return fig


# ---------- Patient explorer tab ----------
# Subscale traces share the timeline figure with SCIM_total.  Use distinct hues
# but keep them muted so the total line stays the prominent signal.
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

    timeline_present = [tp for tp in PATIENT_TIMELINE if pt_total.loc[tp] == pt_total.loc[tp]]  # NaN-safe
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
                    + ("80% PI" if lang == "en" else "80% PI")
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


# ---------- Calibration / performance (Methods tab, F8) ----------


def fig_pred_vs_observed(
    shap_pack: dict,
    schema: Schema,
    lang: str,
    *,
    clip_min: float = 0.0,
    clip_max: float | None = 100.0,
    axis_label: str | None = None,
) -> go.Figure | None:
    y_true = shap_pack.get("y_test")
    y_pred = shap_pack.get("y_pred")
    if y_true is None or y_pred is None:
        return None
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if axis_label is None:
        axis_label = "SCIM-III"
    obs_lbl = "実測値" if lang == "ja" else "Observed"
    pred_lbl = "予測値" if lang == "ja" else "Predicted"
    lo = clip_min if clip_min is not None else float(min(y_true.min(), y_pred.min()) - 1)
    hi = clip_max if clip_max is not None else float(max(y_true.max(), y_pred.max()) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        line=dict(color=INK["200"], dash="dash", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    residuals = y_pred - y_true
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        marker=dict(
            size=6, opacity=0.7,
            color=residuals,
            colorscale=[[0, "#117a8b"], [0.5, INK["100"]], [1, "#a3354e"]],
            cmid=0,
            colorbar=dict(
                title=dict(text=("誤差" if lang == "ja" else "Error"), side="right"),
                thickness=10, len=0.6,
            ),
        ),
        hovertemplate=(
            f"{obs_lbl}: %{{x:.1f}}<br>{pred_lbl}: %{{y:.1f}}"
            "<extra></extra>"
        ),
        showlegend=False,
    ))
    from sklearn.metrics import r2_score as _r2
    r2 = _r2(y_true, y_pred)
    fig.add_annotation(
        x=0.03, y=0.97, xref="paper", yref="paper",
        text=f"R² = {r2:.3f}  (n={len(y_true)})",
        showarrow=False, font=dict(size=12, color=INK["700"]),
        xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
    )
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=20, t=28, b=44),
        xaxis=dict(title=f"{obs_lbl} ({axis_label})", range=[lo, hi]),
        yaxis=dict(title=f"{pred_lbl} ({axis_label})", range=[lo, hi],
                   scaleanchor="x", scaleratio=1),
    )
    return fig


def fig_residual_hist(
    shap_pack: dict,
    schema: Schema,
    lang: str,
    *,
    axis_label: str | None = None,
) -> go.Figure | None:
    y_true = shap_pack.get("y_test")
    y_pred = shap_pack.get("y_pred")
    if y_true is None or y_pred is None:
        return None
    residuals = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    if axis_label is None:
        axis_label = "SCIM-III"
    res_lbl = "残差" if lang == "ja" else "Residual"
    freq_lbl = "頻度" if lang == "ja" else "Frequency"
    mu = float(residuals.mean())
    sigma = float(residuals.std())
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals,
        marker=dict(color=PALETTE_CATEGORICAL[0], line=dict(width=0)),
        hovertemplate=f"{res_lbl}: %{{x:.1f}}<br>{freq_lbl}: %{{y}}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=INK["300"], line_width=1.5)
    fig.add_annotation(
        x=0.97, y=0.97, xref="paper", yref="paper",
        text=f"μ = {mu:.1f}   σ = {sigma:.1f}",
        showarrow=False, font=dict(size=12, color=INK["700"]),
        xanchor="right", yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
    )
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=20, t=28, b=44),
        xaxis=dict(title=f"{res_lbl} ({axis_label})"),
        yaxis=dict(title=freq_lbl),
    )
    return fig


def fig_confusion_matrix(
    shap_pack: dict,
    schema: Schema,
    lang: str,
) -> go.Figure | None:
    y_true = shap_pack.get("y_test")
    y_pred = shap_pack.get("y_pred")
    labels = shap_pack.get("class_labels")
    if y_true is None or y_pred is None or labels is None:
        return None
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    k = len(labels)
    cm = np.zeros((k, k), dtype=int)
    for t_idx, p_idx in zip(y_true, y_pred, strict=False):
        cm[t_idx, p_idx] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm / row_sums * 100, 0)
    actual_lbl = "実際" if lang == "ja" else "Actual"
    pred_lbl = "予測" if lang == "ja" else "Predicted"
    text = [[f"{cm[i, j]}<br>({cm_pct[i, j]:.0f}%)" for j in range(k)] for i in range(k)]
    fig = go.Figure(go.Heatmap(
        z=cm_pct,
        x=[f"AIS {l}" for l in labels],
        y=[f"AIS {l}" for l in labels],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorscale=[[0, INK["paper"]], [1, PALETTE_CATEGORICAL[0]]],
        showscale=False,
        hovertemplate=(
            f"{actual_lbl}: %{{y}}<br>{pred_lbl}: %{{x}}<br>"
            "%{text}<extra></extra>"
        ),
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=60, r=20, t=28, b=50),
        xaxis=dict(title=pred_lbl, side="bottom"),
        yaxis=dict(title=actual_lbl, autorange="reversed"),
    )
    return fig


def fig_calibration_curve(
    shap_pack: dict,
    schema: Schema,
    lang: str,
    *,
    n_bins: int = 5,
) -> go.Figure | None:
    y_true = shap_pack.get("y_test")
    proba = shap_pack.get("y_pred_proba")
    labels = shap_pack.get("class_labels")
    if y_true is None or proba is None or labels is None:
        return None
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    k = len(labels)
    conf_lbl = "予測確率" if lang == "ja" else "Predicted probability"
    obs_lbl = "実測頻度" if lang == "ja" else "Observed frequency"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color=INK["200"], dash="dash", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    ais_colors = list(PALETTE_AIS.values())
    for c in range(k):
        p_c = proba[:, c]
        y_c = (y_true == c).astype(float)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers, bin_freqs = [], []
        for b in range(n_bins):
            mask = (p_c >= bin_edges[b]) & (p_c < bin_edges[b + 1])
            if b == n_bins - 1:
                mask = (p_c >= bin_edges[b]) & (p_c <= bin_edges[b + 1])
            if mask.sum() >= 2:
                bin_centers.append(float(p_c[mask].mean()))
                bin_freqs.append(float(y_c[mask].mean()))
        if len(bin_centers) >= 2:
            fig.add_trace(go.Scatter(
                x=bin_centers, y=bin_freqs,
                mode="lines+markers",
                name=f"AIS {labels[c]}",
                marker=dict(size=7),
                line=dict(color=ais_colors[c], width=2),
            ))
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=20, t=28, b=44),
        xaxis=dict(title=conf_lbl, range=[0, 1]),
        yaxis=dict(title=obs_lbl, range=[0, 1]),
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)", font=dict(size=11)),
    )
    return fig


# ---------- Patient similarity (F16) ----------
AIS_ORD_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}


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
    band_lbl = t(schema, "sim_prediction_interval", lang)
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


# ---------- Recovery archetypes (F18) ----------
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

