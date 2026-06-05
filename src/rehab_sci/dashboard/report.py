"""PDF patient report generator.

Generates a 2-page bilingual (JA/EN) PDF report for a single patient episode:
  Page 1: demographics, all 6 discharge predictions, recovery trajectory chart.
  Page 2: top SHAP drivers chart, methodology note, disclaimer.

Uses fpdf2 for PDF assembly and kaleido for Plotly→PNG chart export.
Font: IPAexGothic (single-weight sans-serif CJK). Visual hierarchy via size only.
"""

from __future__ import annotations

import copy
import io
from datetime import date

from fpdf import FPDF

from rehab_sci.constants import AIS_ORD_TO_LETTER

_FONT_PATH = "/usr/local/share/fonts/truetype/ipaexg.ttf"
_FONT_NAME = "IPAexGothic"

_S = {
    "title": ("患者レポート", "Patient Report"),
    "generated": ("生成日", "Generated"),
    "patient_id": ("患者 ID", "Patient ID"),
    "episode": ("症例", "Episode"),
    "demographics": ("患者情報", "Patient Information"),
    "age": ("年齢", "Age"),
    "sex": ("性別", "Sex"),
    "paralysis": ("麻痺型", "Paralysis"),
    "ais_admit": ("入院時 AIS", "Admission AIS"),
    "nli_admit": ("入院時 NLI", "Admission NLI"),
    "los": ("在院日数", "Length of stay"),
    "predictions": ("退院時予測", "Discharge Predictions"),
    "pred_outcome": ("アウトカム", "Outcome"),
    "pred_value": ("予測値", "Predicted"),
    "pred_pi": ("80% PI", "80% PI"),
    "pred_observed": ("実測値", "Observed"),
    "pred_set": ("予測集合", "Pred. set"),
    "trajectory": ("予測回復軌道 (SCIM-III)", "Predicted Recovery Trajectory (SCIM-III)"),
    "shap": (
        "主要予測因子 (SHAP — SCIM-III 総合点)",
        "Key Predictive Factors (SHAP — SCIM-III Total)",
    ),
    "methods": ("方法論", "Methodology"),
    "methods_text": (
        "LightGBM 勾配ブースティングモデルにより、入院時特徴量から退院時アウトカムを予測。"
        "80% 分割共形予測区間 (Mondrian per-AIS / per-paralysis) により不確実性を定量化。"
        "SHAP (TreeExplainer) により個別患者への寄与要因を分解。",
        "Discharge outcomes predicted from admission features using LightGBM gradient boosting. "
        "Uncertainty quantified via 80% split-conformal prediction intervals (Mondrian per-AIS / "
        "per-paralysis). Individual patient contributions decomposed using SHAP (TreeExplainer).",
    ),
    "disclaimer": (
        "本レポートは研究目的のデモであり、個別の臨床判断には用いられません。",
        "This report is a research demonstration; it is not intended for individual clinical "
        "decision-making.",
    ),
    "days": ("日", "days"),
    "na": ("–", "–"),
}


def _t(key: str, lang: str) -> str:
    pair = _S.get(key)
    if pair is None:
        return key
    return pair[0] if lang == "ja" else pair[1]


# ---------- PDF builder ----------


class _ReportPDF(FPDF):
    def __init__(self, lang: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.lang = lang
        self.add_font(_FONT_NAME, "", _FONT_PATH)
        self.set_auto_page_break(auto=True, margin=18)

    def _font(self, size: float = 10):
        self.set_font(_FONT_NAME, size=size)

    def header(self):
        self._font(8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, "SCI Rehab Suite", align="L")
        self.cell(
            0, 5,
            f"{_t('generated', self.lang)}: {date.today().isoformat()}",
            align="R", new_x="LMARGIN", new_y="NEXT",
        )
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self._font(7)
        self.set_text_color(140, 140, 140)
        self.cell(0, 4, _t("disclaimer", self.lang), align="C",
                  new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 4, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_heading(self, text: str):
        self._font(12)
        self.set_text_color(12, 90, 102)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

    def kv_pair(self, label: str, value: str, label_w: float = 30):
        self._font(9)
        self.set_text_color(80, 80, 80)
        self.cell(label_w, 5.5, label, new_x="END")
        self.set_text_color(0, 0, 0)
        self.cell(0, 5.5, value, new_x="LMARGIN", new_y="NEXT")


def _fig_to_png(fig, width: int = 700, height: int = 280) -> bytes:
    return fig.to_image(format="png", width=width, height=height, scale=2)


def _shap_fig_for_pdf(fig):
    """Return a copy of the SHAP figure with margins adjusted for PDF rendering.

    The dashboard SHAP figure uses left margin=260px, designed for browser viewport.
    For PDF export at 1000px width, we widen the left margin so long feature labels
    (e.g. "Ossified posterior longitudinal ligament") render fully.
    """
    fig2 = copy.deepcopy(fig)
    fig2.update_layout(margin=dict(l=340, r=30, t=10, b=50))
    return fig2


def _safe(v, na: str = "–", fmt: str = "{:.0f}"):
    if v is None:
        return na
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def generate_patient_report(
    meta: dict,
    predictions: dict,
    trajectory_fig,
    shap_fig,
    outcome_labels: list[tuple[str, str, str]],
    lang: str,
) -> bytes:
    """Build a 2-page PDF report for one patient episode.

    Parameters
    ----------
    meta : dict
        From ``patient_meta()`` — keys: id_number, key_record, age, sex,
        paralysis, ais_admit, nli_admit, los_days, y_discharge_scim, y_discharge_ais.
    predictions : dict
        ``{outcome_key: {task, pred, lo, hi, pred_class, pred_prob, proba, observed,
        conformal_set_letters, ...}}``
    trajectory_fig : plotly Figure or None
        SCIM-III timeline chart with trajectory overlay.
    shap_fig : plotly Figure or None
        Local SHAP bar chart for SCIM-total.
    outcome_labels : list of (key, display_label, unit_label)
        Display name and unit for each outcome, in registry order.
    lang : str
        ``"ja"`` or ``"en"``.

    Returns
    -------
    bytes
        PDF file content.
    """
    pdf = _ReportPDF(lang)
    pdf.alias_nb_pages()
    na = _t("na", lang)

    # ===== PAGE 1 =====
    pdf.add_page()

    # -- Title --
    pdf._font(18)
    pdf.set_text_color(12, 90, 102)
    pdf.cell(0, 10, _t("title", lang), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    pid = meta.get("id_number")
    kr = meta.get("key_record")
    pdf._font(10)
    pid_str = str(int(pid)) if pid is not None else na
    kr_str = str(int(kr)) if kr is not None else na
    pdf.cell(
        0, 6,
        f"{_t('patient_id', lang)}: #{pid_str}    {_t('episode', lang)}: #{kr_str}",
        align="C", new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(3)

    # -- Demographics (two-column key-value) --
    pdf.section_heading(_t("demographics", lang))

    age_str = _safe(meta.get("age"), na)
    if meta.get("age") is not None:
        age_str += " 歳" if lang == "ja" else ""

    sex_str = meta.get("sex") or na
    para_str = meta.get("paralysis") or na
    ais_str = f"AIS {meta['ais_admit']}" if meta.get("ais_admit") else na
    nli_str = str(meta["nli_admit"]) if meta.get("nli_admit") else na
    los_str = (
        f"{meta['los_days']:.0f} {_t('days', lang)}" if meta.get("los_days") is not None else na
    )

    col_w = 63
    y_start = pdf.get_y()
    pdf.kv_pair(_t("age", lang), age_str)
    pdf.kv_pair(_t("paralysis", lang), para_str)
    pdf.kv_pair(_t("nli_admit", lang), nli_str)
    y_col1 = pdf.get_y()

    pdf.set_y(y_start)
    pdf.set_x(10 + col_w)
    pdf.kv_pair(_t("sex", lang), sex_str)
    pdf.set_x(10 + col_w)
    pdf.kv_pair(_t("ais_admit", lang), ais_str)
    pdf.set_x(10 + col_w)
    pdf.kv_pair(_t("los", lang), los_str)

    pdf.set_y(max(y_col1, pdf.get_y()) + 2)

    # -- Predictions table --
    pdf.section_heading(_t("predictions", lang))

    # Column widths (mm): outcome label gets the most space.
    col_widths = [70, 24, 34, 26, 36]
    headers = [
        _t("pred_outcome", lang),
        _t("pred_value", lang),
        _t("pred_pi", lang),
        _t("pred_observed", lang),
        _t("pred_set", lang),
    ]

    pdf._font(8)
    pdf.set_fill_color(235, 245, 247)
    pdf.set_text_color(12, 90, 102)
    for i, (hdr, w) in enumerate(zip(headers, col_widths, strict=True)):
        pdf.cell(
            w, 6, hdr, border=1, fill=True,
            new_x="END" if i < len(headers) - 1 else "LMARGIN",
            new_y="TOP" if i < len(headers) - 1 else "NEXT",
        )
    pdf.set_text_color(0, 0, 0)

    pdf._font(8)
    for key, display_label, _unit_label in outcome_labels:
        p = predictions.get(key, {})
        task = p.get("task", "regression")

        if task == "regression":
            pred_val = _safe(p.get("pred"), na)
            lo_val = p.get("lo")
            hi_val = p.get("hi")
            pi_str = f"{_safe(lo_val, na)} – {_safe(hi_val, na)}" if lo_val is not None else na
            obs = p.get("observed")
            obs_str = _safe(obs, na) if obs is not None else na
            set_str = ""
        else:
            pred_class = p.get("pred_class", na)
            pred_prob = p.get("pred_prob")
            pred_val = f"AIS {pred_class}" + (f" ({pred_prob:.0%})" if pred_prob else "")
            pi_str = ""
            obs_ord = p.get("observed")
            if obs_ord is not None:
                obs_letter = AIS_ORD_TO_LETTER.get(int(obs_ord), "?")
                obs_str = f"AIS {obs_letter}"
            else:
                obs_str = na
            set_letters = p.get("conformal_set_letters")
            set_str = f"{{{', '.join(set_letters)}}}" if set_letters else ""

        row_data = [display_label, pred_val, pi_str, obs_str, set_str]
        for i, (val, w) in enumerate(zip(row_data, col_widths, strict=True)):
            pdf.cell(
                w, 5.5, val, border=1,
                new_x="END" if i < len(row_data) - 1 else "LMARGIN",
                new_y="TOP" if i < len(row_data) - 1 else "NEXT",
            )

    pdf.ln(4)

    # -- Trajectory chart --
    if trajectory_fig is not None:
        pdf.section_heading(_t("trajectory", lang))
        traj_png = _fig_to_png(trajectory_fig, width=800, height=320)
        img_w = 190
        img_h = 190 * 320 / 800
        pdf.image(io.BytesIO(traj_png), x=10, w=img_w, h=img_h)
        pdf.ln(2)

    # ===== PAGE 2 =====
    pdf.add_page()

    # -- SHAP chart (with adjusted margins for label visibility) --
    if shap_fig is not None:
        pdf.section_heading(_t("shap", lang))
        pdf_shap = _shap_fig_for_pdf(shap_fig)
        shap_png = _fig_to_png(pdf_shap, width=1000, height=400)
        img_w = 190
        img_h = 190 * 400 / 1000
        pdf.image(io.BytesIO(shap_png), x=10, w=img_w, h=img_h)
        pdf.ln(5)

    # -- Methodology --
    pdf.section_heading(_t("methods", lang))
    pdf._font(9)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 5, _t("methods_text", lang))
    pdf.set_text_color(0, 0, 0)

    # -- Output --
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
