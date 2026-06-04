"""Plotly figure factories, split by dashboard tab.

Import as `from rehab_sci.dashboard import figures as fg` (back-compat).
Submodules: overview, insights, patient, simulator, methods (+ _common helpers).
"""

from rehab_sci.dashboard.figures.insights import (
    fig_dependence,
    fig_global_shap_importance,
    fig_interaction_dependence,
    fig_interaction_heatmap,
    fig_subgroup_box,
)
from rehab_sci.dashboard.figures.methods import (
    fig_calibration_curve,
    fig_confusion_matrix,
    fig_dataquality_overview,
    fig_pred_vs_observed,
    fig_residual_hist,
    fig_temporal_drift,
)
from rehab_sci.dashboard.figures.overview import (
    ARCHETYPE_NAMES_EN,
    ARCHETYPE_NAMES_JA,
    PALETTE_ARCHETYPE,
    fig_age_distribution,
    fig_ais_admit_discharge_sankey,
    fig_archetype_curves,
    fig_archetype_demographics,
    fig_discharge_scim,
    fig_injury_treemap,
    fig_mechanism,
    fig_recovery_curves,
    fig_sex_donut,
)
from rehab_sci.dashboard.figures.patient import (
    fig_neighbor_ais_distribution,
    fig_neighbor_outcomes,
    fig_patient_prediction,
    fig_patient_scim_timeline,
)
from rehab_sci.dashboard.figures.simulator import (
    fig_sim_trajectory,
)

__all__ = [
    "ARCHETYPE_NAMES_EN",
    "ARCHETYPE_NAMES_JA",
    "PALETTE_ARCHETYPE",
    "fig_age_distribution",
    "fig_ais_admit_discharge_sankey",
    "fig_archetype_curves",
    "fig_archetype_demographics",
    "fig_calibration_curve",
    "fig_confusion_matrix",
    "fig_dataquality_overview",
    "fig_dependence",
    "fig_discharge_scim",
    "fig_global_shap_importance",
    "fig_injury_treemap",
    "fig_interaction_dependence",
    "fig_interaction_heatmap",
    "fig_mechanism",
    "fig_neighbor_ais_distribution",
    "fig_neighbor_outcomes",
    "fig_patient_prediction",
    "fig_patient_scim_timeline",
    "fig_pred_vs_observed",
    "fig_recovery_curves",
    "fig_residual_hist",
    "fig_sex_donut",
    "fig_sim_trajectory",
    "fig_subgroup_box",
    "fig_temporal_drift",
]
