"""Behavioral / alignment tripwires — the severe-vs-mild contrast a correctly
aligned model must preserve.

Per the §0b lesson, a row-scramble bug kept per-segment AUC high yet flattened
the personalization gap, so AUC alone missed it.  These assertions check the
*gap* between an extreme-complete (AIS-A, zero motor) and a mild-incomplete
(AIS-D, full motor) episode, not a metric — a re-introduced misalignment that
flattens personalization trips them.  Thresholds carry wide margins around the
observed values (independence 0.9 vs 16.7; topography 0.27 vs 0.98).
"""

from __future__ import annotations

import numpy as np
import pytest


def _row(state, kr):
    from rehab_sci.dashboard import compute as C

    return C.episode_row_for_model(kr)


def test_independence_monotone_with_severity(state, contrast_episodes):
    from rehab_sci.dashboard import compute as C

    if state.INDEPENDENCE_BUNDLE is None:
        pytest.skip("independence bundle absent")
    sev_kr, mil_kr = contrast_episodes
    sev = C.predict_independence(_row(state, sev_kr))["expected_count"]
    mil = C.predict_independence(_row(state, mil_kr))["expected_count"]
    # AIS-A complete expects very few independent functions; AIS-D many (of 18).
    assert sev < 5.0
    assert mil > 12.0
    assert mil - sev > 8.0  # a scramble would collapse this gap toward the cohort mean


def test_topography_antigravity_gap(state, contrast_episodes):
    from rehab_sci.dashboard import compute as C

    if state.TOPOGRAPHY_BUNDLE is None:
        pytest.skip("topography bundle absent")
    sev_kr, mil_kr = contrast_episodes

    def mean_seg_prob(kr):
        res = C.predict_topography(_row(state, kr), C.topography_admission_grades(kr))
        probs = [s["prob"] for s in res["segments"] if s.get("prob") is not None]
        return float(np.mean(probs))

    sev, mil = mean_seg_prob(sev_kr), mean_seg_prob(mil_kr)
    assert sev < 0.5 < mil
    assert mil - sev > 0.3


def test_conversion_complete_unlikely_ambulatory(state, contrast_episodes):
    from rehab_sci.dashboard import compute as C

    if state.CONVERSION_BUNDLE is None:
        pytest.skip("conversion bundle absent")
    sev_kr, _ = contrast_episodes
    res = C.predict_conversion(_row(state, sev_kr))
    amb = res["endpoints"]["ambulatory"]
    assert amb["applicable"]  # AIS-A is in the ambulatory (A–C) at-risk cohort
    assert amb["prob"] < 0.3  # a motor-complete injury rarely converts to ambulatory
