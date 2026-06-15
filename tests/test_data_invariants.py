"""§1 data invariants — exact cohort shapes a loader / ghost-filter regression would break.

Every constant here is load-bearing and cited in AGENT_NOTES §1.  A drift means
``build_analysis_dataset`` changed behavior (or the raw file changed), not that
the test is wrong — confirm against the data before editing a number.
"""

from __future__ import annotations

import pandas as pd

from rehab_sci.data.dataset import ADMISSION_FEATURES

N_EPISODES = 899  # post ghost-filter universe
N_PATIENTS = 866  # unique non-null IDNumber
N_ORPHANS = 27  # admission features present but IDNumber null
N_TIMEPOINTS = 26  # rectangular grid: every episode has all 26 slots
N_LONG = N_EPISODES * N_TIMEPOINTS  # 23374
N_FEATURES = 30  # 2 demographics + 9 injury/admin + 15 ISNCSCI/AIS + 4 SCIM-ADL
CARDINALITY = {  # discharge-outcome availability over the 899-episode universe
    "y_discharge_scim": 507,
    "y_discharge_ais": 638,
    "y_discharge_wisci": 50,  # too sparse to model — stays dropped
    "LOS_days": 682,
}


def test_episode_and_patient_counts(ep):
    assert len(ep) == N_EPISODES
    assert int(ep["IDNumber"].nunique()) == N_PATIENTS
    assert int(ep["IDNumber"].isna().sum()) == N_ORPHANS


def test_long_frame_is_rectangular(long_df):
    assert len(long_df) == N_LONG
    # every episode has exactly the 26 timepoint slots — the perfectly rectangular grid
    assert long_df.groupby("KeyRecordNumber").size().unique().tolist() == [N_TIMEPOINTS]


def test_admission_feature_partition(af):
    assert len(ADMISSION_FEATURES) == N_FEATURES
    assert len(af.feature_cols) == N_FEATURES
    # numeric + categorical partition the feature set exactly (no overlap, no gap)
    assert len(af.numeric_cols) + len(af.categorical_cols) == N_FEATURES
    assert set(af.numeric_cols) | set(af.categorical_cols) == set(af.feature_cols)
    assert not (set(af.numeric_cols) & set(af.categorical_cols))


def test_no_outcome_leakage_into_features(af):
    # no target leaks into the feature set; BusinessYear is a passport, never a feature (§1)
    assert all(not c.startswith("y_") for c in af.feature_cols)
    assert "BusinessYear" not in af.feature_cols
    assert "LOS_days" not in af.feature_cols


def test_outcome_cardinalities(ep):
    for col, expected in CARDINALITY.items():
        assert int(ep[col].notna().sum()) == expected, col


def test_numeric_features_have_numeric_dtype(af):
    # a surviving cp932 sentinel ("_", "NT", "ND", "1*"…) would leave an object column;
    # normalize() coerces them to NaN so every numeric feature stays numeric-dtyped (§1).
    ep = af.df
    for col in af.numeric_cols:
        assert pd.api.types.is_numeric_dtype(ep[col]), col


def test_ais_ordinal_domain(ep):
    # AIS_ord is the A=1…E=5 mapping (constants.AIS_LETTER_TO_ORD); nothing else may appear
    vals = set(pd.to_numeric(ep["AIS_ord"], errors="coerce").dropna().unique().tolist())
    assert vals <= {1, 2, 3, 4, 5}
