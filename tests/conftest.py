"""Shared fixtures + skip gates for the F26 invariant harness.

The raw CSV (``ALL_SCIDATA.csv``) and the trained model bundles are gitignored,
so the whole suite is a *no-op* (every data/model test skips, none fails) on a
checkout that lacks them.  Two gates:

* ``af`` / ``ep`` / ``long_df`` / ``schema`` need only the raw CSV.
* ``state`` (and everything built on it) needs the CSV **and** the core model
  bundles; it skips when importing ``rehab_sci.dashboard.state`` raises.

Fixtures are session-scoped: the CSV is parsed once and the dashboard state is
imported once for the whole run.  Pure-registry tests (no fixture) always run.
"""

from __future__ import annotations

import pytest

from rehab_sci.data.loader import RAW_PATH_DEFAULT

RAW_PRESENT = RAW_PATH_DEFAULT.exists()
_RAW_REASON = "raw CSV absent (gitignored) — data-dependent tests skipped"


def _require_raw() -> None:
    if not RAW_PRESENT:
        pytest.skip(_RAW_REASON)


@pytest.fixture(scope="session")
def af():
    """The analysis frame (episode + longitudinal). Needs the raw CSV only."""
    _require_raw()
    from rehab_sci.data.dataset import build_analysis_dataset

    return build_analysis_dataset()


@pytest.fixture(scope="session")
def ep(af):
    return af.df


@pytest.fixture(scope="session")
def long_df(af):
    return af.longitudinal


@pytest.fixture(scope="session")
def schema(af):
    return af.schema


@pytest.fixture(scope="session")
def state():
    """The imported dashboard state (data + all model bundles loaded once).

    Skips when the CSV or a core bundle is absent — i.e. when
    ``rehab_sci.dashboard.state`` cannot import.
    """
    _require_raw()
    try:
        import rehab_sci.dashboard.state as S
    except Exception as exc:  # missing core bundle / metrics json
        pytest.skip(f"dashboard state import failed (model bundles absent?): {exc!r}")
    return S


@pytest.fixture(scope="session")
def sample_key_record(state):
    """A real KeyRecordNumber with admission features and a recorded admission AIS grade —
    drives every per-patient inference path (conversion/level-descent gate on a real AIS)."""
    import pandas as pd

    ep = state.EP
    mask = ep["IDNumber"].notna() & pd.to_numeric(ep["AIS_ord"], errors="coerce").notna()
    if not mask.any():
        pytest.skip("no episode with a recorded admission AIS")
    return int(ep.loc[mask, "KeyRecordNumber"].iloc[0])


@pytest.fixture(scope="session")
def contrast_episodes(state):
    """(severe, mild) KeyRecordNumbers: most-complete AIS-A (min total motor) vs AIS-D (max total
    motor).  The extreme severity contrast any correctly-aligned model must separate sharply — a
    row-misalignment regression collapses the gap while leaving aggregate AUC high (§0b)."""
    import pandas as pd

    ep = state.EP
    ais = pd.to_numeric(ep["AIS_ord"], errors="coerce")
    tm = pd.to_numeric(ep["TotalMotor"], errors="coerce")
    have_id = ep["IDNumber"].notna()
    sev = ep[have_id & (ais == 1)].assign(_tm=tm).sort_values("_tm")
    mil = ep[have_id & (ais == 4)].assign(_tm=tm).sort_values("_tm", ascending=False)
    if sev.empty or mil.empty:
        pytest.skip("cohort lacks an AIS-A and/or AIS-D episode for the behavioral contrast")
    return int(sev["KeyRecordNumber"].iloc[0]), int(mil["KeyRecordNumber"].iloc[0])
