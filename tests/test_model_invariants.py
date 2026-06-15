"""§3 model invariants — registry/metrics contracts checkable without retraining.

Registry-only tests always run.  Metric tests read the committed
``models/training_metrics.json`` (the performance source of truth) and skip if
absent.  Bundle-shape tests use the loaded dashboard ``state`` and skip when the
gitignored bundles are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rehab_sci.models.outcomes import OUTCOMES, get

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "models" / "training_metrics.json"

RANDOM_STATE = 20260518
N_FEATURES = 30
EXPECTED_KEYS = (
    "scim_total",
    "scim_self_care",
    "scim_resp_sphincter",
    "scim_mobility",
    "ais_discharge",
    "los_days",
    "delta_uems",
    "delta_lems",
    "delta_totalmotor",
    "delta_lighttouch",
    "delta_pinprick",
)
DELTA_KEYS = tuple(k for k in EXPECTED_KEYS if k.startswith("delta_"))
ABS_SCIM_KEYS = ("scim_total", "scim_self_care", "scim_resp_sphincter", "scim_mobility")


@pytest.fixture(scope="session")
def metrics():
    if not METRICS_PATH.exists():
        pytest.skip("training_metrics.json absent")
    with METRICS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


# ---- registry (no artifacts) ------------------------------------------------


def test_outcome_registry_shape():
    assert tuple(s.key for s in OUTCOMES) == EXPECTED_KEYS
    assert len({s.key for s in OUTCOMES}) == len(OUTCOMES) == 11
    # exactly one multiclass head (AIS); every other head is regression
    assert sum(s.task == "multiclass" for s in OUTCOMES) == 1
    assert get("ais_discharge").task == "multiclass"
    assert all(get(k).task == "regression" for k in (*ABS_SCIM_KEYS, "los_days", *DELTA_KEYS))


def test_ais_classes_severity_ordered():
    spec = get("ais_discharge")
    assert spec.class_codes == (1, 2, 3, 4, 5)
    assert spec.class_labels == ("A", "B", "C", "D", "E")


def test_transform_and_clip_contracts():
    # LOS is the only log1p head; Δ heads alone allow deterioration (clip_min < 0); the
    # absolute SCIM heads floor at 0 (§3 — the clip_min<0 flag drives the signed Δ UI).
    assert get("los_days").transform == "log1p"
    assert all(get(k).transform is None for k in (*ABS_SCIM_KEYS, *DELTA_KEYS))
    for k in DELTA_KEYS:
        assert get(k).clip_min < 0, k
    for k in ABS_SCIM_KEYS:
        assert get(k).clip_min == 0, k


# ---- training metrics (committed json) --------------------------------------


def test_metrics_match_registry(metrics):
    assert metrics["random_state"] == RANDOM_STATE
    assert metrics["n_features"] == N_FEATURES
    assert set(metrics["outcome_keys"]) == set(EXPECTED_KEYS)
    assert set(metrics["outcomes"]) == set(EXPECTED_KEYS)
    for s in OUTCOMES:
        block = metrics["outcomes"][s.key]
        assert block["task"] == s.task, s.key
        assert block["target_col"] == s.target_col, s.key


def test_regression_conformal_coverage_sane(metrics):
    # the split-conformal 80% PI must land in a believable band on the holdout; the
    # conformal layer is REQUIRED (raw quantile heads give ~0.41 coverage, §3).
    for s in OUTCOMES:
        if s.task != "regression":
            continue
        cov = metrics["outcomes"][s.key]["test"]["conformal_coverage_80"]
        assert 0.6 <= cov <= 0.95, (s.key, cov)


def test_ais_ordinal_metrics_present(metrics):
    test = metrics["outcomes"]["ais_discharge"]["test"]
    assert "accuracy" in test
    assert "kappa_quadratic" in test  # ordinal-aware metric reported alongside accuracy
    assert test["aps_coverage_80"] >= 0.8  # APS sets run conservative for K=5 (§3)


# ---- loaded bundle shapes (gitignored artifacts) ----------------------------


def test_every_outcome_bundle_loads(state):
    for s in OUTCOMES:
        bundle = state.OUTCOME_BUNDLES[s.key]
        fspec = bundle["feature_spec"]
        assert len(fspec["feature_cols"]) == N_FEATURES, s.key
        if s.task == "regression":
            assert "median" in bundle, s.key
            assert any(k.startswith("conformal_q") for k in fspec), s.key  # REQUIRED layer
            assert fspec["transform"] == s.transform, s.key
        else:
            assert "clf" in bundle, s.key
            assert any(k.startswith("aps_q") for k in fspec), s.key


def test_shared_feature_spec(state):
    assert len(state.FEATURE_SPEC["feature_cols"]) == N_FEATURES
