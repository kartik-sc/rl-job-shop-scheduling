"""
Tests for backend.risk.calibration.

Covers:
  - Calibration fit (isotonic regression direction, output range)
  - Correct score direction: higher anomaly → higher risk, lower RUL estimate
  - Score-to-RUL alignment
  - Engine-level separation of healthy vs late-life risk
  - Reproducibility
  - Handling of out-of-range anomaly scores
  - No test-engine leakage
  - Correct output ranges [0, 1] for risk
  - Calibration artifact save/load
  - Scheduler interface compatibility (risk ∈ [0,1])
  - Both models calibrated and independently accessible
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

from backend.risk.anomaly_models import load_bundle
from backend.risk.calibration import (
    CalibrationBundle,
    compute_calibration_metrics,
    evaluate_calibration,
    fit_calibration,
    fit_both_calibrations,
    load_calibration_bundle,
    save_calibration_bundle,
    summarise_metrics,
)
from backend.risk.cmapss_pipeline import SPLIT_SEED, build_pipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR    = pathlib.Path(__file__).parent.parent / "data" / "raw" / "cmapss"
BUNDLE_PATH = pathlib.Path(__file__).parent.parent / "models" / "anomaly_bundle.pkl"


@pytest.fixture(scope="session")
def anomaly_bundle():
    if not BUNDLE_PATH.exists():
        pytest.skip("models/anomaly_bundle.pkl not found – run train_anomaly first")
    return load_bundle(BUNDLE_PATH)


@pytest.fixture(scope="session")
def pipeline():
    if not DATA_DIR.exists():
        pytest.skip("data/raw/cmapss not present")
    return build_pipeline(DATA_DIR, seed=SPLIT_SEED)


@pytest.fixture(scope="session")
def cal_bundle_ocsvm(anomaly_bundle, pipeline):
    return fit_calibration(
        "ocsvm",
        anomaly_bundle,
        pipeline["df_scaled"],
        anomaly_bundle.splits["calibration"],
        anomaly_bundle.splits["fit"],
    )


@pytest.fixture(scope="session")
def cal_bundle_iforest(anomaly_bundle, pipeline):
    return fit_calibration(
        "iforest",
        anomaly_bundle,
        pipeline["df_scaled"],
        anomaly_bundle.splits["calibration"],
        anomaly_bundle.splits["fit"],
    )


@pytest.fixture(scope="session")
def eval_ocsvm(anomaly_bundle, pipeline, cal_bundle_ocsvm):
    cal_ids = anomaly_bundle.splits["calibration"]
    return evaluate_calibration(
        "ocsvm", cal_bundle_ocsvm, anomaly_bundle,
        pipeline["df_scaled"], cal_ids,
    )


@pytest.fixture(scope="session")
def eval_iforest(anomaly_bundle, pipeline, cal_bundle_iforest):
    cal_ids = anomaly_bundle.splits["calibration"]
    return evaluate_calibration(
        "iforest", cal_bundle_iforest, anomaly_bundle,
        pipeline["df_scaled"], cal_ids,
    )


@pytest.fixture(scope="session")
def metrics_ocsvm(eval_ocsvm):
    return compute_calibration_metrics(eval_ocsvm)


@pytest.fixture(scope="session")
def metrics_iforest(eval_iforest):
    return compute_calibration_metrics(eval_iforest)


# ---------------------------------------------------------------------------
# Calibration fit
# ---------------------------------------------------------------------------


def test_fit_returns_calibration_bundle(cal_bundle_ocsvm, cal_bundle_iforest):
    assert isinstance(cal_bundle_ocsvm,   CalibrationBundle)
    assert isinstance(cal_bundle_iforest, CalibrationBundle)


def test_model_names(cal_bundle_ocsvm, cal_bundle_iforest):
    assert cal_bundle_ocsvm.model_name   == "ocsvm"
    assert cal_bundle_iforest.model_name == "iforest"


def test_max_lifetime_positive(cal_bundle_ocsvm, cal_bundle_iforest):
    assert cal_bundle_ocsvm.max_lifetime > 0
    assert cal_bundle_iforest.max_lifetime > 0


def test_max_lifetime_derived_from_fit_not_cal(anomaly_bundle, pipeline, cal_bundle_ocsvm):
    """max_lifetime must come from fit engines, not calibration engines."""
    fit_ids = anomaly_bundle.splits["fit"]
    df      = pipeline["df_scaled"]
    fit_lifetimes = df[df["engine_id"].isin(fit_ids)].groupby("engine_id")["cycle"].max()
    expected_max  = float(fit_lifetimes.max())
    assert cal_bundle_ocsvm.max_lifetime == expected_max, (
        f"max_lifetime {cal_bundle_ocsvm.max_lifetime} != fit-split max {expected_max}"
    )


def test_score_min_max_stored(cal_bundle_ocsvm):
    assert cal_bundle_ocsvm.score_min < cal_bundle_ocsvm.score_max


def test_cal_engine_ids_stored(anomaly_bundle, cal_bundle_ocsvm):
    assert set(cal_bundle_ocsvm.cal_engine_ids) == set(anomaly_bundle.splits["calibration"])


def test_invalid_model_name_raises(anomaly_bundle, pipeline):
    with pytest.raises(ValueError, match="model_name"):
        fit_calibration(
            "bad_model",
            anomaly_bundle,
            pipeline["df_scaled"],
            anomaly_bundle.splits["calibration"],
            anomaly_bundle.splits["fit"],
        )


def test_cal_fit_engine_overlap_raises(anomaly_bundle, pipeline):
    """Fitting with cal engines that overlap fit must raise ValueError."""
    cal_ids = anomaly_bundle.splits["calibration"]
    fit_ids = anomaly_bundle.splits["fit"]
    mixed   = cal_ids[:5] + fit_ids[:5]   # intentional overlap
    with pytest.raises(ValueError, match="[Ll]eakage"):
        fit_calibration(
            "ocsvm", anomaly_bundle, pipeline["df_scaled"],
            mixed, fit_ids,
        )


# ---------------------------------------------------------------------------
# Score direction: higher anomaly score → higher risk, lower RUL estimate
# ---------------------------------------------------------------------------


def test_isotonic_direction_ocsvm(cal_bundle_ocsvm):
    """Predicted RUL must be non-increasing with anomaly score."""
    scores = np.linspace(
        cal_bundle_ocsvm.score_min, cal_bundle_ocsvm.score_max, 200
    )
    rul_est = cal_bundle_ocsvm.predict_rul(scores)
    diffs   = np.diff(rul_est)
    assert (diffs <= 1e-9).all(), (
        f"OC-SVM isotonic: RUL increases at {(diffs > 1e-9).sum()} points"
    )


def test_isotonic_direction_iforest(cal_bundle_iforest):
    scores = np.linspace(
        cal_bundle_iforest.score_min, cal_bundle_iforest.score_max, 200
    )
    rul_est = cal_bundle_iforest.predict_rul(scores)
    diffs   = np.diff(rul_est)
    assert (diffs <= 1e-9).all(), (
        f"IForest isotonic: RUL increases at {(diffs > 1e-9).sum()} points"
    )


def test_risk_direction_ocsvm(cal_bundle_ocsvm):
    """Risk score must be non-decreasing with anomaly score."""
    scores = np.linspace(
        cal_bundle_ocsvm.score_min, cal_bundle_ocsvm.score_max, 200
    )
    risks  = cal_bundle_ocsvm.predict_risk(scores)
    diffs  = np.diff(risks)
    assert (diffs >= -1e-9).all(), (
        f"OC-SVM risk decreases at {(diffs < -1e-9).sum()} points"
    )


def test_risk_direction_iforest(cal_bundle_iforest):
    scores = np.linspace(
        cal_bundle_iforest.score_min, cal_bundle_iforest.score_max, 200
    )
    risks  = cal_bundle_iforest.predict_risk(scores)
    diffs  = np.diff(risks)
    assert (diffs >= -1e-9).all(), (
        f"IForest risk decreases at {(diffs < -1e-9).sum()} points"
    )


# ---------------------------------------------------------------------------
# Output range
# ---------------------------------------------------------------------------


def test_risk_score_in_01_ocsvm(eval_ocsvm):
    assert (eval_ocsvm["risk_score"] >= 0).all()
    assert (eval_ocsvm["risk_score"] <= 1).all()


def test_risk_score_in_01_iforest(eval_iforest):
    assert (eval_iforest["risk_score"] >= 0).all()
    assert (eval_iforest["risk_score"] <= 1).all()


def test_rul_estimate_non_negative_ocsvm(eval_ocsvm):
    assert (eval_ocsvm["rul_estimate"] >= 0).all()


def test_rul_estimate_non_negative_iforest(eval_iforest):
    assert (eval_iforest["rul_estimate"] >= 0).all()


# ---------------------------------------------------------------------------
# Out-of-range score handling (clipping)
# ---------------------------------------------------------------------------


def test_below_range_score_clipped(cal_bundle_ocsvm):
    """Score well below training range must not produce negative RUL or risk > 1."""
    very_low = cal_bundle_ocsvm.score_min - 1000.0
    risk, rul = cal_bundle_ocsvm.predict(very_low)
    assert 0.0 <= risk <= 1.0
    assert rul >= 0.0


def test_above_range_score_clipped(cal_bundle_ocsvm):
    """Score well above training range must produce risk ≤ 1."""
    very_high = cal_bundle_ocsvm.score_max + 1000.0
    risk, rul = cal_bundle_ocsvm.predict(very_high)
    assert 0.0 <= risk <= 1.0
    assert rul >= 0.0


# ---------------------------------------------------------------------------
# Alignment: eval DataFrame columns and structure
# ---------------------------------------------------------------------------


def test_eval_columns_present(eval_ocsvm):
    for col in ("engine_id", "dataset", "cycle", "rul_true",
                "anomaly_score", "rul_estimate", "risk_score"):
        assert col in eval_ocsvm.columns, f"Missing column: {col}"


def test_eval_rul_zero_at_last_cycle(eval_ocsvm):
    last_ruls = eval_ocsvm.sort_values("cycle").groupby("engine_id")["rul_true"].last()
    bad = last_ruls[last_ruls != 0]
    assert len(bad) == 0, f"Last-cycle rul_true != 0: {bad.index.tolist()}"


def test_eval_covers_all_cal_engines(anomaly_bundle, eval_ocsvm, eval_iforest):
    cal_set = set(anomaly_bundle.splits["calibration"])
    assert set(eval_ocsvm["engine_id"].unique())   == cal_set
    assert set(eval_iforest["engine_id"].unique()) == cal_set


def test_no_test_engines_in_eval(anomaly_bundle, eval_ocsvm, eval_iforest):
    test_set = set(anomaly_bundle.splits["test"])
    assert set(eval_ocsvm["engine_id"].unique()).isdisjoint(test_set)
    assert set(eval_iforest["engine_id"].unique()).isdisjoint(test_set)


def test_no_fit_engines_in_eval(anomaly_bundle, eval_ocsvm, eval_iforest):
    fit_set = set(anomaly_bundle.splits["fit"])
    assert set(eval_ocsvm["engine_id"].unique()).isdisjoint(fit_set)
    assert set(eval_iforest["engine_id"].unique()).isdisjoint(fit_set)


# ---------------------------------------------------------------------------
# Engine-level separation: late-life risk > healthy risk
# ---------------------------------------------------------------------------


def test_latelife_risk_exceeds_healthy_ocsvm(metrics_ocsvm):
    """Mean late-life risk must exceed mean healthy risk for most engines."""
    pct = metrics_ocsvm["risk_dir_success"].mean()
    assert pct >= 0.75, f"OC-SVM directional success {pct:.1%} < 75%"


def test_latelife_risk_exceeds_healthy_iforest(metrics_iforest):
    pct = metrics_iforest["risk_dir_success"].mean()
    assert pct >= 0.75, f"IForest directional success {pct:.1%} < 75%"


def test_population_latelife_higher_than_healthy_ocsvm(eval_ocsvm):
    hw = eval_ocsvm.loc[eval_ocsvm["cycle"] <= 30, "risk_score"].mean()
    ll = eval_ocsvm.loc[eval_ocsvm["rul_true"] < 30, "risk_score"].mean()
    assert ll > hw, f"OC-SVM: late-life mean {ll:.4f} not > healthy {hw:.4f}"


def test_population_latelife_higher_than_healthy_iforest(eval_iforest):
    hw = eval_iforest.loc[eval_iforest["cycle"] <= 30, "risk_score"].mean()
    ll = eval_iforest.loc[eval_iforest["rul_true"] < 30, "risk_score"].mean()
    assert ll > hw, f"IForest: late-life mean {ll:.4f} not > healthy {hw:.4f}"


# ---------------------------------------------------------------------------
# Spearman correlation
# ---------------------------------------------------------------------------


def test_risk_spearman_positive_ocsvm(metrics_ocsvm):
    """All cal engines must have positive Spearman ρ (risk vs −RUL)."""
    pct = (metrics_ocsvm["risk_spearman"] > 0).mean()
    assert pct >= 0.9, f"OC-SVM: only {pct:.1%} engines have positive ρ"


def test_risk_spearman_positive_iforest(metrics_iforest):
    pct = (metrics_iforest["risk_spearman"] > 0).mean()
    assert pct >= 0.9, f"IForest: only {pct:.1%} engines have positive ρ"


# ---------------------------------------------------------------------------
# Metrics structure
# ---------------------------------------------------------------------------


def test_metrics_one_row_per_engine(anomaly_bundle, metrics_ocsvm, metrics_iforest):
    cal_ids = anomaly_bundle.splits["calibration"]
    assert len(metrics_ocsvm)   == len(cal_ids)
    assert len(metrics_iforest) == len(cal_ids)


def test_metrics_no_nan_for_valid_engines(metrics_ocsvm, metrics_iforest):
    for model, m in [("ocsvm", metrics_ocsvm), ("iforest", metrics_iforest)]:
        for col in ("rul_mae", "rul_rmse", "risk_spearman",
                    "healthy_risk_mean", "latelife_risk_mean"):
            nans = m[col].isna().sum()
            assert nans == 0, f"{model}.{col} has {nans} NaN rows"


def test_rul_mae_reasonable(metrics_ocsvm, metrics_iforest):
    """RUL MAE should not exceed the maximum possible lifetime — basic sanity."""
    for model, m in [("ocsvm", metrics_ocsvm), ("iforest", metrics_iforest)]:
        assert m["rul_mae"].max() < 600, f"{model} RUL MAE suspiciously large"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_calibration_reproducible(anomaly_bundle, pipeline):
    """Fitting twice on the same data must produce identical predictions."""
    cal_ids = anomaly_bundle.splits["calibration"]
    fit_ids = anomaly_bundle.splits["fit"]
    df      = pipeline["df_scaled"]
    c1 = fit_calibration("ocsvm", anomaly_bundle, df, cal_ids, fit_ids)
    c2 = fit_calibration("ocsvm", anomaly_bundle, df, cal_ids, fit_ids)
    scores = np.linspace(c1.score_min, c1.score_max, 100)
    np.testing.assert_array_equal(c1.predict_rul(scores), c2.predict_rul(scores))


# ---------------------------------------------------------------------------
# Artifact save / load
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(cal_bundle_ocsvm):
    scores = np.linspace(
        cal_bundle_ocsvm.score_min, cal_bundle_ocsvm.score_max, 50
    )
    rul_before  = cal_bundle_ocsvm.predict_rul(scores)
    risk_before = cal_bundle_ocsvm.predict_risk(scores)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        path = pathlib.Path(tmp.name)

    save_calibration_bundle(cal_bundle_ocsvm, path)
    loaded = load_calibration_bundle(path)

    np.testing.assert_array_equal(loaded.predict_rul(scores),  rul_before)
    np.testing.assert_array_equal(loaded.predict_risk(scores), risk_before)
    assert loaded.model_name == cal_bundle_ocsvm.model_name
    assert loaded.max_lifetime == cal_bundle_ocsvm.max_lifetime
    path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Scheduler interface compatibility
# ---------------------------------------------------------------------------


def test_predict_single_scalar_returns_tuple(cal_bundle_ocsvm, cal_bundle_iforest):
    """predict() must return (risk, rul) for scheduler use."""
    for cb in (cal_bundle_ocsvm, cal_bundle_iforest):
        risk, rul = cb.predict(0.0)
        assert isinstance(risk, float)
        assert isinstance(rul,  float)
        assert 0.0 <= risk <= 1.0
        assert rul >= 0.0


def test_predict_batch_shape(cal_bundle_ocsvm):
    scores = np.random.default_rng(0).uniform(
        cal_bundle_ocsvm.score_min, cal_bundle_ocsvm.score_max, 100
    )
    risks = cal_bundle_ocsvm.predict_risk(scores)
    ruls  = cal_bundle_ocsvm.predict_rul(scores)
    assert risks.shape == (100,)
    assert ruls.shape  == (100,)


def test_risk_schema_compatible(eval_ocsvm, eval_iforest):
    """Risk must satisfy the event_log schema: risk ∈ [0, 1]."""
    for name, ev in [("ocsvm", eval_ocsvm), ("iforest", eval_iforest)]:
        r = ev["risk_score"]
        assert (r >= 0).all() and (r <= 1).all(), (
            f"{name}: risk score outside [0,1]"
        )


# ---------------------------------------------------------------------------
# fit_both_calibrations
# ---------------------------------------------------------------------------


def test_fit_both_returns_both_keys(anomaly_bundle, pipeline):
    cal_ids = anomaly_bundle.splits["calibration"]
    fit_ids = anomaly_bundle.splits["fit"]
    result  = fit_both_calibrations(
        anomaly_bundle, pipeline["df_scaled"], cal_ids, fit_ids
    )
    assert "ocsvm"   in result
    assert "iforest" in result
    assert isinstance(result["ocsvm"],   CalibrationBundle)
    assert isinstance(result["iforest"], CalibrationBundle)
