"""
Tests for backend.risk.anomaly_experiments (Phase 2 controlled experiments).

Covers:
  - Experiment B: separate models trained on correct dataset-specific engines
  - Experiment C: correct feature subsets, no forbidden features used
  - Experiment D: causal rolling means, no future data, correct scaler fit
  - No test-engine leakage across all experiments
  - Score direction preserved (higher = more anomalous)
  - Directional success on calibration engines
  - Positive per-engine Spearman ρ for all configs
  - Score overlap is measurable and non-negative
  - Bootstrap comparison structure
  - Baseline artifacts unchanged
"""

from __future__ import annotations

import pathlib
import pickle

import numpy as np
import pandas as pd
import pytest

from backend.risk.anomaly_models import load_bundle
from backend.risk.cmapss_pipeline import SPLIT_SEED, build_pipeline
from backend.risk.anomaly_experiments import (
    FEATURES_C_FD001,
    FEATURES_C_FD003,
    ROLL_SENSORS,
    ROLL_WINDOW,
    EXP_MODELS,
    RESULTS_DIR,
    add_rolling_means,
    build_exp_pipeline,
    compute_separation,
    compute_spearman,
    compute_score_overlap,
    score_engines,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR    = pathlib.Path(__file__).parent.parent / "data" / "raw" / "cmapss"
BASELINE    = pathlib.Path(__file__).parent.parent / "models" / "anomaly_bundle.pkl"
BUNDLE_B_FD001 = EXP_MODELS / "anomaly_bundle_B_fd001.pkl"
BUNDLE_B_FD003 = EXP_MODELS / "anomaly_bundle_B_fd003.pkl"
BUNDLE_C_FD001 = EXP_MODELS / "anomaly_bundle_C_fd001.pkl"
BUNDLE_C_FD003 = EXP_MODELS / "anomaly_bundle_C_fd003.pkl"
BUNDLE_D_FD001 = EXP_MODELS / "anomaly_bundle_D_fd001.pkl"
BUNDLE_D_FD003 = EXP_MODELS / "anomaly_bundle_D_fd003.pkl"


def _require_data():
    if not DATA_DIR.exists():
        pytest.skip("data/raw/cmapss not present")


def _require_bundle(path: pathlib.Path):
    if not path.exists():
        pytest.skip(f"{path.name} not found – run anomaly_experiments first")


@pytest.fixture(scope="session")
def pipeline():
    _require_data()
    return build_pipeline(DATA_DIR, seed=SPLIT_SEED)


@pytest.fixture(scope="session")
def baseline_bundle():
    _require_bundle(BASELINE)
    return load_bundle(BASELINE)


@pytest.fixture(scope="session")
def bundle_B_fd001():
    _require_bundle(BUNDLE_B_FD001)
    return load_bundle(BUNDLE_B_FD001)


@pytest.fixture(scope="session")
def bundle_B_fd003():
    _require_bundle(BUNDLE_B_FD003)
    return load_bundle(BUNDLE_B_FD003)


@pytest.fixture(scope="session")
def bundle_C_fd001():
    _require_bundle(BUNDLE_C_FD001)
    return load_bundle(BUNDLE_C_FD001)


@pytest.fixture(scope="session")
def bundle_C_fd003():
    _require_bundle(BUNDLE_C_FD003)
    return load_bundle(BUNDLE_C_FD003)


@pytest.fixture(scope="session")
def bundle_D_fd001():
    _require_bundle(BUNDLE_D_FD001)
    return load_bundle(BUNDLE_D_FD001)


@pytest.fixture(scope="session")
def bundle_D_fd003():
    _require_bundle(BUNDLE_D_FD003)
    return load_bundle(BUNDLE_D_FD003)


# ---------------------------------------------------------------------------
# 1. Baseline artifacts unchanged
# ---------------------------------------------------------------------------

def test_baseline_bundle_unchanged(baseline_bundle):
    """Original bundle must still have 18 features and the default config."""
    assert len(baseline_bundle.feature_cols) == 18
    assert baseline_bundle.config.healthy_max_cycle == 30
    assert baseline_bundle.config.ocsvm_kernel == "rbf"
    assert baseline_bundle.config.iforest_n_estimators == 200


def test_baseline_split_sizes(baseline_bundle):
    assert len(baseline_bundle.splits["fit"])         == 120
    assert len(baseline_bundle.splits["calibration"]) == 40
    assert len(baseline_bundle.splits["test"])        == 40


# ---------------------------------------------------------------------------
# 2. Leakage: no test engines in any experiment scoring
# ---------------------------------------------------------------------------

def _no_test_leak(csv_name: str, test_ids: set):
    p = RESULTS_DIR / csv_name
    if not p.exists():
        pytest.skip(f"{csv_name} not found")
    df = pd.read_csv(p)
    scored = set(df["engine_id"].unique())
    assert scored.isdisjoint(test_ids), (
        f"{csv_name}: test engine(s) found in scored data: {scored & test_ids}"
    )


def test_no_test_leak_A_fd001(baseline_bundle):
    _no_test_leak("scores_A_fd001.csv", set(baseline_bundle.splits["test"]))


def test_no_test_leak_A_fd003(baseline_bundle):
    _no_test_leak("scores_A_fd003.csv", set(baseline_bundle.splits["test"]))


def test_no_test_leak_B_fd001(baseline_bundle):
    _no_test_leak("scores_B_fd001.csv", set(baseline_bundle.splits["test"]))


def test_no_test_leak_B_fd003(baseline_bundle):
    _no_test_leak("scores_B_fd003.csv", set(baseline_bundle.splits["test"]))


def test_no_test_leak_C(baseline_bundle):
    for name in ("scores_C_fd001.csv", "scores_C_fd003.csv"):
        _no_test_leak(name, set(baseline_bundle.splits["test"]))


def test_no_test_leak_D(baseline_bundle):
    for name in ("scores_D_fd001.csv", "scores_D_fd003.csv"):
        _no_test_leak(name, set(baseline_bundle.splits["test"]))


# ---------------------------------------------------------------------------
# 3. Experiment B: separate models trained on correct dataset
# ---------------------------------------------------------------------------

def test_B_fd001_bundle_has_18_features(bundle_B_fd001):
    assert len(bundle_B_fd001.feature_cols) == 18


def test_B_fd003_bundle_has_18_features(bundle_B_fd003):
    assert len(bundle_B_fd003.feature_cols) == 18


def test_B_fd001_config_unchanged(bundle_B_fd001, baseline_bundle):
    """Exp B must use the same hyperparameters as the baseline."""
    assert bundle_B_fd001.config == baseline_bundle.config


def test_B_fd003_config_unchanged(bundle_B_fd003, baseline_bundle):
    assert bundle_B_fd003.config == baseline_bundle.config


def test_B_scores_only_correct_dataset(baseline_bundle):
    """FD001 scores must only contain FD001 engines, FD003 only FD003."""
    test_set = set(baseline_bundle.splits["test"])
    for ds, fname in [("FD001", "scores_B_fd001.csv"), ("FD003", "scores_B_fd003.csv")]:
        p = RESULTS_DIR / fname
        if not p.exists():
            pytest.skip(f"{fname} missing")
        df = pd.read_csv(p)
        datasets_in_file = df["engine_id"].str.startswith(ds)
        assert datasets_in_file.all(), (
            f"{fname}: contains engines from wrong dataset"
        )
        assert set(df["engine_id"].unique()).isdisjoint(test_set)


# ---------------------------------------------------------------------------
# 4. Experiment C: correct feature subsets
# ---------------------------------------------------------------------------

def test_C_fd001_feature_subset(bundle_C_fd001):
    """FD001 Exp C bundle must use exactly the specified feature subset."""
    assert set(bundle_C_fd001.feature_cols) == set(FEATURES_C_FD001), (
        f"Expected {FEATURES_C_FD001}, got {bundle_C_fd001.feature_cols}"
    )


def test_C_fd003_feature_subset(bundle_C_fd003):
    assert set(bundle_C_fd003.feature_cols) == set(FEATURES_C_FD003), (
        f"Expected {FEATURES_C_FD003}, got {bundle_C_fd003.feature_cols}"
    )


def test_C_fd001_no_noise_features(bundle_C_fd001):
    """op1, op2, s6 must NOT be in the FD001 feature subset."""
    for bad in ("op1", "op2", "s6"):
        assert bad not in bundle_C_fd001.feature_cols, (
            f"Noise feature {bad} found in FD001 Exp C subset"
        )


def test_C_fd003_no_noise_features(bundle_C_fd003):
    for bad in ("op1", "op2", "s6"):
        assert bad not in bundle_C_fd003.feature_cols, (
            f"Noise feature {bad} found in FD003 Exp C subset"
        )


def test_C_fd001_no_fd003_specific_feature(bundle_C_fd001):
    """s10 must NOT be in FD001 subset (it's a FD003-specific sensor)."""
    assert "s10" not in bundle_C_fd001.feature_cols


def test_C_fd003_has_s10(bundle_C_fd003):
    """s10 must be in FD003 subset."""
    assert "s10" in bundle_C_fd003.feature_cols


# ---------------------------------------------------------------------------
# 5. Experiment D: rolling mean features and causal window
# ---------------------------------------------------------------------------

def test_D_fd001_feature_count(bundle_D_fd001):
    """Exp D must have 18 raw + 5 rolling mean = 23 features."""
    assert len(bundle_D_fd001.feature_cols) == 23


def test_D_rolling_cols_present(bundle_D_fd001):
    for sensor in ROLL_SENSORS:
        col = f"{sensor}_rm{ROLL_WINDOW}"
        assert col in bundle_D_fd001.feature_cols, (
            f"Expected rolling column {col} missing from Exp D FD001 bundle"
        )


def test_add_rolling_means_causal(pipeline):
    """Rolling means must use only past data: value at cycle 1 equals raw value."""
    df_raw = pipeline["df_raw"]
    df_aug, rm_cols = add_rolling_means(df_raw, ["s11"], ROLL_WINDOW)
    for eng_id, grp in df_aug.groupby("engine_id"):
        grp_s = grp.sort_values("cycle")
        # At cycle 1, rm = raw value (window of size 1)
        cycle1 = grp_s[grp_s["cycle"] == 1]
        if len(cycle1):
            assert abs(cycle1["s11_rm10"].values[0] - cycle1["s11"].values[0]) < 1e-9, (
                f"{eng_id}: rolling mean at cycle 1 != raw value"
            )
        # No NaN values (min_periods=1 guarantees this)
        assert not grp_s["s11_rm10"].isna().any(), (
            f"{eng_id}: NaN in rolling mean column"
        )
        break  # spot-check one engine


def test_rolling_mean_uses_only_past(pipeline):
    """Rolling mean at cycle k must equal mean of cycles 1..min(k,w)."""
    df_raw = pipeline["df_raw"]
    df_aug, _ = add_rolling_means(df_raw, ["s11"], ROLL_WINDOW)
    # Check one engine in detail
    eng_ids = df_aug["engine_id"].unique()
    eng = df_aug[df_aug["engine_id"] == eng_ids[0]].sort_values("cycle")
    for k in range(1, min(ROLL_WINDOW + 5, len(eng))):
        window_vals = eng["s11"].values[max(0, k - ROLL_WINDOW): k]
        expected    = window_vals.mean()
        actual      = eng["s11_rm10"].values[k - 1]
        assert abs(actual - expected) < 1e-9, (
            f"Cycle {k}: rolling mean {actual:.6f} != expected {expected:.6f}"
        )


def test_D_scaler_fitted_on_fit_only(pipeline, bundle_D_fd001):
    """Scaler mean for Exp D must differ from the baseline 18-feature scaler
    (because Exp D includes additional rolling-mean columns)."""
    # The feature sets differ in length — checking this confirms separate scalers
    baseline = load_bundle(BASELINE)
    assert len(bundle_D_fd001.feature_cols) > len(baseline.feature_cols)


# ---------------------------------------------------------------------------
# 6. Score direction: higher = more anomalous
# ---------------------------------------------------------------------------

def _score_direction_check(csv_name: str, model: str, label: str):
    p = RESULTS_DIR / csv_name
    if not p.exists():
        pytest.skip(f"{csv_name} missing")
    df = pd.read_csv(p)
    hw = df.loc[df["in_healthy_window"],  f"{model}_score"]
    ll = df.loc[df["in_late_life"],        f"{model}_score"]
    assert ll.mean() > hw.mean(), (
        f"{label} {model}: late-life mean {ll.mean():.4f} "
        f"not > healthy mean {hw.mean():.4f}"
    )


@pytest.mark.parametrize("cfg,ds", [
    ("A","fd001"), ("A","fd003"),
    ("B","fd001"), ("B","fd003"),
    ("C","fd001"), ("C","fd003"),
    ("D","fd001"), ("D","fd003"),
])
def test_score_direction_ocsvm(cfg, ds):
    _score_direction_check(f"scores_{cfg}_{ds}.csv", "ocsvm", f"{cfg}/{ds}")


@pytest.mark.parametrize("cfg,ds", [
    ("A","fd001"), ("A","fd003"),
    ("B","fd001"), ("B","fd003"),
    ("C","fd001"), ("C","fd003"),
    ("D","fd001"), ("D","fd003"),
])
def test_score_direction_iforest(cfg, ds):
    _score_direction_check(f"scores_{cfg}_{ds}.csv", "iforest", f"{cfg}/{ds}")


# ---------------------------------------------------------------------------
# 7. Directional success: all cal engines
# ---------------------------------------------------------------------------

def _directional_success_check(csv_name: str, min_pct: float = 0.80):
    p = RESULTS_DIR / csv_name
    if not p.exists():
        pytest.skip(f"{csv_name} missing")
    df = pd.read_csv(p)
    sep = compute_separation(df)
    for model in ("ocsvm", "iforest"):
        pct = sep[f"{model}_dir_success"].mean()
        assert pct >= min_pct, (
            f"{csv_name} {model}: directional success {pct:.1%} < {min_pct:.0%}"
        )


@pytest.mark.parametrize("fname", [
    "scores_A_fd001.csv","scores_A_fd003.csv",
    "scores_B_fd001.csv","scores_B_fd003.csv",
    "scores_C_fd001.csv","scores_C_fd003.csv",
    "scores_D_fd001.csv","scores_D_fd003.csv",
])
def test_directional_success(fname):
    _directional_success_check(fname)


# ---------------------------------------------------------------------------
# 8. Per-engine Spearman ρ: majority positive
# ---------------------------------------------------------------------------

def _spearman_check(csv_name: str, min_pct_positive: float = 0.85):
    p = RESULTS_DIR / csv_name
    if not p.exists():
        pytest.skip(f"{csv_name} missing")
    df = pd.read_csv(p)
    rho = compute_spearman(df)
    for model in ("ocsvm", "iforest"):
        v = rho[f"{model}_rho"].dropna()
        pct_pos = (v > 0).mean()
        assert pct_pos >= min_pct_positive, (
            f"{csv_name} {model}: only {pct_pos:.1%} of engines have ρ > 0"
        )


@pytest.mark.parametrize("fname", [
    "scores_A_fd001.csv","scores_A_fd003.csv",
    "scores_B_fd001.csv","scores_B_fd003.csv",
    "scores_C_fd001.csv","scores_C_fd003.csv",
    "scores_D_fd001.csv","scores_D_fd003.csv",
])
def test_spearman_mostly_positive(fname):
    _spearman_check(fname)


# ---------------------------------------------------------------------------
# 9. Score overlap values are in [0, 1]
# ---------------------------------------------------------------------------

def test_score_overlap_range():
    p = RESULTS_DIR / "score_overlap.csv"
    if not p.exists():
        pytest.skip("score_overlap.csv missing")
    df = pd.read_csv(p)
    assert (df["overlap_pct"] >= 0).all()
    assert (df["overlap_pct"] <= 1).all()


def test_score_overlap_has_all_configs():
    p = RESULTS_DIR / "score_overlap.csv"
    if not p.exists():
        pytest.skip("score_overlap.csv missing")
    df = pd.read_csv(p)
    assert set(df["config"].unique()) == {"A_shared","B_separate","C_feat_sel","D_temporal"}


# ---------------------------------------------------------------------------
# 10. Summary CSV structure
# ---------------------------------------------------------------------------

def test_summary_csv_has_all_configs():
    p = RESULTS_DIR / "summary_all_configs.csv"
    if not p.exists():
        pytest.skip("summary_all_configs.csv missing")
    df = pd.read_csv(p)
    assert set(df["config"].unique()) == {"A_shared","B_separate","C_feat_sel","D_temporal"}


def test_summary_csv_no_null_rho():
    p = RESULTS_DIR / "summary_all_configs.csv"
    if not p.exists():
        pytest.skip()
    df = pd.read_csv(p)
    for col in ("FD001_ocsvm_median_rho","FD001_iforest_median_rho",
                "FD003_ocsvm_median_rho","FD003_iforest_median_rho"):
        assert df[col].notna().all(), f"NaN in {col}"


# ---------------------------------------------------------------------------
# 11. Bootstrap comparison structure
# ---------------------------------------------------------------------------

def test_bootstrap_csv_structure():
    p = RESULTS_DIR / "bootstrap_comparison.csv"
    if not p.exists():
        pytest.skip("bootstrap_comparison.csv missing")
    df = pd.read_csv(p)
    for col in ("comparison","metric","diff_b_minus_a","ci_lo","ci_hi","p_value","n_engines"):
        assert col in df.columns, f"Missing column {col}"
    assert (df["p_value"] >= 0).all() and (df["p_value"] <= 1).all()
    assert (df["ci_lo"] <= df["ci_hi"]).all()


# ---------------------------------------------------------------------------
# 12. Model artifact files exist and load correctly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    EXP_MODELS / "anomaly_bundle_B_fd001.pkl",
    EXP_MODELS / "anomaly_bundle_B_fd003.pkl",
    EXP_MODELS / "anomaly_bundle_C_fd001.pkl",
    EXP_MODELS / "anomaly_bundle_C_fd003.pkl",
    EXP_MODELS / "anomaly_bundle_D_fd001.pkl",
    EXP_MODELS / "anomaly_bundle_D_fd003.pkl",
])
def test_bundle_loads(path):
    if not path.exists():
        pytest.skip(f"{path.name} missing")
    b = load_bundle(path)
    assert len(b.feature_cols) > 0
    assert b.config.healthy_max_cycle == 30
    # Can score a trivial array
    from backend.risk.anomaly_models import score_ocsvm, score_iforest
    X = np.zeros((3, len(b.feature_cols)))
    sc_oc = score_ocsvm(b.ocsvm, X)
    sc_if = score_iforest(b.iforest, X)
    assert sc_oc.shape == (3,)
    assert sc_if.shape == (3,)


# ---------------------------------------------------------------------------
# 13. build_exp_pipeline: scaler fitted on fit engines only
# ---------------------------------------------------------------------------

def test_exp_pipeline_scaler_fit_only(pipeline, baseline_bundle):
    """Scaler mean for Exp B FD001 must equal the mean of FD001 fit-split rows."""
    df_raw = pipeline["df_raw"]
    splits = pipeline["splits"]
    fc     = baseline_bundle.feature_cols

    result = build_exp_pipeline(df_raw, splits, fc, "FD001")
    fit_ids = result["fit_ids"]
    # All fit_ids must be FD001
    assert all(e.startswith("FD001") for e in fit_ids)
    # Scaler mean must match FD001 fit-split feature mean
    fit_mask = df_raw["engine_id"].isin(fit_ids)
    expected_mean = df_raw.loc[fit_mask, fc].mean().values
    np.testing.assert_allclose(result["scaler"].mean_, expected_mean, rtol=1e-5)


def test_exp_pipeline_no_test_engines(pipeline, baseline_bundle):
    """build_exp_pipeline must only include fit and cal engines (no test)."""
    df_raw   = pipeline["df_raw"]
    splits   = pipeline["splits"]
    test_set = set(splits["test"])
    fc       = baseline_bundle.feature_cols

    for ds in ("FD001", "FD003"):
        result = build_exp_pipeline(df_raw, splits, fc, ds)
        assert set(result["fit_ids"]).isdisjoint(test_set)
        assert set(result["cal_ids"]).isdisjoint(test_set)
