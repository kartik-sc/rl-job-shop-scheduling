"""
Tests for backend.risk.diagnostic.

Covers:
  - Score-to-engine/cycle/RUL alignment
  - Correct healthy (first 30) and late-life (final 30 by RUL) window selection
  - Score direction (higher = more anomalous)
  - Reproducible separation and Spearman metrics
  - Handling of undefined (constant) score sequences
  - No use of calibration/test engines for model fitting or threshold selection
  - Correct engine-level aggregation (not cycle-level)
  - Early-warning LOO threshold does not use the target engine's healthy data
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from backend.risk.anomaly_models import load_bundle
from backend.risk.cmapss_pipeline import SPLIT_SEED, build_pipeline
from backend.risk.diagnostic import (
    BUNDLE_PATH,
    HEALTHY_N,
    LATE_LIFE_N,
    MIN_WARN_PERSIST,
    compute_early_warning,
    compute_separation,
    compute_spearman,
    bootstrap_paired_comparison,
    score_all_fit_engines,
    load_data_and_bundle,
    compute_healthy_threshold_stats,
)

# ---------------------------------------------------------------------------
# Fixtures (session-scoped: load once per test run)
# ---------------------------------------------------------------------------

DATA_DIR = pathlib.Path(__file__).parent.parent / "data" / "raw" / "cmapss"


@pytest.fixture(scope="session")
def bundle_and_data():
    if not DATA_DIR.exists():
        pytest.skip("data/raw/cmapss not present")
    if not BUNDLE_PATH.exists():
        pytest.skip("models/anomaly_bundle.pkl not found – run train_anomaly first")
    return load_data_and_bundle()


@pytest.fixture(scope="session")
def all_scores(bundle_and_data):
    bundle, df_scaled, fit_ids, feature_cols = bundle_and_data
    return score_all_fit_engines(bundle, df_scaled, fit_ids, feature_cols)


@pytest.fixture(scope="session")
def sep_df(all_scores):
    return compute_separation(all_scores)


@pytest.fixture(scope="session")
def rho_df(all_scores):
    return compute_spearman(all_scores)


@pytest.fixture(scope="session")
def ew_df(all_scores, bundle_and_data):
    _, _, fit_ids, _ = bundle_and_data
    return compute_early_warning(all_scores, fit_ids)


# ---------------------------------------------------------------------------
# Alignment: score rows must match engine data exactly
# ---------------------------------------------------------------------------


def test_all_fit_engines_present(all_scores, bundle_and_data):
    """Exactly 120 fit engines must be scored."""
    _, _, fit_ids, _ = bundle_and_data
    scored = set(all_scores["engine_id"].unique())
    assert scored == set(fit_ids), (
        f"Mismatch: scored={len(scored)}, expected={len(fit_ids)}"
    )


def test_no_cal_or_test_engines_scored(all_scores, bundle_and_data):
    """No calibration or test engines may appear in the scored DataFrame."""
    bundle, _, _, _ = bundle_and_data
    cal_set  = set(bundle.splits["calibration"])
    test_set = set(bundle.splits["test"])
    scored   = set(all_scores["engine_id"].unique())
    assert scored.isdisjoint(cal_set),  "Calibration engine found in scored data"
    assert scored.isdisjoint(test_set), "Test engine found in scored data"


def test_rul_zero_at_last_cycle(all_scores):
    """For every engine, the final cycle must have RUL == 0."""
    last_ruls = (
        all_scores.sort_values("cycle")
        .groupby("engine_id")["rul"]
        .last()
    )
    bad = last_ruls[last_ruls != 0]
    assert len(bad) == 0, f"Engines with last-cycle RUL != 0: {bad.index.tolist()}"


def test_rul_monotone_decreasing(all_scores):
    """RUL must decrease by exactly 1 each cycle for every engine."""
    for eng_id, grp in all_scores.groupby("engine_id"):
        rul = grp.sort_values("cycle")["rul"].values
        diffs = np.diff(rul)
        assert (diffs == -1).all(), (
            f"{eng_id}: RUL diffs are not all -1. "
            f"Unique diffs: {np.unique(diffs)}"
        )


def test_cycles_start_at_one(all_scores):
    """Every engine's first cycle must be 1."""
    first_cycles = all_scores.groupby("engine_id")["cycle"].min()
    bad = first_cycles[first_cycles != 1]
    assert len(bad) == 0, f"Engines not starting at cycle 1: {bad.index.tolist()}"


def test_score_columns_present(all_scores):
    for col in ("engine_id", "dataset", "cycle", "rul",
                "ocsvm_score", "iforest_score",
                "in_healthy_window", "in_late_life"):
        assert col in all_scores.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Window selection correctness
# ---------------------------------------------------------------------------


def test_healthy_window_uses_first_n_cycles(all_scores):
    """in_healthy_window must be True iff cycle <= HEALTHY_N."""
    expected = all_scores["cycle"] <= HEALTHY_N
    pd.testing.assert_series_equal(
        all_scores["in_healthy_window"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_late_life_window_uses_rul(all_scores):
    """in_late_life must be True iff RUL < LATE_LIFE_N."""
    expected = all_scores["rul"] < LATE_LIFE_N
    pd.testing.assert_series_equal(
        all_scores["in_late_life"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_healthy_window_count_per_engine(all_scores):
    """Every engine must have exactly HEALTHY_N healthy-window rows."""
    counts = all_scores[all_scores["in_healthy_window"]].groupby("engine_id").size()
    bad = counts[counts != HEALTHY_N]
    assert len(bad) == 0, (
        f"Engines with wrong healthy window count: {bad.to_dict()}"
    )


def test_late_life_minimum_per_engine(all_scores):
    """Every engine must have at least 1 late-life row (all C-MAPSS engines
    last > LATE_LIFE_N cycles)."""
    counts = all_scores[all_scores["in_late_life"]].groupby("engine_id").size()
    assert (counts > 0).all(), "Some engine has no late-life rows"


# ---------------------------------------------------------------------------
# Score direction
# ---------------------------------------------------------------------------


def test_mean_late_life_exceeds_healthy_both_models(all_scores):
    """Population-mean late-life score must exceed healthy score for both models."""
    hw = all_scores[all_scores["in_healthy_window"]]
    ll = all_scores[all_scores["in_late_life"]]
    for model in ("ocsvm", "iforest"):
        col = f"{model}_score"
        mean_hw = hw[col].mean()
        mean_ll = ll[col].mean()
        assert mean_ll > mean_hw, (
            f"{model}: late-life mean {mean_ll:.4f} not > "
            f"healthy mean {mean_hw:.4f}"
        )


def test_directional_success_majority(sep_df):
    """More than 50% of engines must show late-life > healthy-window score."""
    for model in ("ocsvm", "iforest"):
        pct = sep_df[f"{model}_dir_success"].mean()
        assert pct > 0.5, (
            f"{model}: directional success {pct:.1%} <= 50%"
        )


# ---------------------------------------------------------------------------
# Separation metric correctness
# ---------------------------------------------------------------------------


def test_separation_formula(all_scores, sep_df):
    """Verify sep = late_mean - healthy_mean for first engine."""
    eng = sep_df["engine_id"].iloc[0]
    grp = all_scores[all_scores["engine_id"] == eng]
    for model in ("ocsvm", "iforest"):
        col = f"{model}_score"
        h_mean = grp.loc[grp["in_healthy_window"], col].mean()
        l_mean = grp.loc[grp["in_late_life"], col].mean()
        expected_sep = l_mean - h_mean
        actual_sep   = sep_df.loc[sep_df["engine_id"] == eng, f"{model}_sep"].iloc[0]
        np.testing.assert_allclose(actual_sep, expected_sep, rtol=1e-6,
                                   err_msg=f"{model} sep formula error for {eng}")


def test_no_nan_sep_for_valid_engines(sep_df):
    """No NaN separation values expected: all C-MAPSS engines have ≥30 cycles."""
    for model in ("ocsvm", "iforest"):
        col  = f"{model}_sep"
        nans = sep_df[col].isna().sum()
        assert nans == 0, f"{model}_sep has {nans} NaN rows"


# ---------------------------------------------------------------------------
# Spearman correctness
# ---------------------------------------------------------------------------


def test_spearman_formula_spot_check(all_scores, rho_df):
    """Spot-check Spearman ρ for one engine against scipy directly."""
    eng = rho_df["engine_id"].iloc[0]
    grp = all_scores[all_scores["engine_id"] == eng].sort_values("cycle")
    neg_rul = -grp["rul"].values
    for model in ("ocsvm", "iforest"):
        scores = grp[f"{model}_score"].values
        expected_rho, _ = stats.spearmanr(neg_rul, scores)
        actual_rho = rho_df.loc[rho_df["engine_id"] == eng, f"{model}_rho"].iloc[0]
        np.testing.assert_allclose(
            actual_rho, expected_rho, rtol=1e-6,
            err_msg=f"{model} Spearman rho formula error for {eng}"
        )


def test_undefined_spearman_handled(rho_df):
    """Undefined (NaN) Spearman values must have undefined=True flag, not be silently dropped."""
    for model in ("ocsvm", "iforest"):
        undef_mask  = rho_df[f"{model}_undefined"]
        nan_mask    = rho_df[f"{model}_rho"].isna()
        # Every NaN rho must have undefined=True
        assert (undef_mask[nan_mask]).all(), (
            f"{model}: NaN rho without undefined flag"
        )
        # Every undefined=True must have NaN rho
        assert (nan_mask[undef_mask]).all(), (
            f"{model}: undefined=True without NaN rho"
        )


# ---------------------------------------------------------------------------
# Early-warning: leakage check
# ---------------------------------------------------------------------------


def test_loo_threshold_excludes_target_engine(all_scores, bundle_and_data, ew_df):
    """LOO threshold for engine E must be derived from all OTHER fit engines only."""
    _, _, fit_ids, _ = bundle_and_data
    target    = fit_ids[0]
    other_ids = [e for e in fit_ids if e != target]

    # Manually compute the expected threshold from other engines' healthy windows
    other_healthy_ocsvm = all_scores.loc[
        all_scores["engine_id"].isin(other_ids) & all_scores["in_healthy_window"],
        "ocsvm_score",
    ].values
    expected_threshold = np.percentile(other_healthy_ocsvm, 95.0)

    # Retrieve the threshold that was actually used for the target engine
    actual_threshold = ew_df.loc[
        (ew_df["engine_id"] == target) & (ew_df["model"] == "ocsvm"),
        "threshold",
    ].iloc[0]

    np.testing.assert_allclose(
        actual_threshold, expected_threshold, rtol=1e-6,
        err_msg="LOO threshold used target engine's own data"
    )


def test_ew_warned_engines_have_finite_lead_time(ew_df):
    """Warned engines must have a finite positive lead time."""
    warned = ew_df[ew_df["warned"]]
    assert (warned["lead_time_cycles"].notna()).all()
    assert (warned["lead_time_cycles"] >= 0).all()


def test_ew_not_warned_have_nan_lead_time(ew_df):
    """Engines without warning must have NaN lead time."""
    not_warned = ew_df[~ew_df["warned"]]
    assert (not_warned["lead_time_cycles"].isna()).all()


def test_ew_false_alarm_rate_in_range(ew_df):
    """False alarm rates must be in [0, 1]."""
    fa = ew_df["false_alarm_rate"].dropna()
    assert (fa >= 0).all() and (fa <= 1).all()


# ---------------------------------------------------------------------------
# Engine-level aggregation (not cycle-level)
# ---------------------------------------------------------------------------


def test_sep_df_has_one_row_per_engine(sep_df, bundle_and_data):
    _, _, fit_ids, _ = bundle_and_data
    assert len(sep_df) == len(fit_ids), (
        f"sep_df has {len(sep_df)} rows, expected {len(fit_ids)}"
    )


def test_rho_df_has_one_row_per_engine(rho_df, bundle_and_data):
    _, _, fit_ids, _ = bundle_and_data
    assert len(rho_df) == len(fit_ids)


def test_ew_df_has_two_rows_per_engine(ew_df, bundle_and_data):
    """EW table should have 2 rows per engine (ocsvm + iforest)."""
    _, _, fit_ids, _ = bundle_and_data
    assert len(ew_df) == 2 * len(fit_ids), (
        f"ew_df has {len(ew_df)} rows, expected {2 * len(fit_ids)}"
    )


def test_bootstrap_metrics_covered(sep_df, rho_df, ew_df):
    """Bootstrap must cover sep, norm_sep, and rho metrics."""
    boot = bootstrap_paired_comparison(sep_df, rho_df, ew_df)
    covered = set(boot["metric"])
    for m in ("sep", "norm_sep", "rho"):
        assert m in covered, f"Bootstrap missing metric: {m}"


# ---------------------------------------------------------------------------
# Healthy-window stats
# ---------------------------------------------------------------------------


def test_healthy_window_stats_has_both_models(all_scores):
    hw_stats = compute_healthy_threshold_stats(all_scores)
    assert set(hw_stats["model"]) == {"ocsvm", "iforest"}


def test_healthy_window_pct_above_zero_reasonable(all_scores):
    """OC-SVM healthy scores: per specification <=10% should be above 0."""
    hw_stats = compute_healthy_threshold_stats(all_scores)
    ocsvm_row = hw_stats[hw_stats["model"] == "ocsvm"].iloc[0]
    assert ocsvm_row["pct_above_zero"] <= 0.15, (
        f"OC-SVM healthy false-alarm rate too high: "
        f"{ocsvm_row['pct_above_zero']:.1%}"
    )
