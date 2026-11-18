"""
Tests for backend.risk.anomaly_models.

Covers: healthy-cycle selection, score shape/direction, fit-split isolation,
        reproducibility, RUL alignment, bundle roundtrip.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

from backend.risk.anomaly_models import (
    DEFAULT_CONFIG,
    AnomalyBundle,
    AnomalyConfig,
    load_bundle,
    save_bundle,
    score_engine,
    score_iforest,
    score_ocsvm,
    select_healthy_cycles,
    train_iforest,
    train_ocsvm,
)
from backend.risk.cmapss_pipeline import (
    SPLIT_SEED,
    build_pipeline,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR = pathlib.Path(__file__).parent.parent / "data" / "raw" / "cmapss"


@pytest.fixture(scope="session")
def pipeline():
    if not DATA_DIR.exists():
        pytest.skip("data/raw/cmapss not present")
    return build_pipeline(DATA_DIR, seed=SPLIT_SEED)


@pytest.fixture(scope="session")
def X_healthy(pipeline):
    return select_healthy_cycles(
        pipeline["df_scaled"],
        pipeline["splits"]["fit"],
        DEFAULT_CONFIG.healthy_max_cycle,
        pipeline["feature_cols"],
    )


@pytest.fixture(scope="session")
def trained_ocsvm(X_healthy):
    return train_ocsvm(X_healthy, DEFAULT_CONFIG)


@pytest.fixture(scope="session")
def trained_iforest(X_healthy):
    return train_iforest(X_healthy, DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Healthy-cycle selection
# ---------------------------------------------------------------------------


def test_healthy_only_fit_engines(pipeline) -> None:
    """Healthy rows must come only from fit-split engines."""
    splits   = pipeline["splits"]
    df       = pipeline["df_scaled"]
    cal_ids  = set(splits["calibration"])
    test_ids = set(splits["test"])
    mask = (
        df["engine_id"].isin(splits["fit"])
        & (df["cycle"] <= DEFAULT_CONFIG.healthy_max_cycle)
    )
    eids = set(df.loc[mask, "engine_id"].unique())
    assert eids.isdisjoint(cal_ids),  "Cal engines in healthy set"
    assert eids.isdisjoint(test_ids), "Test engines in healthy set"


def test_healthy_row_count(pipeline) -> None:
    """Each fit engine contributes exactly healthy_max_cycle rows."""
    splits = pipeline["splits"]
    df     = pipeline["df_scaled"]
    fcols  = pipeline["feature_cols"]
    hmc    = DEFAULT_CONFIG.healthy_max_cycle
    X        = select_healthy_cycles(df, splits["fit"], hmc, fcols)
    expected = len(splits["fit"]) * hmc
    assert len(X) == expected, f"Expected {expected} rows, got {len(X)}"


def test_healthy_cycle_range(pipeline) -> None:
    """No healthy row should have cycle > healthy_max_cycle."""
    splits = pipeline["splits"]
    df     = pipeline["df_scaled"]
    hmc    = DEFAULT_CONFIG.healthy_max_cycle
    mask   = df["engine_id"].isin(splits["fit"]) & (df["cycle"] <= hmc)
    assert df.loc[mask, "cycle"].max() <= hmc


def test_healthy_empty_raises(pipeline) -> None:
    """healthy_max_cycle=0 must raise ValueError."""
    splits = pipeline["splits"]
    df     = pipeline["df_scaled"]
    fcols  = pipeline["feature_cols"]
    with pytest.raises(ValueError, match="No healthy rows"):
        select_healthy_cycles(df, splits["fit"], 0, fcols)


# ---------------------------------------------------------------------------
# Score shape
# ---------------------------------------------------------------------------


def test_ocsvm_score_shape(trained_ocsvm, X_healthy) -> None:
    assert score_ocsvm(trained_ocsvm, X_healthy).shape == (len(X_healthy),)


def test_iforest_score_shape(trained_iforest, X_healthy) -> None:
    assert score_iforest(trained_iforest, X_healthy).shape == (len(X_healthy),)


# ---------------------------------------------------------------------------
# Score direction
# ---------------------------------------------------------------------------


def test_ocsvm_healthy_mostly_negative(trained_ocsvm, X_healthy) -> None:
    """On healthy training data, <=10 pct of OC-SVM scores should exceed 0."""
    scores = score_ocsvm(trained_ocsvm, X_healthy)
    frac   = (scores > 0).mean()
    assert frac <= 0.10, f"Too many anomalous-scoring healthy rows: {frac:.2%}"


def test_ocsvm_score_direction(pipeline, trained_ocsvm) -> None:
    """Late-life mean score (RUL<=10) must exceed healthy mean score."""
    df    = pipeline["df_scaled"]
    fcols = pipeline["feature_cols"]
    splits = pipeline["splits"]
    hmc   = DEFAULT_CONFIG.healthy_max_cycle
    fit_m = df["engine_id"].isin(splits["fit"])
    X_h = df.loc[fit_m & (df["cycle"] <= hmc), fcols].values
    X_l = df.loc[fit_m & (df["rul"]   <= 10),  fcols].values
    if len(X_l) == 0:
        pytest.skip("No late-life rows with RUL<=10 in fit split")
    sh = score_ocsvm(trained_ocsvm, X_h).mean()
    sl = score_ocsvm(trained_ocsvm, X_l).mean()
    assert sl > sh, f"OC-SVM: late mean {sl:.4f} not > healthy mean {sh:.4f}"


def test_iforest_score_direction(pipeline, trained_iforest) -> None:
    """Late-life mean score (RUL<=10) must exceed healthy mean score."""
    df    = pipeline["df_scaled"]
    fcols = pipeline["feature_cols"]
    splits = pipeline["splits"]
    hmc   = DEFAULT_CONFIG.healthy_max_cycle
    fit_m = df["engine_id"].isin(splits["fit"])
    X_h = df.loc[fit_m & (df["cycle"] <= hmc), fcols].values
    X_l = df.loc[fit_m & (df["rul"]   <= 10),  fcols].values
    if len(X_l) == 0:
        pytest.skip("No late-life rows with RUL<=10 in fit split")
    sh = score_iforest(trained_iforest, X_h).mean()
    sl = score_iforest(trained_iforest, X_l).mean()
    assert sl > sh, f"IForest: late mean {sl:.4f} not > healthy mean {sh:.4f}"


# ---------------------------------------------------------------------------
# score_engine alignment
# ---------------------------------------------------------------------------


def test_score_engine_row_count(pipeline, trained_ocsvm, trained_iforest) -> None:
    """score_engine returns one row per observed cycle."""
    df    = pipeline["df_scaled"]
    fcols = pipeline["feature_cols"]
    eng   = pipeline["splits"]["fit"][0]
    scores   = score_engine(df, eng, trained_ocsvm, trained_iforest, fcols)
    n_cycles = df[df["engine_id"] == eng]["cycle"].nunique()
    assert len(scores) == n_cycles


def test_score_engine_columns(pipeline, trained_ocsvm, trained_iforest) -> None:
    df    = pipeline["df_scaled"]
    fcols = pipeline["feature_cols"]
    eng   = pipeline["splits"]["fit"][0]
    scores = score_engine(df, eng, trained_ocsvm, trained_iforest, fcols)
    for col in ("cycle", "rul", "ocsvm_score", "iforest_score"):
        assert col in scores.columns


def test_score_engine_rul_zero_at_last_cycle(pipeline, trained_ocsvm, trained_iforest) -> None:
    """Last cycle must have rul == 0."""
    df    = pipeline["df_scaled"]
    fcols = pipeline["feature_cols"]
    eng   = pipeline["splits"]["fit"][0]
    scores   = score_engine(df, eng, trained_ocsvm, trained_iforest, fcols)
    last_rul = scores.sort_values("cycle")["rul"].iloc[-1]
    assert last_rul == 0, f"Last-cycle RUL is {last_rul}, expected 0"


def test_score_engine_cycles_sorted(pipeline, trained_ocsvm, trained_iforest) -> None:
    df    = pipeline["df_scaled"]
    fcols = pipeline["feature_cols"]
    eng   = pipeline["splits"]["fit"][0]
    scores = score_engine(df, eng, trained_ocsvm, trained_iforest, fcols)
    assert (scores["cycle"].diff().dropna() > 0).all()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_iforest_reproducible(X_healthy) -> None:
    """Same seed -> identical IForest scores."""
    m1 = train_iforest(X_healthy, DEFAULT_CONFIG)
    m2 = train_iforest(X_healthy, DEFAULT_CONFIG)
    np.testing.assert_array_equal(
        score_iforest(m1, X_healthy), score_iforest(m2, X_healthy),
    )


def test_ocsvm_reproducible(X_healthy) -> None:
    """OC-SVM is deterministic; same data -> identical scores."""
    m1 = train_ocsvm(X_healthy, DEFAULT_CONFIG)
    m2 = train_ocsvm(X_healthy, DEFAULT_CONFIG)
    np.testing.assert_array_equal(
        score_ocsvm(m1, X_healthy), score_ocsvm(m2, X_healthy),
    )


def test_iforest_different_seed_differs(X_healthy) -> None:
    """Different random seeds produce different IForest scores."""
    m1 = train_iforest(X_healthy, AnomalyConfig(iforest_random_state=42))
    m2 = train_iforest(X_healthy, AnomalyConfig(iforest_random_state=99))
    assert not np.array_equal(
        score_iforest(m1, X_healthy), score_iforest(m2, X_healthy),
    )


# ---------------------------------------------------------------------------
# Bundle save / load
# ---------------------------------------------------------------------------


def test_bundle_roundtrip(pipeline, trained_ocsvm, trained_iforest) -> None:
    """save_bundle / load_bundle must preserve model predictions."""
    bundle = AnomalyBundle(
        ocsvm=trained_ocsvm,
        iforest=trained_iforest,
        feature_cols=pipeline["feature_cols"],
        config=DEFAULT_CONFIG,
        splits=pipeline["splits"],
    )
    X = pipeline["df_scaled"][pipeline["feature_cols"]].values[:50]
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        path = tmp.name
    save_bundle(bundle, path)
    loaded = load_bundle(path)
    np.testing.assert_array_equal(
        score_ocsvm(bundle.ocsvm, X), score_ocsvm(loaded.ocsvm, X),
    )
    np.testing.assert_array_equal(
        score_iforest(bundle.iforest, X), score_iforest(loaded.iforest, X),
    )
    assert loaded.feature_cols == bundle.feature_cols
    assert loaded.config == bundle.config
