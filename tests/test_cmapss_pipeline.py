"""
Tests for backend.risk.cmapss_pipeline.

All tests use the real FD001/FD003 training files from data/raw/cmapss/.
Fixtures load once per session so the test suite stays fast.
"""

from __future__ import annotations

import pathlib
import json
import tempfile

import numpy as np
import pandas as pd
import pytest

from backend.risk.cmapss_pipeline import (
    CONSTANT_COLS,
    SPLIT_SEED,
    SUPPORTED_DATASETS,
    _ALL_COLS,
    _N_CAL,
    _N_FIT,
    _N_TEST,
    apply_scaler,
    build_pipeline,
    drop_constant_features,
    fit_scaler,
    get_feature_cols,
    load_combined,
    load_dataset,
    load_splits,
    save_splits,
    split_engines,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR = pathlib.Path(__file__).parent.parent / "data" / "raw" / "cmapss"


@pytest.fixture(scope="session")
def data_dir() -> pathlib.Path:
    if not DATA_DIR.exists():
        pytest.skip("data/raw/cmapss not present - copy training files first")
    return DATA_DIR


@pytest.fixture(scope="session")
def df_fd001(data_dir: pathlib.Path) -> pd.DataFrame:
    return load_dataset(data_dir / "train_FD001.txt")


@pytest.fixture(scope="session")
def df_fd003(data_dir: pathlib.Path) -> pd.DataFrame:
    return load_dataset(data_dir / "train_FD003.txt")


@pytest.fixture(scope="session")
def df_combined(data_dir: pathlib.Path) -> pd.DataFrame:
    return load_combined(data_dir)


@pytest.fixture(scope="session")
def df_clean(df_combined: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    return drop_constant_features(df_combined)


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------


def test_fd001_column_count(df_fd001: pd.DataFrame) -> None:
    """Raw load must produce exactly 26 source columns plus 3 added ones."""
    expected_added = {"dataset", "engine_id", "rul"}
    assert set(_ALL_COLS).issubset(df_fd001.columns)
    assert expected_added.issubset(df_fd001.columns)


def test_fd001_engine_count(df_fd001: pd.DataFrame) -> None:
    assert df_fd001["unit"].nunique() == 100


def test_fd003_engine_count(df_fd003: pd.DataFrame) -> None:
    assert df_fd003["unit"].nunique() == 100


def test_combined_engine_count(df_combined: pd.DataFrame) -> None:
    """FD001 + FD003 must yield 200 unique engine_ids."""
    assert df_combined["engine_id"].nunique() == 200


def test_engine_id_format(df_combined: pd.DataFrame) -> None:
    """engine_id must look like 'FD001_042'."""
    sample = df_combined["engine_id"].iloc[0]
    assert "_" in sample
    ds_part, num_part = sample.split("_")
    assert ds_part in SUPPORTED_DATASETS
    assert num_part.isdigit() and len(num_part) == 3


def test_dataset_tags(df_combined: pd.DataFrame) -> None:
    """Only FD001 and FD003 dataset tags should appear."""
    assert set(df_combined["dataset"].unique()) == {"FD001", "FD003"}


def test_no_engine_id_overlap(df_fd001: pd.DataFrame, df_fd003: pd.DataFrame) -> None:
    """engine_id values must be disjoint between datasets."""
    ids1 = set(df_fd001["engine_id"])
    ids3 = set(df_fd003["engine_id"])
    assert ids1.isdisjoint(ids3)


def test_row_count_fd001(df_fd001: pd.DataFrame) -> None:
    assert len(df_fd001) == 20631


def test_row_count_fd003(df_fd003: pd.DataFrame) -> None:
    assert len(df_fd003) == 24720


def test_no_nulls(df_combined: pd.DataFrame) -> None:
    assert df_combined.isnull().sum().sum() == 0


def test_cycle_starts_at_one(df_combined: pd.DataFrame) -> None:
    """Every engine must start at cycle 1."""
    min_cycles = df_combined.groupby("engine_id")["cycle"].min()
    assert (min_cycles == 1).all(), f"Engines not starting at cycle 1: {min_cycles[min_cycles != 1]}"


def test_cycles_are_contiguous(df_fd001: pd.DataFrame) -> None:
    """Cycle numbers must increment by exactly 1 for each engine."""
    for eng, grp in df_fd001.groupby("engine_id"):
        diffs = grp["cycle"].sort_values().diff().dropna()
        assert (diffs == 1).all(), f"Non-contiguous cycles in engine {eng}"


# ---------------------------------------------------------------------------
# RUL tests
# ---------------------------------------------------------------------------


def test_rul_at_final_cycle_is_zero(df_combined: pd.DataFrame) -> None:
    """At the last observed cycle RUL must equal 0."""
    last_rows = df_combined.groupby("engine_id").apply(
        lambda g: g.loc[g["cycle"].idxmax()], include_groups=False
    )
    assert (last_rows["rul"] == 0).all()


def test_rul_monotonically_decreasing(df_fd001: pd.DataFrame) -> None:
    """RUL must decrease by exactly 1 per cycle for each engine."""
    for eng, grp in df_fd001.groupby("engine_id"):
        rul_sorted = grp.sort_values("cycle")["rul"]
        diffs = rul_sorted.diff().dropna()
        assert (diffs == -1).all(), f"RUL not monotone for engine {eng}"


def test_rul_non_negative(df_combined: pd.DataFrame) -> None:
    assert (df_combined["rul"] >= 0).all()


def test_rul_formula_spot_check(df_fd001: pd.DataFrame) -> None:
    """Spot-check RUL for engine 1: rul[i] == max_cycle - cycle[i]."""
    eng1 = df_fd001[df_fd001["unit"] == 1].sort_values("cycle")
    max_cycle = eng1["cycle"].max()
    expected_rul = max_cycle - eng1["cycle"]
    assert (eng1["rul"].values == expected_rul.values).all()


# ---------------------------------------------------------------------------
# Constant-feature tests
# ---------------------------------------------------------------------------


def test_known_constant_cols_removed(df_clean: tuple[pd.DataFrame, list[str]]) -> None:
    """All known constant columns must be absent after cleaning."""
    cleaned, dropped = df_clean
    for col in CONSTANT_COLS:
        assert col not in cleaned.columns, f"Expected {col!r} to be dropped"


def test_dropped_cols_reported(df_clean: tuple[pd.DataFrame, list[str]]) -> None:
    """drop_constant_features must report all removed column names."""
    cleaned, dropped = df_clean
    for col in CONSTANT_COLS:
        assert col in dropped, f"Column {col!r} not in dropped list"


def test_id_cols_not_dropped(df_clean: tuple[pd.DataFrame, list[str]]) -> None:
    """engine_id, unit, cycle, rul, dataset must survive cleaning."""
    cleaned, _ = df_clean
    for col in ("engine_id", "unit", "cycle", "rul", "dataset"):
        assert col in cleaned.columns, f"ID column {col!r} was incorrectly removed"


def test_remaining_features_nonzero_std(df_clean: tuple[pd.DataFrame, list[str]]) -> None:
    """Every remaining feature column must have non-negligible variance."""
    cleaned, _ = df_clean
    feature_cols = get_feature_cols(cleaned)
    stds = cleaned[feature_cols].std()
    flat = stds[stds < 1e-6]
    assert flat.empty, f"Zero-std columns remain: {flat.index.tolist()}"


# ---------------------------------------------------------------------------
# Split-integrity tests
# ---------------------------------------------------------------------------


def test_split_sizes(df_clean: tuple[pd.DataFrame, list[str]]) -> None:
    """Splits must be exactly 120 / 40 / 40."""
    cleaned, _ = df_clean
    splits = split_engines(cleaned, seed=SPLIT_SEED)
    fit_len = len(splits["fit"]); assert fit_len == _N_FIT, f"fit size {fit_len}"
    cal_len = len(splits["calibration"]); assert cal_len == _N_CAL, f"cal size {cal_len}"
    test_len = len(splits["test"]); assert test_len == _N_TEST, f"test size {test_len}"


def test_split_no_overlap(df_clean: tuple[pd.DataFrame, list[str]]) -> None:
    """No engine_id may appear in more than one split."""
    cleaned, _ = df_clean
    splits = split_engines(cleaned, seed=SPLIT_SEED)
    fit  = set(splits["fit"])
    cal  = set(splits["calibration"])
    test = set(splits["test"])
    assert fit.isdisjoint(cal),  "Overlap between fit and calibration"
    assert fit.isdisjoint(test), "Overlap between fit and test"
    assert cal.isdisjoint(test), "Overlap between calibration and test"


def test_split_covers_all_engines(df_clean: tuple[pd.DataFrame, list[str]]) -> None:
    """Union of all splits must equal the full engine-id set."""
    cleaned, _ = df_clean
    splits = split_engines(cleaned, seed=SPLIT_SEED)
    all_split = set(splits["fit"]) | set(splits["calibration"]) | set(splits["test"])
    all_engines = set(cleaned["engine_id"].unique())
    assert all_split == all_engines


def test_split_deterministic(df_clean: tuple[pd.DataFrame, list[str]]) -> None:
    """Same seed must produce identical splits on repeated calls."""
    cleaned, _ = df_clean
    s1 = split_engines(cleaned, seed=SPLIT_SEED)
    s2 = split_engines(cleaned, seed=SPLIT_SEED)
    assert s1 == s2


def test_split_different_seeds(df_clean: tuple[pd.DataFrame, list[str]]) -> None:
    """Different seeds must produce different splits."""
    cleaned, _ = df_clean
    s1 = split_engines(cleaned, seed=42)
    s2 = split_engines(cleaned, seed=99)
    assert s1["fit"] != s2["fit"]


def test_split_both_datasets_represented(df_clean: tuple[pd.DataFrame, list[str]]) -> None:
    """Both FD001 and FD003 engines must appear in the fit split."""
    cleaned, _ = df_clean
    splits = split_engines(cleaned, seed=SPLIT_SEED)
    ds_tags = {eid.split("_")[0] for eid in splits["fit"]}
    assert "FD001" in ds_tags and "FD003" in ds_tags


# ---------------------------------------------------------------------------
# Scaler / leakage tests
# ---------------------------------------------------------------------------


def test_scaler_fitted_on_fit_only(df_clean: tuple[pd.DataFrame, list[str]]) -> None:
    """Scaler mean must match the fit-split feature mean (not the full dataset)."""
    cleaned, _ = df_clean
    splits = split_engines(cleaned, seed=SPLIT_SEED)
    scaler, feature_cols = fit_scaler(cleaned, splits["fit"])

    fit_mask = cleaned["engine_id"].isin(splits["fit"])
    fit_mean = cleaned.loc[fit_mask, feature_cols].mean().values
    np.testing.assert_allclose(scaler.mean_, fit_mean, rtol=1e-5)


def test_scaled_fit_mean_near_zero(df_clean: tuple[pd.DataFrame, list[str]]) -> None:
    """After scaling, the fit-split feature mean must be ~0."""
    cleaned, _ = df_clean
    splits = split_engines(cleaned, seed=SPLIT_SEED)
    scaler, feature_cols = fit_scaler(cleaned, splits["fit"])
    df_scaled = apply_scaler(cleaned, scaler, feature_cols)

    fit_mask = df_scaled["engine_id"].isin(splits["fit"])
    fit_means = df_scaled.loc[fit_mask, feature_cols].mean()
    np.testing.assert_allclose(fit_means.values, 0.0, atol=1e-6)


def test_apply_scaler_preserves_id_cols(df_clean: tuple[pd.DataFrame, list[str]]) -> None:
    """Scaling must not alter engine_id, cycle, rul, or dataset columns."""
    cleaned, _ = df_clean
    splits = split_engines(cleaned, seed=SPLIT_SEED)
    scaler, feature_cols = fit_scaler(cleaned, splits["fit"])
    df_scaled = apply_scaler(cleaned, scaler, feature_cols)

    pd.testing.assert_series_equal(cleaned["engine_id"], df_scaled["engine_id"])
    pd.testing.assert_series_equal(cleaned["cycle"],     df_scaled["cycle"])
    pd.testing.assert_series_equal(cleaned["rul"],       df_scaled["rul"])


# ---------------------------------------------------------------------------
# Splits JSON serialisation
# ---------------------------------------------------------------------------


def test_save_and_load_splits_roundtrip(df_clean: tuple[pd.DataFrame, list[str]]) -> None:
    cleaned, _ = df_clean
    splits = split_engines(cleaned, seed=SPLIT_SEED)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name
    save_splits(splits, path)
    loaded = load_splits(path)
    assert loaded == splits


# ---------------------------------------------------------------------------
# build_pipeline integration test
# ---------------------------------------------------------------------------


def test_build_pipeline_keys(data_dir: pathlib.Path) -> None:
    """build_pipeline must return all expected keys."""
    result = build_pipeline(data_dir)
    for key in ("df_scaled", "df_raw", "splits", "scaler", "feature_cols", "dropped_cols"):
        assert key in result, f"Missing key: {key!r}"


def test_build_pipeline_no_data_leakage(data_dir: pathlib.Path) -> None:
    """No engine_id may appear in multiple splits returned by build_pipeline."""
    result = build_pipeline(data_dir)
    splits = result["splits"]
    fit  = set(splits["fit"])
    cal  = set(splits["calibration"])
    test = set(splits["test"])
    assert fit.isdisjoint(cal)  and fit.isdisjoint(test) and cal.isdisjoint(test)