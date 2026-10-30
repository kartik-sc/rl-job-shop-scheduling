"""
C-MAPSS data-loading and preprocessing pipeline.

Supports FD001 and FD003 only (single operating condition, NASA C-MAPSS).
Reference: Saxena et al., PHM08.

Column layout (26 columns, space-separated, no header):
  0  unit          - engine number within the dataset file (1-based)
  1  cycle         - time in cycles
  2  op1           - operational setting 1
  3  op2           - operational setting 2
  4  op3           - operational setting 3  [constant, dropped]
  5  s1            - sensor 1               [constant, dropped]
  6  s2  7  s3  8  s4
  9  s5            - sensor 5               [constant, dropped]
 10  s6 ... 19  s15
 20  s16           - sensor 16              [constant, dropped]
 21  s17
 22  s18           - sensor 18              [constant, dropped]
 23  s19           - sensor 19              [constant, dropped]
 24  s20  25  s21
"""

from __future__ import annotations

import json
import pathlib
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Column metadata
# ---------------------------------------------------------------------------

_ALL_COLS: list[str] = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

# Columns with std < 1e-6 across combined FD001+FD003 training data.
# Removing them prevents zero-variance warnings and uninformative features.
CONSTANT_COLS: tuple[str, ...] = ("op3", "s1", "s5", "s16", "s18", "s19")

# Identifier/label columns - not used as model inputs
_ID_COLS: tuple[str, ...] = ("engine_id", "dataset", "unit", "cycle", "rul")

SUPPORTED_DATASETS: tuple[str, ...] = ("FD001", "FD003")

_N_FIT  = 120
_N_CAL  = 40
_N_TEST = 40
SPLIT_SEED = 42


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_dataset(path: str | pathlib.Path) -> pd.DataFrame:
    """Load one C-MAPSS training file; add engine_id, dataset, and rul."""
    path = pathlib.Path(path)
    dataset = _infer_dataset(path)
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=_ALL_COLS,
        dtype=float,
        engine="python",
    )
    df["unit"]  = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    df["dataset"]   = dataset
    # Unique engine ID stable across dataset merges
    df["engine_id"] = dataset + "_" + df["unit"].astype(str).str.zfill(3)
    # RUL = last observed cycle for this engine minus current cycle
    df["rul"] = (
        df.groupby("engine_id")["cycle"].transform("max") - df["cycle"]
    )
    return df


def _infer_dataset(path: pathlib.Path) -> str:
    """Extract dataset tag (e.g. 'FD001') from a filename like train_FD001.txt."""
    for ds in SUPPORTED_DATASETS:
        if ds in path.stem:
            return ds
    raise ValueError(
        f"Cannot infer dataset from filename {path.name!r}. "
        f"Expected one of {SUPPORTED_DATASETS}."
    )


def load_combined(data_dir: str | pathlib.Path) -> pd.DataFrame:
    """Load FD001 and FD003, concatenate, sort by engine_id then cycle."""
    data_dir = pathlib.Path(data_dir)
    frames = []
    for ds in SUPPORTED_DATASETS:
        fpath = data_dir / f"train_{ds}.txt"
        if not fpath.exists():
            raise FileNotFoundError(
                f"Training file not found: {fpath}. "
                "Place train_FD001.txt and train_FD003.txt in data/raw/cmapss/."
            )
        frames.append(load_dataset(fpath))
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["engine_id", "cycle"])
    combined = combined.reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Constant-feature removal
# ---------------------------------------------------------------------------


def drop_constant_features(
    df: pd.DataFrame,
    threshold: float = 1e-6,
    extra_cols: Sequence[str] = (),
) -> tuple[pd.DataFrame, list[str]]:
    """Remove sensor/op columns whose global std is below *threshold*.

    Identifier and label columns (engine_id, dataset, unit, cycle, rul)
    are never considered for removal.

    Returns
    -------
    (cleaned_df, dropped_col_names)
    """
    feature_cols = [
        c for c in df.columns
        if c not in _ID_COLS and c not in ("dataset", "unit")
    ]
    stds = df[feature_cols].std()
    auto_drop = stds[stds < threshold].index.tolist()
    to_drop = list(dict.fromkeys(list(auto_drop) + list(extra_cols)))
    cleaned = df.drop(columns=[c for c in to_drop if c in df.columns])
    return cleaned, to_drop


# ---------------------------------------------------------------------------
# Engine-level split (no leakage)
# ---------------------------------------------------------------------------


def split_engines(
    df: pd.DataFrame,
    n_fit:  int = _N_FIT,
    n_cal:  int = _N_CAL,
    n_test: int = _N_TEST,
    seed:   int = SPLIT_SEED,
) -> dict[str, list[str]]:
    """Randomly assign engine_ids to fit / calibration / test splits.

    Every row for a given engine lands in exactly one split.
    The split is deterministic given *seed*.

    Returns
    -------
    dict with keys "fit", "calibration", "test", each a sorted list of engine_id.
    """
    engine_ids = sorted(df["engine_id"].unique().tolist())
    total    = len(engine_ids)
    required = n_fit + n_cal + n_test
    if total != required:
        raise ValueError(
            f"Expected {required} unique engines ({n_fit}+{n_cal}+{n_test}) "
            f"but found {total}."
        )
    rng      = np.random.default_rng(seed)
    shuffled = rng.permutation(engine_ids).tolist()
    return {
        "fit":         sorted(shuffled[:n_fit]),
        "calibration": sorted(shuffled[n_fit : n_fit + n_cal]),
        "test":        sorted(shuffled[n_fit + n_cal :]),
    }


def save_splits(splits: dict[str, list[str]], out_path: str | pathlib.Path) -> None:
    """Persist the engine-ID split to JSON (intended to be committed)."""
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(splits, fh, indent=2)


def load_splits(path: str | pathlib.Path) -> dict[str, list[str]]:
    """Load a previously saved splits.json."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Preprocessing (parameters fitted on fit-split only)
# ---------------------------------------------------------------------------


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return model-input feature column names from a cleaned DataFrame."""
    return [
        c for c in df.columns
        if c not in _ID_COLS and c not in ("dataset", "unit")
    ]


def fit_scaler(
    df: pd.DataFrame,
    fit_engine_ids: list[str],
) -> tuple[StandardScaler, list[str]]:
    """Fit a StandardScaler **only on fit-split engines**.

    Parameters
    ----------
    df              : cleaned DataFrame (constant cols already removed)
    fit_engine_ids  : engine IDs belonging to the fit split

    Returns
    -------
    (fitted_scaler, feature_col_names)
    """
    feature_cols = get_feature_cols(df)
    fit_mask = df["engine_id"].isin(fit_engine_ids)
    scaler   = StandardScaler()
    scaler.fit(df.loc[fit_mask, feature_cols].values)
    return scaler, feature_cols


def apply_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Return a copy of *df* with feature columns z-scored by *scaler*."""
    out = df.copy()
    out[feature_cols] = scaler.transform(df[feature_cols].values)
    return out


# ---------------------------------------------------------------------------
# Convenience: full pipeline in one call
# ---------------------------------------------------------------------------


def build_pipeline(
    data_dir:    str | pathlib.Path,
    splits_path: str | pathlib.Path | None = None,
    seed:        int = SPLIT_SEED,
) -> dict:
    """End-to-end pipeline: load -> clean -> split -> scale.

    Parameters
    ----------
    data_dir    : directory containing train_FD001.txt and train_FD003.txt
    splits_path : if given and the file exists, load splits from JSON;
                  otherwise generate and (if path given) save them.
    seed        : RNG seed used only when generating a new split

    Returns
    -------
    dict with keys: df_scaled, df_raw, splits, scaler, feature_cols, dropped_cols
    """
    df_raw_all = load_combined(data_dir)
    df_clean, dropped = drop_constant_features(df_raw_all)

    if splits_path is not None and pathlib.Path(splits_path).exists():
        splits = load_splits(splits_path)
    else:
        splits = split_engines(df_clean, seed=seed)
        if splits_path is not None:
            save_splits(splits, splits_path)

    scaler, feature_cols = fit_scaler(df_clean, splits["fit"])
    df_scaled = apply_scaler(df_clean, scaler, feature_cols)

    return {
        "df_scaled":    df_scaled,
        "df_raw":       df_clean,
        "splits":       splits,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "dropped_cols": dropped,
    }