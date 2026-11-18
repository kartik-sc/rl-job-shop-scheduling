"""
Layer-1 anomaly detection for C-MAPSS engine health monitoring.

Two models are provided:
  - One-Class SVM  (sklearn.svm.OneClassSVM)
  - Isolation Forest (sklearn.ensemble.IsolationForest)

Training convention
-------------------
Both models are fitted **only** on healthy early-cycle observations
from fit-split engines.  "Healthy" means cycle <= healthy_max_cycle.
The rationale: early cycles are assumed to represent normal operation
before any degradation has accumulated.

Score convention
----------------
Raw sklearn decision functions return *higher* values for inliers.
This module negates them so that higher score == more anomalous,
which is the natural direction for a risk/health signal.
"""

from __future__ import annotations

import dataclasses
import pathlib
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from backend.risk.cmapss_pipeline import get_feature_cols


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AnomalyConfig:
    """Hyperparameters and healthy-window definition for both detectors.

    Attributes
    ----------
    healthy_max_cycle : int
        Cycles 1..healthy_max_cycle are used as healthy training data.
        Must be well below the minimum observed engine lifetime.
    ocsvm_kernel : str
        Kernel for OneClassSVM. 'rbf' works well for sensor data.
    ocsvm_nu : float
        Upper bound on training outlier fraction (OneClassSVM nu param).
    ocsvm_gamma : str | float
        Kernel coefficient for OneClassSVM.
    iforest_n_estimators : int
        Number of trees in IsolationForest.
    iforest_contamination : float | str
        Expected fraction of outliers. Use 'auto' to follow sklearn default.
    iforest_random_state : int
        Random seed for IsolationForest reproducibility.
    """

    healthy_max_cycle:      int   = 30
    ocsvm_kernel:           str   = "rbf"
    ocsvm_nu:               float = 0.05
    ocsvm_gamma:            str   = "scale"
    iforest_n_estimators:   int   = 200
    iforest_contamination:  Any   = "auto"
    iforest_random_state:   int   = 42


DEFAULT_CONFIG = AnomalyConfig()


# ---------------------------------------------------------------------------
# Healthy-window selection
# ---------------------------------------------------------------------------


def select_healthy_cycles(
    df: pd.DataFrame,
    fit_engine_ids: list[str],
    healthy_max_cycle: int,
    feature_cols: list[str],
) -> np.ndarray:
    """Return a 2-D feature array of healthy early-cycle rows.

    Selects rows where:
      - engine_id is in fit_engine_ids  (no cal/test leakage)
      - cycle <= healthy_max_cycle       (early, healthy operation)

    Parameters
    ----------
    df                : scaled DataFrame from build_pipeline
    fit_engine_ids    : engine IDs belonging to the fit split
    healthy_max_cycle : upper cycle limit for healthy window
    feature_cols      : list of feature column names

    Returns
    -------
    X_healthy : np.ndarray of shape (n_healthy_rows, n_features)
    """
    mask = (
        df["engine_id"].isin(fit_engine_ids)
        & (df["cycle"] <= healthy_max_cycle)
    )
    X = df.loc[mask, feature_cols].values
    if len(X) == 0:
        raise ValueError(
            f"No healthy rows found for healthy_max_cycle={healthy_max_cycle} "
            f"and {len(fit_engine_ids)} fit engines."
        )
    return X


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_ocsvm(
    X_healthy: np.ndarray,
    config: AnomalyConfig = DEFAULT_CONFIG,
) -> OneClassSVM:
    """Fit a One-Class SVM on healthy observations.

    The model is fitted on *X_healthy* only (pre-selected by caller).
    """
    model = OneClassSVM(
        kernel=config.ocsvm_kernel,
        nu=config.ocsvm_nu,
        gamma=config.ocsvm_gamma,
    )
    model.fit(X_healthy)
    return model


def train_iforest(
    X_healthy: np.ndarray,
    config: AnomalyConfig = DEFAULT_CONFIG,
) -> IsolationForest:
    """Fit an Isolation Forest on healthy observations.

    The model is fitted on *X_healthy* only (pre-selected by caller).
    """
    model = IsolationForest(
        n_estimators=config.iforest_n_estimators,
        contamination=config.iforest_contamination,
        random_state=config.iforest_random_state,
    )
    model.fit(X_healthy)
    return model


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_ocsvm(
    model: OneClassSVM,
    X: np.ndarray,
) -> np.ndarray:
    """Return anomaly scores for rows in X.

    Score convention: **higher == more anomalous**.
    Achieved by negating sklearn decision_function (which returns
    positive values for inliers).
    """
    return -model.decision_function(X)


def score_iforest(
    model: IsolationForest,
    X: np.ndarray,
) -> np.ndarray:
    """Return anomaly scores for rows in X.

    Score convention: **higher == more anomalous**.
    Achieved by negating sklearn score_samples (which returns
    negative average path length: less negative = more normal).
    """
    return -model.score_samples(X)


def score_engine(
    df: pd.DataFrame,
    engine_id: str,
    ocsvm: OneClassSVM,
    iforest: IsolationForest,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Score all cycles of one engine with both models.

    Returns a DataFrame with columns: cycle, rul, ocsvm_score, iforest_score.
    Scores are in the higher-is-more-anomalous convention.
    """
    eng = df[df["engine_id"] == engine_id].sort_values("cycle")
    X = eng[feature_cols].values
    out = pd.DataFrame({
        "cycle":         eng["cycle"].values,
        "rul":           eng["rul"].values,
        "ocsvm_score":   score_ocsvm(ocsvm, X),
        "iforest_score": score_iforest(iforest, X),
    })
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AnomalyBundle:
    """Everything needed to reproduce or apply the anomaly detectors."""
    ocsvm:        OneClassSVM
    iforest:      IsolationForest
    feature_cols: list[str]
    config:       AnomalyConfig
    splits:       dict[str, list[str]]


def save_bundle(bundle: AnomalyBundle, path: str | pathlib.Path) -> None:
    """Pickle the full anomaly bundle to *path* (models/ dir, git-ignored)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)


def load_bundle(path: str | pathlib.Path) -> AnomalyBundle:
    """Load a previously saved AnomalyBundle."""
    with open(path, "rb") as fh:
        return pickle.load(fh)