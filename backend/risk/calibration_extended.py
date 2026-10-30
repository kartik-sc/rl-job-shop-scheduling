"""
Extended CalibrationBundle types for Phase 3 experiments.

These classes are defined here (not in calibration_experiments.py) so that
pickle can locate them via a stable module path when deserialising artifacts
from tests or other scripts.
"""
from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclasses.dataclass
class PiecewiseCalibrationBundle:
    """Two isotonic regressors: bulk (low score) + critical (high score).

    The boundary_score separates the two zones.
    Both regressors map: score → predicted RUL.
    The mapping is monotone within each piece and (by construction) at
    the boundary after the continuity correction in fit_cal_C.
    """
    model_name:     Literal["ocsvm", "iforest"]
    iso_bulk:       IsotonicRegression   # lower scores → higher RUL
    iso_critical:   IsotonicRegression   # higher scores → lower RUL
    boundary_score: float
    max_lifetime:   float
    score_min:      float
    score_max:      float
    cal_engine_ids: list[str]
    fit_engine_ids: list[str]

    def predict_rul(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float).ravel()
        scores_clipped = np.clip(scores, self.score_min, self.score_max)
        rul_est = np.empty_like(scores_clipped)
        bulk_mask     = scores_clipped <= self.boundary_score
        critical_mask = ~bulk_mask
        if bulk_mask.any():
            rul_est[bulk_mask] = self.iso_bulk.predict(
                scores_clipped[bulk_mask])
        if critical_mask.any():
            rul_est[critical_mask] = self.iso_critical.predict(
                scores_clipped[critical_mask])
        return np.maximum(rul_est, 0.0)

    def predict_risk(self, scores: np.ndarray) -> np.ndarray:
        rul_est  = self.predict_rul(scores)
        rul_norm = np.clip(rul_est / self.max_lifetime, 0.0, 1.0)
        return 1.0 - rul_norm

    def predict(self, score: float) -> tuple[float, float]:
        arr = np.array([score])
        return float(self.predict_risk(arr)[0]), float(self.predict_rul(arr)[0])


@dataclasses.dataclass
class QuantileCalibrationBundle:
    """Isotonic regressor fitted on the q-th quantile of RUL (conservative).

    This is a lower-bound estimate, NOT the expected RUL.
    It is stored alongside the mean-estimate CalibrationBundle for comparison.
    The risk score is still derived with the same formula:
        risk = 1 - clamp(rul_estimate / max_lifetime, 0, 1)
    """
    model_name:     Literal["ocsvm", "iforest"]
    iso_q25:        IsotonicRegression
    max_lifetime:   float
    score_min:      float
    score_max:      float
    quantile:       float   # e.g. 0.25
    cal_engine_ids: list[str]
    fit_engine_ids: list[str]

    def predict_rul(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float).ravel()
        scores_clipped = np.clip(scores, self.score_min, self.score_max)
        return np.maximum(self.iso_q25.predict(scores_clipped), 0.0)

    def predict_risk(self, scores: np.ndarray) -> np.ndarray:
        rul_est  = self.predict_rul(scores)
        rul_norm = np.clip(rul_est / self.max_lifetime, 0.0, 1.0)
        return 1.0 - rul_norm

    def predict(self, score: float) -> tuple[float, float]:
        arr = np.array([score])
        return float(self.predict_risk(arr)[0]), float(self.predict_rul(arr)[0])
