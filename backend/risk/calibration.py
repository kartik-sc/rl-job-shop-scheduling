"""
Layer 2: Risk Calibration for RAJS.

Converts raw OC-SVM and Isolation Forest anomaly scores into two outputs:
  1. risk_score ∈ [0, 1]   – degradation-risk signal for the RAJS scheduler.
  2. rul_estimate (cycles)  – estimated remaining useful life.

Design
------
The downstream RAJS scheduler (event_log.schema.json) consumes a ``risk`` field
in [0, 1] on ``risk_update`` events.  The calibration layer must produce a
value in that range that is:
  - Monotone: higher anomaly score → higher risk.
  - Bounded:  0 = healthy (no imminent concern), 1 = near failure.
  - Reproducible across calls.

We deliberately do NOT label the output as a failure probability.  It is a
calibrated degradation-risk score.  Calling it a probability would require a
well-defined prediction horizon and verified calibration on held-out data,
neither of which is available until the test-split evaluation.

Calibration methodology
-----------------------
Two components are fitted using ONLY the 40 calibration engines.

A. Isotonic RUL mapping (score → expected RUL):
   Isotonic regression (PAVA) maps raw anomaly score to expected RUL.  This
   enforces the monotone constraint (higher score → lower expected RUL) and
   is non-parametric, so it can capture the non-linear score-RUL relationship
   visible in the data (OC-SVM shows a nearly piecewise-linear step profile;
   IForest shows a gentler sigmoid).  The fit is engine-aware: we concatenate
   all (score, RUL) pairs from cal engines and use the isotonic regressor in
   increasing=False mode (higher score → lower RUL).

B. Risk score derivation:
   risk = 1 - clamp(rul_estimate / max_lifetime, 0, 1)
   where max_lifetime is taken from fit-split engine lifetimes (already
   captured in the bundle's split metadata) to avoid any look-ahead into the
   calibration engines' actual lifetimes.  This transforms the raw RUL
   estimate into a [0, 1] risk value: 0 when the engine is brand new,
   approaching 1 as estimated RUL approaches 0.

Leakage safeguards
------------------
- Anomaly detection models are loaded from bundle (already fit on fit-split).
- All calibration fits (isotonic, RUL normalisation constant) use cal-split
  engines only.
- Test-split engines are never accessed.
- The max_lifetime normalisation constant is derived from the fit-split to
  avoid any calibration-engine leakage into the normalisation denominator.

Usage
-----
Train:
    from backend.risk.calibration import CalibrationPipeline
    cal = CalibrationPipeline.train(bundle, df_scaled, cal_engine_ids,
                                    fit_engine_ids)
    cal.save('models/calibration_bundle.pkl')

Inference (single engine observation):
    risk, rul_est = cal.predict(anomaly_score_scalar)

Batch:
    risks, rul_ests = cal.predict_batch(score_array)
"""

from __future__ import annotations

import dataclasses
import pathlib
import pickle
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.validation import check_is_fitted

from backend.risk.anomaly_models import (
    AnomalyBundle,
    score_ocsvm,
    score_iforest,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT        = pathlib.Path(__file__).parent.parent.parent
CAL_BUNDLE_PATH  = REPO_ROOT / "models" / "calibration_bundle.pkl"

# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CalibrationBundle:
    """Serialisable container for a fitted calibration pipeline.

    Attributes
    ----------
    model_name       : 'ocsvm' or 'iforest' — which detector this calibrates.
    iso_reg          : fitted IsotonicRegression (score → expected_rul).
    max_lifetime     : normalisation constant for risk score derivation,
                       derived from fit-split engine lifetimes only.
    score_min        : minimum score seen during calibration fit (for clipping).
    score_max        : maximum score seen during calibration fit (for clipping).
    cal_engine_ids   : list of calibration engine IDs used during fitting.
    fit_engine_ids   : list of fit engine IDs (stored for provenance).
    """
    model_name:      Literal["ocsvm", "iforest"]
    iso_reg:         IsotonicRegression
    max_lifetime:    float
    score_min:       float
    score_max:       float
    cal_engine_ids:  list[str]
    fit_engine_ids:  list[str]

    def predict_rul(self, scores: np.ndarray) -> np.ndarray:
        """Map raw anomaly scores to estimated RUL (cycles).

        Clips inputs to the training score range before applying isotonic
        regression to avoid extrapolation artefacts.

        Parameters
        ----------
        scores : 1-D array of raw anomaly scores (higher = more anomalous).

        Returns
        -------
        rul_est : 1-D array of estimated RUL values, non-negative.
        """
        scores = np.asarray(scores, dtype=float).ravel()
        scores_clipped = np.clip(scores, self.score_min, self.score_max)
        rul_est = self.iso_reg.predict(scores_clipped)
        return np.maximum(rul_est, 0.0)

    def predict_risk(self, scores: np.ndarray) -> np.ndarray:
        """Map raw anomaly scores to degradation-risk scores in [0, 1].

        risk = 1 - clamp(rul_estimate / max_lifetime, 0, 1)

        Interpretation:
          0.0 → engine is estimated to be like-new (RUL ≈ max_lifetime).
          1.0 → engine is estimated to be near or at failure (RUL ≈ 0).

        This is a calibrated degradation-risk score, NOT a failure probability.
        It should not be interpreted as P(failure within N cycles).

        Parameters
        ----------
        scores : 1-D array of raw anomaly scores.

        Returns
        -------
        risk : 1-D array of risk scores in [0, 1].
        """
        rul_est = self.predict_rul(scores)
        rul_norm = np.clip(rul_est / self.max_lifetime, 0.0, 1.0)
        return 1.0 - rul_norm

    def predict(self, score: float) -> tuple[float, float]:
        """Convenience: predict risk and RUL for a single scalar score.

        Returns
        -------
        (risk_score, rul_estimate)
        """
        arr = np.array([score])
        return float(self.predict_risk(arr)[0]), float(self.predict_rul(arr)[0])


def save_calibration_bundle(
    bundle: CalibrationBundle,
    path: str | pathlib.Path,
) -> None:
    """Pickle a CalibrationBundle to *path*."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)


def load_calibration_bundle(
    path: str | pathlib.Path,
) -> CalibrationBundle:
    """Load a previously saved CalibrationBundle."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _compute_max_lifetime(
    df: pd.DataFrame,
    fit_engine_ids: list[str],
) -> float:
    """Derive the normalisation constant from fit-split engine lifetimes.

    Using the fit split (not calibration) prevents any leakage from cal
    engines' actual lifetimes into the normalisation denominator.

    Returns
    -------
    float : maximum observed lifetime (cycles) across fit-split engines.
    """
    fit_df = df[df["engine_id"].isin(fit_engine_ids)]
    lifetimes = fit_df.groupby("engine_id")["cycle"].max()
    return float(lifetimes.max())


def fit_calibration(
    model_name: Literal["ocsvm", "iforest"],
    anomaly_bundle: AnomalyBundle,
    df_scaled: pd.DataFrame,
    cal_engine_ids: list[str],
    fit_engine_ids: list[str],
) -> CalibrationBundle:
    """Fit the calibration mapping for one anomaly model.

    Uses ONLY cal_engine_ids to learn the score → RUL mapping.
    Derives max_lifetime from fit_engine_ids only.

    Parameters
    ----------
    model_name      : 'ocsvm' or 'iforest'.
    anomaly_bundle  : loaded AnomalyBundle from Layer 1.
    df_scaled       : full scaled DataFrame from build_pipeline.
    cal_engine_ids  : list of calibration engine IDs.
    fit_engine_ids  : list of fit engine IDs.

    Returns
    -------
    CalibrationBundle
    """
    if model_name not in ("ocsvm", "iforest"):
        raise ValueError(f"model_name must be 'ocsvm' or 'iforest', got {model_name!r}")

    feature_cols = anomaly_bundle.feature_cols

    # Leakage check: cal engines must not appear in fit or test splits
    fit_set  = set(fit_engine_ids)
    cal_set  = set(cal_engine_ids)
    test_set = set(anomaly_bundle.splits["test"])
    if not cal_set.isdisjoint(fit_set):
        raise ValueError("Leakage: calibration engines overlap with fit split")
    if not cal_set.isdisjoint(test_set):
        raise ValueError("Leakage: calibration engines overlap with test split")

    # Collect (score, RUL) pairs from all calibration engines
    cal_df = df_scaled[df_scaled["engine_id"].isin(cal_engine_ids)].copy()
    cal_df = cal_df.sort_values(["engine_id", "cycle"])

    X   = cal_df[feature_cols].values
    rul = cal_df["rul"].values.astype(float)

    if model_name == "ocsvm":
        scores = score_ocsvm(anomaly_bundle.ocsvm, X)
    else:
        scores = score_iforest(anomaly_bundle.iforest, X)

    # Fit isotonic regression: higher score → lower RUL (increasing=False)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(scores, rul)

    max_lifetime = _compute_max_lifetime(df_scaled, fit_engine_ids)
    score_min    = float(scores.min())
    score_max    = float(scores.max())

    return CalibrationBundle(
        model_name=model_name,
        iso_reg=iso,
        max_lifetime=max_lifetime,
        score_min=score_min,
        score_max=score_max,
        cal_engine_ids=list(cal_engine_ids),
        fit_engine_ids=list(fit_engine_ids),
    )


def fit_both_calibrations(
    anomaly_bundle: AnomalyBundle,
    df_scaled: pd.DataFrame,
    cal_engine_ids: list[str],
    fit_engine_ids: list[str],
) -> dict[str, CalibrationBundle]:
    """Fit calibration pipelines for both OC-SVM and Isolation Forest.

    Returns
    -------
    dict with keys 'ocsvm' and 'iforest', each a CalibrationBundle.
    """
    return {
        name: fit_calibration(
            name, anomaly_bundle, df_scaled, cal_engine_ids, fit_engine_ids
        )
        for name in ("ocsvm", "iforest")
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_calibration(
    model_name: Literal["ocsvm", "iforest"],
    cal_bundle: CalibrationBundle,
    anomaly_bundle: AnomalyBundle,
    df_scaled: pd.DataFrame,
    eval_engine_ids: list[str],
) -> pd.DataFrame:
    """Score every cycle of eval engines; attach calibrated risk and RUL.

    Returns DataFrame with columns:
        engine_id, dataset, cycle, rul_true, anomaly_score,
        rul_estimate, risk_score

    IMPORTANT: This function can be called with cal_engine_ids for
    in-calibration evaluation.  Never call it with test_engine_ids
    before the final held-out evaluation is approved.
    """
    feature_cols = anomaly_bundle.feature_cols
    rows = []
    for eng_id in eval_engine_ids:
        eng = df_scaled[df_scaled["engine_id"] == eng_id].sort_values("cycle")
        X   = eng[feature_cols].values
        if model_name == "ocsvm":
            scores = score_ocsvm(anomaly_bundle.ocsvm, X)
        else:
            scores = score_iforest(anomaly_bundle.iforest, X)

        rul_est  = cal_bundle.predict_rul(scores)
        risk_sc  = cal_bundle.predict_risk(scores)

        chunk = pd.DataFrame({
            "engine_id":     eng_id,
            "dataset":       eng_id.split("_")[0],
            "cycle":         eng["cycle"].values,
            "rul_true":      eng["rul"].values,
            "anomaly_score": scores,
            "rul_estimate":  rul_est,
            "risk_score":    risk_sc,
        })
        rows.append(chunk)

    return pd.concat(rows, ignore_index=True)


def compute_calibration_metrics(
    eval_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-engine calibration quality metrics.

    Metrics
    -------
    - rul_mae         : mean absolute error of RUL estimate.
    - rul_rmse        : root-mean-square error of RUL estimate.
    - risk_spearman   : Spearman ρ between risk_score and −rul_true.
    - risk_dir_success: 1 if mean late-life risk > mean healthy risk.
    - healthy_risk_mean : mean risk score in cycles 1–30.
    - latelife_risk_mean: mean risk score in final 30 cycles (RUL < 30).
    - healthy_risk_std  : std of risk score in cycles 1–30.
    - monotone_violations: fraction of consecutive cycle pairs where
                           risk decreases (risk[t+1] < risk[t]).
    """
    from scipy import stats as sp_stats

    HEALTHY_N   = 30
    LATE_LIFE_N = 30

    rows = []
    for eng_id, grp in eval_df.groupby("engine_id"):
        grp = grp.sort_values("cycle")
        rul_t  = grp["rul_true"].values
        rul_e  = grp["rul_estimate"].values
        risk   = grp["risk_score"].values
        neg_rul = -rul_t

        mae  = np.mean(np.abs(rul_e - rul_t))
        rmse = np.sqrt(np.mean((rul_e - rul_t) ** 2))

        if np.std(risk) < 1e-12:
            rho = np.nan
        else:
            rho, _ = sp_stats.spearmanr(neg_rul, risk)

        hw  = risk[grp["cycle"].values <= HEALTHY_N]
        ll  = risk[rul_t < LATE_LIFE_N]
        dir_ok = float(ll.mean() > hw.mean()) if len(hw) > 0 and len(ll) > 0 else np.nan

        mono_viols = (np.diff(risk) < 0).mean() if len(risk) > 1 else 0.0

        rows.append({
            "engine_id":         eng_id,
            "dataset":           grp["dataset"].iloc[0],
            "rul_mae":           mae,
            "rul_rmse":          rmse,
            "risk_spearman":     rho,
            "risk_dir_success":  dir_ok,
            "healthy_risk_mean": hw.mean() if len(hw) > 0 else np.nan,
            "latelife_risk_mean": ll.mean() if len(ll) > 0 else np.nan,
            "healthy_risk_std":  hw.std(ddof=1) if len(hw) > 1 else 0.0,
            "monotone_violations": mono_viols,
            "n_cycles":          len(grp),
        })

    return pd.DataFrame(rows)


def summarise_metrics(
    metrics_df: pd.DataFrame,
    scope_label: str = "overall",
) -> pd.DataFrame:
    """Aggregate per-engine calibration metrics to mean/median/std/IQR."""
    records = []
    for col in ("rul_mae", "rul_rmse", "risk_spearman",
                "risk_dir_success", "healthy_risk_mean",
                "latelife_risk_mean", "monotone_violations"):
        vals = metrics_df[col].dropna()
        records.append({
            "scope":   scope_label,
            "metric":  col,
            "mean":    vals.mean(),
            "median":  vals.median(),
            "std":     vals.std(ddof=1),
            "q25":     vals.quantile(0.25),
            "q75":     vals.quantile(0.75),
            "n":       len(vals),
        })
    return pd.DataFrame(records)
