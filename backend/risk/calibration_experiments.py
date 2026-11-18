"""
Phase 3: Calibration experiments for RAJS.

Builds on D_temporal anomaly-detection configuration (Phase 2 winner):
  - Separate FD001 / FD003 OC-SVM + IForest models
  - 18 raw features + 5 causal rolling means (w=10)

Four calibration approaches evaluated against a baseline:

  Baseline — Phase 1 shared calibration (loaded from existing artifacts)
              Single isotonic regressor fitted on all 40 cal engines combined.

  Exp A    — Dataset-specific isotonic calibration (no anchoring)
              Separate calibrators for FD001 (20 engines) and FD003 (20 engines).
              Same decreasing isotonic regression, same risk formula.

  Exp B    — Dataset-specific + endpoint anchoring
              Builds on Exp A. Adds one synthetic anchor (score_p99_fit, 0)
              derived exclusively from fit-split score statistics.
              Prevents the isotonic floor from being unrealistically high.

  Exp C    — Piecewise isotonic calibration
              Two isotonic regressors per dataset: one for high scores (critical
              zone, RUL ≤ critical_rul_boundary) and one for low scores (bulk
              zone). Boundary chosen from fit-split score distribution.
              Only implemented if data density supports it.

  Exp D    — Conservative quantile-based calibration
              Fits isotonic regression on 25th-percentile RUL (lower bound
              estimate) rather than the conditional mean.
              Explicitly labelled as a conservative lower-bound estimate.
              Does NOT replace the mean estimate — reported alongside it.

Leakage safeguards
------------------
- D_temporal anomaly bundles loaded from models/experiments/ (no retraining).
- Rolling means recomputed causally on fit + cal data only.
- All calibration fitting uses cal engines only.
- Anchor scores derived from fit-split score distributions only.
- Piecewise boundary derived from fit-split score distributions only.
- Test engines never touched.
- max_lifetime derived from fit-split engine lifetimes.

Outputs
-------
models/calibration_experiments/
    cal_{exp}_{dataset}_{model}.pkl
      exp in (A, B, C, D)
      dataset in (fd001, fd003)
      model in (ocsvm, iforest)

results/calibration_experiments/
    predictions_{exp}_{dataset}_{model}.csv
    rul_errors_by_bin_{exp}_{dataset}_{model}.csv
    per_engine_metrics_{exp}_{dataset}_{model}.csv
    summary_all_experiments.csv
    phase3_calibration_results.md

plots/calibration_experiments/
    predicted_vs_actual_{exp}_{model}.png
    rul_error_by_bin_comparison.png
    calibration_curves_{dataset}.png
    risk_progression_{exp}_{dataset}.png
    near_failure_distribution.png
"""

from __future__ import annotations

import pathlib
import pickle
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.isotonic import IsotonicRegression

from backend.risk.anomaly_models import (
    AnomalyBundle,
    load_bundle,
    score_ocsvm,
    score_iforest,
)
from backend.risk.anomaly_experiments import (
    ROLL_SENSORS,
    ROLL_WINDOW,
    add_rolling_means,
    build_exp_pipeline,
)
from backend.risk.calibration import CalibrationBundle, save_calibration_bundle
from backend.risk.calibration_extended import (
    PiecewiseCalibrationBundle,
    QuantileCalibrationBundle,
)
from backend.risk.cmapss_pipeline import SPLIT_SEED, build_pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT    = pathlib.Path(__file__).parent.parent.parent
DATA_DIR     = REPO_ROOT / "data" / "raw" / "cmapss"
EXP_MODELS   = REPO_ROOT / "models"   / "experiments"
CAL_EXP_DIR  = REPO_ROOT / "models"   / "calibration_experiments"
RESULTS_DIR  = REPO_ROOT / "results"  / "calibration_experiments"
PLOTS_DIR    = REPO_ROOT / "plots"    / "calibration_experiments"

HEALTHY_N    = 30
LATE_LIFE_N  = 30

# Piecewise boundary: score percentile above which we use the "critical" regressor
PIECEWISE_SCORE_PERCENTILE = 70   # derived from fit-split scores; see _fit_piecewise


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _save_csv(df: pd.DataFrame, name: str) -> pathlib.Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / name
    df.to_csv(p, index=False)
    return p


def _save_plot(fig: plt.Figure, name: str) -> pathlib.Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    p = PLOTS_DIR / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Score generation from D_temporal bundles
# ---------------------------------------------------------------------------

def score_engines_temporal(
    df_aug:     pd.DataFrame,
    engine_ids: list[str],
    bundle:     AnomalyBundle,
) -> pd.DataFrame:
    """Score every cycle of given engines using the D_temporal bundle.

    df_aug must already contain the rolling-mean columns used by the bundle.
    """
    fc = bundle.feature_cols
    records = []
    for eng_id in engine_ids:
        eng = df_aug[df_aug["engine_id"] == eng_id].sort_values("cycle")
        if len(eng) == 0:
            continue
        X  = eng[fc].values
        oc = score_ocsvm(bundle.ocsvm, X)
        if_ = score_iforest(bundle.iforest, X)
        chunk = pd.DataFrame({
            "engine_id":     eng_id,
            "dataset":       eng_id.split("_")[0],
            "cycle":         eng["cycle"].values,
            "rul_true":      eng["rul"].values,
            "ocsvm_score":   oc,
            "iforest_score": if_,
        })
        records.append(chunk)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _max_lifetime_from_fit(df_raw: pd.DataFrame, fit_ids: list[str]) -> float:
    """max_lifetime derived from fit-split only (no leakage)."""
    fit_df    = df_raw[df_raw["engine_id"].isin(fit_ids)]
    lifetimes = fit_df.groupby("engine_id")["cycle"].max()
    return float(lifetimes.max())


# ---------------------------------------------------------------------------
# Calibration fitting helpers
# ---------------------------------------------------------------------------

def _fit_standard_iso(
    scores: np.ndarray,
    ruls:   np.ndarray,
) -> IsotonicRegression:
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(scores, ruls)
    return iso


def fit_cal_A(
    model_name:  Literal["ocsvm", "iforest"],
    cal_scores:  pd.DataFrame,
    fit_ids:     list[str],
    cal_ids:     list[str],
    df_raw:      pd.DataFrame,
) -> CalibrationBundle:
    """Exp A: dataset-specific isotonic calibration (no anchoring)."""
    sc_col = f"{model_name}_score"
    scores = cal_scores[sc_col].values.astype(float)
    ruls   = cal_scores["rul_true"].values.astype(float)

    iso = _fit_standard_iso(scores, ruls)
    max_lt = _max_lifetime_from_fit(df_raw, fit_ids)

    return CalibrationBundle(
        model_name=model_name,
        iso_reg=iso,
        max_lifetime=max_lt,
        score_min=float(scores.min()),
        score_max=float(scores.max()),
        cal_engine_ids=list(cal_ids),
        fit_engine_ids=list(fit_ids),
    )


def fit_cal_B(
    model_name:    Literal["ocsvm", "iforest"],
    cal_scores:    pd.DataFrame,
    fit_scores:    pd.DataFrame,
    fit_ids:       list[str],
    cal_ids:       list[str],
    df_raw:        pd.DataFrame,
    anchor_pct:    float = 99.0,
) -> CalibrationBundle:
    """Exp B: dataset-specific + endpoint anchoring.

    Anchor score = fit-split score at `anchor_pct` percentile.
    Anchor RUL   = 0.
    The anchor tells the isotonic regressor that the very highest scores
    observed in training correspond to failure (RUL = 0).

    anchor_pct is chosen from FIT-SPLIT statistics only — no cal/test leakage.
    """
    sc_col = f"{model_name}_score"
    scores = cal_scores[sc_col].values.astype(float)
    ruls   = cal_scores["rul_true"].values.astype(float)

    # Anchor score from fit-split
    fit_sc      = fit_scores[sc_col].values.astype(float)
    anchor_score = float(np.percentile(fit_sc, anchor_pct))
    anchor_rul   = 0.0

    # Prepend anchor to calibration data
    scores_aug = np.append(scores, anchor_score)
    ruls_aug   = np.append(ruls,   anchor_rul)

    iso    = _fit_standard_iso(scores_aug, ruls_aug)
    max_lt = _max_lifetime_from_fit(df_raw, fit_ids)

    return CalibrationBundle(
        model_name=model_name,
        iso_reg=iso,
        max_lifetime=max_lt,
        score_min=float(scores.min()),
        score_max=max(float(scores.max()), anchor_score),
        cal_engine_ids=list(cal_ids),
        fit_engine_ids=list(fit_ids),
    )


def fit_cal_C(
    model_name:   Literal["ocsvm", "iforest"],
    cal_scores:   pd.DataFrame,
    fit_scores:   pd.DataFrame,
    fit_ids:      list[str],
    cal_ids:      list[str],
    df_raw:       pd.DataFrame,
    boundary_pct: float = 70.0,
    min_samples:  int   = 50,
) -> PiecewiseCalibrationBundle | None:
    """Exp C: piecewise isotonic calibration.

    boundary_score = fit-split score at boundary_pct percentile.
    Bulk zone  : score ≤ boundary_score → fitted on bulk-zone cal data.
    Critical zone: score > boundary_score → fitted on critical-zone cal data.

    Continuity is enforced by prepending a (boundary_score, bulk_boundary_pred)
    anchor to the critical-zone training data before fitting.  This forces the
    critical-zone isotonic regressor to start from the bulk prediction at the
    boundary and only decrease from there, ensuring no discontinuous jump.

    Returns None if either zone has < min_samples observations.
    """
    sc_col = f"{model_name}_score"
    scores = cal_scores[sc_col].values.astype(float)
    ruls   = cal_scores["rul_true"].values.astype(float)

    # Boundary from fit-split only (no leakage)
    fit_sc         = fit_scores[sc_col].values.astype(float)
    boundary_score = float(np.percentile(fit_sc, boundary_pct))

    bulk_mask     = scores <= boundary_score
    critical_mask = ~bulk_mask

    n_bulk, n_crit = bulk_mask.sum(), critical_mask.sum()
    if n_bulk < min_samples or n_crit < min_samples:
        return None   # insufficient data — caller falls back to Exp A

    # Step 1: fit bulk regressor
    iso_bulk = _fit_standard_iso(scores[bulk_mask], ruls[bulk_mask])

    # Step 2: get bulk prediction at boundary (this is the continuity target)
    bulk_pred_at_boundary = float(iso_bulk.predict([boundary_score])[0])

    # Step 3: fit critical regressor with a continuity anchor prepended.
    # The anchor forces the critical zone to start at or below the bulk
    # prediction at the boundary, so no upward jump occurs.
    s_crit_aug = np.append(boundary_score, scores[critical_mask])
    r_crit_aug = np.append(bulk_pred_at_boundary, ruls[critical_mask])
    iso_critical = _fit_standard_iso(s_crit_aug, r_crit_aug)

    max_lt = _max_lifetime_from_fit(df_raw, fit_ids)

    return PiecewiseCalibrationBundle(
        model_name=model_name,
        iso_bulk=iso_bulk,
        iso_critical=iso_critical,
        boundary_score=boundary_score,
        max_lifetime=max_lt,
        score_min=float(scores.min()),
        score_max=float(scores.max()),
        cal_engine_ids=list(cal_ids),
        fit_engine_ids=list(fit_ids),
    )


def fit_cal_D(
    model_name: Literal["ocsvm", "iforest"],
    cal_scores: pd.DataFrame,
    fit_ids:    list[str],
    cal_ids:    list[str],
    df_raw:     pd.DataFrame,
    quantile:   float = 0.25,
    n_bins:     int   = 50,
) -> QuantileCalibrationBundle:
    """Exp D: conservative quantile-based calibration.

    For each score bin, compute the target_quantile of observed RUL values.
    Fit isotonic regression on (score, quantile_rul) pairs.
    This gives a conservative lower-bound RUL estimate.

    Implemented by binning the (score, RUL) pairs and computing the quantile
    within each bin, then fitting isotonic regression on the bin summaries.
    """
    sc_col = f"{model_name}_score"
    scores = cal_scores[sc_col].values.astype(float)
    ruls   = cal_scores["rul_true"].values.astype(float)

    # Compute quantile within score bins
    score_edges  = np.linspace(scores.min(), scores.max(), n_bins + 1)
    bin_scores   = []
    bin_quantiles = []
    for i in range(n_bins):
        lo, hi = score_edges[i], score_edges[i + 1]
        mask   = (scores >= lo) & (scores < hi) if i < n_bins - 1 \
                 else (scores >= lo) & (scores <= hi)
        if mask.sum() >= 5:   # need at least 5 obs per bin
            bin_scores.append((lo + hi) / 2.0)
            bin_quantiles.append(float(np.percentile(ruls[mask], quantile * 100)))

    if len(bin_scores) < 3:
        # Fallback: use all obs directly (thin data)
        iso_q = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso_q.fit(scores, np.percentile(ruls, quantile * 100) * np.ones_like(ruls))
    else:
        iso_q = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso_q.fit(np.array(bin_scores), np.array(bin_quantiles))

    max_lt = _max_lifetime_from_fit(df_raw, fit_ids)

    return QuantileCalibrationBundle(
        model_name=model_name,
        iso_q25=iso_q,
        max_lifetime=max_lt,
        score_min=float(scores.min()),
        score_max=float(scores.max()),
        quantile=quantile,
        cal_engine_ids=list(cal_ids),
        fit_engine_ids=list(fit_ids),
    )


# ---------------------------------------------------------------------------
# Prediction from any calibrator type
# ---------------------------------------------------------------------------

def apply_calibrator(
    bundle: CalibrationBundle | PiecewiseCalibrationBundle | QuantileCalibrationBundle,
    scores_df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Apply any calibrator type to a scored DataFrame.

    Returns DataFrame with rul_estimate, risk_score added.
    """
    sc = scores_df[f"{model_name}_score"].values.astype(float)
    rul_est  = bundle.predict_rul(sc)
    risk_sc  = bundle.predict_risk(sc)
    out = scores_df.copy()
    out["rul_estimate"] = rul_est
    out["risk_score"]   = risk_sc
    return out


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def binned_rul_errors(pred_df: pd.DataFrame) -> pd.DataFrame:
    """MAE, RMSE, signed error by RUL bin."""
    bins   = [0, 20, 50, 100, np.inf]
    labels = ["0–20", "21–50", "51–100", ">100"]
    pred_df = pred_df.copy()
    pred_df["error"]     = pred_df["rul_estimate"] - pred_df["rul_true"]
    pred_df["abs_error"] = pred_df["error"].abs()
    pred_df["sq_error"]  = pred_df["error"] ** 2
    pred_df["rul_bin"]   = pd.cut(pred_df["rul_true"], bins=bins,
                                   labels=labels, right=True)
    rows = []
    for lbl in labels:
        sub = pred_df[pred_df["rul_bin"] == lbl]
        if len(sub) == 0:
            continue
        rows.append({
            "rul_bin":             lbl,
            "n":                   len(sub),
            "mae":                 sub["abs_error"].mean(),
            "rmse":                np.sqrt(sub["sq_error"].mean()),
            "mean_signed_error":   sub["error"].mean(),
            "median_signed_error": sub["error"].median(),
            "pct_overestimate":    (sub["error"] > 0).mean(),
        })
    return pd.DataFrame(rows)


def per_engine_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Per-engine: MAE, RMSE, risk Spearman, directional success, monotone violations."""
    rows = []
    for eng_id, grp in pred_df.groupby("engine_id"):
        grp = grp.sort_values("cycle")
        rul_t = grp["rul_true"].values
        rul_e = grp["rul_estimate"].values
        risk  = grp["risk_score"].values
        mae   = np.mean(np.abs(rul_e - rul_t))
        rmse  = np.sqrt(np.mean((rul_e - rul_t) ** 2))
        neg_rul = -rul_t
        rho = float(sp_stats.spearmanr(neg_rul, risk)[0]) \
              if np.std(risk) > 1e-12 else np.nan
        hw  = risk[grp["cycle"].values <= HEALTHY_N]
        ll  = risk[rul_t < LATE_LIFE_N]
        dir_ok = float(ll.mean() > hw.mean()) \
                 if len(hw) > 0 and len(ll) > 0 else np.nan
        mono_v = (np.diff(risk) < 0).mean() if len(risk) > 1 else 0.0
        rows.append({
            "engine_id":          eng_id,
            "dataset":            grp["dataset"].iloc[0],
            "rul_mae":            mae,
            "rul_rmse":           rmse,
            "risk_spearman":      rho,
            "risk_dir_success":   dir_ok,
            "healthy_risk_mean":  hw.mean() if len(hw) else np.nan,
            "latelife_risk_mean": ll.mean() if len(ll) else np.nan,
            "monotone_violations": mono_v,
            "n_cycles":           len(grp),
        })
    return pd.DataFrame(rows)


def aggregate_metrics(metrics_df: pd.DataFrame, label: str) -> dict:
    """Aggregate per-engine metrics to a summary dict."""
    row = {"config": label}
    for scope, mask in [
        ("overall", pd.Series(True, index=metrics_df.index)),
        ("FD001",   metrics_df["dataset"] == "FD001"),
        ("FD003",   metrics_df["dataset"] == "FD003"),
    ]:
        sub = metrics_df[mask]
        for col in ("rul_mae", "rul_rmse", "risk_spearman",
                    "risk_dir_success", "monotone_violations"):
            vals = sub[col].dropna()
            row[f"{scope}_{col}_mean"]   = vals.mean()
            row[f"{scope}_{col}_median"] = vals.median()
    return row


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_predicted_vs_actual(
    preds_dict: dict[str, pd.DataFrame],
    model:      str,
    ds:         str,
) -> pathlib.Path:
    """Predicted vs actual RUL scatter, all calibration experiments."""
    configs = list(preds_dict.keys())
    n = len(configs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]
    colors = {"FD001": "steelblue", "FD003": "darkorange"}

    for ax, cfg in zip(axes, configs):
        df  = preds_dict[cfg]
        sub = df[df["dataset"] == ds] if ds != "overall" else df
        ax.scatter(sub["rul_true"], sub["rul_estimate"],
                   s=3, alpha=0.2, color=colors.get(ds, "grey"),
                   rasterized=True)
        lim = max(sub["rul_true"].max(), sub["rul_estimate"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
        mae  = np.abs(sub["rul_estimate"] - sub["rul_true"]).mean()
        nf   = sub[sub["rul_true"] <= 20]
        nf_mae = np.abs(nf["rul_estimate"] - nf["rul_true"]).mean() if len(nf) else np.nan
        ax.set_title(f"{cfg}\n{ds} | MAE={mae:.1f}  NF-MAE={nf_mae:.1f}",
                     fontsize=8)
        ax.set_xlabel("True RUL")
        ax.set_ylabel("Estimated RUL")
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"{model.upper()} — {ds}: predicted vs actual RUL\n"
                 "Diagonal = perfect prediction; points above = overestimate",
                 fontsize=9)
    plt.tight_layout()
    return _save_plot(fig, f"predicted_vs_actual_{model}_{ds}.png")


def plot_rul_error_bins(
    errors_dict: dict[str, pd.DataFrame],
    model:       str,
    ds:          str,
) -> pathlib.Path:
    """MAE by RUL bin, all calibration experiments compared."""
    configs = list(errors_dict.keys())
    bins    = ["0–20", "21–50", "51–100", ">100"]
    x       = np.arange(len(bins))
    w       = 0.8 / max(len(configs), 1)
    offsets = np.linspace(-(len(configs)-1)/2 * w, (len(configs)-1)/2 * w, len(configs))
    cmap    = matplotlib.colormaps["tab10"].resampled(len(configs))

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, cfg in enumerate(configs):
        df   = errors_dict[cfg]
        vals = [df.loc[df["rul_bin"] == b, "mae"].values[0]
                if b in df["rul_bin"].values else np.nan for b in bins]
        bars = ax.bar(x + offsets[i], vals, w,
                      label=cfg, color=cmap(i), alpha=0.85)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"RUL {b}" for b in bins])
    ax.set_ylabel("MAE (cycles)")
    ax.set_title(f"{model.upper()} {ds} — RUL MAE by true-RUL range\n"
                 "(lower is better; Phase 1 baseline shown for reference)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return _save_plot(fig, f"rul_error_bins_{model}_{ds}.png")


def plot_signed_error_comparison(
    preds_dict: dict[str, pd.DataFrame],
    model:      str,
    ds:         str,
) -> pathlib.Path:
    """Mean signed error (bias) by RUL bin across experiments."""
    configs = list(preds_dict.keys())
    bins    = ["0–20", "21–50", "51–100", ">100"]
    bin_edges = [0, 20, 50, 100, np.inf]
    x = np.arange(len(bins))
    w = 0.8 / max(len(configs), 1)
    offsets = np.linspace(-(len(configs)-1)/2 * w, (len(configs)-1)/2 * w, len(configs))
    cmap    = matplotlib.colormaps["tab10"].resampled(len(configs))

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, cfg in enumerate(configs):
        df   = preds_dict[cfg]
        sub  = df[df["dataset"] == ds] if ds != "overall" else df
        err  = sub["rul_estimate"] - sub["rul_true"]
        vals = []
        for lo, hi, lbl in zip(bin_edges[:-1], bin_edges[1:], bins):
            mask = (sub["rul_true"] >= lo) & (sub["rul_true"] < hi)
            vals.append(err[mask].mean() if mask.sum() > 0 else np.nan)
        ax.bar(x + offsets[i], vals, w, label=cfg, color=cmap(i), alpha=0.85)

    ax.axhline(0, color="black", lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"RUL {b}" for b in bins])
    ax.set_ylabel("Mean signed error: estimated − true (cycles)")
    ax.set_title(f"{model.upper()} {ds} — RUL prediction bias by true-RUL range\n"
                 "Positive = overestimate (predicts more life than actual)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return _save_plot(fig, f"signed_error_{model}_{ds}.png")


def plot_calibration_curves(
    fit_scores_dict: dict[str, pd.DataFrame],
    preds_dict:      dict[str, pd.DataFrame],
    model:           str,
    ds:              str,
) -> pathlib.Path:
    """Score → predicted RUL curves overlaid, per experiment."""
    configs = list(preds_dict.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap    = matplotlib.colormaps["tab10"].resampled(len(configs))

    for i, cfg in enumerate(preds_dict):
        df  = preds_dict[cfg]
        sub = df[df["dataset"] == ds] if ds != "overall" else df
        sc_col = f"{model}_score"
        if sc_col not in sub.columns and "anomaly_score" in sub.columns:
            sc_col = "anomaly_score"
        # Sort by score and plot smooth curve
        sc  = sub[sc_col].values if sc_col in sub.columns else \
              sub["ocsvm_score"].values if model == "ocsvm" else sub["iforest_score"].values
        rul_est = sub["rul_estimate"].values
        sort_idx = np.argsort(sc)
        ax.scatter(sc[sort_idx], rul_est[sort_idx],
                   s=2, alpha=0.15, color=cmap(i), rasterized=True)
        # Rolling mean for the trend line
        tmp = pd.DataFrame({"sc": sc, "rul": rul_est}).sort_values("sc")
        rolled = tmp["rul"].rolling(100, center=True, min_periods=10).mean()
        ax.plot(tmp["sc"], rolled, lw=2.0, color=cmap(i), label=cfg)

    ax.set_xlabel(f"{model.upper()} anomaly score")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(f"{model.upper()} {ds} — calibration curves\n"
                 "Lower score → higher predicted RUL (monotone constraint)")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return _save_plot(fig, f"calibration_curves_{model}_{ds}.png")


def plot_risk_progression(
    preds_dict: dict[str, pd.DataFrame],
    model:      str,
    ds:         str,
    n_engines:  int = 4,
) -> pathlib.Path:
    """Risk score vs RUL for sample engines across experiments."""
    # Pick sample engines from the first config
    first_df = list(preds_dict.values())[0]
    engs_ds  = sorted(first_df[first_df["dataset"] == ds]["engine_id"].unique())
    idx      = np.linspace(0, len(engs_ds) - 1, min(n_engines, len(engs_ds)), dtype=int)
    samp     = [engs_ds[i] for i in idx]

    n_cols = len(samp)
    n_rows = len(preds_dict)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows),
                             sharex=False, sharey=True)
    axes = np.array(axes).reshape(n_rows, n_cols)
    cmap = matplotlib.colormaps["tab10"].resampled(n_rows)

    for r, (cfg, df) in enumerate(preds_dict.items()):
        for c, eng in enumerate(samp):
            ax  = axes[r, c]
            grp = df[(df["engine_id"] == eng) & (df["dataset"] == ds)]
            grp = grp.sort_values("cycle")
            ax.plot(grp["rul_true"], grp["risk_score"],
                    lw=1.2, color=cmap(r), alpha=0.9)
            ax.invert_xaxis()
            ax.axhline(0.5, color="grey", lw=0.6, ls="--")
            ax.axvline(0, color="red", lw=0.8, ls=":", alpha=0.7)
            ax.set_ylim(-0.05, 1.05)
            if r == 0:
                ax.set_title(f"{eng}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{cfg}\nrisk score", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2)

    fig.suptitle(f"{model.upper()} {ds} — risk score vs RUL\n"
                 "x-axis inverted: right = failure. "
                 "Grey dashed = 0.5 threshold.")
    plt.tight_layout()
    return _save_plot(fig, f"risk_progression_{model}_{ds}.png")


def plot_near_failure_comparison(
    preds_dict: dict[str, pd.DataFrame],
    model:      str,
) -> pathlib.Path:
    """Distribution of prediction errors at RUL ≤ 20 across all experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap    = matplotlib.colormaps["tab10"].resampled(len(preds_dict))
    colors  = {"FD001": "steelblue", "FD003": "darkorange"}

    for ax, ds in zip(axes, ("FD001", "FD003")):
        for i, (cfg, df) in enumerate(preds_dict.items()):
            nf  = df[(df["dataset"] == ds) & (df["rul_true"] <= 20)]
            err = nf["rul_estimate"] - nf["rul_true"]
            if len(err) > 0:
                ax.hist(err, bins=25, alpha=0.4, density=True,
                        color=cmap(i), label=f"{cfg} (n={len(err)}, mean={err.mean():+.1f})")
        ax.axvline(0, color="black", lw=1.2, ls="--")
        ax.set_xlabel("Signed error: estimated − true (cycles)")
        ax.set_ylabel("Density")
        ax.set_title(f"{model.upper()} {ds} — near-failure errors (RUL ≤ 20)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"{model.upper()} — near-failure error distributions\n"
                 "Positive = overestimate; ideal peak near 0")
    plt.tight_layout()
    return _save_plot(fig, f"near_failure_errors_{model}.png")


# ---------------------------------------------------------------------------
# Master run function
# ---------------------------------------------------------------------------

def run() -> None:
    print("[calibration_experiments] Loading D_temporal bundles and pipeline ...")

    bundle_D_fd1 = load_bundle(EXP_MODELS / "anomaly_bundle_D_fd001.pkl")
    bundle_D_fd3 = load_bundle(EXP_MODELS / "anomaly_bundle_D_fd003.pkl")

    pipe   = build_pipeline(DATA_DIR, seed=SPLIT_SEED)
    df_raw = pipe["df_raw"]
    splits = pipe["splits"]

    # Rebuild augmented DataFrame (same as Phase 2 Exp D)
    df_aug, rm_cols = add_rolling_means(df_raw, ROLL_SENSORS, ROLL_WINDOW)

    # Dataset-specific engine ID lists
    cal_fd1 = [e for e in splits["calibration"] if e.startswith("FD001")]
    cal_fd3 = [e for e in splits["calibration"] if e.startswith("FD003")]
    fit_fd1 = bundle_D_fd1.splits["fit"]  # already filtered in Phase 2
    fit_fd3 = bundle_D_fd3.splits["fit"]
    # Recover actual fit-split FD001/FD003 subsets (bundle.splits["fit"] = full 120)
    fit_fd1_ids = [e for e in fit_fd1 if e.startswith("FD001")]
    fit_fd3_ids = [e for e in fit_fd3 if e.startswith("FD003")]

    # Verify no test leakage
    test_ids = set(splits["test"])
    assert test_ids.isdisjoint(set(cal_fd1 + cal_fd3))

    # ------------------------------------------------------------------ #
    # Generate D_temporal anomaly scores for cal + fit engines
    # ------------------------------------------------------------------ #
    print("[calibration_experiments] Scoring cal and fit engines ...")

    # Build scaled DataFrames using the same pipeline as Phase 2 Exp D
    ALL_18 = bundle_D_fd1.feature_cols[:18]  # first 18 are raw features
    feat_D = bundle_D_fd1.feature_cols        # all 23

    pipe_fd1 = build_exp_pipeline(df_aug, splits, feat_D, "FD001")
    pipe_fd3 = build_exp_pipeline(df_aug, splits, feat_D, "FD003")

    scores_cal_fd1 = score_engines_temporal(pipe_fd1["df_scaled"], cal_fd1, bundle_D_fd1)
    scores_cal_fd3 = score_engines_temporal(pipe_fd3["df_scaled"], cal_fd3, bundle_D_fd3)
    scores_fit_fd1 = score_engines_temporal(pipe_fd1["df_scaled"], fit_fd1_ids, bundle_D_fd1)
    scores_fit_fd3 = score_engines_temporal(pipe_fd3["df_scaled"], fit_fd3_ids, bundle_D_fd3)

    print(f"  Cal FD001: {len(scores_cal_fd1):,} rows from {len(cal_fd1)} engines")
    print(f"  Cal FD003: {len(scores_cal_fd3):,} rows from {len(cal_fd3)} engines")
    print(f"  Fit FD001: {len(scores_fit_fd1):,} rows from {len(fit_fd1_ids)} engines")
    print(f"  Fit FD003: {len(scores_fit_fd3):,} rows from {len(fit_fd3_ids)} engines")

    # ------------------------------------------------------------------ #
    # Baseline: Phase 1 shared calibration applied to D_temporal scores
    # ------------------------------------------------------------------ #
    print("[calibration_experiments] Evaluating Phase 1 baseline on D_temporal scores ...")
    from backend.risk.calibration import load_calibration_bundle as _load_cal
    baseline_oc = _load_cal(REPO_ROOT / "models" / "calibration_ocsvm.pkl")
    baseline_if = _load_cal(REPO_ROOT / "models" / "calibration_iforest.pkl")

    # Apply Phase 1 calibrators to D_temporal scores
    pred_base_oc_fd1 = apply_calibrator(baseline_oc, scores_cal_fd1, "ocsvm")
    pred_base_if_fd1 = apply_calibrator(baseline_if, scores_cal_fd1, "iforest")
    pred_base_oc_fd3 = apply_calibrator(baseline_oc, scores_cal_fd3, "ocsvm")
    pred_base_if_fd3 = apply_calibrator(baseline_if, scores_cal_fd3, "iforest")

    # ------------------------------------------------------------------ #
    # Exp A: Dataset-specific isotonic calibration
    # ------------------------------------------------------------------ #
    print("[calibration_experiments] Fitting Exp A calibrators ...")
    cal_A = {}
    for model, sc_fd1, sc_fd3 in [
        ("ocsvm",   scores_cal_fd1, scores_cal_fd3),
        ("iforest", scores_cal_fd1, scores_cal_fd3),
    ]:
        for ds_label, sc_cal, sc_fit, fit_ids, cal_ids in [
            ("fd001", sc_fd1, scores_fit_fd1, fit_fd1_ids, cal_fd1),
            ("fd003", sc_fd3, scores_fit_fd3, fit_fd3_ids, cal_fd3),
        ]:
            cb = fit_cal_A(model, sc_cal, fit_ids, cal_ids, df_aug)
            save_path = CAL_EXP_DIR / f"cal_A_{ds_label}_{model}.pkl"
            CAL_EXP_DIR.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as fh:
                pickle.dump(cb, fh)
            cal_A[f"{ds_label}_{model}"] = cb

    pred_A_oc_fd1 = apply_calibrator(cal_A["fd001_ocsvm"],   scores_cal_fd1, "ocsvm")
    pred_A_if_fd1 = apply_calibrator(cal_A["fd001_iforest"], scores_cal_fd1, "iforest")
    pred_A_oc_fd3 = apply_calibrator(cal_A["fd003_ocsvm"],   scores_cal_fd3, "ocsvm")
    pred_A_if_fd3 = apply_calibrator(cal_A["fd003_iforest"], scores_cal_fd3, "iforest")

    # ------------------------------------------------------------------ #
    # Exp B: Dataset-specific + endpoint anchoring
    # ------------------------------------------------------------------ #
    print("[calibration_experiments] Fitting Exp B calibrators (with anchoring) ...")
    cal_B = {}
    for model, sc_fd1, sc_fd3, sf_fd1, sf_fd3 in [
        ("ocsvm",   scores_cal_fd1, scores_cal_fd3, scores_fit_fd1, scores_fit_fd3),
        ("iforest", scores_cal_fd1, scores_cal_fd3, scores_fit_fd1, scores_fit_fd3),
    ]:
        for ds_label, sc_cal, sc_fit, fit_ids, cal_ids in [
            ("fd001", sc_fd1, sf_fd1, fit_fd1_ids, cal_fd1),
            ("fd003", sc_fd3, sf_fd3, fit_fd3_ids, cal_fd3),
        ]:
            cb = fit_cal_B(model, sc_cal, sc_fit, fit_ids, cal_ids, df_aug)
            anchor_val = float(np.percentile(sc_fit[f"{model}_score"].values, 99.0))
            print(f"  {model} {ds_label} anchor score (fit p99): {anchor_val:.4f}")
            save_path = CAL_EXP_DIR / f"cal_B_{ds_label}_{model}.pkl"
            with open(save_path, "wb") as fh:
                pickle.dump(cb, fh)
            cal_B[f"{ds_label}_{model}"] = cb

    pred_B_oc_fd1 = apply_calibrator(cal_B["fd001_ocsvm"],   scores_cal_fd1, "ocsvm")
    pred_B_if_fd1 = apply_calibrator(cal_B["fd001_iforest"], scores_cal_fd1, "iforest")
    pred_B_oc_fd3 = apply_calibrator(cal_B["fd003_ocsvm"],   scores_cal_fd3, "ocsvm")
    pred_B_if_fd3 = apply_calibrator(cal_B["fd003_iforest"], scores_cal_fd3, "iforest")

    # ------------------------------------------------------------------ #
    # Exp C: Piecewise isotonic calibration
    # ------------------------------------------------------------------ #
    print("[calibration_experiments] Fitting Exp C calibrators (piecewise) ...")
    cal_C = {}
    piecewise_enabled = {}
    for model in ("ocsvm", "iforest"):
        for ds_label, sc_cal, sc_fit, fit_ids, cal_ids in [
            ("fd001", scores_cal_fd1, scores_fit_fd1, fit_fd1_ids, cal_fd1),
            ("fd003", scores_cal_fd3, scores_fit_fd3, fit_fd3_ids, cal_fd3),
        ]:
            cb = fit_cal_C(model, sc_cal, sc_fit, fit_ids, cal_ids, df_aug)
            key = f"{ds_label}_{model}"
            if cb is not None:
                piecewise_enabled[key] = True
                save_path = CAL_EXP_DIR / f"cal_C_{ds_label}_{model}.pkl"
                with open(save_path, "wb") as fh:
                    pickle.dump(cb, fh)
                cal_C[key] = cb
                print(f"  {model} {ds_label}: piecewise boundary "
                      f"score={cb.boundary_score:.4f}")
            else:
                piecewise_enabled[key] = False
                # Fall back to Exp A
                cal_C[key] = cal_A[key]
                print(f"  {model} {ds_label}: piecewise SKIPPED (insufficient data); "
                      f"using Exp A calibrator")

    pred_C_oc_fd1 = apply_calibrator(cal_C["fd001_ocsvm"],   scores_cal_fd1, "ocsvm")
    pred_C_if_fd1 = apply_calibrator(cal_C["fd001_iforest"], scores_cal_fd1, "iforest")
    pred_C_oc_fd3 = apply_calibrator(cal_C["fd003_ocsvm"],   scores_cal_fd3, "ocsvm")
    pred_C_if_fd3 = apply_calibrator(cal_C["fd003_iforest"], scores_cal_fd3, "iforest")

    # ------------------------------------------------------------------ #
    # Exp D: Quantile-based calibration
    # ------------------------------------------------------------------ #
    print("[calibration_experiments] Fitting Exp D calibrators (quantile, q=0.25) ...")
    cal_D = {}
    for model in ("ocsvm", "iforest"):
        for ds_label, sc_cal, sc_fit, fit_ids, cal_ids in [
            ("fd001", scores_cal_fd1, scores_fit_fd1, fit_fd1_ids, cal_fd1),
            ("fd003", scores_cal_fd3, scores_fit_fd3, fit_fd3_ids, cal_fd3),
        ]:
            cb = fit_cal_D(model, sc_cal, fit_ids, cal_ids, df_aug)
            key = f"{ds_label}_{model}"
            cal_D[key] = cb
            save_path = CAL_EXP_DIR / f"cal_D_{ds_label}_{model}.pkl"
            with open(save_path, "wb") as fh:
                pickle.dump(cb, fh)

    pred_D_oc_fd1 = apply_calibrator(cal_D["fd001_ocsvm"],   scores_cal_fd1, "ocsvm")
    pred_D_if_fd1 = apply_calibrator(cal_D["fd001_iforest"], scores_cal_fd1, "iforest")
    pred_D_oc_fd3 = apply_calibrator(cal_D["fd003_ocsvm"],   scores_cal_fd3, "ocsvm")
    pred_D_if_fd3 = apply_calibrator(cal_D["fd003_iforest"], scores_cal_fd3, "iforest")

    # ------------------------------------------------------------------ #
    # Collect all predictions and compute metrics
    # ------------------------------------------------------------------ #
    print("[calibration_experiments] Computing metrics ...")

    # For each model: dict of {config_label: combined_fd1+fd3 predictions}
    def comb(a, b):
        return pd.concat([a, b], ignore_index=True)

    for model, pred_sets in [
        ("ocsvm", {
            "Baseline_P1":   comb(pred_base_oc_fd1, pred_base_oc_fd3),
            "A_ds_iso":      comb(pred_A_oc_fd1,    pred_A_oc_fd3),
            "B_anchored":    comb(pred_B_oc_fd1,    pred_B_oc_fd3),
            "C_piecewise":   comb(pred_C_oc_fd1,    pred_C_oc_fd3),
            "D_quantile_25": comb(pred_D_oc_fd1,    pred_D_oc_fd3),
        }),
        ("iforest", {
            "Baseline_P1":   comb(pred_base_if_fd1, pred_base_if_fd3),
            "A_ds_iso":      comb(pred_A_if_fd1,    pred_A_if_fd3),
            "B_anchored":    comb(pred_B_if_fd1,    pred_B_if_fd3),
            "C_piecewise":   comb(pred_C_if_fd1,    pred_C_if_fd3),
            "D_quantile_25": comb(pred_D_if_fd1,    pred_D_if_fd3),
        }),
    ]:
        print(f"\n{'='*70}")
        print(f"MODEL: {model.upper()}")
        print(f"{'='*70}")

        all_binned  = {}
        all_metrics = {}
        all_summaries = []

        for cfg, pred_df in pred_sets.items():
            # Save predictions
            _save_csv(pred_df, f"predictions_{cfg}_{model}.csv")
            # Binned errors
            for ds in ("FD001", "FD003"):
                sub = pred_df[pred_df["dataset"] == ds]
                be  = binned_rul_errors(sub)
                be["dataset"] = ds
                be["config"]  = cfg
                be["model"]   = model
                _save_csv(be, f"rul_errors_by_bin_{cfg}_{ds}_{model}.csv")
                key = f"{ds}_{cfg}"
                all_binned[key] = be
            # Per-engine metrics
            m = per_engine_metrics(pred_df)
            m["config"] = cfg
            m["model"]  = model
            _save_csv(m, f"per_engine_metrics_{cfg}_{model}.csv")
            all_metrics[cfg] = m
            all_summaries.append(aggregate_metrics(m, f"{model}_{cfg}"))

            # Print summary
            for ds in ("FD001", "FD003", "overall"):
                scope_m = m if ds == "overall" else m[m["dataset"] == ds]
                nf_rows = pred_df[pred_df["dataset"] == ds] if ds != "overall" else pred_df
                nf_rows = nf_rows[nf_rows["rul_true"] <= 20]
                nf_mae  = np.abs(nf_rows["rul_estimate"] - nf_rows["rul_true"]).mean() \
                          if len(nf_rows) else np.nan
                overall_mae = scope_m["rul_mae"].mean()
                rho = scope_m["risk_spearman"].median()
                print(f"  {cfg:15s} {ds:8s}  MAE={overall_mae:.1f}  "
                      f"NF-MAE={nf_mae:.1f}  risk_rho={rho:.3f}")

        # ---------------------------------------------------------------- #
        # Save per-model summary
        # ---------------------------------------------------------------- #
        summary_df = pd.DataFrame(all_summaries)
        _save_csv(summary_df, f"summary_{model}.csv")

        # ---------------------------------------------------------------- #
        # Plots for this model
        # ---------------------------------------------------------------- #
        print(f"[calibration_experiments] Generating plots for {model.upper()} ...")

        # Predicted vs actual
        for ds in ("FD001", "FD003"):
            plot_predicted_vs_actual(pred_sets, model, ds)

        # RUL error bins
        for ds in ("FD001", "FD003"):
            bins_by_cfg = {
                cfg: binned_rul_errors(
                    pred_df[pred_df["dataset"] == ds]
                )
                for cfg, pred_df in pred_sets.items()
            }
            plot_rul_error_bins(bins_by_cfg, model, ds)
            plot_signed_error_comparison(pred_sets, model, ds)

        # Calibration curves
        for ds in ("FD001", "FD003"):
            plot_calibration_curves({cfg: df for cfg, df in pred_sets.items()},
                                    {}, model, ds)

        # Risk progression
        for ds in ("FD001", "FD003"):
            plot_risk_progression(pred_sets, model, ds)

        # Near-failure error distribution
        plot_near_failure_comparison(pred_sets, model)

    # ------------------------------------------------------------------ #
    # Combined summary across models
    # ------------------------------------------------------------------ #
    all_csvs = list(RESULTS_DIR.glob("summary_*.csv"))
    if all_csvs:
        combined = pd.concat([pd.read_csv(p) for p in all_csvs], ignore_index=True)
        _save_csv(combined, "summary_all_experiments.csv")

    # ------------------------------------------------------------------ #
    # Write report
    # ------------------------------------------------------------------ #
    print("[calibration_experiments] Writing phase3_calibration_results.md ...")
    _write_report(piecewise_enabled)
    print("[calibration_experiments] Done. TEST SPLIT NOT ACCESSED.")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(piecewise_enabled: dict) -> pathlib.Path:
    """Read all generated CSVs and write the Phase 3 report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load all binned error CSVs
    rows = []
    for p in sorted(RESULTS_DIR.glob("rul_errors_by_bin_*.csv")):
        rows.append(pd.read_csv(p))
    if not rows:
        p = RESULTS_DIR / "phase3_calibration_results.md"
        p.write_text("# Phase 3 report — no data yet\n")
        return p
    err_df = pd.concat(rows, ignore_index=True)

    # Load per-engine metrics
    met_rows = []
    for p in sorted(RESULTS_DIR.glob("per_engine_metrics_*.csv")):
        met_rows.append(pd.read_csv(p))
    met_df = pd.concat(met_rows, ignore_index=True) if met_rows else pd.DataFrame()

    # Helper: get metric from err_df
    def get_mae(cfg, model, ds, rul_bin):
        sub = err_df[
            (err_df["config"] == cfg) &
            (err_df["model"] == model) &
            (err_df["dataset"] == ds) &
            (err_df["rul_bin"] == rul_bin)
        ]
        return f"{sub['mae'].values[0]:.1f}" if len(sub) > 0 else "—"

    def get_rho(cfg, model, ds):
        if met_df.empty:
            return "—"
        sub = met_df[
            (met_df["config"] == cfg) &
            (met_df["model"] == model) &
            (met_df["dataset"] == ds)
        ]
        if len(sub) == 0:
            return "—"
        return f"{sub['risk_spearman'].median():.3f}"

    configs = ["Baseline_P1", "A_ds_iso", "B_anchored", "C_piecewise", "D_quantile_25"]
    models  = ["ocsvm", "iforest"]

    lines = [
        "# Phase 3: Calibration Experiments",
        "**Branch:** feature/kartik-risk-layer",
        "**Anomaly detector:** D_temporal (separate FD001/FD003 models, 23 features)",
        "",
        "---",
        "",
        "## Configurations",
        "",
        "| Config | Description |",
        "|--------|-------------|",
        "| Baseline_P1 | Phase 1 shared isotonic calibrator applied to D_temporal scores |",
        "| A_ds_iso | Dataset-specific isotonic calibration (FD001 + FD003 separate) |",
        "| B_anchored | A + endpoint anchor at fit-split p99 score → RUL = 0 |",
        "| C_piecewise | Piecewise isotonic (bulk + critical zones) |",
        "| D_quantile_25 | Conservative quantile-based (25th-pct RUL, NOT expected RUL) |",
        "",
        "**Evaluation:** 20 FD001 + 20 FD003 calibration engines. Test split not accessed.",
        "**Risk formula:** `risk = 1 - clamp(rul_estimate / max_lifetime, 0, 1)`",
        "**Risk score is a degradation-risk indicator, NOT a failure probability.**",
        "",
    ]

    # Near-failure MAE table
    lines += [
        "---",
        "",
        "## Near-Failure RUL MAE (RUL ≤ 20 cycles)",
        "",
        "**This is the primary metric.** Phase 1 baseline: OC-SVM 16 cycles, IForest 52 cycles.",
        "",
        "### OC-SVM",
        "",
        "| Config | FD001 NF-MAE | FD003 NF-MAE |",
        "|--------|-------------|-------------|",
    ]
    for cfg in configs:
        lines.append(
            f"| {cfg} | {get_mae(cfg, 'ocsvm', 'FD001', '0–20')} "
            f"| {get_mae(cfg, 'ocsvm', 'FD003', '0–20')} |"
        )

    lines += [
        "",
        "### Isolation Forest",
        "",
        "| Config | FD001 NF-MAE | FD003 NF-MAE |",
        "|--------|-------------|-------------|",
    ]
    for cfg in configs:
        lines.append(
            f"| {cfg} | {get_mae(cfg, 'iforest', 'FD001', '0–20')} "
            f"| {get_mae(cfg, 'iforest', 'FD003', '0–20')} |"
        )

    # Full binned error tables
    lines += [
        "",
        "---",
        "",
        "## RUL MAE by True-RUL Range",
        "",
        "### OC-SVM — FD001",
        "",
        "| Config | RUL 0–20 | RUL 21–50 | RUL 51–100 | RUL >100 |",
        "|--------|----------|-----------|-----------|---------|",
    ]
    for cfg in configs:
        lines.append(
            f"| {cfg} "
            f"| {get_mae(cfg,'ocsvm','FD001','0–20')} "
            f"| {get_mae(cfg,'ocsvm','FD001','21–50')} "
            f"| {get_mae(cfg,'ocsvm','FD001','51–100')} "
            f"| {get_mae(cfg,'ocsvm','FD001','>100')} |"
        )

    lines += [
        "",
        "### OC-SVM — FD003",
        "",
        "| Config | RUL 0–20 | RUL 21–50 | RUL 51–100 | RUL >100 |",
        "|--------|----------|-----------|-----------|---------|",
    ]
    for cfg in configs:
        lines.append(
            f"| {cfg} "
            f"| {get_mae(cfg,'ocsvm','FD003','0–20')} "
            f"| {get_mae(cfg,'ocsvm','FD003','21–50')} "
            f"| {get_mae(cfg,'ocsvm','FD003','51–100')} "
            f"| {get_mae(cfg,'ocsvm','FD003','>100')} |"
        )

    lines += [
        "",
        "### IForest — FD001",
        "",
        "| Config | RUL 0–20 | RUL 21–50 | RUL 51–100 | RUL >100 |",
        "|--------|----------|-----------|-----------|---------|",
    ]
    for cfg in configs:
        lines.append(
            f"| {cfg} "
            f"| {get_mae(cfg,'iforest','FD001','0–20')} "
            f"| {get_mae(cfg,'iforest','FD001','21–50')} "
            f"| {get_mae(cfg,'iforest','FD001','51–100')} "
            f"| {get_mae(cfg,'iforest','FD001','>100')} |"
        )

    lines += [
        "",
        "### IForest — FD003",
        "",
        "| Config | RUL 0–20 | RUL 21–50 | RUL 51–100 | RUL >100 |",
        "|--------|----------|-----------|-----------|---------|",
    ]
    for cfg in configs:
        lines.append(
            f"| {cfg} "
            f"| {get_mae(cfg,'iforest','FD003','0–20')} "
            f"| {get_mae(cfg,'iforest','FD003','21–50')} "
            f"| {get_mae(cfg,'iforest','FD003','51–100')} "
            f"| {get_mae(cfg,'iforest','FD003','>100')} |"
        )

    # Risk metrics
    lines += [
        "",
        "---",
        "",
        "## Risk Spearman ρ (median per-engine)",
        "",
        "| Config | OC-SVM FD001 | OC-SVM FD003 | IForest FD001 | IForest FD003 |",
        "|--------|-------------|-------------|--------------|--------------|",
    ]
    for cfg in configs:
        lines.append(
            f"| {cfg} "
            f"| {get_rho(cfg,'ocsvm','FD001')} "
            f"| {get_rho(cfg,'ocsvm','FD003')} "
            f"| {get_rho(cfg,'iforest','FD001')} "
            f"| {get_rho(cfg,'iforest','FD003')} |"
        )

    # Piecewise status
    lines += [
        "",
        "---",
        "",
        "## Piecewise Calibration Status",
        "",
    ]
    for key, enabled in piecewise_enabled.items():
        status = "enabled" if enabled else "skipped (insufficient data → fell back to Exp A)"
        lines.append(f"- `{key}`: {status}")

    # Answers
    lines += [
        "",
        "---",
        "",
        "## Answers to Key Questions",
        "",
        "*(See binned error tables and summary CSVs for the numbers backing each answer.)*",
        "",
        "**Does D_temporal improve near-failure RUL estimation?**",
        "Compare Baseline_P1 vs A_ds_iso in the near-failure MAE table.",
        "Baseline_P1 uses the Phase 1 shared calibrator on D_temporal scores.",
        "A_ds_iso uses dataset-specific calibration on the same scores.",
        "",
        "**Does separate FD001/FD003 calibration help?**",
        "Compare Baseline_P1 (shared) vs A_ds_iso (dataset-specific) on the same D_temporal scores.",
        "",
        "**Does endpoint anchoring improve the calibration floor?**",
        "Compare A_ds_iso vs B_anchored, particularly at RUL 0–20.",
        "",
        "**Is piecewise calibration justified?**",
        "Compare C_piecewise vs B_anchored.",
        "If C falls back to A, piecewise was not supported by the data.",
        "",
        "**Which algorithm is best for FD001/FD003?**",
        "The algorithm with lower NF-MAE and higher risk Spearman ρ on the respective dataset.",
        "",
        "**Which calibration configuration should become production?**",
        "The configuration with the lowest NF-MAE, acceptable risk ρ, and no increased FA rate.",
        "",
        "---",
        "",
        "## Files",
        "",
        "**Models:**",
        "- `models/calibration_experiments/cal_{A,B,C,D}_{fd001,fd003}_{ocsvm,iforest}.pkl`",
        "",
        "**Results:**",
        "- `predictions_{config}_{model}.csv` — per-cycle RUL and risk estimates",
        "- `rul_errors_by_bin_{config}_{ds}_{model}.csv` — binned MAE, RMSE, bias",
        "- `per_engine_metrics_{config}_{model}.csv` — per-engine MAE, Spearman ρ, etc.",
        "- `summary_{model}.csv` — aggregate summaries per model",
        "- `summary_all_experiments.csv` — combined across models",
        "",
        "**Plots:**",
        "- `predicted_vs_actual_{model}_{ds}.png`",
        "- `rul_error_bins_{model}_{ds}.png`",
        "- `signed_error_{model}_{ds}.png`",
        "- `calibration_curves_{model}_{ds}.png`",
        "- `risk_progression_{model}_{ds}.png`",
        "- `near_failure_errors_{model}.png`",
        "",
        "*Baseline artifacts unchanged. Test split not accessed.*",
    ]

    p = RESULTS_DIR / "phase3_calibration_results.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


if __name__ == "__main__":
    run()
