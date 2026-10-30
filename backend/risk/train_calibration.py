"""
Train calibration pipelines and evaluate on calibration engines.

Usage (from repo root, with .venv active):
    python -m backend.risk.train_calibration

What this does
--------------
1. Loads the existing anomaly bundle (Layer 1 — no retraining).
2. Rebuilds the scaled DataFrame via build_pipeline (same seed → same split).
3. Fits CalibrationBundle for OC-SVM and Isolation Forest using ONLY the
   40 calibration engines.
4. Evaluates calibrated outputs in-calibration (for diagnostic purposes).
5. Saves calibration artifacts to models/.
6. Generates plots and CSV results.

Stopping condition
------------------
Test-split engines are NEVER accessed.  This script stops after calibration
evaluation on the 40 calibration engines.

Outputs (all git-ignored):
    models/calibration_ocsvm.pkl
    models/calibration_iforest.pkl
    results/cal_scores_ocsvm.csv
    results/cal_scores_iforest.csv
    results/cal_metrics_ocsvm_{overall,FD001,FD003}.csv
    results/cal_metrics_iforest_{overall,FD001,FD003}.csv
    results/cal_comparison_summary.csv
    plots/calibration/*.png
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from backend.risk.anomaly_models import load_bundle
from backend.risk.calibration import (
    CalibrationBundle,
    evaluate_calibration,
    fit_both_calibrations,
    compute_calibration_metrics,
    summarise_metrics,
    save_calibration_bundle,
    load_calibration_bundle,
)
from backend.risk.cmapss_pipeline import SPLIT_SEED, build_pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT        = pathlib.Path(__file__).parent.parent.parent
DATA_DIR         = REPO_ROOT / "data" / "raw" / "cmapss"
BUNDLE_PATH      = REPO_ROOT / "models" / "anomaly_bundle.pkl"
CAL_OC_PATH      = REPO_ROOT / "models" / "calibration_ocsvm.pkl"
CAL_IF_PATH      = REPO_ROOT / "models" / "calibration_iforest.pkl"
RESULTS_DIR      = REPO_ROOT / "results"
PLOTS_DIR        = REPO_ROOT / "plots" / "calibration"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_plot(fig: plt.Figure, name: str) -> pathlib.Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    p = PLOTS_DIR / name
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def _save_csv(df: pd.DataFrame, name: str) -> pathlib.Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / name
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_score_vs_rul(
    eval_ocsvm: pd.DataFrame,
    eval_iforest: pd.DataFrame,
    n_engines: int = 6,
) -> list[pathlib.Path]:
    """Raw anomaly score and calibrated risk score vs true RUL."""
    saved = []
    for ds in ("FD001", "FD003"):
        engs = sorted(eval_ocsvm[eval_ocsvm["dataset"] == ds]["engine_id"].unique())
        idx  = np.linspace(0, len(engs) - 1, min(n_engines, len(engs)), dtype=int)
        sample = [engs[i] for i in idx]

        fig, axes = plt.subplots(
            len(sample), 4,
            figsize=(20, 3.5 * len(sample)),
            sharex=False,
        )
        if len(sample) == 1:
            axes = axes[np.newaxis, :]

        for row, eng in enumerate(sample):
            for col, (ev, label, score_col, color) in enumerate([
                (eval_ocsvm,   "OC-SVM",  "anomaly_score", "steelblue"),
                (eval_ocsvm,   "OC-SVM",  "risk_score",    "steelblue"),
                (eval_iforest, "IForest", "anomaly_score", "darkorange"),
                (eval_iforest, "IForest", "risk_score",    "darkorange"),
            ]):
                ax  = axes[row, col]
                grp = ev[ev["engine_id"] == eng].sort_values("cycle")
                rul = grp["rul_true"].values
                sc  = grp[score_col].values

                ax.plot(rul, sc, lw=1.0, color=color, alpha=0.85)
                ax.invert_xaxis()
                ax.axvline(0, color="red", lw=1.0, ls=":", alpha=0.7)

                if score_col == "anomaly_score":
                    ax.axhline(0, color="grey", lw=0.6, ls="--")
                    ylabel = f"{label}\nanomaly score\n(↑ anomalous)"
                else:
                    ax.set_ylim(-0.05, 1.05)
                    ax.axhline(0.5, color="grey", lw=0.6, ls="--")
                    ylabel = f"{label}\nrisk score\n[0=healthy, 1=failure]"

                ax.set_ylabel(ylabel, fontsize=7)
                ax.set_xlabel("True RUL", fontsize=7)
                ax.set_title(f"{eng}", fontsize=7)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.25)

        fig.suptitle(
            f"{ds} – raw anomaly score and calibrated risk score vs true RUL\n"
            "(x-axis inverted: right = failure, red dashed = RUL 0)",
            fontsize=10,
        )
        plt.tight_layout()
        saved.append(_save_plot(fig, f"{ds}_score_vs_rul.png"))
    return saved


def plot_rul_estimate_vs_true(
    eval_ocsvm: pd.DataFrame,
    eval_iforest: pd.DataFrame,
) -> pathlib.Path:
    """Scatter: estimated RUL vs actual RUL for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"FD001": "steelblue", "FD003": "darkorange"}

    for ax, ev, title in [
        (axes[0], eval_ocsvm,   "OC-SVM"),
        (axes[1], eval_iforest, "Isolation Forest"),
    ]:
        for ds in ("FD001", "FD003"):
            sub = ev[ev["dataset"] == ds]
            ax.scatter(
                sub["rul_true"], sub["rul_estimate"],
                s=4, alpha=0.25, color=colors[ds], label=ds, rasterized=True,
            )
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5, label="perfect")
        ax.set_xlabel("True RUL (cycles)")
        ax.set_ylabel("Estimated RUL (cycles)")
        ax.set_title(f"{title}: estimated vs true RUL")
        ax.legend(fontsize=8, markerscale=3)
        ax.grid(True, alpha=0.25)

    fig.suptitle("RUL estimation on calibration engines\n"
                 "(blue = FD001, orange = FD003)")
    plt.tight_layout()
    return _save_plot(fig, "rul_estimate_vs_true.png")


def plot_risk_score_vs_rul_aggregate(
    eval_ocsvm: pd.DataFrame,
    eval_iforest: pd.DataFrame,
) -> pathlib.Path:
    """Mean risk score by RUL bucket for both models and datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"FD001": "steelblue", "FD003": "darkorange"}

    bins   = list(range(0, 550, 10))
    labels = [b + 5 for b in bins[:-1]]

    for ax, ev, title in [
        (axes[0], eval_ocsvm,   "OC-SVM"),
        (axes[1], eval_iforest, "Isolation Forest"),
    ]:
        for ds in ("FD001", "FD003"):
            sub = ev[ev["dataset"] == ds].copy()
            sub["rul_bucket"] = pd.cut(sub["rul_true"], bins=bins, labels=labels)
            agg = sub.groupby("rul_bucket", observed=True)["risk_score"].mean()
            ax.plot(
                agg.index.astype(float), agg.values,
                color=colors[ds], lw=1.5, label=ds,
            )
        ax.invert_xaxis()
        ax.set_xlabel("True RUL (cycles, right = failure)")
        ax.set_ylabel("Mean calibrated risk score [0, 1]")
        ax.set_title(f"{title}: mean risk vs true RUL (10-cycle buckets)")
        ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.6)
        ax.axvline(0, color="red", lw=1.0, ls=":", alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Calibrated risk score vs RUL (calibration engines)\n"
                 "Higher risk = more degraded; right side = near failure")
    plt.tight_layout()
    return _save_plot(fig, "risk_score_vs_rul_aggregate.png")


def plot_calibration_curves(
    eval_ocsvm: pd.DataFrame,
    eval_iforest: pd.DataFrame,
) -> pathlib.Path:
    """Isotonic calibration curve: anomaly score → estimated RUL."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"FD001": "steelblue", "FD003": "darkorange"}

    for ax, ev, title in [
        (axes[0], eval_ocsvm,   "OC-SVM"),
        (axes[1], eval_iforest, "Isolation Forest"),
    ]:
        for ds in ("FD001", "FD003"):
            sub = ev[ev["dataset"] == ds]
            ax.scatter(
                sub["anomaly_score"], sub["rul_estimate"],
                s=3, alpha=0.2, color=colors[ds], label=ds, rasterized=True,
            )
        ax.set_xlabel("Raw anomaly score (higher = more anomalous)")
        ax.set_ylabel("Isotonic estimated RUL (cycles)")
        ax.set_title(f"{title}: calibration curve")
        ax.legend(fontsize=8, markerscale=4)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Isotonic calibration curve: anomaly score → estimated RUL\n"
                 "(monotone decreasing by construction)")
    plt.tight_layout()
    return _save_plot(fig, "calibration_curves.png")


def plot_model_comparison(
    metrics_oc: pd.DataFrame,
    metrics_if: pd.DataFrame,
) -> pathlib.Path:
    """Side-by-side comparison of per-engine metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"FD001": "steelblue", "FD003": "darkorange"}

    for ax, m_oc, m_if, metric, xlabel in [
        (axes[0],
         metrics_oc["rul_mae"], metrics_if["rul_mae"],
         "rul_mae", "RUL MAE (cycles)"),
        (axes[1],
         metrics_oc["risk_spearman"], metrics_if["risk_spearman"],
         "risk_spearman", "Risk score Spearman ρ"),
        (axes[2],
         metrics_oc["monotone_violations"], metrics_if["monotone_violations"],
         "monotone_violations", "Monotone violation rate"),
    ]:
        ax.scatter(
            m_oc.values, m_if.values,
            s=18, alpha=0.6,
            c=[colors.get(d, "grey") for d in metrics_oc["dataset"]],
        )
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel(f"OC-SVM {xlabel}")
        ax.set_ylabel(f"IForest {xlabel}")
        ax.set_title(f"Per-engine {metric}")
        ax.grid(True, alpha=0.25)

    # colour legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[ds], label=ds) for ds in ("FD001", "FD003")]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle("OC-SVM vs IForest calibration: per-engine metrics\n"
                 "(blue = FD001, orange = FD003; points above y=x → IForest better)")
    plt.tight_layout()
    return _save_plot(fig, "model_comparison.png")


def plot_risk_distribution(
    eval_ocsvm: pd.DataFrame,
    eval_iforest: pd.DataFrame,
) -> pathlib.Path:
    """Violin plots: risk score distribution at healthy vs late-life."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    HEALTHY_N   = 30
    LATE_LIFE_N = 30

    for ax, ev, title in [
        (axes[0], eval_ocsvm,   "OC-SVM"),
        (axes[1], eval_iforest, "Isolation Forest"),
    ]:
        hw = ev.loc[ev["cycle"] <= HEALTHY_N, "risk_score"].values
        ll = ev.loc[ev["rul_true"] < LATE_LIFE_N, "risk_score"].values
        parts = ax.violinplot([hw, ll], positions=[0, 1],
                              showmedians=True, showextrema=True)
        parts["bodies"][0].set_facecolor("mediumseagreen")
        parts["bodies"][1].set_facecolor("tomato")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Healthy\n(cycle ≤ 30)", "Late-life\n(RUL < 30)"])
        ax.set_ylabel("Calibrated risk score [0, 1]")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.set_title(f"{title}: risk score distribution")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Risk score distributions: healthy window vs late-life\n"
                 "(all calibration engines)")
    plt.tight_layout()
    return _save_plot(fig, "risk_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    print("[train_calibration] Loading anomaly bundle ...")
    anomaly_bundle = load_bundle(BUNDLE_PATH)

    print("[train_calibration] Rebuilding scaled data ...")
    pipe         = build_pipeline(DATA_DIR, seed=SPLIT_SEED)
    df_scaled    = pipe["df_scaled"]
    cal_ids      = anomaly_bundle.splits["calibration"]
    fit_ids      = anomaly_bundle.splits["fit"]
    test_ids     = anomaly_bundle.splits["test"]

    fd1_cal = [e for e in cal_ids if e.startswith("FD001")]
    fd3_cal = [e for e in cal_ids if e.startswith("FD003")]
    print(f"  Cal engines: {len(cal_ids)} "
          f"(FD001: {len(fd1_cal)}, FD003: {len(fd3_cal)})")
    print(f"  Test engines remain untouched: {len(test_ids)}")

    # Verify test engines are not touched
    assert set(cal_ids).isdisjoint(set(test_ids)), "LEAKAGE: cal/test overlap"
    assert set(cal_ids).isdisjoint(set(fit_ids)),  "LEAKAGE: cal/fit overlap"

    # ------------------------------------------------------------------
    # Fit calibration pipelines (cal engines only)
    # ------------------------------------------------------------------
    print("[train_calibration] Fitting calibration pipelines ...")
    cals = fit_both_calibrations(anomaly_bundle, df_scaled, cal_ids, fit_ids)
    cal_oc = cals["ocsvm"]
    cal_if = cals["iforest"]
    print(f"  OC-SVM  max_lifetime: {cal_oc.max_lifetime:.0f} cycles")
    print(f"  IForest max_lifetime: {cal_if.max_lifetime:.0f} cycles")

    # Save artifacts
    save_calibration_bundle(cal_oc, CAL_OC_PATH)
    save_calibration_bundle(cal_if, CAL_IF_PATH)
    print(f"  Saved: {CAL_OC_PATH}")
    print(f"  Saved: {CAL_IF_PATH}")

    # ------------------------------------------------------------------
    # Evaluate on calibration engines (in-calibration diagnostic)
    # ------------------------------------------------------------------
    print("[train_calibration] Evaluating on calibration engines ...")
    eval_oc = evaluate_calibration("ocsvm",   cal_oc, anomaly_bundle, df_scaled, cal_ids)
    eval_if = evaluate_calibration("iforest", cal_if, anomaly_bundle, df_scaled, cal_ids)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    metrics_oc = compute_calibration_metrics(eval_oc)
    metrics_if = compute_calibration_metrics(eval_if)

    def full_summary(metrics: pd.DataFrame, model_name: str) -> dict[str, pd.DataFrame]:
        summaries = {}
        for scope, mask in [
            ("overall", pd.Series(True, index=metrics.index)),
            ("FD001",   metrics["dataset"] == "FD001"),
            ("FD003",   metrics["dataset"] == "FD003"),
        ]:
            summaries[scope] = summarise_metrics(
                metrics[mask], scope_label=f"{model_name}_{scope}"
            )
        return summaries

    summ_oc = full_summary(metrics_oc, "ocsvm")
    summ_if = full_summary(metrics_if, "iforest")

    # Print summaries
    print("\n" + "=" * 70)
    print("OC-SVM CALIBRATION METRICS (overall, calibration engines)")
    print("=" * 70)
    print(summ_oc["overall"].to_string(index=False))

    print("\n" + "=" * 70)
    print("IFOREST CALIBRATION METRICS (overall, calibration engines)")
    print("=" * 70)
    print(summ_if["overall"].to_string(index=False))

    for ds in ("FD001", "FD003"):
        print(f"\n--- OC-SVM: {ds} ---")
        print(summ_oc[ds].to_string(index=False))
        print(f"\n--- IForest: {ds} ---")
        print(summ_if[ds].to_string(index=False))

    # Side-by-side comparison table
    oc_row = summ_oc["overall"].set_index("metric")
    if_row = summ_if["overall"].set_index("metric")
    cmp_rows = []
    for metric in oc_row.index:
        cmp_rows.append({
            "metric":       metric,
            "ocsvm_mean":   oc_row.loc[metric, "mean"],
            "ocsvm_median": oc_row.loc[metric, "median"],
            "iforest_mean": if_row.loc[metric, "mean"],
            "iforest_median": if_row.loc[metric, "median"],
        })
    cmp_df = pd.DataFrame(cmp_rows)
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON (calibration engines)")
    print("=" * 70)
    print(cmp_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------------
    print("\n[train_calibration] Saving CSV results ...")
    _save_csv(eval_oc,    "cal_scores_ocsvm.csv")
    _save_csv(eval_if,    "cal_scores_iforest.csv")
    _save_csv(metrics_oc, "cal_metrics_ocsvm.csv")
    _save_csv(metrics_if, "cal_metrics_iforest.csv")
    _save_csv(cmp_df,     "cal_comparison_summary.csv")
    for scope in ("overall", "FD001", "FD003"):
        _save_csv(summ_oc[scope], f"cal_summary_ocsvm_{scope}.csv")
        _save_csv(summ_if[scope], f"cal_summary_iforest_{scope}.csv")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("[train_calibration] Generating plots ...")
    traj_paths = plot_score_vs_rul(eval_oc, eval_if)
    rul_path   = plot_rul_estimate_vs_true(eval_oc, eval_if)
    agg_path   = plot_risk_score_vs_rul_aggregate(eval_oc, eval_if)
    curve_path = plot_calibration_curves(eval_oc, eval_if)
    cmp_path   = plot_model_comparison(metrics_oc, metrics_if)
    dist_path  = plot_risk_distribution(eval_oc, eval_if)

    print(f"  Score vs RUL : {[str(p) for p in traj_paths]}")
    print(f"  RUL scatter  : {rul_path}")
    print(f"  Risk agg     : {agg_path}")
    print(f"  Cal curves   : {curve_path}")
    print(f"  Model cmp    : {cmp_path}")
    print(f"  Risk dist    : {dist_path}")

    print("\n[train_calibration] Done. TEST ENGINES WERE NOT TOUCHED.")


if __name__ == "__main__":
    run()
