"""
RUL prediction error analysis — Layer 2 calibration investigation.

Uses only the already-generated calibration CSVs (cal_scores_ocsvm.csv,
cal_scores_iforest.csv) and the saved calibration artifacts.
Nothing is retrained, recalibrated or tuned. Test data is never touched.

Outputs
-------
results/rul_error_by_bin_{ocsvm,iforest}.csv
results/rul_error_by_dataset_{ocsvm,iforest}.csv
results/rul_signed_error_by_bin_{ocsvm,iforest}.csv
plots/rul_analysis/*.png
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT   = pathlib.Path(__file__).parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR   = REPO_ROOT / "plots" / "rul_analysis"

# RUL bins requested
BINS       = [0, 20, 50, 100, np.inf]
BIN_LABELS = ["0–20", "21–50", "51–100", ">100"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> pathlib.Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    p = PLOTS_DIR / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def _save_csv(df: pd.DataFrame, name: str) -> pathlib.Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / name
    df.to_csv(p, index=False)
    return p


def _assign_bin(rul_series: pd.Series) -> pd.Series:
    return pd.cut(rul_series, bins=BINS, labels=BIN_LABELS, right=True)


def binned_errors(df: pd.DataFrame) -> pd.DataFrame:
    """MAE, RMSE, mean signed error and n per RUL bin."""
    df = df.copy()
    df["error"]        = df["rul_estimate"] - df["rul_true"]   # signed: + = overestimate
    df["abs_error"]    = df["error"].abs()
    df["sq_error"]     = df["error"] ** 2
    df["rul_bin"]      = _assign_bin(df["rul_true"])
    rows = []
    for lbl in BIN_LABELS:
        sub = df[df["rul_bin"] == lbl]
        if len(sub) == 0:
            continue
        rows.append({
            "rul_bin":          lbl,
            "n":                len(sub),
            "mae":              sub["abs_error"].mean(),
            "rmse":             np.sqrt(sub["sq_error"].mean()),
            "mean_signed_error": sub["error"].mean(),
            "median_signed_error": sub["error"].median(),
            "pct_overestimate": (sub["error"] > 0).mean(),
        })
    return pd.DataFrame(rows)


def dataset_errors(df: pd.DataFrame) -> pd.DataFrame:
    """MAE and RMSE per dataset × RUL bin."""
    df = df.copy()
    df["error"]     = df["rul_estimate"] - df["rul_true"]
    df["abs_error"] = df["error"].abs()
    df["sq_error"]  = df["error"] ** 2
    df["rul_bin"]   = _assign_bin(df["rul_true"])
    rows = []
    for ds in ("FD001", "FD003"):
        sub_ds = df[df["dataset"] == ds]
        for lbl in BIN_LABELS:
            sub = sub_ds[sub_ds["rul_bin"] == lbl]
            if len(sub) == 0:
                continue
            rows.append({
                "dataset":           ds,
                "rul_bin":           lbl,
                "n":                 len(sub),
                "mae":               sub["abs_error"].mean(),
                "rmse":              np.sqrt(sub["sq_error"].mean()),
                "mean_signed_error": sub["error"].mean(),
                "pct_overestimate":  (sub["error"] > 0).mean(),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot 1: Predicted vs actual RUL scatter (both models, both datasets)
# ---------------------------------------------------------------------------

def plot_predicted_vs_actual(oc: pd.DataFrame, if_: pd.DataFrame) -> pathlib.Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    colors = {"FD001": "steelblue", "FD003": "darkorange"}
    model_data = [("OC-SVM", oc), ("Isolation Forest", if_)]

    for row, (model_name, df) in enumerate(model_data):
        for col, ds in enumerate(("FD001", "FD003")):
            ax  = axes[row, col]
            sub = df[df["dataset"] == ds]
            ax.scatter(
                sub["rul_true"], sub["rul_estimate"],
                s=3, alpha=0.18, color=colors[ds], rasterized=True,
            )
            lim = max(sub["rul_true"].max(), sub["rul_estimate"].max()) * 1.05
            ax.plot([0, lim], [0, lim], "k--", lw=1.0, label="perfect")
            ax.set_xlim(-5, lim)
            ax.set_ylim(-5, lim)
            ax.set_xlabel("True RUL (cycles)")
            ax.set_ylabel("Estimated RUL (cycles)")
            ax.set_title(f"{model_name} — {ds}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25)

            # annotate MAE
            mae  = np.abs(sub["rul_estimate"] - sub["rul_true"]).mean()
            rmse = np.sqrt(((sub["rul_estimate"] - sub["rul_true"]) ** 2).mean())
            ax.text(
                0.04, 0.96,
                f"MAE={mae:.1f}  RMSE={rmse:.1f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )

    fig.suptitle(
        "Estimated vs true RUL — calibration engines\n"
        "Points above the diagonal = overestimation; below = underestimation",
        fontsize=11,
    )
    plt.tight_layout()
    return _save(fig, "predicted_vs_actual_rul.png")


# ---------------------------------------------------------------------------
# Plot 2: Signed error vs true RUL (bias profile)
# ---------------------------------------------------------------------------

def plot_signed_error_vs_rul(oc: pd.DataFrame, if_: pd.DataFrame) -> pathlib.Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    colors = {"FD001": "steelblue", "FD003": "darkorange"}
    model_data = [("OC-SVM", oc), ("Isolation Forest", if_)]

    for row, (model_name, df) in enumerate(model_data):
        for col, ds in enumerate(("FD001", "FD003")):
            ax  = axes[row, col]
            sub = df[df["dataset"] == ds].copy()
            err = sub["rul_estimate"] - sub["rul_true"]
            ax.scatter(
                sub["rul_true"], err,
                s=3, alpha=0.18, color=colors[ds], rasterized=True,
            )
            # Smoothed mean with 20-cycle rolling window
            tmp = pd.DataFrame({"rul": sub["rul_true"].values, "err": err.values})
            tmp = tmp.sort_values("rul")
            tmp["smooth"] = tmp["err"].rolling(window=50, center=True,
                                               min_periods=10).mean()
            ax.plot(tmp["rul"], tmp["smooth"], color="black", lw=1.5,
                    label="rolling mean (50 obs)")
            ax.axhline(0, color="red", lw=1.2, ls="--", alpha=0.8, label="zero bias")
            ax.set_xlabel("True RUL (cycles)")
            ax.set_ylabel("Signed error: estimated − true (cycles)")
            ax.set_title(f"{model_name} — {ds}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25)

            # Shade near-failure zone
            ax.axvspan(0, 20, alpha=0.08, color="red", label="near-failure (RUL≤20)")

    fig.suptitle(
        "Signed prediction error vs true RUL — calibration engines\n"
        "Positive = overestimate (predicts more life remaining than actual)\n"
        "Negative = underestimate | red band = near-failure zone (RUL ≤ 20)",
        fontsize=10,
    )
    plt.tight_layout()
    return _save(fig, "signed_error_vs_rul.png")


# ---------------------------------------------------------------------------
# Plot 3: MAE and RMSE by RUL bin (grouped bar)
# ---------------------------------------------------------------------------

def plot_error_by_bin(bin_oc: pd.DataFrame, bin_if: pd.DataFrame) -> pathlib.Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x      = np.arange(len(BIN_LABELS))
    width  = 0.35

    for ax, metric, ylabel in [
        (axes[0], "mae",  "Mean Absolute Error (cycles)"),
        (axes[1], "rmse", "RMSE (cycles)"),
    ]:
        oc_vals = [bin_oc.loc[bin_oc["rul_bin"] == b, metric].values[0]
                   if b in bin_oc["rul_bin"].values else 0 for b in BIN_LABELS]
        if_vals = [bin_if.loc[bin_if["rul_bin"] == b, metric].values[0]
                   if b in bin_if["rul_bin"].values else 0 for b in BIN_LABELS]

        bars_oc = ax.bar(x - width / 2, oc_vals, width, label="OC-SVM",
                         color="steelblue", alpha=0.85)
        bars_if = ax.bar(x + width / 2, if_vals, width, label="IForest",
                         color="darkorange", alpha=0.85)
        ax.bar_label(bars_oc, fmt="%.1f", fontsize=8, padding=2)
        ax.bar_label(bars_if, fmt="%.1f", fontsize=8, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"RUL\n{b}" for b in BIN_LABELS])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric.upper()} by RUL bin")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "RUL prediction errors by true-RUL range (all calibration engines)\n"
        "Lower is better",
        fontsize=11,
    )
    plt.tight_layout()
    return _save(fig, "error_by_rul_bin.png")


# ---------------------------------------------------------------------------
# Plot 4: Mean signed error by bin (bias)
# ---------------------------------------------------------------------------

def plot_bias_by_bin(bin_oc: pd.DataFrame, bin_if: pd.DataFrame) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(BIN_LABELS))
    width = 0.35

    oc_vals = [bin_oc.loc[bin_oc["rul_bin"] == b, "mean_signed_error"].values[0]
               if b in bin_oc["rul_bin"].values else 0 for b in BIN_LABELS]
    if_vals = [bin_if.loc[bin_if["rul_bin"] == b, "mean_signed_error"].values[0]
               if b in bin_if["rul_bin"].values else 0 for b in BIN_LABELS]

    bars_oc = ax.bar(x - width / 2, oc_vals, width, label="OC-SVM",
                     color="steelblue", alpha=0.85)
    bars_if = ax.bar(x + width / 2, if_vals, width, label="IForest",
                     color="darkorange", alpha=0.85)
    ax.bar_label(bars_oc, fmt="%+.1f", fontsize=9, padding=2)
    ax.bar_label(bars_if, fmt="%+.1f", fontsize=9, padding=2)
    ax.axhline(0, color="black", lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"RUL {b}" for b in BIN_LABELS])
    ax.set_ylabel("Mean signed error: estimated − true (cycles)")
    ax.set_title(
        "RUL prediction bias by true-RUL range\n"
        "Positive = systematic overestimate, Negative = underestimate"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return _save(fig, "bias_by_rul_bin.png")


# ---------------------------------------------------------------------------
# Plot 5: FD001 vs FD003 error comparison
# ---------------------------------------------------------------------------

def plot_error_by_dataset(ds_oc: pd.DataFrame, ds_if: pd.DataFrame) -> pathlib.Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x      = np.arange(len(BIN_LABELS))
    width  = 0.2
    hatch  = {"FD001": "", "FD003": "///"}
    colors = {"OC-SVM": "steelblue", "IForest": "darkorange"}

    for ax, metric, ylabel in [
        (axes[0], "mae",  "MAE (cycles)"),
        (axes[1], "rmse", "RMSE (cycles)"),
    ]:
        for i, (model_name, ds_df) in enumerate([("OC-SVM", ds_oc), ("IForest", ds_if)]):
            for j, ds in enumerate(("FD001", "FD003")):
                sub = ds_df[ds_df["dataset"] == ds]
                vals = [sub.loc[sub["rul_bin"] == b, metric].values[0]
                        if b in sub["rul_bin"].values else 0 for b in BIN_LABELS]
                offset = (i * 2 + j) * width - 1.5 * width
                bars = ax.bar(x + offset, vals, width,
                              color=colors[model_name], alpha=0.75,
                              hatch=hatch[ds],
                              label=f"{model_name} {ds}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"RUL {b}" for b in BIN_LABELS])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric.upper()} by RUL bin: FD001 vs FD003")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "RUL prediction errors by dataset and RUL range\n"
        "(solid = FD001, hatched = FD003)",
        fontsize=11,
    )
    plt.tight_layout()
    return _save(fig, "error_by_dataset_and_bin.png")


# ---------------------------------------------------------------------------
# Plot 6: Near-failure zoom — error distribution RUL 0–20
# ---------------------------------------------------------------------------

def plot_near_failure_detail(oc: pd.DataFrame, if_: pd.DataFrame) -> pathlib.Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_ds = {"FD001": "steelblue", "FD003": "darkorange"}

    for ax, model_name, df in [
        (axes[0], "OC-SVM",          oc),
        (axes[1], "Isolation Forest", if_),
    ]:
        sub = df[df["rul_true"] <= 20].copy()
        err = sub["rul_estimate"] - sub["rul_true"]

        for ds in ("FD001", "FD003"):
            e = err[sub["dataset"] == ds]
            ax.hist(e, bins=30, alpha=0.55, color=colors_ds[ds],
                    label=f"{ds} (n={len(e)}, mean={e.mean():+.1f})",
                    density=True)

        ax.axvline(0, color="black", lw=1.2, ls="--", label="zero error")
        ax.axvline(err.mean(), color="red", lw=1.2, ls="-",
                   label=f"overall mean {err.mean():+.1f}")
        ax.set_xlabel("Signed error: estimated − true (cycles)")
        ax.set_ylabel("Density")
        ax.set_title(f"{model_name}: error distribution, RUL ≤ 20\n"
                     "(near-failure zone)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Near-failure (RUL ≤ 20) RUL prediction error distribution\n"
        "Positive = overestimate (engine thought to have more life than it does)",
        fontsize=10,
    )
    plt.tight_layout()
    return _save(fig, "near_failure_error_detail.png")


# ---------------------------------------------------------------------------
# Plot 7: Per-engine MAE vs engine lifetime (size effect)
# ---------------------------------------------------------------------------

def plot_mae_vs_lifetime(oc_scores: pd.DataFrame, if_scores: pd.DataFrame) -> pathlib.Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_ds = {"FD001": "steelblue", "FD003": "darkorange"}

    for ax, model_name, df in [
        (axes[0], "OC-SVM", oc_scores),
        (axes[1], "Isolation Forest", if_scores),
    ]:
        per_eng = df.groupby(["engine_id", "dataset"]).apply(
            lambda g: pd.Series({
                "lifetime": g["cycle"].max(),
                "mae": np.abs(g["rul_estimate"] - g["rul_true"]).mean(),
            })
        ).reset_index()

        for ds in ("FD001", "FD003"):
            sub = per_eng[per_eng["dataset"] == ds]
            ax.scatter(sub["lifetime"], sub["mae"],
                       s=35, alpha=0.7, color=colors_ds[ds], label=ds)

        # trend line
        rho, pval = sp_stats.spearmanr(per_eng["lifetime"], per_eng["mae"])
        ax.set_xlabel("Engine lifetime (cycles)")
        ax.set_ylabel("Per-engine MAE (cycles)")
        ax.set_title(f"{model_name}: MAE vs engine lifetime\n"
                     f"Spearman ρ={rho:.3f}, p={pval:.3f}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Does longer-lived engine → higher RUL prediction error?\n"
        "(FD003 engines tend to live longer)",
        fontsize=10,
    )
    plt.tight_layout()
    return _save(fig, "mae_vs_lifetime.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    print("[rul_error_analysis] Loading saved calibration score CSVs ...")
    oc  = pd.read_csv(RESULTS_DIR / "cal_scores_ocsvm.csv")
    if_ = pd.read_csv(RESULTS_DIR / "cal_scores_iforest.csv")

    # Sanity: no test data
    for df, name in [(oc, "ocsvm"), (if_, "iforest")]:
        assert df["rul_true"].min() == 0, f"{name}: last cycle not rul=0"
        print(f"  {name}: {len(df):,} rows, "
              f"{df['engine_id'].nunique()} engines, "
              f"RUL range [{df['rul_true'].min()}, {df['rul_true'].max()}]")

    # ------------------------------------------------------------------ #
    # Binned error tables
    # ------------------------------------------------------------------ #
    print("[rul_error_analysis] Computing binned errors ...")
    bin_oc  = binned_errors(oc)
    bin_if  = binned_errors(if_)
    ds_oc   = dataset_errors(oc)
    ds_if   = dataset_errors(if_)

    # Signed error tables
    def signed_by_bin(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["error"]   = df["rul_estimate"] - df["rul_true"]
        df["rul_bin"] = _assign_bin(df["rul_true"])
        agg = (
            df.groupby("rul_bin", observed=True)["error"]
            .agg(
                mean="mean",
                median="median",
                std="std",
                n="count",
            )
            .reset_index()
        )
        q_vals = (
            df.groupby("rul_bin", observed=True)["error"]
            .quantile([0.25, 0.75])
            .unstack(level=1)
            .rename(columns={0.25: "p25", 0.75: "p75"})
            .reset_index()
        )
        return agg.merge(q_vals, on="rul_bin", how="left")

    signed_oc  = signed_by_bin(oc)
    signed_if  = signed_by_bin(if_)

    # ------------------------------------------------------------------ #
    # Print tables
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 65)
    print("OC-SVM: MAE / RMSE / Signed Error by RUL bin (all engines)")
    print("=" * 65)
    print_oc = bin_oc.copy()
    print_oc["mean_err"]    = signed_oc["mean"].values
    print_oc["median_err"]  = signed_oc["median"].values
    print(print_oc[["rul_bin", "n", "mae", "rmse",
                    "mean_err", "median_err", "pct_overestimate"]].to_string(index=False))

    print("\n" + "=" * 65)
    print("IForest: MAE / RMSE / Signed Error by RUL bin (all engines)")
    print("=" * 65)
    print_if = bin_if.copy()
    print_if["mean_err"]    = signed_if["mean"].values
    print_if["median_err"]  = signed_if["median"].values
    print(print_if[["rul_bin", "n", "mae", "rmse",
                    "mean_err", "median_err", "pct_overestimate"]].to_string(index=False))

    print("\n" + "=" * 65)
    print("OC-SVM: MAE / RMSE by dataset × RUL bin")
    print("=" * 65)
    print(ds_oc[["dataset", "rul_bin", "n", "mae", "rmse",
                 "mean_signed_error", "pct_overestimate"]].to_string(index=False))

    print("\n" + "=" * 65)
    print("IForest: MAE / RMSE by dataset × RUL bin")
    print("=" * 65)
    print(ds_if[["dataset", "rul_bin", "n", "mae", "rmse",
                 "mean_signed_error", "pct_overestimate"]].to_string(index=False))

    # Near-failure summary
    print("\n" + "=" * 65)
    print("NEAR-FAILURE SUMMARY (RUL ≤ 20)")
    print("=" * 65)
    for name, df in [("OC-SVM", oc), ("IForest", if_)]:
        nf  = df[df["rul_true"] <= 20].copy()
        err = nf["rul_estimate"] - nf["rul_true"]
        print(f"\n  {name}:")
        print(f"    n cycles          : {len(nf)}")
        print(f"    MAE               : {err.abs().mean():.2f} cycles")
        print(f"    RMSE              : {np.sqrt((err**2).mean()):.2f} cycles")
        print(f"    Mean signed error : {err.mean():+.2f} cycles")
        print(f"    Median signed error:{err.median():+.2f} cycles")
        print(f"    % overestimate    : {(err > 0).mean():.1%}")
        for ds in ("FD001", "FD003"):
            e_ds = err[nf["dataset"] == ds]
            print(f"    {ds}: MAE={e_ds.abs().mean():.1f}  "
                  f"mean_err={e_ds.mean():+.1f}  "
                  f"pct_over={( e_ds > 0).mean():.1%}")

    # ------------------------------------------------------------------ #
    # Save CSVs
    # ------------------------------------------------------------------ #
    print("\n[rul_error_analysis] Saving CSVs ...")
    _save_csv(bin_oc,  "rul_error_by_bin_ocsvm.csv")
    _save_csv(bin_if,  "rul_error_by_bin_iforest.csv")
    _save_csv(ds_oc,   "rul_error_by_dataset_ocsvm.csv")
    _save_csv(ds_if,   "rul_error_by_dataset_iforest.csv")
    _save_csv(signed_oc, "rul_signed_error_by_bin_ocsvm.csv")
    _save_csv(signed_if, "rul_signed_error_by_bin_iforest.csv")

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    print("[rul_error_analysis] Generating plots ...")
    p1 = plot_predicted_vs_actual(oc, if_)
    p2 = plot_signed_error_vs_rul(oc, if_)
    p3 = plot_error_by_bin(bin_oc, bin_if)
    p4 = plot_bias_by_bin(bin_oc, bin_if)
    p5 = plot_error_by_dataset(ds_oc, ds_if)
    p6 = plot_near_failure_detail(oc, if_)
    p7 = plot_mae_vs_lifetime(oc, if_)

    for p in (p1, p2, p3, p4, p5, p6, p7):
        print(f"  {p}")

    print("\n[rul_error_analysis] Done.")


if __name__ == "__main__":
    run()
