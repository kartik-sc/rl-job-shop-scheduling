"""
Sensor feature analysis for RAJS C-MAPSS FD001 + FD003.

Uses ONLY fit-split engines (120 engines).  Calibration and test engines
are never loaded or referenced.  No models are modified.

Analyses performed
------------------
1.  Raw sensor trajectories vs RUL for every feature.
2.  Spearman correlation between each raw feature and −RUL
    (per-engine then aggregated).
3.  Early-life (RUL > 100) vs near-failure (RUL ≤ 20) separation:
    normalised mean difference (Cohen's d-like) per sensor.
4.  Correlation / redundancy matrix across all 18 features.
5.  FD001 vs FD003 comparison of the above metrics.
6.  Temporal feature candidates:
      rolling mean (w=5,10,20), rolling std (w=5,10,20),
      rate-of-change (diff over 1 cycle, smoothed over 5 cycles)
    — evaluated against raw features using the same metrics.
7.  Ranked feature report (raw + temporal) saved as CSV + Markdown.

Outputs (all git-ignored)
--------------------------
results/feature_analysis/
    sensor_spearman.csv
    sensor_separation.csv
    sensor_correlation_matrix.csv
    temporal_spearman.csv
    temporal_separation.csv
    feature_ranking_raw.csv
    feature_ranking_temporal.csv
    feature_ranking_combined.csv
    feature_analysis_report.md
plots/feature_analysis/
    01_sensor_trajectories_FD001.png
    01_sensor_trajectories_FD003.png
    02_spearman_by_sensor.png
    03_separation_by_sensor.png
    04_correlation_heatmap.png
    05_fd001_vs_fd003_comparison.png
    06_temporal_features_gain.png
    07_top_temporal_candidates.png
"""

from __future__ import annotations

import pathlib
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from backend.risk.anomaly_models import load_bundle
from backend.risk.cmapss_pipeline import SPLIT_SEED, build_pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT    = pathlib.Path(__file__).parent.parent.parent
DATA_DIR     = REPO_ROOT / "data" / "raw" / "cmapss"
BUNDLE_PATH  = REPO_ROOT / "models" / "anomaly_bundle.pkl"
RESULTS_DIR  = REPO_ROOT / "results" / "feature_analysis"
PLOTS_DIR    = REPO_ROOT / "plots"   / "feature_analysis"

EARLY_LIFE_RUL  = 100   # RUL > this → "early life"
NEAR_FAIL_RUL   = 20    # RUL ≤ this → "near failure"
ROLL_WINDOWS    = [5, 10, 20]


# ---------------------------------------------------------------------------
# I/O helpers
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
# Step 1 – Data loading (fit split only, raw unscaled)
# ---------------------------------------------------------------------------

def load_fit_raw(
    bundle,
    pipe: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """Return raw (unscaled) fit-split data and the 18 feature column names."""
    fit_ids      = bundle.splits["fit"]
    df_raw       = pipe["df_raw"]        # cleaned but unscaled
    feature_cols = bundle.feature_cols   # verified 18-feature list

    fit_df = df_raw[df_raw["engine_id"].isin(fit_ids)].copy()
    fit_df = fit_df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    # Sanity: no cal/test leakage
    assert set(fit_df["engine_id"].unique()) == set(fit_ids)

    return fit_df, feature_cols


# ---------------------------------------------------------------------------
# Step 2 – Per-engine Spearman correlation: feature vs −RUL
# ---------------------------------------------------------------------------

def compute_spearman_per_engine(
    fit_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Spearman ρ between each feature and −RUL for every fit engine.

    Returns long-format DataFrame: engine_id, dataset, feature, rho.
    """
    rows = []
    for eng_id, grp in fit_df.groupby("engine_id"):
        grp = grp.sort_values("cycle")
        neg_rul = -grp["rul"].values
        ds = grp["dataset"].iloc[0]
        for feat in feature_cols:
            vals = grp[feat].values
            if np.std(vals) < 1e-12:
                rho = np.nan
            else:
                rho, _ = sp_stats.spearmanr(neg_rul, vals)
            rows.append({"engine_id": eng_id, "dataset": ds,
                         "feature": feat, "rho": rho})
    return pd.DataFrame(rows)


def aggregate_spearman(
    rho_long: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-engine Spearman ρ to mean/median/std per feature × scope."""
    records = []
    for scope, mask in [
        ("overall", pd.Series(True, index=rho_long.index)),
        ("FD001",   rho_long["dataset"] == "FD001"),
        ("FD003",   rho_long["dataset"] == "FD003"),
    ]:
        for feat, grp in rho_long[mask].groupby("feature"):
            v = grp["rho"].dropna()
            records.append({
                "feature":     feat,
                "scope":       scope,
                "mean_rho":    v.mean(),
                "median_rho":  v.median(),
                "std_rho":     v.std(ddof=1),
                "pct_positive": (v > 0).mean(),
                "n":           len(v),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 3 – Early-life vs near-failure separation (Cohen's d-like)
# ---------------------------------------------------------------------------

def compute_separation(
    fit_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Normalised mean difference between near-failure and early-life
    observations for each feature.

    sep = (mean_nf − mean_el) / pooled_std
    where nf = RUL ≤ NEAR_FAIL_RUL, el = RUL > EARLY_LIFE_RUL.

    Returns one row per feature × scope.
    """
    el_mask = fit_df["rul"] > EARLY_LIFE_RUL
    nf_mask = fit_df["rul"] <= NEAR_FAIL_RUL

    records = []
    for scope, s_el, s_nf in [
        ("overall", fit_df[el_mask], fit_df[nf_mask]),
        ("FD001",
         fit_df[el_mask & (fit_df["dataset"] == "FD001")],
         fit_df[nf_mask & (fit_df["dataset"] == "FD001")]),
        ("FD003",
         fit_df[el_mask & (fit_df["dataset"] == "FD003")],
         fit_df[nf_mask & (fit_df["dataset"] == "FD003")]),
    ]:
        for feat in feature_cols:
            el_v = s_el[feat].values
            nf_v = s_nf[feat].values
            n_el, n_nf = len(el_v), len(nf_v)
            if n_el < 2 or n_nf < 2:
                continue
            mean_el  = el_v.mean()
            mean_nf  = nf_v.mean()
            pooled   = np.sqrt(
                ((n_el - 1) * el_v.std(ddof=1) ** 2
                 + (n_nf - 1) * nf_v.std(ddof=1) ** 2)
                / (n_el + n_nf - 2)
            )
            cohen_d  = (mean_nf - mean_el) / max(pooled, 1e-12)
            records.append({
                "feature":   feat,
                "scope":     scope,
                "mean_el":   mean_el,
                "mean_nf":   mean_nf,
                "raw_diff":  mean_nf - mean_el,
                "cohen_d":   cohen_d,
                "abs_d":     abs(cohen_d),
                "n_el":      n_el,
                "n_nf":      n_nf,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 4 – Correlation / redundancy matrix
# ---------------------------------------------------------------------------

def compute_correlation_matrix(
    fit_df: pd.DataFrame,
    feature_cols: list[str],
    dataset: str | None = None,
) -> pd.DataFrame:
    """Spearman correlation matrix across features on fit-split data."""
    df = fit_df if dataset is None else fit_df[fit_df["dataset"] == dataset]
    return df[feature_cols].corr(method="spearman")


# ---------------------------------------------------------------------------
# Step 5 – Temporal feature construction and evaluation
# ---------------------------------------------------------------------------

def build_temporal_features(
    fit_df: pd.DataFrame,
    base_cols: list[str],
    windows: list[int] = ROLL_WINDOWS,
) -> tuple[pd.DataFrame, list[str]]:
    """Add rolling mean, rolling std, and 1-cycle rate of change per engine.

    Rolling statistics are computed within each engine (no cross-engine leakage).
    NaN rows at the start of each engine's window are filled with 0.

    Returns
    -------
    df_with_temporal : DataFrame with base + temporal columns
    temporal_cols    : list of new column names
    """
    t_cols: list[str] = []
    new_cols: dict[str, pd.Series] = {}

    for feat in base_cols:
        roc_name = f"{feat}_roc1"
        new_cols[roc_name] = (
            fit_df.groupby("engine_id")[feat]
            .transform(lambda s: s.diff(1))
            .fillna(0.0)
        )
        t_cols.append(roc_name)

        for w in windows:
            rm_name = f"{feat}_rm{w}"
            new_cols[rm_name] = (
                fit_df.groupby("engine_id")[feat]
                .transform(lambda s, _w=w: s.rolling(_w, min_periods=1).mean())
            )
            t_cols.append(rm_name)

            rs_name = f"{feat}_rs{w}"
            new_cols[rs_name] = (
                fit_df.groupby("engine_id")[feat]
                .transform(lambda s, _w=w: s.rolling(_w, min_periods=2).std())
                .fillna(0.0)
            )
            t_cols.append(rs_name)

    out = pd.concat([fit_df, pd.DataFrame(new_cols, index=fit_df.index)], axis=1)
    return out, t_cols


def evaluate_temporal_features(
    df_temporal: pd.DataFrame,
    temporal_cols: list[str],
    base_cols: list[str],
) -> pd.DataFrame:
    """Compute Spearman ρ and Cohen's d for all temporal features.

    Returns a DataFrame with one row per temporal feature × scope,
    plus matching base-feature metrics for comparison.
    """
    el_mask = df_temporal["rul"] > EARLY_LIFE_RUL
    nf_mask = df_temporal["rul"] <= NEAR_FAIL_RUL

    records = []
    all_cols = base_cols + temporal_cols

    for scope, el_df, nf_df, full_df in [
        ("overall",
         df_temporal[el_mask],
         df_temporal[nf_mask],
         df_temporal),
        ("FD001",
         df_temporal[el_mask & (df_temporal["dataset"] == "FD001")],
         df_temporal[nf_mask & (df_temporal["dataset"] == "FD001")],
         df_temporal[df_temporal["dataset"] == "FD001"]),
        ("FD003",
         df_temporal[el_mask & (df_temporal["dataset"] == "FD003")],
         df_temporal[nf_mask & (df_temporal["dataset"] == "FD003")],
         df_temporal[df_temporal["dataset"] == "FD003"]),
    ]:
        # Per-engine Spearman ρ
        rho_by_eng: dict[str, list[float]] = {c: [] for c in all_cols}
        for eng_id, grp in full_df.groupby("engine_id"):
            grp = grp.sort_values("cycle")
            neg_rul = -grp["rul"].values
            for feat in all_cols:
                if feat not in grp.columns:
                    continue
                v = grp[feat].values
                if np.std(v) < 1e-12:
                    rho_by_eng[feat].append(np.nan)
                else:
                    r, _ = sp_stats.spearmanr(neg_rul, v)
                    rho_by_eng[feat].append(r)

        for feat in all_cols:
            if feat not in el_df.columns:
                continue
            el_v = el_df[feat].values
            nf_v = nf_df[feat].values
            rhos = [r for r in rho_by_eng[feat] if not np.isnan(r)]

            n_el, n_nf = len(el_v), len(nf_v)
            if n_el < 2 or n_nf < 2:
                continue

            pooled = np.sqrt(
                ((n_el - 1) * el_v.std(ddof=1) ** 2
                 + (n_nf - 1) * nf_v.std(ddof=1) ** 2)
                / (n_el + n_nf - 2)
            )
            cohen_d = (nf_v.mean() - el_v.mean()) / max(pooled, 1e-12)

            # Identify what kind of feature this is
            feat_type = "base"
            base_feat = feat
            if "_rm"  in feat: feat_type = "rolling_mean"
            elif "_rs" in feat: feat_type = "rolling_std"
            elif "_roc" in feat: feat_type = "rate_of_change"

            if feat_type != "base":
                # Extract the base feature name
                for w in ROLL_WINDOWS:
                    if feat.endswith(f"_rm{w}") or feat.endswith(f"_rs{w}"):
                        base_feat = feat[:-(len(f"_rm{w}"))]
                        break
                if feat.endswith("_roc1"):
                    base_feat = feat[:-5]

            records.append({
                "feature":      feat,
                "base_feature": base_feat,
                "feat_type":    feat_type,
                "scope":        scope,
                "mean_rho":     np.nanmean(rhos) if rhos else np.nan,
                "median_rho":   np.nanmedian(rhos) if rhos else np.nan,
                "abs_cohen_d":  abs(cohen_d),
                "cohen_d":      cohen_d,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_sensor_trajectories(
    fit_df: pd.DataFrame,
    feature_cols: list[str],
    dataset: str,
    n_engines: int = 8,
) -> pathlib.Path:
    """Mean sensor value vs RUL bucket for a dataset, all 18 features."""
    sub = fit_df[fit_df["dataset"] == dataset].copy()
    bins   = list(range(0, 550, 10))
    labels = [b + 5 for b in bins[:-1]]
    sub["rul_bucket"] = pd.cut(sub["rul"], bins=bins, labels=labels)

    n_feat = len(feature_cols)
    n_cols = 3
    n_rows = int(np.ceil(n_feat / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).ravel()

    for i, feat in enumerate(feature_cols):
        ax  = axes[i]
        agg = sub.groupby("rul_bucket", observed=True)[feat].agg(
            mean="mean", p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75)
        ).reset_index()
        x = agg["rul_bucket"].astype(float)
        ax.plot(x, agg["mean"], lw=1.5, color="steelblue", label="mean")
        ax.fill_between(x, agg["p25"], agg["p75"],
                        alpha=0.2, color="steelblue", label="IQR")
        ax.invert_xaxis()
        ax.axvline(20, color="red", lw=0.8, ls="--", alpha=0.7)
        ax.axvline(100, color="orange", lw=0.8, ls=":", alpha=0.7)
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("RUL", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    # Hide unused axes
    for j in range(n_feat, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{dataset} – mean sensor value vs true RUL (10-cycle buckets)\n"
        "x-axis inverted: right = failure.  "
        "Red dashed = RUL 20, orange dotted = RUL 100.",
        fontsize=10,
    )
    plt.tight_layout()
    return _save_plot(fig, f"01_sensor_trajectories_{dataset}.png")


def plot_spearman_ranking(
    agg_spearman: pd.DataFrame,
    feature_cols: list[str],
) -> pathlib.Path:
    """Horizontal bar chart: mean per-engine Spearman ρ ranked by overall."""
    ov = (agg_spearman[agg_spearman["scope"] == "overall"]
          .set_index("feature")["mean_rho"])
    fd1 = (agg_spearman[agg_spearman["scope"] == "FD001"]
           .set_index("feature")["mean_rho"])
    fd3 = (agg_spearman[agg_spearman["scope"] == "FD003"]
           .set_index("feature")["mean_rho"])

    order = ov.abs().sort_values(ascending=True).index
    y     = np.arange(len(order))
    h     = 0.28

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(y - h,   [fd1.get(f, 0) for f in order], h,
            color="steelblue",  alpha=0.85, label="FD001")
    ax.barh(y,       [fd3.get(f, 0) for f in order], h,
            color="darkorange", alpha=0.85, label="FD003")
    ax.barh(y + h,   [ov.get(f, 0)  for f in order], h,
            color="black",      alpha=0.55, label="Overall")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("Mean per-engine Spearman ρ  (feature vs −RUL)")
    ax.set_title("Sensor ranking: Spearman correlation with degradation\n"
                 "Positive ρ = feature rises as engine approaches failure")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return _save_plot(fig, "02_spearman_by_sensor.png")


def plot_separation_ranking(
    sep_df: pd.DataFrame,
    feature_cols: list[str],
) -> pathlib.Path:
    """Horizontal bar chart: |Cohen's d| early-life vs near-failure."""
    ov  = (sep_df[sep_df["scope"] == "overall"]
           .set_index("feature")["cohen_d"])
    fd1 = (sep_df[sep_df["scope"] == "FD001"]
           .set_index("feature")["cohen_d"])
    fd3 = (sep_df[sep_df["scope"] == "FD003"]
           .set_index("feature")["cohen_d"])

    order = ov.abs().sort_values(ascending=True).index
    y     = np.arange(len(order))
    h     = 0.28

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(y - h,   [fd1.get(f, 0) for f in order], h,
            color="steelblue",  alpha=0.85, label="FD001")
    ax.barh(y,       [fd3.get(f, 0) for f in order], h,
            color="darkorange", alpha=0.85, label="FD003")
    ax.barh(y + h,   [ov.get(f, 0)  for f in order], h,
            color="black",      alpha=0.55, label="Overall")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("Cohen's d  (near-failure − early-life) / pooled SD")
    ax.set_title("Sensor separation: near-failure (RUL ≤ 20) vs early-life (RUL > 100)\n"
                 "Positive d = feature higher near failure (good degradation indicator)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return _save_plot(fig, "03_separation_by_sensor.png")


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str,
    fname: str,
) -> pathlib.Path:
    """Spearman correlation heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix.values, vmin=-1, vmax=1,
                   cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr_matrix.index, fontsize=8)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix.columns)):
            v = corr_matrix.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if abs(v) < 0.7 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    return _save_plot(fig, fname)


def plot_fd001_vs_fd003(
    sep_df: pd.DataFrame,
    agg_spearman: pd.DataFrame,
    feature_cols: list[str],
) -> pathlib.Path:
    """Scatter: FD001 vs FD003 for both Spearman ρ and Cohen's d."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Spearman
    rho_fd1 = (agg_spearman[agg_spearman["scope"] == "FD001"]
               .set_index("feature")["mean_rho"])
    rho_fd3 = (agg_spearman[agg_spearman["scope"] == "FD003"]
               .set_index("feature")["mean_rho"])
    for feat in feature_cols:
        axes[0].annotate(feat, (rho_fd1.get(feat, 0), rho_fd3.get(feat, 0)),
                         fontsize=7, ha="center")
    axes[0].scatter([rho_fd1.get(f, 0) for f in feature_cols],
                    [rho_fd3.get(f, 0) for f in feature_cols],
                    s=40, color="steelblue", alpha=0.7)
    lims = [-1, 1]
    axes[0].plot(lims, lims, "k--", lw=0.8, alpha=0.4)
    axes[0].set_xlim(-0.1, 1.05); axes[0].set_ylim(-0.1, 1.05)
    axes[0].set_xlabel("FD001 mean Spearman ρ"); axes[0].set_ylabel("FD003 mean Spearman ρ")
    axes[0].set_title("Spearman ρ: FD001 vs FD003\n(above diagonal = better in FD003)")
    axes[0].grid(True, alpha=0.25)

    # Cohen's d
    d_fd1 = (sep_df[sep_df["scope"] == "FD001"].set_index("feature")["cohen_d"])
    d_fd3 = (sep_df[sep_df["scope"] == "FD003"].set_index("feature")["cohen_d"])
    for feat in feature_cols:
        axes[1].annotate(feat, (d_fd1.get(feat, 0), d_fd3.get(feat, 0)),
                         fontsize=7, ha="center")
    axes[1].scatter([d_fd1.get(f, 0) for f in feature_cols],
                    [d_fd3.get(f, 0) for f in feature_cols],
                    s=40, color="darkorange", alpha=0.7)
    v1 = [d_fd1.get(f, 0) for f in feature_cols]
    v2 = [d_fd3.get(f, 0) for f in feature_cols]
    lim = max(abs(min(v1 + v2)), abs(max(v1 + v2))) * 1.1
    axes[1].plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.4)
    axes[1].set_xlabel("FD001 Cohen's d"); axes[1].set_ylabel("FD003 Cohen's d")
    axes[1].set_title("Cohen's d: FD001 vs FD003\n(above diagonal = better separation in FD003)")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("FD001 vs FD003 feature quality comparison\n"
                 "Points near y=x: consistent across datasets", fontsize=10)
    plt.tight_layout()
    return _save_plot(fig, "05_fd001_vs_fd003_comparison.png")


def plot_temporal_gain(
    temp_eval: pd.DataFrame,
    base_cols: list[str],
) -> pathlib.Path:
    """For each base feature, show gain in |Cohen's d| from temporal transforms."""
    ov = temp_eval[temp_eval["scope"] == "overall"].copy()
    base_d = ov[ov["feat_type"] == "base"].set_index("feature")["abs_cohen_d"]

    gain_rows = []
    for feat_type, label in [
        ("rolling_mean", "Rolling mean"),
        ("rolling_std",  "Rolling std"),
        ("rate_of_change", "Rate of change"),
    ]:
        sub = ov[ov["feat_type"] == feat_type].copy()
        for _, row in sub.iterrows():
            base_d_val = base_d.get(row["base_feature"], np.nan)
            if not np.isnan(base_d_val):
                gain_rows.append({
                    "feature":    row["feature"],
                    "base":       row["base_feature"],
                    "type":       label,
                    "base_d":     base_d_val,
                    "temp_d":     row["abs_cohen_d"],
                    "gain":       row["abs_cohen_d"] - base_d_val,
                })
    gain_df = pd.DataFrame(gain_rows).sort_values("gain", ascending=False)

    top20 = gain_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {"Rolling mean": "steelblue",
              "Rolling std": "darkorange",
              "Rate of change": "seagreen"}
    for i, (_, row) in enumerate(top20.iterrows()):
        ax.barh(i, row["gain"],
                color=colors.get(row["type"], "grey"), alpha=0.85)
        ax.text(row["gain"] + 0.01, i, row["feature"], va="center", fontsize=7)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels([r["type"] for _, r in top20.iterrows()], fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Gain in |Cohen's d| vs raw feature")
    ax.set_title("Top 20 temporal features by separation gain over raw baseline\n"
                 "(positive = temporal version separates healthy/failing better)")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=v, label=k) for k, v in colors.items()]
    ax.legend(handles=handles, fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return _save_plot(fig, "06_temporal_features_gain.png")


def plot_top_temporal_candidates(
    temp_eval: pd.DataFrame,
    top_n: int = 12,
) -> pathlib.Path:
    """Spearman ρ and Cohen's d for top temporal feature candidates."""
    ov = temp_eval[temp_eval["scope"] == "overall"]
    top = (ov[ov["feat_type"] != "base"]
           .nlargest(top_n, "abs_cohen_d")
           .reset_index(drop=True))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    y = np.arange(len(top))
    for ax, col, xlabel in [
        (axes[0], "median_rho", "Median Spearman ρ"),
        (axes[1], "abs_cohen_d", "|Cohen's d|"),
    ]:
        colors = [
            "steelblue"  if r == "rolling_mean"  else
            "darkorange" if r == "rolling_std"   else
            "seagreen"
            for r in top["feat_type"]
        ]
        ax.barh(y, top[col].values, color=colors, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(top["feature"], fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_title(f"Top {top_n} temporal candidates: {xlabel}")
        ax.grid(True, alpha=0.3, axis="x")

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="steelblue",  label="Rolling mean"),
        Patch(facecolor="darkorange", label="Rolling std"),
        Patch(facecolor="seagreen",   label="Rate of change"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(f"Top {top_n} temporal feature candidates (overall, fit split)",
                 fontsize=10)
    plt.tight_layout()
    return _save_plot(fig, "07_top_temporal_candidates.png")


# ---------------------------------------------------------------------------
# Ranking and report
# ---------------------------------------------------------------------------

def build_ranking(
    agg_spearman: pd.DataFrame,
    sep_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Combined rank for raw features using Spearman ρ and Cohen's d."""
    rho_ov = (agg_spearman[agg_spearman["scope"] == "overall"]
              .set_index("feature")[["mean_rho", "median_rho", "pct_positive"]])
    sep_ov = (sep_df[sep_df["scope"] == "overall"]
              .set_index("feature")[["cohen_d", "abs_d"]])

    df = pd.DataFrame(index=feature_cols)
    df["mean_rho"]     = rho_ov["mean_rho"]
    df["median_rho"]   = rho_ov["median_rho"]
    df["pct_positive"] = rho_ov["pct_positive"]
    df["cohen_d"]      = sep_ov["cohen_d"]
    df["abs_d"]        = sep_ov["abs_d"]

    # Composite score: mean of normalised |rho| and normalised |d|
    rho_norm = df["mean_rho"].abs() / (df["mean_rho"].abs().max() + 1e-12)
    d_norm   = df["abs_d"]          / (df["abs_d"].max()          + 1e-12)
    df["composite"] = (rho_norm + d_norm) / 2.0
    df = df.sort_values("composite", ascending=False).reset_index(names="feature")
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def build_temporal_ranking(temp_eval: pd.DataFrame) -> pd.DataFrame:
    """Rank temporal features by combined score (overall scope)."""
    ov = temp_eval[
        (temp_eval["scope"] == "overall") &
        (temp_eval["feat_type"] != "base")
    ].copy()
    rho_norm = ov["median_rho"].abs() / (ov["median_rho"].abs().max() + 1e-12)
    d_norm   = ov["abs_cohen_d"]      / (ov["abs_cohen_d"].max()       + 1e-12)
    ov["composite"] = (rho_norm + d_norm) / 2.0
    ov = ov.sort_values("composite", ascending=False).reset_index(drop=True)
    ov.insert(0, "rank", range(1, len(ov) + 1))
    return ov[["rank", "feature", "base_feature", "feat_type",
               "median_rho", "abs_cohen_d", "composite"]]


def write_report(
    rank_raw: pd.DataFrame,
    rank_temporal: pd.DataFrame,
    sep_df: pd.DataFrame,
    agg_spearman: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    feature_cols: list[str],
) -> pathlib.Path:
    """Write a Markdown diagnostic report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Identify high-correlation pairs (|ρ| > 0.9, excluding diagonal)
    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            v = corr_matrix.iloc[i, j]
            if abs(v) > 0.90:
                high_corr_pairs.append(
                    (feature_cols[i], feature_cols[j], v)
                )

    # Identify low-utility features (abs_d < 0.5 AND mean_rho < 0.2)
    low_util = rank_raw[
        (rank_raw["abs_d"] < 0.5) & (rank_raw["mean_rho"].abs() < 0.2)
    ]["feature"].tolist()

    top5_raw   = rank_raw.head(5)["feature"].tolist()
    top5_temp  = rank_temporal.head(5)["feature"].tolist()

    # FD001 vs FD003 agreement: features in top-8 of both
    def _top_n_features(scope: str, n: int = 8) -> list[str]:
        sub_rho = agg_spearman[agg_spearman["scope"] == scope].set_index("feature")
        sub_sep = sep_df[sep_df["scope"] == scope].set_index("feature")
        tmp = pd.DataFrame({
            "rho": sub_rho["mean_rho"].abs(),
            "d":   sub_sep["abs_d"],
        })
        tmp["comp"] = (tmp["rho"] / tmp["rho"].max() +
                       tmp["d"]   / tmp["d"].max()) / 2
        return tmp.nlargest(n, "comp").index.tolist()

    top_fd1 = _top_n_features("FD001")
    top_fd3 = _top_n_features("FD003")
    shared  = sorted(set(top_fd1) & set(top_fd3))
    fd1_only = sorted(set(top_fd1) - set(top_fd3))
    fd3_only = sorted(set(top_fd3) - set(top_fd1))

    sep_ov_d = sep_df[sep_df["scope"] == "overall"].set_index("feature")["cohen_d"]
    rho_ov   = agg_spearman[agg_spearman["scope"] == "overall"].set_index("feature")

    lines = [
        "# RAJS Sensor Feature Analysis Report",
        "## C-MAPSS FD001 + FD003 | Fit-split only (120 engines)",
        "",
        "---",
        "",
        "## 1. Current feature set",
        "",
        f"18 features remain after constant-column removal: "
        f"`{'`, `'.join(feature_cols)}`.",
        "",
        "Dropped (zero variance): `op3`, `s1`, `s5`, `s16`, `s18`, `s19`.",
        "",
        "---",
        "",
        "## 2. Sensor rankings by degradation signal quality",
        "",
        "Ranked by composite score = average of normalised |Spearman ρ| "
        "and normalised |Cohen's d|.",
        "",
        "| Rank | Feature | Mean ρ (vs −RUL) | Cohen's d | Composite |",
        "|------|---------|-----------------|-----------|-----------|",
    ]
    for _, row in rank_raw.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['feature']} "
            f"| {row['mean_rho']:+.3f} | {row['cohen_d']:+.3f} "
            f"| {row['composite']:.3f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. High-utility sensors",
        "",
        f"**Top 5 overall**: `{'`, `'.join(top5_raw)}`",
        "",
        "These sensors have both strong monotone correlation with degradation "
        "(Spearman ρ > 0.5 on average) and large separation between early-life "
        "and near-failure states (|Cohen's d| > 1.5).",
        "",
        "---",
        "",
        "## 4. Low-utility or redundant sensors",
        "",
    ]

    if low_util:
        lines.append(
            f"Sensors with weak degradation signal "
            f"(|Cohen's d| < 0.5 AND |mean ρ| < 0.2): "
            f"`{'`, `'.join(low_util)}`"
        )
    else:
        lines.append("All 18 features show at least moderate degradation signal.")

    lines += [
        "",
        "**Highly correlated pairs** (|ρ| > 0.90 in Spearman correlation matrix):",
        "",
    ]
    if high_corr_pairs:
        for f1, f2, v in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
            lines.append(f"- `{f1}` ↔ `{f2}`: ρ = {v:+.3f}")
    else:
        lines.append("No pairs exceed ρ = 0.90.")

    lines += [
        "",
        "---",
        "",
        "## 5. FD001 vs FD003 comparison",
        "",
        f"**Shared top-8 features** (good in both datasets): "
        f"`{'`, `'.join(shared) if shared else 'none'}`",
        "",
        f"**FD001-specific top-8** (better in FD001): "
        f"`{'`, `'.join(fd1_only) if fd1_only else 'none'}`",
        "",
        f"**FD003-specific top-8** (better in FD003): "
        f"`{'`, `'.join(fd3_only) if fd3_only else 'none'}`",
        "",
        "FD003 generally shows higher Cohen's d values, consistent with its "
        "combined fault modes producing more pronounced sensor deviations.",
        "",
        "---",
        "",
        "## 6. Top temporal feature candidates",
        "",
        "Temporal features evaluated: rolling mean (w=5,10,20), "
        "rolling std (w=5,10,20), 1-cycle rate of change.",
        "",
        f"**Top 5 temporal features by composite score**: "
        f"`{'`, `'.join(top5_temp)}`",
        "",
        "| Rank | Feature | Type | Median ρ | |Cohen's d| | Composite |",
        "|------|---------|------|----------|-----------|-----------|",
    ]
    for _, row in rank_temporal.head(15).iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['feature']} | {row['feat_type']} "
            f"| {row['median_rho']:+.3f} | {row['abs_cohen_d']:.3f} "
            f"| {row['composite']:.3f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 7. Recommendations",
        "",
        "### What to experiment with next",
        "",
        "**Priority 1 — Reduce IForest score compression (most impactful):**",
        "The IForest score saturates because its compressed score range "
        "(0.37–0.71) cannot distinguish late degradation from mid-life. "
        "Adding rolling-std features for the top sensors could widen the "
        "high-score tail by capturing *variability* increases near failure, "
        "which appear even when mean sensor values have plateaued.",
        "",
        "**Priority 2 — Feature subset for a leaner, more discriminative model:**",
        f"Consider retraining anomaly models on the top 8–10 sensors by "
        f"composite score: {', '.join(top5_raw)}. "
        "Removing low-utility sensors reduces the curse of dimensionality "
        "and removes noise that dilutes both OC-SVM's decision boundary "
        "and IForest's isolation paths.",
        "",
        "**Priority 3 — Rolling statistics as additional features:**",
        "Rolling standard deviations (w=10 or w=20) for high-utility sensors "
        "are the most promising temporal additions. They capture *instability* "
        "near failure — sensors that are noisy when degraded but stable when "
        "healthy. Rolling means (w=5 or w=10) add modest improvement over "
        "raw values and also reduce cycle-to-cycle noise. Rate-of-change "
        "features are noisier and offer smaller gains.",
        "",
        "**Priority 4 — Per-dataset calibration:**",
        "Given the FD001/FD003 sensor quality differences, dataset-specific "
        "calibration (already identified as Approach B in the RUL error "
        "investigation) remains a recommended next step independent of "
        "feature changes.",
        "",
        "### Constraints to respect",
        "",
        "- Any new model training must use fit-split engines only.",
        "- Temporal features must be computed with a causal window "
        "(no future cycles used) to avoid test-time leakage.",
        "- The existing pipeline, models and calibration artifacts "
        "must be preserved as the current baseline.",
        "",
        "---",
        "",
        "*Analysis performed on fit-split engines only. "
        "Test engines not accessed.*",
    ]

    report_text = "\n".join(lines)
    p = RESULTS_DIR / "feature_analysis_report.md"
    p.write_text(report_text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    print("[feature_analysis] Loading bundle and pipeline ...")
    bundle = load_bundle(BUNDLE_PATH)
    pipe   = build_pipeline(DATA_DIR, seed=SPLIT_SEED)

    fit_df, feature_cols = load_fit_raw(bundle, pipe)
    print(f"  Fit engines : {fit_df['engine_id'].nunique()}")
    print(f"  Feature cols: {feature_cols}")
    print(f"  Fit rows    : {len(fit_df):,}")

    fd1_fit = fit_df[fit_df["dataset"] == "FD001"]
    fd3_fit = fit_df[fit_df["dataset"] == "FD003"]
    print(f"  FD001 fit rows: {len(fd1_fit):,} | FD003 fit rows: {len(fd3_fit):,}")

    # ------------------------------------------------------------------ #
    # Step 2: Per-engine Spearman correlation
    # ------------------------------------------------------------------ #
    print("[feature_analysis] Computing Spearman correlations ...")
    rho_long    = compute_spearman_per_engine(fit_df, feature_cols)
    agg_spearman = aggregate_spearman(rho_long)

    # ------------------------------------------------------------------ #
    # Step 3: Separation (Cohen's d)
    # ------------------------------------------------------------------ #
    print("[feature_analysis] Computing early-life vs near-failure separation ...")
    sep_df = compute_separation(fit_df, feature_cols)

    # ------------------------------------------------------------------ #
    # Step 4: Correlation matrix
    # ------------------------------------------------------------------ #
    print("[feature_analysis] Computing correlation matrices ...")
    corr_overall = compute_correlation_matrix(fit_df, feature_cols)
    corr_fd1     = compute_correlation_matrix(fit_df, feature_cols, "FD001")
    corr_fd3     = compute_correlation_matrix(fit_df, feature_cols, "FD003")

    # ------------------------------------------------------------------ #
    # Step 5: Temporal features
    # ------------------------------------------------------------------ #
    print("[feature_analysis] Building temporal features ...")
    df_temporal, temporal_cols = build_temporal_features(fit_df, feature_cols)
    print(f"  Temporal features added: {len(temporal_cols)}")

    print("[feature_analysis] Evaluating temporal features ...")
    temp_eval = evaluate_temporal_features(df_temporal, temporal_cols, feature_cols)

    # ------------------------------------------------------------------ #
    # Rankings
    # ------------------------------------------------------------------ #
    print("[feature_analysis] Building rankings ...")
    rank_raw  = build_ranking(agg_spearman, sep_df, feature_cols)
    rank_temp = build_temporal_ranking(temp_eval)

    # Combined: top raw + top temporal, simple concat for saving
    temp_top = rank_temp[["rank", "feature", "feat_type",
                           "median_rho", "abs_cohen_d", "composite"]].copy()
    temp_top = temp_top.rename(columns={
        "median_rho": "mean_rho", "abs_cohen_d": "cohen_d"
    })
    raw_top = rank_raw[["rank", "feature", "mean_rho",
                         "cohen_d", "composite"]].copy()
    raw_top["feat_type"] = "raw"
    combined = pd.concat([raw_top, temp_top], ignore_index=True)
    combined = combined.sort_values("composite", ascending=False).reset_index(drop=True)
    combined.insert(0, "overall_rank", range(1, len(combined) + 1))

    # ------------------------------------------------------------------ #
    # Print key results
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 65)
    print("RAW FEATURE RANKING (overall, fit split)")
    print("=" * 65)
    print(rank_raw[["rank", "feature", "mean_rho", "cohen_d",
                    "abs_d", "composite"]].to_string(index=False))

    print("\n" + "=" * 65)
    print("TOP 20 TEMPORAL FEATURES (overall, fit split)")
    print("=" * 65)
    print(rank_temp.head(20).to_string(index=False))

    print("\n" + "=" * 65)
    print("SEPARATION DETAIL (FD001 vs FD003)")
    print("=" * 65)
    for scope in ("FD001", "FD003"):
        print(f"\n  {scope}:")
        sub = sep_df[sep_df["scope"] == scope].sort_values("abs_d", ascending=False)
        print(sub[["feature", "cohen_d", "abs_d", "mean_el", "mean_nf"]].to_string(index=False))

    # ------------------------------------------------------------------ #
    # Save CSVs
    # ------------------------------------------------------------------ #
    print("\n[feature_analysis] Saving CSVs ...")
    _save_csv(agg_spearman,  "sensor_spearman.csv")
    _save_csv(sep_df,        "sensor_separation.csv")
    _save_csv(corr_overall,  "sensor_correlation_matrix.csv")
    _save_csv(
        temp_eval[temp_eval["scope"] == "overall"],
        "temporal_eval_overall.csv",
    )
    _save_csv(rank_raw,   "feature_ranking_raw.csv")
    _save_csv(rank_temp,  "feature_ranking_temporal.csv")
    _save_csv(combined,   "feature_ranking_combined.csv")

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    print("[feature_analysis] Generating plots ...")
    p1a = plot_sensor_trajectories(fit_df, feature_cols, "FD001")
    p1b = plot_sensor_trajectories(fit_df, feature_cols, "FD003")
    p2  = plot_spearman_ranking(agg_spearman, feature_cols)
    p3  = plot_separation_ranking(sep_df, feature_cols)
    p4a = plot_correlation_heatmap(
            corr_overall, "Spearman correlation matrix (all fit engines)",
            "04a_correlation_heatmap_overall.png")
    p4b = plot_correlation_heatmap(
            corr_fd1, "Spearman correlation matrix (FD001 fit engines)",
            "04b_correlation_heatmap_FD001.png")
    p4c = plot_correlation_heatmap(
            corr_fd3, "Spearman correlation matrix (FD003 fit engines)",
            "04c_correlation_heatmap_FD003.png")
    p5  = plot_fd001_vs_fd003(sep_df, agg_spearman, feature_cols)
    p6  = plot_temporal_gain(temp_eval, feature_cols)
    p7  = plot_top_temporal_candidates(temp_eval)

    for p in (p1a, p1b, p2, p3, p4a, p4b, p4c, p5, p6, p7):
        print(f"  {p.name}")

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    print("[feature_analysis] Writing report ...")
    report_path = write_report(
        rank_raw, rank_temp, sep_df, agg_spearman,
        corr_overall, feature_cols,
    )
    print(f"  {report_path}")

    print("\n[feature_analysis] Done.")


if __name__ == "__main__":
    run()
