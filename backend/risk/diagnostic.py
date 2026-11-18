"""
OC-SVM vs Isolation Forest diagnostic analysis for C-MAPSS fit-split engines.

Steps performed (mirroring the RAJS diagnostic specification):
  1. Load existing bundle (no retraining).
  2. Rebuild scaled data via build_pipeline (same seed → same split).
  3. Score all 120 fit-engine cycles with both models.
  4. Healthy-window (cycle ≤ 30) vs late-life (final 30 cycles by RUL) separation.
  5. Spearman rank correlation between score and -RUL (degradation stage).
  6. Early-warning analysis: leave-one-engine-out threshold on fit data only.
  7. Save per-engine CSVs, aggregate summary tables, and comparison plots.

Score convention (both models): higher score = more anomalous.
  OC-SVM  : -decision_function(X)   (positive for inliers → negated)
  IForest : -score_samples(X)        (less negative = more normal → negated)

Leakage safeguards
------------------
- Bundle is loaded, not retrained.
- Thresholds for early-warning are estimated engine-by-engine using only the
  remaining 119 fit engines' healthy windows (leave-one-out within fit split).
- Calibration and test engines are never touched.

Usage (from repo root):
    python -m backend.risk.diagnostic
"""

from __future__ import annotations

import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

from backend.risk.anomaly_models import (
    load_bundle,
    score_engine,
    score_ocsvm,
    score_iforest,
)
from backend.risk.cmapss_pipeline import (
    SPLIT_SEED,
    build_pipeline,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT    = pathlib.Path(__file__).parent.parent.parent
DATA_DIR     = REPO_ROOT / "data" / "raw" / "cmapss"
BUNDLE_PATH  = REPO_ROOT / "models" / "anomaly_bundle.pkl"
PLOTS_DIR    = REPO_ROOT / "plots" / "diagnostic"
RESULTS_DIR  = REPO_ROOT / "results"

HEALTHY_N    = 30   # first N cycles = healthy window
LATE_LIFE_N  = 30   # last N cycles (by RUL) = late-life window
MIN_WARN_PERSIST = 3  # consecutive cycles above threshold to count as warning


# ---------------------------------------------------------------------------
# Step 1 & 2: Load bundle + rebuild scaled data
# ---------------------------------------------------------------------------

def load_data_and_bundle() -> tuple:
    """Load existing bundle and rebuild scaled DataFrame.

    Returns
    -------
    bundle, df_scaled, fit_ids, feature_cols
    """
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Bundle not found at {BUNDLE_PATH}. "
            "Run: python -m backend.risk.train_anomaly"
        )
    bundle = load_bundle(BUNDLE_PATH)

    # Rebuild pipeline with the same seed – this guarantees the same split.
    # We pass splits_path=None so it regenerates (deterministic via seed).
    pipe = build_pipeline(DATA_DIR, seed=SPLIT_SEED)
    df_scaled    = pipe["df_scaled"]
    feature_cols = pipe["feature_cols"]

    # Verify feature columns match the bundle
    if feature_cols != bundle.feature_cols:
        raise RuntimeError(
            f"Bundle feature_cols {bundle.feature_cols} != "
            f"pipeline feature_cols {feature_cols}. "
            "Bundle may have been trained on different data."
        )

    fit_ids = bundle.splits["fit"]

    # Verify no calibration/test engines are present in fit_ids
    cal_ids  = set(bundle.splits["calibration"])
    test_ids = set(bundle.splits["test"])
    fit_set  = set(fit_ids)
    assert fit_set.isdisjoint(cal_ids),  "Leakage: cal engine in fit split"
    assert fit_set.isdisjoint(test_ids), "Leakage: test engine in fit split"

    return bundle, df_scaled, fit_ids, feature_cols


# ---------------------------------------------------------------------------
# Step 3: Score all fit engines
# ---------------------------------------------------------------------------

def score_all_fit_engines(
    bundle,
    df_scaled: pd.DataFrame,
    fit_ids: list[str],
    feature_cols: list[str],
) -> pd.DataFrame:
    """Score every cycle of all 120 fit engines; retain metadata.

    Returns DataFrame with columns:
        engine_id, dataset, cycle, rul, ocsvm_score, iforest_score,
        in_healthy_window, in_late_life
    """
    records = []
    for eng_id in fit_ids:
        eng_df = df_scaled[df_scaled["engine_id"] == eng_id].sort_values("cycle")

        # Sanity checks per engine
        cycles = eng_df["cycle"].values
        ruls   = eng_df["rul"].values
        assert cycles[0] == 1,          f"{eng_id}: cycle does not start at 1"
        assert ruls[-1] == 0,           f"{eng_id}: last cycle RUL != 0"
        assert (np.diff(ruls) == -1).all(), f"{eng_id}: RUL not monotone -1/cycle"
        assert (np.diff(cycles) == 1).all(),f"{eng_id}: cycles not contiguous"

        scores = score_engine(
            df_scaled, eng_id,
            bundle.ocsvm, bundle.iforest, feature_cols,
        )

        # Flags
        max_cycle  = cycles[-1]
        # late-life: last LATE_LIFE_N cycles by RUL (rul < LATE_LIFE_N)
        in_late   = scores["rul"] < LATE_LIFE_N
        in_healthy = scores["cycle"] <= HEALTHY_N

        scores["engine_id"]        = eng_id
        scores["dataset"]          = eng_id.split("_")[0]
        scores["in_healthy_window"] = in_healthy
        scores["in_late_life"]      = in_late

        records.append(scores)

    all_scores = pd.concat(records, ignore_index=True)

    # Verify no cal/test engines slipped in
    assert set(all_scores["engine_id"].unique()).issubset(set(fit_ids))

    return all_scores


# ---------------------------------------------------------------------------
# Step 4: Healthy vs late-life separation per engine
# ---------------------------------------------------------------------------

def compute_separation(all_scores: pd.DataFrame) -> pd.DataFrame:
    """Compute healthy/late-life metrics per engine for both models.

    Returns one row per engine with columns for each model:
        engine_id, dataset,
        ocsvm_healthy_mean, ocsvm_late_mean, ocsvm_sep,
        ocsvm_norm_sep, ocsvm_directional_success,
        iforest_healthy_mean, iforest_late_mean, iforest_sep,
        iforest_norm_sep, iforest_directional_success
    """
    rows = []
    for eng_id, grp in all_scores.groupby("engine_id"):
        healthy = grp[grp["in_healthy_window"]]
        late    = grp[grp["in_late_life"]]

        row = {"engine_id": eng_id, "dataset": grp["dataset"].iloc[0]}

        for model in ("ocsvm", "iforest"):
            col = f"{model}_score"
            h_vals = healthy[col].values
            l_vals = late[col].values

            if len(h_vals) == 0 or len(l_vals) == 0:
                # Shouldn't happen with 30-cycle windows on C-MAPSS, but be safe
                row.update({
                    f"{model}_healthy_mean": np.nan,
                    f"{model}_late_mean":    np.nan,
                    f"{model}_sep":          np.nan,
                    f"{model}_norm_sep":     np.nan,
                    f"{model}_dir_success":  np.nan,
                    f"{model}_healthy_std":  np.nan,
                    f"{model}_late_std":     np.nan,
                    f"{model}_healthy_n":    0,
                    f"{model}_late_n":       0,
                })
                continue

            h_mean = h_vals.mean()
            l_mean = l_vals.mean()
            sep    = l_mean - h_mean

            # Pooled std of healthy + late-life observations for this engine
            # Use denominator n-1 (ddof=1); guard against zero variance
            h_std = h_vals.std(ddof=1)
            l_std = l_vals.std(ddof=1)
            n_h, n_l = len(h_vals), len(l_vals)
            pooled_var = (
                ((n_h - 1) * h_std**2 + (n_l - 1) * l_std**2)
                / (n_h + n_l - 2)
            )
            pooled_std = np.sqrt(max(pooled_var, 1e-12))
            norm_sep   = sep / pooled_std

            row.update({
                f"{model}_healthy_mean": h_mean,
                f"{model}_late_mean":    l_mean,
                f"{model}_sep":          sep,
                f"{model}_norm_sep":     norm_sep,
                f"{model}_dir_success":  float(l_mean > h_mean),
                f"{model}_healthy_std":  h_std,
                f"{model}_late_std":     l_std,
                f"{model}_healthy_n":    n_h,
                f"{model}_late_n":       n_l,
            })

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_separation(sep_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate separation metrics by dataset and overall."""
    summaries = {}
    for scope, mask in [
        ("overall", pd.Series([True] * len(sep_df), index=sep_df.index)),
        ("FD001",   sep_df["dataset"] == "FD001"),
        ("FD003",   sep_df["dataset"] == "FD003"),
    ]:
        subset = sep_df[mask]
        records = []
        for model in ("ocsvm", "iforest"):
            for metric in ("sep", "norm_sep", "dir_success"):
                col = f"{model}_{metric}"
                vals = subset[col].dropna()
                records.append({
                    "model":    model,
                    "metric":   metric,
                    "scope":    scope,
                    "mean":     vals.mean(),
                    "median":   vals.median(),
                    "std":      vals.std(ddof=1),
                    "q25":      vals.quantile(0.25),
                    "q75":      vals.quantile(0.75),
                    "iqr":      vals.quantile(0.75) - vals.quantile(0.25),
                    "n":        len(vals),
                })
        summaries[scope] = pd.DataFrame(records)
    return summaries


# ---------------------------------------------------------------------------
# Step 5: Spearman correlation with degradation stage
# ---------------------------------------------------------------------------

def compute_spearman(all_scores: pd.DataFrame) -> pd.DataFrame:
    """Per-engine Spearman correlation between anomaly score and -RUL.

    Degradation stage = -RUL (larger → closer to failure).
    Positive correlation = score increases as engine degrades = good.

    Returns DataFrame with one row per engine, columns:
        engine_id, dataset, ocsvm_rho, iforest_rho,
        ocsvm_undefined, iforest_undefined
    """
    rows = []
    for eng_id, grp in all_scores.groupby("engine_id"):
        grp_sorted  = grp.sort_values("cycle")
        neg_rul     = -grp_sorted["rul"].values

        row = {"engine_id": eng_id, "dataset": grp_sorted["dataset"].iloc[0]}
        for model in ("ocsvm", "iforest"):
            scores = grp_sorted[f"{model}_score"].values
            # Spearman undefined if score is constant
            if np.std(scores) < 1e-12:
                row[f"{model}_rho"]       = np.nan
                row[f"{model}_undefined"] = True
            else:
                rho, _ = stats.spearmanr(neg_rul, scores)
                row[f"{model}_rho"]       = rho
                row[f"{model}_undefined"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_spearman(rho_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate Spearman metrics by dataset and overall."""
    summaries = {}
    for scope, mask in [
        ("overall", pd.Series([True] * len(rho_df), index=rho_df.index)),
        ("FD001",   rho_df["dataset"] == "FD001"),
        ("FD003",   rho_df["dataset"] == "FD003"),
    ]:
        subset = rho_df[mask]
        records = []
        for model in ("ocsvm", "iforest"):
            col     = f"{model}_rho"
            undef   = f"{model}_undefined"
            valid   = subset[~subset[undef]][col]
            n_undef = subset[undef].sum()
            records.append({
                "model":           model,
                "scope":           scope,
                "mean_rho":        valid.mean(),
                "median_rho":      valid.median(),
                "std_rho":         valid.std(ddof=1),
                "q25_rho":         valid.quantile(0.25),
                "q75_rho":         valid.quantile(0.75),
                "iqr_rho":         valid.quantile(0.75) - valid.quantile(0.25),
                "pct_positive":    (valid > 0).mean(),
                "n_valid":         len(valid),
                "n_undefined":     int(n_undef),
            })
        summaries[scope] = pd.DataFrame(records)
    return summaries


# ---------------------------------------------------------------------------
# Step 6: Early-warning analysis (leave-one-engine-out threshold)
# ---------------------------------------------------------------------------

def compute_early_warning(
    all_scores: pd.DataFrame,
    fit_ids: list[str],
    percentile: float = 95.0,
    min_persist: int = MIN_WARN_PERSIST,
) -> pd.DataFrame:
    """Leave-one-engine-out early-warning analysis within the fit split.

    Threshold procedure (leakage-free):
    For each target engine i, the threshold is the `percentile`-th percentile
    of healthy-window (cycle ≤ HEALTHY_N) scores across all OTHER fit engines.
    The engine under evaluation is never used to set its own threshold.

    A warning is issued when `min_persist` consecutive cycles exceed the
    threshold. Lead time = cycle count from first warning until failure.

    Returns one row per engine × model with:
        engine_id, dataset, model, threshold,
        warned (bool), first_warn_cycle, last_cycle, lead_time_cycles,
        false_alarm_rate (fraction of healthy cycles above threshold)
    """
    # Pre-build lookup: healthy scores per engine
    healthy_ocsvm  = {}
    healthy_iforest = {}
    for eng_id, grp in all_scores.groupby("engine_id"):
        h = grp[grp["in_healthy_window"]]
        healthy_ocsvm[eng_id]   = h["ocsvm_score"].values
        healthy_iforest[eng_id] = h["iforest_score"].values

    fit_set = set(fit_ids)
    rows = []

    for eng_id in fit_ids:
        eng_data = all_scores[all_scores["engine_id"] == eng_id].sort_values("cycle")
        last_cycle = eng_data["cycle"].max()
        ds = eng_id.split("_")[0]

        for model, h_lookup in [
            ("ocsvm",   healthy_ocsvm),
            ("iforest", healthy_iforest),
        ]:
            score_col = f"{model}_score"

            # Pool healthy scores from all OTHER fit engines
            other_healthy = np.concatenate([
                v for k, v in h_lookup.items()
                if k != eng_id and k in fit_set
            ])
            if len(other_healthy) == 0:
                continue
            threshold = np.percentile(other_healthy, percentile)

            # False alarm: healthy window cycles above threshold
            healthy_cycles = eng_data[eng_data["in_healthy_window"]]
            fa_n   = (healthy_cycles[score_col] > threshold).sum()
            fa_tot = len(healthy_cycles)
            fa_rate = fa_n / fa_tot if fa_tot > 0 else np.nan

            # Warning detection: min_persist consecutive cycles above threshold
            scores_ts = eng_data[score_col].values
            cycles_ts = eng_data["cycle"].values
            above     = scores_ts > threshold

            # Find first run of min_persist consecutive True values
            first_warn_cycle = None
            run = 0
            for i, val in enumerate(above):
                if val:
                    run += 1
                    if run >= min_persist:
                        # First warn cycle is the start of this run
                        first_warn_cycle = int(cycles_ts[i - min_persist + 1])
                        break
                else:
                    run = 0

            warned      = first_warn_cycle is not None
            lead_time   = (last_cycle - first_warn_cycle) if warned else np.nan

            rows.append({
                "engine_id":          eng_id,
                "dataset":            ds,
                "model":              model,
                "threshold":          threshold,
                "warned":             warned,
                "first_warn_cycle":   first_warn_cycle,
                "last_cycle":         last_cycle,
                "lead_time_cycles":   lead_time,
                "false_alarm_rate":   fa_rate,
            })

    return pd.DataFrame(rows)


def summarize_early_warning(ew_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate early-warning metrics by model × dataset and overall."""
    summaries = {}
    for scope, mask in [
        ("overall", pd.Series([True] * len(ew_df), index=ew_df.index)),
        ("FD001",   ew_df["dataset"] == "FD001"),
        ("FD003",   ew_df["dataset"] == "FD003"),
    ]:
        subset = ew_df[mask]
        records = []
        for model in ("ocsvm", "iforest"):
            m = subset[subset["model"] == model]
            warned     = m["warned"]
            lead_valid = m.loc[m["warned"], "lead_time_cycles"]
            fa_valid   = m["false_alarm_rate"].dropna()
            records.append({
                "model":              model,
                "scope":              scope,
                "n_engines":          len(m),
                "pct_warned":         warned.mean(),
                "lead_mean":          lead_valid.mean(),
                "lead_median":        lead_valid.median(),
                "lead_std":           lead_valid.std(ddof=1),
                "lead_q25":           lead_valid.quantile(0.25),
                "lead_q75":           lead_valid.quantile(0.75),
                "mean_fa_rate":       fa_valid.mean(),
                "median_fa_rate":     fa_valid.median(),
                "missed_detections":  int((~warned).sum()),
            })
        summaries[scope] = pd.DataFrame(records)
    return summaries


# ---------------------------------------------------------------------------
# Step 7: Statistical comparison (bootstrap, paired)
# ---------------------------------------------------------------------------

def bootstrap_paired_comparison(
    sep_df: pd.DataFrame,
    rho_df: pd.DataFrame,
    ew_df: pd.DataFrame,
    n_boot: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    """Engine-level bootstrap for paired OC-SVM vs IForest comparison.

    Metrics compared (higher = better for all):
      - sep:       score separation (late - healthy mean)
      - norm_sep:  normalised separation (Cohen's d-like)
      - rho:       Spearman correlation with degradation
      - lead_time: early-warning lead time (warned engines only)

    Bootstrap resamples engines with replacement.

    Returns a DataFrame with columns:
        metric, ocsvm_mean, iforest_mean, diff_mean (ocsvm-iforest),
        diff_ci_lo, diff_ci_hi, p_value (fraction of bootstrap diffs <= 0 or >= 0)
    """
    rng = np.random.default_rng(seed)

    # Merge all per-engine metrics into one aligned table
    base = sep_df[["engine_id", "ocsvm_sep", "iforest_sep",
                    "ocsvm_norm_sep", "iforest_norm_sep"]].copy()
    rho_sub = rho_df[["engine_id", "ocsvm_rho", "iforest_rho"]]
    base = base.merge(rho_sub, on="engine_id", how="inner")

    # Lead time: pivot ew_df to wide
    ew_wide = ew_df[ew_df["warned"]].pivot_table(
        index="engine_id", columns="model", values="lead_time_cycles"
    ).reset_index()
    if "ocsvm" in ew_wide.columns and "iforest" in ew_wide.columns:
        ew_wide = ew_wide.rename(columns={
            "ocsvm": "ocsvm_lead", "iforest": "iforest_lead"
        })
        base = base.merge(ew_wide[["engine_id", "ocsvm_lead", "iforest_lead"]],
                          on="engine_id", how="left")
    else:
        base["ocsvm_lead"]   = np.nan
        base["iforest_lead"] = np.nan

    n = len(base)
    results = []

    for metric, oc_col, if_col in [
        ("sep",       "ocsvm_sep",     "iforest_sep"),
        ("norm_sep",  "ocsvm_norm_sep","iforest_norm_sep"),
        ("rho",       "ocsvm_rho",     "iforest_rho"),
        ("lead_time", "ocsvm_lead",    "iforest_lead"),
    ]:
        oc_vals = base[oc_col].values.astype(float)
        if_vals = base[if_col].values.astype(float)

        # Only use rows where both values are finite
        mask    = np.isfinite(oc_vals) & np.isfinite(if_vals)
        oc_v    = oc_vals[mask]
        if_v    = if_vals[mask]
        if len(oc_v) < 2:
            continue

        obs_diff = oc_v.mean() - if_v.mean()

        diffs = []
        for _ in range(n_boot):
            idx      = rng.integers(0, len(oc_v), len(oc_v))
            diffs.append(oc_v[idx].mean() - if_v[idx].mean())
        diffs = np.array(diffs)

        ci_lo = np.percentile(diffs, 2.5)
        ci_hi = np.percentile(diffs, 97.5)
        # Two-sided p-value: fraction of bootstrap samples on the opposite side
        if obs_diff >= 0:
            p_val = (diffs <= 0).mean()
        else:
            p_val = (diffs >= 0).mean()
        p_val = float(min(2 * p_val, 1.0))   # two-sided

        results.append({
            "metric":       metric,
            "ocsvm_mean":   oc_v.mean(),
            "iforest_mean": if_v.mean(),
            "diff_mean":    obs_diff,
            "diff_ci_lo":   ci_lo,
            "diff_ci_hi":   ci_hi,
            "p_value":      p_val,
            "n_engines":    int(mask.sum()),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Step 6 (plots): Visualisations
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> pathlib.Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    p = PLOTS_DIR / name
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_score_trajectories(
    all_scores: pd.DataFrame,
    fit_ids: list[str],
    n_per_ds: int = 6,
) -> list[pathlib.Path]:
    """Multi-panel score-vs-RUL for a representative subset of fit engines."""
    saved = []
    for ds in ("FD001", "FD003"):
        eng_list = sorted(e for e in fit_ids if e.startswith(ds))
        # evenly spaced sample
        indices  = np.linspace(0, len(eng_list) - 1, min(n_per_ds, len(eng_list)),
                               dtype=int)
        sample   = [eng_list[i] for i in indices]

        n_cols = 3
        n_rows = int(np.ceil(len(sample) / n_cols)) * 2  # 2 rows per engine col
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 3 * n_rows),
            sharex=False, sharey=False,
        )
        axes = np.array(axes).reshape(n_rows, n_cols)

        for col_i, eng_id in enumerate(sample):
            eng = all_scores[all_scores["engine_id"] == eng_id].sort_values("cycle")
            rul = eng["rul"].values
            for row_off, (model, color) in enumerate([
                ("ocsvm",   "steelblue"),
                ("iforest", "darkorange"),
            ]):
                ax = axes[col_i // n_cols * 2 + row_off, col_i % n_cols]
                sc = eng[f"{model}_score"].values
                ax.plot(rul, sc, lw=1.0, color=color, alpha=0.85)
                ax.invert_xaxis()
                ax.axvline(0, color="red", lw=1.2, ls=":", alpha=0.7)
                # shade healthy window
                max_rul = rul.max()
                # healthy window: high RUL end (early cycles)
                hw_rul = rul[eng["cycle"].values <= HEALTHY_N]
                if len(hw_rul):
                    ax.axvspan(hw_rul.min(), hw_rul.max(), alpha=0.10,
                               color="green")
                ax.axhline(0, color="grey", lw=0.6, ls="--")
                ax.set_title(f"{eng_id}\n{model.upper()}", fontsize=7)
                ax.set_xlabel("RUL", fontsize=6)
                ax.set_ylabel("score\n(↑ anomalous)", fontsize=6)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.25)

        # Hide unused axes
        for idx in range(len(sample) * 2, n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            # adjust for double-row layout
        fig.suptitle(
            f"{ds} – anomaly score vs RUL (green=healthy window, red=failure)",
            fontsize=11,
        )
        plt.tight_layout()
        saved.append(_save(fig, f"{ds}_score_trajectories.png"))
    return saved


def plot_score_distributions(
    all_scores: pd.DataFrame,
) -> pathlib.Path:
    """Violin plots: healthy-window vs late-life score distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model, color_h, color_l in [
        (axes[0], "ocsvm",   "mediumseagreen", "tomato"),
        (axes[1], "iforest", "mediumseagreen", "tomato"),
    ]:
        col = f"{model}_score"
        hw  = all_scores.loc[all_scores["in_healthy_window"],  col].values
        ll  = all_scores.loc[all_scores["in_late_life"],        col].values
        parts = ax.violinplot([hw, ll], positions=[0, 1],
                              showmedians=True, showextrema=True)
        parts["bodies"][0].set_facecolor(color_h)
        parts["bodies"][1].set_facecolor(color_l)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Healthy\n(cycle ≤ 30)", "Late-life\n(RUL < 30)"])
        ax.set_ylabel("Anomaly score (higher = more anomalous)")
        ax.set_title(f"{'OC-SVM' if model == 'ocsvm' else 'Isolation Forest'} "
                     f"score distribution")
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Score distributions: healthy window vs late-life (all fit engines)")
    plt.tight_layout()
    return _save(fig, "score_distributions.png")


def plot_separation_distributions(sep_df: pd.DataFrame) -> pathlib.Path:
    """Per-engine score-separation distributions (paired violin)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"FD001": "steelblue", "FD003": "darkorange"}
    for ax, model in [(axes[0], "ocsvm"), (axes[1], "iforest")]:
        col = f"{model}_sep"
        for ds in ("FD001", "FD003"):
            vals = sep_df.loc[sep_df["dataset"] == ds, col].dropna()
            ax.violinplot(
                vals, positions=[0 if ds == "FD001" else 1],
                showmedians=True,
            )
            ax.scatter(
                [0 if ds == "FD001" else 1] * len(vals),
                vals,
                s=8, alpha=0.4, color=colors[ds], label=ds,
            )
        ax.axhline(0, color="grey", lw=0.8, ls="--", label="zero separation")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["FD001", "FD003"])
        ax.set_ylabel("Late-life mean − healthy mean score")
        ax.set_title(f"{'OC-SVM' if model == 'ocsvm' else 'Isolation Forest'} "
                     "score separation per engine")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Per-engine score separation (positive = late-life scored more anomalous)")
    plt.tight_layout()
    return _save(fig, "separation_distributions.png")


def plot_spearman_distributions(rho_df: pd.DataFrame) -> pathlib.Path:
    """Per-engine Spearman ρ distribution comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"FD001": "steelblue", "FD003": "darkorange"}
    for ax, model in [(axes[0], "ocsvm"), (axes[1], "iforest")]:
        col = f"{model}_rho"
        for ds in ("FD001", "FD003"):
            vals = rho_df.loc[rho_df["dataset"] == ds, col].dropna()
            ax.violinplot(
                vals, positions=[0 if ds == "FD001" else 1],
                showmedians=True,
            )
            ax.scatter(
                [0 if ds == "FD001" else 1] * len(vals),
                vals,
                s=8, alpha=0.4, color=colors[ds], label=ds,
            )
        ax.axhline(0,   color="grey",   lw=0.8, ls="--")
        ax.axhline(0.5, color="green",  lw=0.8, ls=":", alpha=0.6,
                   label="ρ = 0.5 reference")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["FD001", "FD003"])
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Spearman ρ (score vs −RUL)")
        ax.set_title(f"{'OC-SVM' if model == 'ocsvm' else 'Isolation Forest'} "
                     "per-engine Spearman ρ")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(
        "Per-engine Spearman ρ: score vs degradation stage (−RUL).\n"
        "Positive ρ = score rises as engine approaches failure."
    )
    plt.tight_layout()
    return _save(fig, "spearman_distributions.png")


def plot_model_comparison(sep_df: pd.DataFrame, rho_df: pd.DataFrame) -> pathlib.Path:
    """Scatter plots: OC-SVM vs IForest on sep and rho, coloured by dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"FD001": "steelblue", "FD003": "darkorange"}
    merged = sep_df.merge(
        rho_df[["engine_id", "ocsvm_rho", "iforest_rho"]],
        on="engine_id",
    )
    for ds in ("FD001", "FD003"):
        sub = merged[merged["dataset"] == ds]
        axes[0].scatter(
            sub["ocsvm_sep"], sub["iforest_sep"],
            s=18, alpha=0.6, color=colors[ds], label=ds,
        )
        axes[1].scatter(
            sub["ocsvm_rho"], sub["iforest_rho"],
            s=18, alpha=0.6, color=colors[ds], label=ds,
        )

    for ax, xlab, ylab, title in [
        (axes[0],
         "OC-SVM separation",
         "IForest separation",
         "Score separation: OC-SVM vs IForest"),
        (axes[1],
         "OC-SVM Spearman ρ",
         "IForest Spearman ρ",
         "Spearman ρ: OC-SVM vs IForest"),
    ]:
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5, label="y = x")
        ax.axhline(0, color="grey", lw=0.6, ls=":")
        ax.axvline(0, color="grey", lw=0.6, ls=":")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Model comparison: FD001 (blue) vs FD003 (orange)")
    plt.tight_layout()
    return _save(fig, "model_comparison.png")


def plot_early_warning_summary(ew_df: pd.DataFrame) -> pathlib.Path:
    """Lead-time and false-alarm-rate comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"FD001": "steelblue", "FD003": "darkorange"}

    warned = ew_df[ew_df["warned"]]

    for ax, metric, ylabel in [
        (axes[0], "lead_time_cycles", "Lead time (cycles before failure)"),
        (axes[1], "false_alarm_rate", "False alarm rate (fraction of healthy cycles)"),
    ]:
        src = warned if metric == "lead_time_cycles" else ew_df
        positions = {"ocsvm": {"FD001": 0, "FD003": 1},
                     "iforest": {"FD001": 3, "FD003": 4}}
        for model in ("ocsvm", "iforest"):
            for ds in ("FD001", "FD003"):
                sub  = src[(src["model"] == model) & (src["dataset"] == ds)]
                vals = sub[metric].dropna().values
                pos  = positions[model][ds]
                if len(vals):
                    ax.boxplot(
                        vals, positions=[pos], widths=0.6,
                        patch_artist=True,
                        boxprops=dict(facecolor=colors[ds], alpha=0.6),
                        medianprops=dict(color="black", lw=1.5),
                        whiskerprops=dict(lw=1.0),
                        capprops=dict(lw=1.0),
                        flierprops=dict(marker=".", ms=3, alpha=0.3),
                    )
        ax.set_xticks([0.5, 3.5])
        ax.set_xticklabels(["OC-SVM", "IForest"])
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_title("Lead time (warned engines only)")
    axes[1].set_title("False alarm rate (all engines, healthy window)")
    # legend proxy
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[ds], label=ds) for ds in ("FD001", "FD003")]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle(
        f"Early-warning summary (LOO threshold at 95th pct, "
        f"min {MIN_WARN_PERSIST}-cycle persistence)"
    )
    plt.tight_layout()
    return _save(fig, "early_warning_summary.png")


# ---------------------------------------------------------------------------
# Pooled Spearman (descriptive only, explicitly warned)
# ---------------------------------------------------------------------------

def compute_pooled_spearman(all_scores: pd.DataFrame) -> dict[str, float]:
    """Pooled cycle-level Spearman correlation (descriptive only).

    WARNING: pooled correlation mixes within-engine trends with between-engine
    differences in lifetime. It can be inflated by engines with long vs short
    lifetimes and must NOT substitute per-engine analysis for model selection.
    """
    neg_rul = -all_scores["rul"].values
    results = {}
    for model in ("ocsvm", "iforest"):
        rho, pval = stats.spearmanr(neg_rul, all_scores[f"{model}_score"].values)
        results[f"{model}_pooled_rho"] = rho
        results[f"{model}_pooled_pval"] = pval
    return results


# ---------------------------------------------------------------------------
# False-alarm threshold reporting (fit healthy data only)
# ---------------------------------------------------------------------------

def compute_healthy_threshold_stats(
    all_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Report score distributions in healthy window to characterise
    false-alarm behaviour without inventing a fixed threshold.

    The LOO threshold used in early-warning already quantifies this properly.
    This function provides additional distributional summaries.
    """
    hw = all_scores[all_scores["in_healthy_window"]]
    rows = []
    for model in ("ocsvm", "iforest"):
        col  = f"{model}_score"
        vals = hw[col].values
        rows.append({
            "model":  model,
            "mean":   vals.mean(),
            "std":    vals.std(ddof=1),
            "p50":    np.percentile(vals, 50),
            "p90":    np.percentile(vals, 90),
            "p95":    np.percentile(vals, 95),
            "p99":    np.percentile(vals, 99),
            "pct_above_zero": (vals > 0).mean(),
            "n":      len(vals),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    all_scores:    pd.DataFrame,
    sep_df:        pd.DataFrame,
    rho_df:        pd.DataFrame,
    ew_df:         pd.DataFrame,
    sep_summaries: dict,
    rho_summaries: dict,
    ew_summaries:  dict,
    boot_df:       pd.DataFrame,
    hw_thresh_df:  pd.DataFrame,
) -> list[pathlib.Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    def _save_csv(df: pd.DataFrame, name: str) -> pathlib.Path:
        p = RESULTS_DIR / name
        df.to_csv(p, index=False)
        saved.append(p)
        return p

    _save_csv(all_scores, "all_fit_scores.csv")
    _save_csv(sep_df,     "per_engine_separation.csv")
    _save_csv(rho_df,     "per_engine_spearman.csv")
    _save_csv(ew_df,      "per_engine_early_warning.csv")
    _save_csv(boot_df,    "bootstrap_comparison.csv")
    _save_csv(hw_thresh_df, "healthy_window_score_stats.csv")

    for scope, df in sep_summaries.items():
        _save_csv(df, f"summary_separation_{scope}.csv")
    for scope, df in rho_summaries.items():
        _save_csv(df, f"summary_spearman_{scope}.csv")
    for scope, df in ew_summaries.items():
        _save_csv(df, f"summary_early_warning_{scope}.csv")

    return saved


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run() -> None:
    """Execute the full diagnostic pipeline."""

    print("[diagnostic] Loading bundle and data ...")
    bundle, df_scaled, fit_ids, feature_cols = load_data_and_bundle()
    print(f"  Bundle config : {bundle.config}")
    print(f"  Feature cols  : {feature_cols}  (n={len(feature_cols)})")
    print(f"  Fit engines   : {len(fit_ids)}"
          f"  (FD001: {sum(1 for e in fit_ids if e.startswith('FD001'))},"
          f"   FD003: {sum(1 for e in fit_ids if e.startswith('FD003'))})")

    print("[diagnostic] Scoring all fit engines ...")
    all_scores = score_all_fit_engines(bundle, df_scaled, fit_ids, feature_cols)
    print(f"  Total scored rows : {len(all_scores):,}")

    print("[diagnostic] Computing healthy vs late-life separation ...")
    sep_df = compute_separation(all_scores)
    sep_summaries = summarize_separation(sep_df)

    print("[diagnostic] Computing Spearman correlations ...")
    rho_df = compute_spearman(all_scores)
    rho_summaries = summarize_spearman(rho_df)

    pooled = compute_pooled_spearman(all_scores)
    print(f"  [WARNING] Pooled Spearman (descriptive only, see docstring):")
    print(f"    OC-SVM  ρ={pooled['ocsvm_pooled_rho']:.4f}  "
          f"p={pooled['ocsvm_pooled_pval']:.2e}")
    print(f"    IForest ρ={pooled['iforest_pooled_rho']:.4f}  "
          f"p={pooled['iforest_pooled_pval']:.2e}")

    print("[diagnostic] Running early-warning analysis (LOO) ...")
    ew_df = compute_early_warning(all_scores, fit_ids)
    ew_summaries = summarize_early_warning(ew_df)

    print("[diagnostic] Bootstrap paired comparison ...")
    boot_df = bootstrap_paired_comparison(sep_df, rho_df, ew_df)

    hw_thresh_df = compute_healthy_threshold_stats(all_scores)

    # ------------------------------------------------------------------ #
    # Print key summary tables to stdout
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("SEPARATION SUMMARY (overall)")
    print("=" * 70)
    print(sep_summaries["overall"].to_string(index=False))

    print("\n" + "=" * 70)
    print("SPEARMAN CORRELATION SUMMARY (overall)")
    print("=" * 70)
    print(rho_summaries["overall"].to_string(index=False))

    print("\n" + "=" * 70)
    print("EARLY-WARNING SUMMARY (overall)")
    print("=" * 70)
    print(ew_summaries["overall"].to_string(index=False))

    print("\n" + "=" * 70)
    print("BOOTSTRAP COMPARISON (OC-SVM minus IForest, positive = OC-SVM better)")
    print("=" * 70)
    print(boot_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("HEALTHY-WINDOW SCORE STATS (fit data only, threshold context)")
    print("=" * 70)
    print(hw_thresh_df.to_string(index=False))

    # FD001 vs FD003 breakdown
    for scope in ("FD001", "FD003"):
        print(f"\n--- Separation: {scope} ---")
        print(sep_summaries[scope].to_string(index=False))
        print(f"\n--- Spearman: {scope} ---")
        print(rho_summaries[scope].to_string(index=False))
        print(f"\n--- Early-warning: {scope} ---")
        print(ew_summaries[scope].to_string(index=False))

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    print("\n[diagnostic] Generating plots ...")
    traj_paths = plot_score_trajectories(all_scores, fit_ids)
    dist_path  = plot_score_distributions(all_scores)
    sep_path   = plot_separation_distributions(sep_df)
    rho_path   = plot_spearman_distributions(rho_df)
    cmp_path   = plot_model_comparison(sep_df, rho_df)
    ew_path    = plot_early_warning_summary(ew_df)

    print(f"  Trajectory plots : {[str(p) for p in traj_paths]}")
    print(f"  Distributions    : {dist_path}")
    print(f"  Separation       : {sep_path}")
    print(f"  Spearman         : {rho_path}")
    print(f"  Model comparison : {cmp_path}")
    print(f"  Early warning    : {ew_path}")

    # ------------------------------------------------------------------ #
    # Save CSVs
    # ------------------------------------------------------------------ #
    print("\n[diagnostic] Saving CSV results ...")
    saved_csvs = save_results(
        all_scores, sep_df, rho_df, ew_df,
        sep_summaries, rho_summaries, ew_summaries,
        boot_df, hw_thresh_df,
    )
    for p in saved_csvs:
        print(f"  {p}")

    print("\n[diagnostic] Done.")


if __name__ == "__main__":
    run()
