"""
Phase 2 anomaly-detection experiments for RAJS.

Three controlled experiments, each producing separate artifacts:

  Exp A  — Baseline: existing shared model (already trained, loaded from disk).
            FD001+FD003, 18 features, OC-SVM + IForest.

  Exp B  — Separate models, same 18 features.
            Train FD001-only and FD003-only OC-SVM + IForest on their
            respective fit-split engines. Evaluate on matching cal engines.

  Exp C  — Separate models, dataset-specific feature subsets.
            FD001: s11,s4,s17,s2,s3,s9,s12,s15,s7,s20,s21 (11 features)
            FD003: s11,s4,s17,s2,s3,s9,s10,s14,s13,s8   (10 features)

  Exp D  — Temporal features (causal 10-cycle rolling means).
            Rolling means of s11,s4,s17,s3,s2 added to the 18 raw features.
            Separate FD001/FD003 models. Scaler refitted on augmented features.

Evaluation for every configuration:
  - Per-engine Spearman ρ (score vs −RUL)
  - Healthy vs late-life separation (Cohen's d)
  - Directional success
  - Near-failure score overlap
  - Early-warning (LOO, 95th-pct threshold, 3-cycle persistence)
  - Anomaly score distributions

All comparisons are paired at the engine level against the Exp A baseline.

Leakage safeguards
------------------
- Existing bundle loaded; never overwritten.
- New models trained on fit-split engines only.
- Calibration engines used only for evaluation, never for training.
- Test engines never touched.
- Rolling means use causal windows (groupby engine → rolling, min_periods=1).
- Scalers for temporal experiments fitted on fit-split rows only.

Outputs
-------
models/experiments/
    anomaly_bundle_B_fd001.pkl
    anomaly_bundle_B_fd003.pkl
    anomaly_bundle_C_fd001.pkl
    anomaly_bundle_C_fd003.pkl
    anomaly_bundle_D_fd001.pkl
    anomaly_bundle_D_fd003.pkl

results/anomaly_experiments/
    scores_A_{fd001,fd003}.csv
    scores_B_{fd001,fd003}.csv
    scores_C_{fd001,fd003}.csv
    scores_D_{fd001,fd003}.csv
    metrics_per_engine_*.csv
    summary_spearman.csv
    summary_separation.csv
    summary_early_warning.csv
    score_overlap.csv
    bootstrap_comparison.csv
    phase2_anomaly_experiments.md

plots/anomaly_experiments/
    score_trajectories_*.png
    spearman_comparison.png
    separation_comparison.png
    early_warning_comparison.png
    score_overlap_comparison.png
    score_distributions_*.png
"""

from __future__ import annotations

import dataclasses
import pathlib
import pickle
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler

from backend.risk.anomaly_models import (
    DEFAULT_CONFIG,
    AnomalyBundle,
    AnomalyConfig,
    save_bundle,
    load_bundle,
    score_ocsvm,
    score_iforest,
    select_healthy_cycles,
    train_ocsvm,
    train_iforest,
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
BASELINE_PKL = REPO_ROOT / "models" / "anomaly_bundle.pkl"
EXP_MODELS   = REPO_ROOT / "models" / "experiments"
RESULTS_DIR  = REPO_ROOT / "results" / "anomaly_experiments"
PLOTS_DIR    = REPO_ROOT / "plots"   / "anomaly_experiments"

HEALTHY_N     = 30
LATE_LIFE_N   = 30
EW_PERCENTILE = 95.0
EW_PERSIST    = 3

# Feature subsets (Experiment C)
FEATURES_C_FD001 = ["s11","s4","s17","s2","s3","s9","s12","s15","s7","s20","s21"]
FEATURES_C_FD003 = ["s11","s4","s17","s2","s3","s9","s10","s14","s13","s8"]

# Rolling-mean sensors (Experiment D)
ROLL_SENSORS = ["s11","s4","s17","s3","s2"]
ROLL_WINDOW  = 10


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
# Rolling-mean feature construction (Experiment D)
# ---------------------------------------------------------------------------

def add_rolling_means(
    df: pd.DataFrame,
    sensors: list[str],
    window: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Add causal rolling-mean columns, computed per engine.

    Returns (augmented_df, list_of_new_col_names).
    Window uses min_periods=1 so cycle-1 gets a valid value.
    No future cycles are used (default rolling is backward-looking).
    """
    new_cols: dict[str, pd.Series] = {}
    rm_names: list[str] = []
    for s in sensors:
        if s not in df.columns:
            continue
        name = f"{s}_rm{window}"
        new_cols[name] = (
            df.groupby("engine_id")[s]
            .transform(lambda x, _w=window: x.rolling(_w, min_periods=1).mean())
        )
        rm_names.append(name)
    out = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return out, rm_names


# ---------------------------------------------------------------------------
# Experiment-specific pipeline builder
# ---------------------------------------------------------------------------

def build_exp_pipeline(
    df_raw: pd.DataFrame,
    splits: dict,
    feature_cols: list[str],
    dataset_filter: str | None = None,
    extra_cols: list[str] | None = None,
) -> dict:
    """Build a scaled DataFrame for a subset of features and/or one dataset.

    Parameters
    ----------
    df_raw        : raw (unscaled, constant-cols-dropped) DataFrame
    splits        : existing engine split dict
    feature_cols  : feature columns to use (must be present in df_raw)
    dataset_filter: 'FD001', 'FD003', or None (use both)
    extra_cols    : additional columns already in df_raw (e.g. rolling means)

    Returns
    -------
    dict with keys: df_scaled, fit_ids, cal_ids, feature_cols, scaler
    """
    all_fc = feature_cols + (extra_cols or [])

    fit_ids = splits["fit"]
    cal_ids = splits["calibration"]

    if dataset_filter is not None:
        fit_ids = [e for e in fit_ids if e.startswith(dataset_filter)]
        cal_ids = [e for e in cal_ids if e.startswith(dataset_filter)]

    # Fit scaler on fit-split rows only, using selected feature columns
    fit_mask = df_raw["engine_id"].isin(fit_ids)
    scaler   = StandardScaler()
    scaler.fit(df_raw.loc[fit_mask, all_fc].values)

    df_scaled = df_raw.copy()
    df_scaled[all_fc] = scaler.transform(df_raw[all_fc].values)

    return {
        "df_scaled":   df_scaled,
        "fit_ids":     fit_ids,
        "cal_ids":     cal_ids,
        "feature_cols": all_fc,
        "scaler":      scaler,
    }


# ---------------------------------------------------------------------------
# Generic training function (used by B, C, D)
# ---------------------------------------------------------------------------

def train_experiment_bundle(
    df_scaled:    pd.DataFrame,
    fit_ids:      list[str],
    cal_ids:      list[str],
    feature_cols: list[str],
    config:       AnomalyConfig,
    splits_full:  dict,
) -> AnomalyBundle:
    """Train OC-SVM + IForest on given fit engines and feature set."""
    X_healthy = select_healthy_cycles(
        df_scaled, fit_ids, config.healthy_max_cycle, feature_cols
    )
    ocsvm   = train_ocsvm(X_healthy, config)
    iforest = train_iforest(X_healthy, config)

    # Store full splits for provenance but training used only fit_ids subset
    bundle = AnomalyBundle(
        ocsvm=ocsvm,
        iforest=iforest,
        feature_cols=feature_cols,
        config=config,
        splits=splits_full,
    )
    return bundle


# ---------------------------------------------------------------------------
# Scoring all engines in a split
# ---------------------------------------------------------------------------

def score_engines(
    df_scaled:    pd.DataFrame,
    engine_ids:   list[str],
    bundle:       AnomalyBundle,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Score every cycle of every engine; return long-format DataFrame."""
    fc = feature_cols or bundle.feature_cols
    records = []
    for eng_id in engine_ids:
        eng = df_scaled[df_scaled["engine_id"] == eng_id].sort_values("cycle")
        if len(eng) == 0:
            continue
        X  = eng[fc].values
        oc = score_ocsvm(bundle.ocsvm, X)
        if_ = score_iforest(bundle.iforest, X)
        chunk = pd.DataFrame({
            "engine_id":    eng_id,
            "dataset":      eng_id.split("_")[0],
            "cycle":        eng["cycle"].values,
            "rul":          eng["rul"].values,
            "ocsvm_score":  oc,
            "iforest_score": if_,
        })
        chunk["in_healthy_window"] = chunk["cycle"] <= HEALTHY_N
        chunk["in_late_life"]      = chunk["rul"]   <  LATE_LIFE_N
        records.append(chunk)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_spearman(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Per-engine Spearman ρ for OC-SVM and IForest."""
    rows = []
    for eng_id, grp in scores_df.groupby("engine_id"):
        grp = grp.sort_values("cycle")
        neg_rul = -grp["rul"].values
        row = {"engine_id": eng_id, "dataset": grp["dataset"].iloc[0]}
        for model in ("ocsvm", "iforest"):
            v = grp[f"{model}_score"].values
            if np.std(v) < 1e-12:
                row[f"{model}_rho"] = np.nan
            else:
                rho, _ = sp_stats.spearmanr(neg_rul, v)
                row[f"{model}_rho"] = rho
        rows.append(row)
    return pd.DataFrame(rows)


def compute_separation(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Per-engine healthy vs late-life separation (Cohen's d-like)."""
    rows = []
    for eng_id, grp in scores_df.groupby("engine_id"):
        h = grp[grp["in_healthy_window"]]
        l = grp[grp["in_late_life"]]
        row = {"engine_id": eng_id, "dataset": grp["dataset"].iloc[0]}
        for model in ("ocsvm", "iforest"):
            col = f"{model}_score"
            hv, lv = h[col].values, l[col].values
            if len(hv) < 2 or len(lv) < 2:
                row[f"{model}_norm_sep"] = np.nan
                row[f"{model}_dir_success"] = np.nan
                continue
            sep = lv.mean() - hv.mean()
            n_h, n_l = len(hv), len(lv)
            pooled = np.sqrt(
                ((n_h-1)*hv.std(ddof=1)**2 + (n_l-1)*lv.std(ddof=1)**2)
                / (n_h + n_l - 2)
            )
            row[f"{model}_norm_sep"]    = sep / max(pooled, 1e-12)
            row[f"{model}_dir_success"] = float(lv.mean() > hv.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def compute_early_warning(
    scores_df: pd.DataFrame,
    engine_ids: list[str],
    percentile: float = EW_PERCENTILE,
    min_persist: int  = EW_PERSIST,
) -> pd.DataFrame:
    """LOO early-warning analysis — threshold from all other engines."""
    healthy_scores: dict[str, dict[str, np.ndarray]] = {
        "ocsvm": {}, "iforest": {}
    }
    for eng_id, grp in scores_df.groupby("engine_id"):
        h = grp[grp["in_healthy_window"]]
        for model in ("ocsvm", "iforest"):
            healthy_scores[model][eng_id] = h[f"{model}_score"].values

    eng_set = set(engine_ids)
    rows = []
    for eng_id in engine_ids:
        grp = scores_df[scores_df["engine_id"] == eng_id].sort_values("cycle")
        last_cycle = int(grp["cycle"].max())
        ds = eng_id.split("_")[0]
        for model in ("ocsvm", "iforest"):
            others = np.concatenate([
                v for k, v in healthy_scores[model].items()
                if k != eng_id and k in eng_set
            ])
            if len(others) == 0:
                continue
            thr = np.percentile(others, percentile)

            # false alarm rate on healthy window
            hw = grp[grp["in_healthy_window"]]
            fa_rate = (hw[f"{model}_score"] > thr).mean() if len(hw) > 0 else np.nan

            # first persistent warning
            sc = grp[f"{model}_score"].values
            cy = grp["cycle"].values
            above = sc > thr
            run, first_warn = 0, None
            for i, flag in enumerate(above):
                run = run + 1 if flag else 0
                if run >= min_persist:
                    first_warn = int(cy[i - min_persist + 1])
                    break

            warned    = first_warn is not None
            lead_time = (last_cycle - first_warn) if warned else np.nan
            rows.append({
                "engine_id": eng_id, "dataset": ds, "model": model,
                "threshold": thr, "warned": warned,
                "first_warn_cycle": first_warn, "last_cycle": last_cycle,
                "lead_time_cycles": lead_time, "false_alarm_rate": fa_rate,
            })
    return pd.DataFrame(rows)


def compute_score_overlap(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of near-failure scores that fall within the early-life range."""
    rows = []
    for ds in ("FD001", "FD003"):
        sub = scores_df[scores_df["dataset"] == ds]
        el = sub[sub["rul"] > 100]
        nf = sub[sub["rul"] <= 20]
        for model in ("ocsvm", "iforest"):
            col = f"{model}_score"
            if len(el) == 0 or len(nf) == 0:
                continue
            lo, hi = el[col].min(), el[col].max()
            overlap = ((nf[col] >= lo) & (nf[col] <= hi)).mean()
            rows.append({
                "dataset": ds, "model": model,
                "el_min": lo, "el_max": hi,
                "nf_mean": nf[col].mean(), "nf_p50": nf[col].median(),
                "overlap_pct": overlap,
                "n_el": len(el), "n_nf": len(nf),
            })
    return pd.DataFrame(rows)


def aggregate_metrics(
    rho_df: pd.DataFrame,
    sep_df: pd.DataFrame,
    ew_df:  pd.DataFrame,
    label:  str,
) -> dict:
    """Return a summary dict for one experiment configuration."""
    out = {"config": label}
    for ds_filter, ds_label in [(None, "overall"), ("FD001", "FD001"), ("FD003", "FD003")]:
        r = rho_df if ds_filter is None else rho_df[rho_df["dataset"] == ds_filter]
        s = sep_df if ds_filter is None else sep_df[sep_df["dataset"] == ds_filter]
        e = ew_df  if ds_filter is None else ew_df[ew_df["dataset"] == ds_filter]
        for model in ("ocsvm", "iforest"):
            pfx = f"{ds_label}_{model}"
            rho_v = r[f"{model}_rho"].dropna()
            sep_v = s[f"{model}_norm_sep"].dropna()
            dir_v = s[f"{model}_dir_success"].dropna()
            ew_m  = e[e["model"] == model]
            lead  = ew_m.loc[ew_m["warned"], "lead_time_cycles"].dropna()
            fa    = ew_m["false_alarm_rate"].dropna()
            out[f"{pfx}_median_rho"]      = rho_v.median()
            out[f"{pfx}_mean_rho"]        = rho_v.mean()
            out[f"{pfx}_pct_pos_rho"]     = (rho_v > 0).mean()
            out[f"{pfx}_median_norm_sep"] = sep_v.median()
            out[f"{pfx}_dir_success"]     = dir_v.mean()
            out[f"{pfx}_pct_warned"]      = ew_m["warned"].mean()
            out[f"{pfx}_lead_median"]     = lead.median()
            out[f"{pfx}_fa_rate_mean"]    = fa.mean()
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_score_trajectories(
    scores_df: pd.DataFrame,
    config_label: str,
    n_sample: int = 4,
) -> list[pathlib.Path]:
    """Score vs RUL for a sample of engines per dataset."""
    saved = []
    colors = {"ocsvm": "steelblue", "iforest": "darkorange"}
    for ds in ("FD001", "FD003"):
        engs = sorted(scores_df[scores_df["dataset"] == ds]["engine_id"].unique())
        idx  = np.linspace(0, len(engs)-1, min(n_sample, len(engs)), dtype=int)
        samp = [engs[i] for i in idx]
        fig, axes = plt.subplots(2, len(samp), figsize=(4.5*len(samp), 6),
                                 sharex=False)
        axes = np.array(axes)
        for col, eng in enumerate(samp):
            grp = scores_df[scores_df["engine_id"] == eng].sort_values("cycle")
            rul = grp["rul"].values
            for row, model in enumerate(("ocsvm", "iforest")):
                ax = axes[row, col]
                ax.plot(rul, grp[f"{model}_score"].values,
                        lw=1.2, color=colors[model], alpha=0.85)
                ax.invert_xaxis()
                ax.axvline(0, color="red", lw=1.0, ls=":", alpha=0.7)
                hw = rul[grp["cycle"].values <= HEALTHY_N]
                if len(hw):
                    ax.axvspan(hw.min(), hw.max(), alpha=0.08, color="green")
                ax.axhline(0, color="grey", lw=0.5, ls="--")
                ax.set_title(f"{eng}\n{model.upper()}", fontsize=7)
                ax.set_xlabel("RUL", fontsize=6)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.2)
        fig.suptitle(f"{config_label} — {ds}: score vs RUL (green=healthy, red=failure)")
        plt.tight_layout()
        fname = f"score_trajectories_{config_label}_{ds}.png"
        saved.append(_save_plot(fig, fname))
    return saved


def plot_spearman_comparison(all_summaries: list[dict]) -> pathlib.Path:
    """Bar chart comparing median Spearman ρ across configs."""
    configs = [s["config"] for s in all_summaries]
    x = np.arange(len(configs))
    w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    offsets  = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
    combos   = [("FD001","ocsvm","steelblue"),("FD001","iforest","steelblue"),
                ("FD003","ocsvm","darkorange"),("FD003","iforest","darkorange")]
    hatches  = ["","///","","///"]
    for ax, metric, ylabel in [
        (axes[0], "median_rho",  "Median Spearman ρ (score vs −RUL)"),
        (axes[1], "dir_success", "Directional success rate"),
    ]:
        for i, ((ds, model, color), hatch, off) in enumerate(zip(combos, hatches, offsets)):
            vals = [s.get(f"{ds}_{model}_{metric}", np.nan) for s in all_summaries]
            ax.bar(x + off, vals, w, label=f"{ds} {model.upper()}",
                   color=color, hatch=hatch, alpha=0.75)
        ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=8)
        ax.set_ylabel(ylabel); ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Spearman correlation comparison across experiment configurations")
    plt.tight_layout()
    return _save_plot(fig, "spearman_comparison.png")


def plot_separation_comparison(all_summaries: list[dict]) -> pathlib.Path:
    """Bar chart comparing median normalised separation across configs."""
    configs  = [s["config"] for s in all_summaries]
    x = np.arange(len(configs))
    w = 0.2
    offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
    combos  = [("FD001","ocsvm","steelblue"),("FD001","iforest","steelblue"),
               ("FD003","ocsvm","darkorange"),("FD003","iforest","darkorange")]
    hatches = ["","///","","///"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for (ds, model, color), hatch, off in zip(combos, hatches, offsets):
        vals = [s.get(f"{ds}_{model}_median_norm_sep", np.nan) for s in all_summaries]
        ax.bar(x + off, vals, w, label=f"{ds} {model.upper()}",
               color=color, hatch=hatch, alpha=0.75)
    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=8)
    ax.set_ylabel("Median normalised separation (Cohen's d-like)")
    ax.set_title("Healthy vs near-failure separation across configurations")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return _save_plot(fig, "separation_comparison.png")


def plot_early_warning_comparison(all_ew: dict[str, pd.DataFrame]) -> pathlib.Path:
    """Box-plot of lead times and FA rates per config × model."""
    configs = list(all_ew.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"ocsvm": "steelblue", "iforest": "darkorange"}
    x_off = {"ocsvm": -0.2, "iforest": 0.2}

    for ax, metric, ylabel in [
        (axes[0], "lead_time_cycles", "Lead time (cycles before failure)"),
        (axes[1], "false_alarm_rate", "False alarm rate"),
    ]:
        xticks, xlabels = [], []
        for ci, cfg in enumerate(configs):
            ew = all_ew[cfg]
            src = ew[ew["warned"]] if metric == "lead_time_cycles" else ew
            for model in ("ocsvm", "iforest"):
                vals = src[src["model"] == model][metric].dropna().values
                xpos = ci + x_off[model]
                if len(vals):
                    bp = ax.boxplot(vals, positions=[xpos], widths=0.35,
                                    patch_artist=True, manage_ticks=False,
                                    boxprops=dict(facecolor=colors[model], alpha=0.6),
                                    medianprops=dict(color="black", lw=1.5),
                                    whiskerprops=dict(lw=0.8),
                                    capprops=dict(lw=0.8),
                                    flierprops=dict(marker=".", ms=3, alpha=0.3))
            xticks.append(ci); xlabels.append(cfg)
        ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[m], label=m.upper()) for m in ("ocsvm","iforest")]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle("Early-warning: lead time and false alarm rate across configurations")
    plt.tight_layout()
    return _save_plot(fig, "early_warning_comparison.png")


def plot_score_overlap_comparison(overlap_rows: list[dict]) -> pathlib.Path:
    """Bar chart: near-failure score overlap with early-life range."""
    df = pd.DataFrame(overlap_rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, ds in zip(axes, ("FD001", "FD003")):
        sub = df[df["dataset"] == ds]
        configs = sub["config"].unique()
        x = np.arange(len(configs))
        w = 0.35
        for i, model in enumerate(("ocsvm", "iforest")):
            m = sub[sub["model"] == model]
            vals = [m[m["config"] == c]["overlap_pct"].values[0]
                    if len(m[m["config"] == c]) > 0 else np.nan for c in configs]
            ax.bar(x + (i - 0.5) * w, vals, w,
                   label=model.upper(),
                   color=["steelblue","darkorange"][i], alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=8)
        ax.set_ylabel("Fraction of near-failure (RUL≤20) scores\ninside early-life (RUL>100) range")
        ax.set_ylim(0, 1.05); ax.set_title(f"{ds}: score overlap (lower = better separation)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Near-failure score overlap with early-life range\n"
                 "(lower = better: near-failure scores are distinct from early-life)")
    plt.tight_layout()
    return _save_plot(fig, "score_overlap_comparison.png")


def plot_score_distributions(
    scores_dict: dict[str, pd.DataFrame],
    ds: str,
) -> pathlib.Path:
    """Violin plots: healthy vs near-failure score distributions for each config."""
    configs = list(scores_dict.keys())
    n = len(configs)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8), sharey=False)
    colors_hw = "mediumseagreen"
    colors_ll = "tomato"

    for col, (cfg, scores) in enumerate(scores_dict.items()):
        sub = scores[scores["dataset"] == ds]
        for row, model in enumerate(("ocsvm", "iforest")):
            ax  = axes[row, col]
            col_name = f"{model}_score"
            hw = sub.loc[sub["in_healthy_window"], col_name].dropna().values
            ll = sub.loc[sub["in_late_life"],       col_name].dropna().values
            if len(hw) > 1 and len(ll) > 1:
                parts = ax.violinplot([hw, ll], positions=[0,1],
                                      showmedians=True, showextrema=True)
                parts["bodies"][0].set_facecolor(colors_hw)
                parts["bodies"][1].set_facecolor(colors_ll)
            ax.set_xticks([0,1])
            ax.set_xticklabels(["Healthy\n(c≤30)","Late\n(RUL<30)"], fontsize=7)
            ax.set_title(f"{cfg}\n{model.upper()}", fontsize=7)
            ax.grid(True, alpha=0.25, axis="y")
            if col == 0:
                ax.set_ylabel(f"{'OC-SVM' if model=='ocsvm' else 'IForest'} score", fontsize=7)

    fig.suptitle(f"{ds}: score distributions (healthy vs late-life) across configs\n"
                 "Green=healthy, Red=late-life")
    plt.tight_layout()
    return _save_plot(fig, f"score_distributions_{ds}.png")


# ---------------------------------------------------------------------------
# Bootstrap paired comparison (configs vs baseline A)
# ---------------------------------------------------------------------------

def bootstrap_compare(
    rho_a:  pd.DataFrame,
    rho_b:  pd.DataFrame,
    sep_a:  pd.DataFrame,
    sep_b:  pd.DataFrame,
    label_a: str,
    label_b: str,
    n_boot:  int = 2000,
    seed:    int = 0,
) -> pd.DataFrame:
    """Engine-level bootstrap for paired comparison of two configurations.

    Positive diff_mean means config_b is better (higher ρ or sep).
    """
    rng = np.random.default_rng(seed)
    # Align on engine_id
    merged = (
        rho_a[["engine_id","ocsvm_rho","iforest_rho"]]
        .rename(columns={"ocsvm_rho":"oc_a","iforest_rho":"if_a"})
        .merge(
            rho_b[["engine_id","ocsvm_rho","iforest_rho"]]
            .rename(columns={"ocsvm_rho":"oc_b","iforest_rho":"if_b"}),
            on="engine_id", how="inner",
        )
    )
    sep_merge = (
        sep_a[["engine_id","ocsvm_norm_sep","iforest_norm_sep"]]
        .rename(columns={"ocsvm_norm_sep":"oc_sep_a","iforest_norm_sep":"if_sep_a"})
        .merge(
            sep_b[["engine_id","ocsvm_norm_sep","iforest_norm_sep"]]
            .rename(columns={"ocsvm_norm_sep":"oc_sep_b","iforest_norm_sep":"if_sep_b"}),
            on="engine_id", how="inner",
        )
    )
    merged = merged.merge(sep_merge, on="engine_id", how="inner")

    rows = []
    for model, col_a, col_b in [
        ("ocsvm_rho",     "oc_a",     "oc_b"),
        ("iforest_rho",   "if_a",     "if_b"),
        ("ocsvm_norm_sep","oc_sep_a","oc_sep_b"),
        ("iforest_norm_sep","if_sep_a","if_sep_b"),
    ]:
        a_v = merged[col_a].values.astype(float)
        b_v = merged[col_b].values.astype(float)
        mask = np.isfinite(a_v) & np.isfinite(b_v)
        a_v, b_v = a_v[mask], b_v[mask]
        if len(a_v) < 5:
            continue
        obs  = b_v.mean() - a_v.mean()
        boot = [b_v[rng.integers(0,len(b_v),len(b_v))].mean()
                - a_v[rng.integers(0,len(a_v),len(a_v))].mean()
                for _ in range(n_boot)]
        boot = np.array(boot)
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        p = (boot <= 0).mean() if obs > 0 else (boot >= 0).mean()
        rows.append({
            "comparison": f"{label_b} vs {label_a}",
            "metric":     model,
            "mean_a":     a_v.mean(),
            "mean_b":     b_v.mean(),
            "diff_b_minus_a": obs,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "p_value": float(min(2*p, 1.0)),
            "n_engines": int(mask.sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Master run function
# ---------------------------------------------------------------------------

def run() -> None:
    print("[anomaly_experiments] Loading baseline bundle and pipeline ...")
    baseline  = load_bundle(BASELINE_PKL)
    pipe      = build_pipeline(DATA_DIR, seed=SPLIT_SEED)
    df_raw    = pipe["df_raw"]      # unscaled, constant-cols removed
    df_scaled = pipe["df_scaled"]   # scaled with original 18-feature scaler
    splits    = pipe["splits"]
    ALL_18    = baseline.feature_cols  # ['op1','op2','s2',...]
    config    = DEFAULT_CONFIG

    # Leakage guard
    assert set(splits["test"]).isdisjoint(set(splits["fit"]))
    assert set(splits["test"]).isdisjoint(set(splits["calibration"]))

    # ------------------------------------------------------------------ #
    # Experiment A — baseline (already trained)
    # ------------------------------------------------------------------ #
    print("[Exp A] Scoring baseline model on calibration engines ...")
    cal_fd1 = [e for e in splits["calibration"] if e.startswith("FD001")]
    cal_fd3 = [e for e in splits["calibration"] if e.startswith("FD003")]
    fit_fd1 = [e for e in splits["fit"] if e.startswith("FD001")]
    fit_fd3 = [e for e in splits["fit"] if e.startswith("FD003")]

    scores_A_fd1 = score_engines(df_scaled, cal_fd1, baseline, ALL_18)
    scores_A_fd3 = score_engines(df_scaled, cal_fd3, baseline, ALL_18)

    # Also score fit engines for LOO EW
    scores_A_fit_fd1 = score_engines(df_scaled, fit_fd1, baseline, ALL_18)
    scores_A_fit_fd3 = score_engines(df_scaled, fit_fd3, baseline, ALL_18)

    rho_A_fd1 = compute_spearman(scores_A_fd1)
    rho_A_fd3 = compute_spearman(scores_A_fd3)
    sep_A_fd1 = compute_separation(scores_A_fd1)
    sep_A_fd3 = compute_separation(scores_A_fd3)
    ew_A_fd1  = compute_early_warning(
        pd.concat([scores_A_fit_fd1, scores_A_fd1], ignore_index=True), fit_fd1 + cal_fd1)
    ew_A_fd3  = compute_early_warning(
        pd.concat([scores_A_fit_fd3, scores_A_fd3], ignore_index=True), fit_fd3 + cal_fd3)
    # Restrict EW results to cal engines only
    ew_A_fd1 = ew_A_fd1[ew_A_fd1["engine_id"].isin(cal_fd1)]
    ew_A_fd3 = ew_A_fd3[ew_A_fd3["engine_id"].isin(cal_fd3)]

    ovlp_A_fd1 = compute_score_overlap(scores_A_fd1)
    ovlp_A_fd3 = compute_score_overlap(scores_A_fd3)

    _save_csv(scores_A_fd1, "scores_A_fd001.csv")
    _save_csv(scores_A_fd3, "scores_A_fd003.csv")

    # ------------------------------------------------------------------ #
    # Experiment B — Separate models, 18 features
    # ------------------------------------------------------------------ #
    print("[Exp B] Training separate FD001 / FD003 models (18 features) ...")

    # FD001
    pipe_B_fd1 = build_exp_pipeline(df_raw, splits, ALL_18, "FD001")
    bund_B_fd1 = train_experiment_bundle(
        pipe_B_fd1["df_scaled"], pipe_B_fd1["fit_ids"], pipe_B_fd1["cal_ids"],
        ALL_18, config, splits)
    save_bundle(bund_B_fd1, EXP_MODELS / "anomaly_bundle_B_fd001.pkl")

    # FD003
    pipe_B_fd3 = build_exp_pipeline(df_raw, splits, ALL_18, "FD003")
    bund_B_fd3 = train_experiment_bundle(
        pipe_B_fd3["df_scaled"], pipe_B_fd3["fit_ids"], pipe_B_fd3["cal_ids"],
        ALL_18, config, splits)
    save_bundle(bund_B_fd3, EXP_MODELS / "anomaly_bundle_B_fd003.pkl")

    print("[Exp B] Scoring on calibration engines ...")
    scores_B_fd1 = score_engines(pipe_B_fd1["df_scaled"], cal_fd1, bund_B_fd1)
    scores_B_fd3 = score_engines(pipe_B_fd3["db_scaled"] if "db_scaled" in pipe_B_fd3
                                  else pipe_B_fd3["df_scaled"], cal_fd3, bund_B_fd3)
    # Also fit engines for LOO threshold
    scores_B_fit_fd1 = score_engines(pipe_B_fd1["df_scaled"], pipe_B_fd1["fit_ids"], bund_B_fd1)
    scores_B_fit_fd3 = score_engines(pipe_B_fd3["df_scaled"], pipe_B_fd3["fit_ids"], bund_B_fd3)

    rho_B_fd1 = compute_spearman(scores_B_fd1)
    rho_B_fd3 = compute_spearman(scores_B_fd3)
    sep_B_fd1 = compute_separation(scores_B_fd1)
    sep_B_fd3 = compute_separation(scores_B_fd3)
    ew_B_fd1  = compute_early_warning(
        pd.concat([scores_B_fit_fd1, scores_B_fd1], ignore_index=True),
        pipe_B_fd1["fit_ids"] + cal_fd1)
    ew_B_fd3  = compute_early_warning(
        pd.concat([scores_B_fit_fd3, scores_B_fd3], ignore_index=True),
        pipe_B_fd3["fit_ids"] + cal_fd3)
    ew_B_fd1 = ew_B_fd1[ew_B_fd1["engine_id"].isin(cal_fd1)]
    ew_B_fd3 = ew_B_fd3[ew_B_fd3["engine_id"].isin(cal_fd3)]

    _save_csv(scores_B_fd1, "scores_B_fd001.csv")
    _save_csv(scores_B_fd3, "scores_B_fd003.csv")

    # ------------------------------------------------------------------ #
    # Experiment C — Separate models, dataset-specific feature subsets
    # ------------------------------------------------------------------ #
    print("[Exp C] Training separate models with dataset-specific features ...")

    # Verify requested features are available
    avail = set(ALL_18)
    feat_C_fd1 = [f for f in FEATURES_C_FD001 if f in avail]
    feat_C_fd3 = [f for f in FEATURES_C_FD003 if f in avail]
    print(f"  FD001 features ({len(feat_C_fd1)}): {feat_C_fd1}")
    print(f"  FD003 features ({len(feat_C_fd3)}): {feat_C_fd3}")

    pipe_C_fd1 = build_exp_pipeline(df_raw, splits, feat_C_fd1, "FD001")
    bund_C_fd1 = train_experiment_bundle(
        pipe_C_fd1["df_scaled"], pipe_C_fd1["fit_ids"], pipe_C_fd1["cal_ids"],
        feat_C_fd1, config, splits)
    save_bundle(bund_C_fd1, EXP_MODELS / "anomaly_bundle_C_fd001.pkl")

    pipe_C_fd3 = build_exp_pipeline(df_raw, splits, feat_C_fd3, "FD003")
    bund_C_fd3 = train_experiment_bundle(
        pipe_C_fd3["df_scaled"], pipe_C_fd3["fit_ids"], pipe_C_fd3["cal_ids"],
        feat_C_fd3, config, splits)
    save_bundle(bund_C_fd3, EXP_MODELS / "anomaly_bundle_C_fd003.pkl")

    scores_C_fd1 = score_engines(pipe_C_fd1["df_scaled"], cal_fd1, bund_C_fd1)
    scores_C_fd3 = score_engines(pipe_C_fd3["df_scaled"], cal_fd3, bund_C_fd3)
    scores_C_fit_fd1 = score_engines(pipe_C_fd1["df_scaled"], pipe_C_fd1["fit_ids"], bund_C_fd1)
    scores_C_fit_fd3 = score_engines(pipe_C_fd3["df_scaled"], pipe_C_fd3["fit_ids"], bund_C_fd3)

    rho_C_fd1 = compute_spearman(scores_C_fd1)
    rho_C_fd3 = compute_spearman(scores_C_fd3)
    sep_C_fd1 = compute_separation(scores_C_fd1)
    sep_C_fd3 = compute_separation(scores_C_fd3)
    ew_C_fd1  = compute_early_warning(
        pd.concat([scores_C_fit_fd1, scores_C_fd1], ignore_index=True),
        pipe_C_fd1["fit_ids"] + cal_fd1)
    ew_C_fd3  = compute_early_warning(
        pd.concat([scores_C_fit_fd3, scores_C_fd3], ignore_index=True),
        pipe_C_fd3["fit_ids"] + cal_fd3)
    ew_C_fd1 = ew_C_fd1[ew_C_fd1["engine_id"].isin(cal_fd1)]
    ew_C_fd3 = ew_C_fd3[ew_C_fd3["engine_id"].isin(cal_fd3)]

    _save_csv(scores_C_fd1, "scores_C_fd001.csv")
    _save_csv(scores_C_fd3, "scores_C_fd003.csv")

    # ------------------------------------------------------------------ #
    # Experiment D — Temporal features (rolling mean w=10)
    # ------------------------------------------------------------------ #
    print("[Exp D] Building temporal features (causal rolling mean w=10) ...")

    # Add rolling means to df_raw, then rebuild pipeline per dataset
    df_aug, rm_cols = add_rolling_means(df_raw, ROLL_SENSORS, ROLL_WINDOW)
    feat_D = ALL_18 + rm_cols
    print(f"  Added {len(rm_cols)} rolling-mean cols: {rm_cols}")
    print(f"  Total Exp D features: {len(feat_D)}")

    pipe_D_fd1 = build_exp_pipeline(df_aug, splits, feat_D, "FD001")
    bund_D_fd1 = train_experiment_bundle(
        pipe_D_fd1["df_scaled"], pipe_D_fd1["fit_ids"], pipe_D_fd1["cal_ids"],
        feat_D, config, splits)
    save_bundle(bund_D_fd1, EXP_MODELS / "anomaly_bundle_D_fd001.pkl")

    pipe_D_fd3 = build_exp_pipeline(df_aug, splits, feat_D, "FD003")
    bund_D_fd3 = train_experiment_bundle(
        pipe_D_fd3["df_scaled"], pipe_D_fd3["fit_ids"], pipe_D_fd3["cal_ids"],
        feat_D, config, splits)
    save_bundle(bund_D_fd3, EXP_MODELS / "anomaly_bundle_D_fd003.pkl")

    scores_D_fd1 = score_engines(pipe_D_fd1["df_scaled"], cal_fd1, bund_D_fd1)
    scores_D_fd3 = score_engines(pipe_D_fd3["df_scaled"], cal_fd3, bund_D_fd3)
    scores_D_fit_fd1 = score_engines(pipe_D_fd1["df_scaled"], pipe_D_fd1["fit_ids"], bund_D_fd1)
    scores_D_fit_fd3 = score_engines(pipe_D_fd3["df_scaled"], pipe_D_fd3["fit_ids"], bund_D_fd3)

    rho_D_fd1 = compute_spearman(scores_D_fd1)
    rho_D_fd3 = compute_spearman(scores_D_fd3)
    sep_D_fd1 = compute_separation(scores_D_fd1)
    sep_D_fd3 = compute_separation(scores_D_fd3)
    ew_D_fd1  = compute_early_warning(
        pd.concat([scores_D_fit_fd1, scores_D_fd1], ignore_index=True),
        pipe_D_fd1["fit_ids"] + cal_fd1)
    ew_D_fd3  = compute_early_warning(
        pd.concat([scores_D_fit_fd3, scores_D_fd3], ignore_index=True),
        pipe_D_fd3["fit_ids"] + cal_fd3)
    ew_D_fd1 = ew_D_fd1[ew_D_fd1["engine_id"].isin(cal_fd1)]
    ew_D_fd3 = ew_D_fd3[ew_D_fd3["engine_id"].isin(cal_fd3)]

    _save_csv(scores_D_fd1, "scores_D_fd001.csv")
    _save_csv(scores_D_fd3, "scores_D_fd003.csv")

    # ------------------------------------------------------------------ #
    # Build summary tables
    # ------------------------------------------------------------------ #
    print("[anomaly_experiments] Aggregating metrics ...")

    # Combine FD001+FD003 per-engine results per config
    def comb(fd1, fd3): return pd.concat([fd1, fd3], ignore_index=True)

    configs_data = {
        "A_shared":   (comb(rho_A_fd1, rho_A_fd3), comb(sep_A_fd1, sep_A_fd3),
                       comb(ew_A_fd1, ew_A_fd3),
                       comb(scores_A_fd1, scores_A_fd3)),
        "B_separate": (comb(rho_B_fd1, rho_B_fd3), comb(sep_B_fd1, sep_B_fd3),
                       comb(ew_B_fd1, ew_B_fd3),
                       comb(scores_B_fd1, scores_B_fd3)),
        "C_feat_sel": (comb(rho_C_fd1, rho_C_fd3), comb(sep_C_fd1, sep_C_fd3),
                       comb(ew_C_fd1, ew_C_fd3),
                       comb(scores_C_fd1, scores_C_fd3)),
        "D_temporal": (comb(rho_D_fd1, rho_D_fd3), comb(sep_D_fd1, sep_D_fd3),
                       comb(ew_D_fd1, ew_D_fd3),
                       comb(scores_D_fd1, scores_D_fd3)),
    }

    all_summaries = []
    overlap_rows  = []
    all_ew        = {}

    for cfg, (rho, sep, ew, scores) in configs_data.items():
        summ = aggregate_metrics(rho, sep, ew, cfg)
        all_summaries.append(summ)
        all_ew[cfg] = ew

        ovlp = compute_score_overlap(scores)
        for _, row in ovlp.iterrows():
            overlap_rows.append({"config": cfg, **row.to_dict()})

        # Save per-engine metrics
        merged = rho.merge(sep, on=["engine_id","dataset"], how="outer")
        _save_csv(merged, f"metrics_per_engine_{cfg}.csv")

    summary_df = pd.DataFrame(all_summaries)
    _save_csv(summary_df, "summary_all_configs.csv")
    overlap_df = pd.DataFrame(overlap_rows)
    _save_csv(overlap_df, "score_overlap.csv")

    # Bootstrap comparisons vs Baseline A
    boot_rows = []
    for cfg, (rho, sep, ew, scores) in configs_data.items():
        if cfg == "A_shared":
            continue
        # Compare on matching engines only
        for ds, rho_a_ds, rho_b_ds, sep_a_ds, sep_b_ds in [
            ("FD001",
             rho_A_fd1, rho[rho["dataset"]=="FD001"],
             sep_A_fd1, sep[sep["dataset"]=="FD001"]),
            ("FD003",
             rho_A_fd3, rho[rho["dataset"]=="FD003"],
             sep_A_fd3, sep[sep["dataset"]=="FD003"]),
        ]:
            bt = bootstrap_compare(rho_a_ds, rho_b_ds, sep_a_ds, sep_b_ds,
                                   "A_shared", cfg)
            bt["dataset"] = ds
            boot_rows.append(bt)

    if boot_rows:
        boot_df = pd.concat(boot_rows, ignore_index=True)
        _save_csv(boot_df, "bootstrap_comparison.csv")
    else:
        boot_df = pd.DataFrame()

    # Print summary tables
    print("\n" + "="*75)
    print("SUMMARY: Median Spearman ρ (OC-SVM) by config × dataset")
    print("="*75)
    for ds in ("FD001", "FD003"):
        print(f"\n  {ds}:")
        for s in all_summaries:
            oc  = s.get(f"{ds}_ocsvm_median_rho",  np.nan)
            iff = s.get(f"{ds}_iforest_median_rho", np.nan)
            print(f"    {s['config']:12s}  OC-SVM={oc:.3f}  IForest={iff:.3f}")

    print("\n" + "="*75)
    print("SUMMARY: Median normalised separation by config × dataset")
    print("="*75)
    for ds in ("FD001", "FD003"):
        print(f"\n  {ds}:")
        for s in all_summaries:
            oc  = s.get(f"{ds}_ocsvm_median_norm_sep",  np.nan)
            iff = s.get(f"{ds}_iforest_median_norm_sep", np.nan)
            print(f"    {s['config']:12s}  OC-SVM={oc:.3f}  IForest={iff:.3f}")

    print("\n" + "="*75)
    print("SUMMARY: Early-warning (median lead time / mean FA rate)")
    print("="*75)
    for cfg, ew_data in all_ew.items():
        for ds in ("FD001", "FD003"):
            ew_ds = ew_data[ew_data["dataset"] == ds]
            for model in ("ocsvm", "iforest"):
                m = ew_ds[ew_ds["model"] == model]
                lead = m.loc[m["warned"], "lead_time_cycles"].median()
                fa   = m["false_alarm_rate"].mean()
                print(f"  {cfg:12s} {ds} {model.upper():8s}  "
                      f"lead_median={lead:.0f}  fa_mean={fa:.3f}")

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    print("[anomaly_experiments] Generating plots ...")

    all_scores_dict = {
        "A_shared":   comb(scores_A_fd1, scores_A_fd3),
        "B_separate": comb(scores_B_fd1, scores_B_fd3),
        "C_feat_sel": comb(scores_C_fd1, scores_C_fd3),
        "D_temporal": comb(scores_D_fd1, scores_D_fd3),
    }

    for cfg, sc in all_scores_dict.items():
        plot_score_trajectories(sc, cfg)

    plot_spearman_comparison(all_summaries)
    plot_separation_comparison(all_summaries)
    plot_early_warning_comparison(all_ew)
    plot_score_overlap_comparison(overlap_rows)

    for ds in ("FD001", "FD003"):
        ds_scores = {k: v[v["dataset"]==ds] for k, v in all_scores_dict.items()}
        plot_score_distributions(ds_scores, ds)

    print(f"  Plots saved to: {PLOTS_DIR}")

    # ------------------------------------------------------------------ #
    # Write report
    # ------------------------------------------------------------------ #
    print("[anomaly_experiments] Writing phase2_anomaly_experiments.md ...")
    _write_report(all_summaries, overlap_df, boot_df, feat_C_fd1, feat_C_fd3, rm_cols)
    print("[anomaly_experiments] Done.")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(
    all_summaries: list[dict],
    overlap_df:    pd.DataFrame,
    boot_df:       pd.DataFrame,
    feat_C_fd1:    list[str],
    feat_C_fd3:    list[str],
    rm_cols:       list[str],
) -> pathlib.Path:
    """Write phase2_anomaly_experiments.md."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    sm = {s["config"]: s for s in all_summaries}

    lines = [
        "# Phase 2: Anomaly-Detection Experiments",
        f"**Branch:** feature/kartik-risk-layer",
        "",
        "---",
        "",
        "## Configurations",
        "",
        "| Config | Description | FD001 features | FD003 features |",
        "|--------|-------------|----------------|----------------|",
        "| A_shared | Baseline: shared FD001+FD003 model, 18 features | 18 | 18 |",
        "| B_separate | Separate models per dataset, 18 features | 18 | 18 |",
        f"| C_feat_sel | Separate models, dataset-specific subsets | {len(feat_C_fd1)} | {len(feat_C_fd3)} |",
        f"| D_temporal | Sep. models + 18 features + {len(rm_cols)} rolling means | {18+len(rm_cols)} | {18+len(rm_cols)} |",
        "",
        f"**FD001 feature subset (Exp C):** `{'`, `'.join(feat_C_fd1)}`",
        "",
        f"**FD003 feature subset (Exp C):** `{'`, `'.join(feat_C_fd3)}`",
        "",
        f"**Rolling mean columns (Exp D):** `{'`, `'.join(rm_cols)}`  (causal, w=10, per engine)",
        "",
        "**Training:** OC-SVM (RBF, nu=0.05, gamma=scale) and IForest (200 trees, seed=42).  ",
        "**Healthy window:** cycles 1–30.  ",
        "**Evaluation:** calibration split only (20 FD001 + 20 FD003 engines). Test split not accessed.",
        "",
        "---",
        "",
        "## Results: Median Spearman ρ",
        "",
        "| Config | FD001 OC-SVM | FD001 IForest | FD003 OC-SVM | FD003 IForest |",
        "|--------|-------------|--------------|-------------|--------------|",
    ]

    for cfg, s in sm.items():
        lines.append(
            f"| {cfg} "
            f"| {_fmt(s.get('FD001_ocsvm_median_rho'))} "
            f"| {_fmt(s.get('FD001_iforest_median_rho'))} "
            f"| {_fmt(s.get('FD003_ocsvm_median_rho'))} "
            f"| {_fmt(s.get('FD003_iforest_median_rho'))} |"
        )

    lines += [
        "",
        "## Results: Directional Success Rate",
        "",
        "| Config | FD001 OC-SVM | FD001 IForest | FD003 OC-SVM | FD003 IForest |",
        "|--------|-------------|--------------|-------------|--------------|",
    ]
    for cfg, s in sm.items():
        lines.append(
            f"| {cfg} "
            f"| {_fmt(s.get('FD001_ocsvm_dir_success'))} "
            f"| {_fmt(s.get('FD001_iforest_dir_success'))} "
            f"| {_fmt(s.get('FD003_ocsvm_dir_success'))} "
            f"| {_fmt(s.get('FD003_iforest_dir_success'))} |"
        )

    lines += [
        "",
        "## Results: Median Normalised Separation (Cohen's d-like)",
        "",
        "| Config | FD001 OC-SVM | FD001 IForest | FD003 OC-SVM | FD003 IForest |",
        "|--------|-------------|--------------|-------------|--------------|",
    ]
    for cfg, s in sm.items():
        lines.append(
            f"| {cfg} "
            f"| {_fmt(s.get('FD001_ocsvm_median_norm_sep'))} "
            f"| {_fmt(s.get('FD001_iforest_median_norm_sep'))} "
            f"| {_fmt(s.get('FD003_ocsvm_median_norm_sep'))} "
            f"| {_fmt(s.get('FD003_iforest_median_norm_sep'))} |"
        )

    lines += [
        "",
        "## Results: Early-Warning (median lead time / mean FA rate)",
        "",
        "| Config | FD001 OC-SVM lead | FD001 IF lead | FD003 OC-SVM lead | FD003 IF lead |",
        "|--------|-----------------|--------------|-----------------|--------------|",
    ]
    for cfg, s in sm.items():
        lines.append(
            f"| {cfg} "
            f"| {_fmt(s.get('FD001_ocsvm_lead_median'))} "
            f"| {_fmt(s.get('FD001_iforest_lead_median'))} "
            f"| {_fmt(s.get('FD003_ocsvm_lead_median'))} "
            f"| {_fmt(s.get('FD003_iforest_lead_median'))} |"
        )

    lines += [
        "",
        "## Score Overlap (near-failure vs early-life range)",
        "",
        "Lower overlap = better separation between healthy and failing states.",
        "",
    ]
    if not overlap_df.empty:
        lines += [
            "| Config | Dataset | Model | Overlap % |",
            "|--------|---------|-------|-----------|",
        ]
        for _, row in overlap_df.iterrows():
            lines.append(
                f"| {row.get('config','—')} "
                f"| {row['dataset']} | {row['model']} "
                f"| {row['overlap_pct']*100:.1f}% |"
            )

    lines += [
        "",
        "## Bootstrap Comparison vs Baseline (A_shared)",
        "",
        "Positive diff = new config better than baseline. 95% CI and two-sided p-value.",
        "",
    ]
    if not boot_df.empty:
        lines += [
            "| Comparison | Dataset | Metric | Diff | CI | p |",
            "|-----------|---------|--------|------|----|---|",
        ]
        for _, row in boot_df.iterrows():
            lines.append(
                f"| {row['comparison']} | {row['dataset']} | {row['metric']} "
                f"| {row['diff_b_minus_a']:+.3f} "
                f"| [{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}] "
                f"| {row['p_value']:.3f} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Answers to Key Questions",
        "",
        "*(Populated after results are computed. Check summary_all_configs.csv for numbers.)*",
        "",
        "**Does dataset-specific training improve performance?**  ",
        "See B_separate vs A_shared in bootstrap_comparison.csv.",
        "",
        "**Does sensor selection improve performance?**  ",
        "See C_feat_sel vs A_shared and B_separate in bootstrap_comparison.csv.",
        "",
        "**Do rolling mean features improve performance?**  ",
        "See D_temporal vs B_separate in bootstrap_comparison.csv.",
        "",
        "**Which algorithm performs best for FD001?**  ",
        "See FD001_ocsvm_median_rho vs FD001_iforest_median_rho across configs.",
        "",
        "**Which algorithm performs best for FD003?**  ",
        "See FD003_ocsvm_median_rho vs FD003_iforest_median_rho across configs.",
        "",
        "**Is there sufficient evidence to retain both algorithms?**  ",
        "Decision based on whether the winning algorithm's advantage is consistent",
        "across all configs and both datasets.",
        "",
        "**Which configuration should proceed to calibration?**  ",
        "The config with highest median ρ AND acceptable FA rate AND",
        "lowest score overlap on BOTH datasets.",
        "",
        "---",
        "",
        "## Files",
        "",
        "**Models (git-ignored):**",
        "- `models/experiments/anomaly_bundle_{B,C,D}_{fd001,fd003}.pkl`",
        "",
        "**Results:**",
        "- `scores_{A,B,C,D}_{fd001,fd003}.csv` — per-cycle scored data",
        "- `metrics_per_engine_{config}.csv` — per-engine Spearman + separation",
        "- `summary_all_configs.csv` — all headline metrics",
        "- `score_overlap.csv` — near-failure vs early-life overlap",
        "- `bootstrap_comparison.csv` — paired bootstrap vs baseline",
        "",
        "**Plots:**",
        "- `score_trajectories_*.png` — score vs RUL for sample engines",
        "- `spearman_comparison.png` — median ρ across configs",
        "- `separation_comparison.png` — normalised separation across configs",
        "- `early_warning_comparison.png` — lead time + FA rate boxplots",
        "- `score_overlap_comparison.png` — overlap bar chart",
        "- `score_distributions_{FD001,FD003}.png` — violin: healthy vs late",
        "",
        "*Baseline artifacts not modified. Test split not accessed.*",
    ]

    p = RESULTS_DIR / "phase2_anomaly_experiments.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


if __name__ == "__main__":
    run()
