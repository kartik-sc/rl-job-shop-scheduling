"""
Train anomaly detectors and generate diagnostic plots.

Usage (from repo root, with .venv active):
    python -m backend.risk.train_anomaly

Outputs (all git-ignored):
    models/anomaly_bundle.pkl  - serialised OC-SVM + IForest + metadata
    plots/anomaly_scores/      - one PNG per engine per model
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe on headless machines
import matplotlib.pyplot as plt
import numpy as np

from backend.risk.anomaly_models import (
    DEFAULT_CONFIG,
    AnomalyBundle,
    AnomalyConfig,
    save_bundle,
    score_engine,
    select_healthy_cycles,
    train_iforest,
    train_ocsvm,
)
from backend.risk.cmapss_pipeline import (
    SPLIT_SEED,
    build_pipeline,
    get_feature_cols,
)

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------

REPO_ROOT  = pathlib.Path(__file__).parent.parent.parent
DATA_DIR   = REPO_ROOT / "data" / "raw" / "cmapss"
MODELS_DIR = REPO_ROOT / "models"
PLOTS_DIR  = REPO_ROOT / "plots" / "anomaly_scores"
BUNDLE_PATH = MODELS_DIR / "anomaly_bundle.pkl"

# Number of fit engines to plot (spread across both datasets)
N_PLOT_ENGINES = 6


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def run(config: AnomalyConfig = DEFAULT_CONFIG) -> AnomalyBundle:
    """Full training pipeline: load data, fit models, save, plot."""

    print(f"[train_anomaly] repo root : {REPO_ROOT}")
    print(f"[train_anomaly] data dir  : {DATA_DIR}")
    print(f"[train_anomaly] config    : {config}")

    # ------------------------------------------------------------------ #
    # 1. Data pipeline (uses fixed SPLIT_SEED so split is stable)
    # ------------------------------------------------------------------ #
    print("[train_anomaly] loading + preprocessing data ...")
    pipe = build_pipeline(DATA_DIR, seed=SPLIT_SEED)
    df_scaled    = pipe["df_scaled"]
    splits       = pipe["splits"]
    feature_cols = pipe["feature_cols"]

    fit_ids = splits["fit"]
    print(f"[train_anomaly] fit engines : {len(fit_ids)}")
    print(f"[train_anomaly] feature cols: {feature_cols}")

    # ------------------------------------------------------------------ #
    # 2. Healthy-cycle selection (fit engines only, cycle <= threshold)
    # ------------------------------------------------------------------ #
    X_healthy = select_healthy_cycles(
        df_scaled, fit_ids, config.healthy_max_cycle, feature_cols
    )
    # 3. Fit detectors
    # ------------------------------------------------------------------ #
    print("[train_anomaly] fitting One-Class SVM ...")
    ocsvm = train_ocsvm(X_healthy, config)
    print("[train_anomaly] fitting Isolation Forest ...")
    iforest = train_iforest(X_healthy, config)

    # ------------------------------------------------------------------ #
    # 4. Save bundle
    # ------------------------------------------------------------------ #
    bundle = AnomalyBundle(
        ocsvm=ocsvm,
        iforest=iforest,
        feature_cols=feature_cols,
        config=config,
        splits=splits,
    )
    save_bundle(bundle, BUNDLE_PATH)
    print(f"[train_anomaly] bundle saved: {BUNDLE_PATH}")

    # ------------------------------------------------------------------ #
    # 5. Diagnostic plots
    # ------------------------------------------------------------------ #
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_engines = _pick_plot_engines(fit_ids, n=N_PLOT_ENGINES)
    print(f"[train_anomaly] plotting {len(plot_engines)} engines: {plot_engines}")

    for eng_id in plot_engines:
        scores = score_engine(df_scaled, eng_id, ocsvm, iforest, feature_cols)
        _plot_scores(scores, eng_id, config.healthy_max_cycle, PLOTS_DIR)

    print(f"[train_anomaly] plots saved to: {PLOTS_DIR}")
    return bundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_plot_engines(fit_ids: list[str], n: int) -> list[str]:
    """Pick n representative engines: mix of FD001 and FD003, spread by index."""
    fd001 = sorted(e for e in fit_ids if e.startswith("FD001"))
    fd003 = sorted(e for e in fit_ids if e.startswith("FD003"))
    half = n // 2
    # evenly spaced indices
    idx1 = np.linspace(0, len(fd001) - 1, half, dtype=int).tolist()
    idx3 = np.linspace(0, len(fd003) - 1, n - half, dtype=int).tolist()
    return [fd001[i] for i in idx1] + [fd003[i] for i in idx3]


def _plot_scores(
    scores: "pd.DataFrame",
    engine_id: str,
    healthy_max_cycle: int,
    plots_dir: pathlib.Path,
) -> None:
    """Save a 2-panel plot: OC-SVM and IForest anomaly scores vs true RUL."""
    import pandas as pd  # local import keeps top-level clean

    rul  = scores["rul"].values
    osc  = scores["ocsvm_score"].values
    ifsc = scores["iforest_score"].values
    cyc  = scores["cycle"].values

    # RUL axis: plot right-to-left so time flows left -> right
    # We plot score vs RUL directly; lower RUL = nearer failure
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        f"Anomaly scores vs true RUL  |  engine {engine_id}",
        fontsize=13,
    )

    healthy_rul_threshold = rul[cyc == min(cyc[cyc <= healthy_max_cycle])].max()

    for ax, scores_arr, label, color in [
        (axes[0], osc,  "OC-SVM",          "steelblue"),
        (axes[1], ifsc, "Isolation Forest", "darkorange"),
    ]:
        ax.plot(rul, scores_arr, lw=1.2, color=color, alpha=0.85)
        # shade the healthy training window
        ymin, ymax = ax.get_ylim()
        ax.axvspan(healthy_rul_threshold, rul.max(), alpha=0.12,
                   color="green", label=f"healthy window (cycle 1-{healthy_max_cycle})")
        ax.axhline(0, color="grey", lw=0.8, linestyle="--")
        ax.set_ylabel(f"{label}\\nanomaly score\\n(higher = more anomalous)", fontsize=9)
        ax.invert_xaxis()  # RUL counts down toward failure on the right
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("True RUL (cycles remaining, 0 = failure)", fontsize=10)

    # Annotate with failure marker
    for ax in axes:
        ax.axvline(0, color="red", lw=1.5, linestyle=":", alpha=0.7, label="failure")

    plt.tight_layout()
    fname = plots_dir / f"{engine_id}_scores.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {fname}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    run()