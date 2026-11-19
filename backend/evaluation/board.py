"""Kaul's risk model and the rule baselines, written as TensorBoard runs for the demo.

    python -m backend.evaluation.board [--out runs/demo]

Reads committed CSVs only; nothing is refit or rerun:
    results/calibration_experiments/   Kaul's production risk config (OC-SVM, A_ds_iso)
    results/demo/summary.csv           SPT, LPT, FIFO, threshold+SPT on that risk model

Writes, under ``--out``:
    kaul_risk/          risk per cycle for a few calibration engines (step = cycle)
    baseline_<policy>/  gap, breakdowns, maintenances (step 0/1/2 = low/medium/high regime)
    report/             tables in the Text tab, figures in the Images tab
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from torch.utils.tensorboard import SummaryWriter

from backend.env.risk_adapter import PRODUCTION_CONFIG, predictions_path
from backend.training.protocol import REPO_ROOT

DETECTOR = "ocsvm"
NEAR_FAILURE = 20
SHOWN_ENGINES = 4
REGIMES = ("low", "medium", "high")
POLICIES = ("spt", "lpt", "fifo", "threshold_spt")
COLORS = dict(zip(POLICIES, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"), strict=True))
SUMMARY = REPO_ROOT / "results" / "demo" / "summary.csv"
PER_ENGINE = REPO_ROOT / "results" / "calibration_experiments" / f"per_engine_metrics_{PRODUCTION_CONFIG}_{DETECTOR}.csv"


def load_predictions() -> pd.DataFrame:
    return pd.read_csv(predictions_path(DETECTOR))


def shown_engines(predictions: pd.DataFrame) -> list[str]:
    """The longest-lived engines of each dataset, so the curves show a full degradation."""
    lifetimes = predictions.groupby(["dataset", "engine_id"])["cycle"].max().reset_index()
    per_dataset = SHOWN_ENGINES // lifetimes["dataset"].nunique()
    picked = lifetimes.sort_values("cycle", ascending=False).groupby("dataset").head(per_dataset)
    return sorted(picked["engine_id"])


def risk_table(predictions: pd.DataFrame, per_engine: pd.DataFrame) -> pd.DataFrame:
    near = predictions[predictions["rul_true"] <= NEAR_FAILURE]
    healthy = predictions[predictions["cycle"] <= 30]
    return pd.DataFrame(
        {
            "engines": predictions.groupby("dataset")["engine_id"].nunique(),
            "near-failure RUL MAE (cycles)": (near["rul_estimate"] - near["rul_true"]).abs().groupby(near["dataset"]).mean(),
            "median risk Spearman": per_engine.groupby("dataset")["risk_spearman"].median(),
            "mean risk, first 30 cycles": healthy.groupby("dataset")["risk_score"].mean(),
            f"mean risk, last {NEAR_FAILURE} cycles": near.groupby("dataset")["risk_score"].mean(),
        }
    ).round(3)


def baseline_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Means over instances (each already a mean over test seeds) per policy and regime."""
    metrics = ["gap_mean", "makespan_mean", "breakdowns_mean", "maintenances_mean"]
    table = summary.groupby(["regime", "policy"])[metrics].mean()
    order = pd.MultiIndex.from_product([REGIMES, POLICIES], names=["regime", "policy"])
    return table.reindex(order).round(2)


def risk_figure(predictions: pd.DataFrame) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for axis, (dataset, rows) in zip(axes, predictions.groupby("dataset"), strict=True):
        binned = rows.assign(bin=(rows["rul_true"] // 10) * 10).groupby("bin")["risk_score"]
        quartiles = binned.quantile([0.25, 0.5, 0.75]).unstack()
        quartiles = quartiles[quartiles.index <= 200]
        axis.fill_between(quartiles.index, quartiles[0.25], quartiles[0.75], color="#9ec5f4", linewidth=0)
        axis.plot(quartiles.index, quartiles[0.5], color="#2a78d6", linewidth=2)
        axis.set_title(f"{dataset}: risk vs true remaining life", fontsize=10, loc="left")
        axis.set_xlabel("true RUL (cycles)")
        axis.invert_xaxis()
        _quiet(axis)
    axes[0].set_ylabel("calibrated risk (median, IQR)")
    figure.tight_layout()
    return figure


def baseline_figure(table: pd.DataFrame) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    width = 0.8 / len(POLICIES)
    for axis, metric, label in ((axes[0], "gap_mean", "gap to best-known makespan"), (axes[1], "breakdowns_mean", "breakdowns per episode")):
        for index, policy in enumerate(POLICIES):
            values = [table.loc[(regime, policy), metric] for regime in REGIMES]
            positions = [regime + (index - 1.5) * width for regime in range(len(REGIMES))]
            axis.bar(positions, values, width * 0.9, color=COLORS[policy], label=policy)
        axis.set_xticks(range(len(REGIMES)), REGIMES)
        axis.set_xlabel("failure regime")
        axis.set_title(label, fontsize=10, loc="left")
        _quiet(axis)
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return figure


def write_board(out: Path) -> None:
    predictions, per_engine = load_predictions(), pd.read_csv(PER_ENGINE)
    summary = pd.read_csv(SUMMARY)
    table = baseline_table(summary)

    with SummaryWriter(str(out / "kaul_risk")) as writer:
        for engine_id in shown_engines(predictions):
            rows = predictions[predictions["engine_id"] == engine_id]
            for cycle, risk in zip(rows["cycle"], rows["risk_score"], strict=True):
                writer.add_scalar(f"kaul_risk/{engine_id}", risk, int(cycle))

    for policy in POLICIES:
        with SummaryWriter(str(out / f"baseline_{policy}")) as writer:
            for step, regime in enumerate(REGIMES):
                row = table.loc[(regime, policy)]
                writer.add_scalar("baselines/gap", row["gap_mean"], step)
                writer.add_scalar("baselines/breakdowns", row["breakdowns_mean"], step)
                writer.add_scalar("baselines/maintenances", row["maintenances_mean"], step)

    with SummaryWriter(str(out / "report")) as writer:
        writer.add_text("kaul_risk_model", _markdown(risk_table(predictions, per_engine), f"{DETECTOR}, {PRODUCTION_CONFIG}, calibration split"))
        writer.add_text("baselines", _markdown(table, "means over ta06/ta11/ta21 x 10 test seeds; baselines/* x-axis: 0=low, 1=medium, 2=high"))
        writer.add_figure("kaul_risk/risk_vs_rul", risk_figure(predictions))
        writer.add_figure("baselines/by_regime", baseline_figure(table))


def _quiet(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    axis.set_axisbelow(True)


def _markdown(table: pd.DataFrame, note: str) -> str:
    flat = table.reset_index()
    rows = [list(flat.columns), ["---"] * len(flat.columns), *flat.astype(str).values.tolist()]
    lines = "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return f"{lines}\n\n_{note}_"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write Kaul's risk model and the baselines to TensorBoard.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "demo")
    args = parser.parse_args(argv)
    write_board(args.out)
    print(f"wrote {args.out}; view with: tensorboard --logdir {args.out}")


if __name__ == "__main__":
    main()
