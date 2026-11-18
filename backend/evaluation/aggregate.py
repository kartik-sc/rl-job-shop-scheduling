"""Turn the test episodes into the summary and paired-comparison tables.

    python -m backend.evaluation.aggregate --protocol configs/protocol.toml

The unit of analysis is the instance: seeds (and PPO training seeds) are
averaged within an instance first, since episodes on one instance are
correlated. Intervals are 95% bootstrap intervals over instances. Every
regime is written, whatever it shows.

Refuses to run on rows from another protocol, truncated episodes, illegal
actions, or contenders that did not see the same (instance, seed) pairs.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from backend.evaluation.episodes import results_dir
from backend.training.protocol import Protocol, load_protocol

METRICS = ("gap", "breakdown_rate", "maintenance_share")
COMPARISONS = (("full_ppo", "blind_ppo"), ("full_ppo", "threshold_spt"), ("threshold_spt", "spt"))
RESAMPLES = 10_000
LEVEL = 0.95


def load_episodes(out: Path, protocol: Protocol) -> pd.DataFrame:
    files = sorted((out / "episodes").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no episode files in {out / 'episodes'}")
    episodes = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    episodes["variant"] = episodes["variant"].fillna("").astype(str)
    check(episodes, protocol)
    return episodes


def check(episodes: pd.DataFrame, protocol: Protocol) -> None:
    if set(episodes["protocol"]) != {protocol.sha256}:
        raise ValueError(f"episodes from other protocols: {sorted(set(episodes['protocol']) - {protocol.sha256})}")
    if episodes["truncated"].any():
        raise ValueError(f"{int(episodes['truncated'].sum())} truncated episodes")
    if episodes["illegal_actions"].sum():
        raise ValueError(f"{int(episodes['illegal_actions'].sum())} illegal actions")
    if episodes["gap"].isna().any():
        raise ValueError("episodes without a reference makespan")
    for regime, rows in episodes.groupby("regime"):
        pairs = rows.groupby(["policy", "variant"])[["instance", "seed"]].apply(
            lambda group: frozenset(zip(group["instance"], group["seed"], strict=True))
        )
        if pairs.nunique() != 1:
            raise ValueError(f"regime {regime}: contenders saw different (instance, seed) pairs")


def per_instance(episodes: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds and PPO training seeds, per policy, regime and instance."""
    return (
        episodes.groupby(["policy", "regime", "size", "instance"])[list(METRICS) + ["makespan"]]
        .mean()
        .reset_index()
    )


def summarise(episodes: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    instances = per_instance(episodes)
    variant_spread = (
        episodes.groupby(["policy", "regime", "variant"])["gap"].mean().groupby(["policy", "regime"]).std(ddof=1)
    )
    rows = []
    for size in ["all", *sorted(instances["size"].unique())]:
        subset = instances if size == "all" else instances[instances["size"] == size]
        for (policy, regime), group in subset.groupby(["policy", "regime"]):
            row = {"policy": policy, "regime": regime, "size": size, "instances": len(group)}
            for metric in METRICS:
                low, high = bootstrap_interval(group[metric].to_numpy(), np.random.default_rng(seed))
                row |= {f"{metric}_mean": group[metric].mean(), f"{metric}_lo": low, f"{metric}_hi": high}
            row["makespan_mean"] = group["makespan"].mean()
            row["variant_sd_gap"] = variant_spread.get((policy, regime), np.nan) if size == "all" else np.nan
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["regime", "size", "gap_mean"], kind="stable")


def compare(episodes: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """``a - b`` per regime and metric; negative means ``a`` is lower.

    ``a_lower`` is the share of (instance, seed) pairs where ``a`` is lower, ties counting half.
    """
    by_seed = episodes.groupby(["policy", "regime", "instance", "seed"])[list(METRICS)].mean()
    rows = []
    for a, b in COMPARISONS:
        if a not in by_seed.index.get_level_values("policy") or b not in by_seed.index.get_level_values("policy"):
            continue
        difference = (by_seed.loc[a] - by_seed.loc[b]).dropna()
        for regime, paired in difference.groupby(level="regime"):
            for metric in METRICS:
                values = paired[metric]
                instance_means = values.groupby(level="instance").mean().to_numpy()
                low, high = bootstrap_interval(instance_means, np.random.default_rng(seed))
                rows.append(
                    {
                        "a": a,
                        "b": b,
                        "regime": regime,
                        "metric": metric,
                        "instances": len(instance_means),
                        "pairs": len(values),
                        "mean_difference": instance_means.mean(),
                        "lo": low,
                        "hi": high,
                        "wilcoxon_p": signed_rank_p(instance_means),
                        "a_lower": float(((values < 0) + 0.5 * (values == 0)).mean()),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) < 2:
        return float(values.mean()), float(values.mean())
    means = values[rng.integers(0, len(values), size=(RESAMPLES, len(values)))].mean(axis=1)
    tail = (1 - LEVEL) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return float(low), float(high)


def signed_rank_p(differences: np.ndarray) -> float:
    """Wilcoxon signed-rank p-value; NaN when every difference is zero."""
    if not np.any(differences):
        return float("nan")
    return float(wilcoxon(differences).pvalue)


def aggregate(protocol: Protocol, *, results_root: Path = Path("results")) -> tuple[Path, Path]:
    out = results_dir(protocol, results_root)
    episodes = load_episodes(out, protocol)
    summary_path, paired_path = out / "summary.csv", out / "paired.csv"
    summary = summarise(episodes)
    summary.to_csv(summary_path, index=False)
    compare(episodes).to_csv(paired_path, index=False)
    return summary_path, paired_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate test episodes into summary and paired tables.")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    summary_path, paired_path = aggregate(load_protocol(args.protocol), results_root=args.results)
    summary = pd.read_csv(summary_path)
    columns = ["regime", "policy", "instances", "gap_mean", "gap_lo", "gap_hi", "breakdown_rate_mean", "maintenance_share_mean"]
    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        print(summary[summary["size"] == "all"][columns].to_string(index=False))
    print(f"wrote {summary_path}\nwrote {paired_path}")


if __name__ == "__main__":
    main()
