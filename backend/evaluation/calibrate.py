"""Describe each regime in policy-free terms, to set its parameters at the freeze.

    python -m backend.evaluation.calibrate --protocol configs/protocol.toml [--workers 8]

Uses only the tuning split and blind SPT, so regimes can be fixed before any
PPO result exists. Per regime it reports machine lifetimes in mean-operation
units, the chance that an operation outlasts a whole lifetime (which forces
repeated breakdowns), and what blind SPT suffers: breakdowns per episode,
breakdown rate, lost work and gap. Writes ``results/<hash>/calibration.csv``.
"""

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import fmean

import numpy as np
import pandas as pd

from backend.baselines.dispatch import shortest_processing_time, solve
from backend.env.health import EngineHealth
from backend.env.instance import Instance, from_benchmark
from backend.evaluation.actors import reference_rule
from backend.evaluation.episodes import Cell, results_dir, run_cells
from backend.training.factory import build_regime, rate_sampler, sample_engine
from backend.training.protocol import Protocol, git_commit, load_protocol

LIFETIMES_PER_MACHINE = 5


def lifetime_stats(protocol: Protocol, regime_name: str) -> dict[str, float]:
    regime = build_regime(protocol.regime(regime_name), protocol.pool("tune"))
    rates = rate_sampler(protocol.rates)
    engine = sample_engine(protocol.engine)
    lifetimes, outlasting = [], []
    for name in protocol.split.train_instances:
        instance = from_benchmark(name)
        durations = _durations_by_machine(instance)
        for seed in protocol.seeds.tune:
            health = EngineHealth(
                regime=regime,
                rates=rates(instance, np.random.default_rng(seed)),
                mean_duration=instance.mean_duration,
                seed=seed,
                sample_engine=engine,
                maintenance_enabled=False,
            )
            for machine in range(instance.num_machines):
                for _ in range(LIFETIMES_PER_MACHINE):
                    lifetime = health.time_to_failure(machine)
                    lifetimes.append(lifetime / instance.mean_duration)
                    outlasting.append(float(np.mean(durations[machine] >= lifetime)))
                    health.reset(machine)
    quantiles = np.quantile(lifetimes, [0.05, 0.5, 0.95])
    return {
        "lifetime_p05": quantiles[0],
        "lifetime_median": quantiles[1],
        "lifetime_p95": quantiles[2],
        "p_operation_outlasts_lifetime": fmean(outlasting),
    }


def calibrate(protocol: Protocol, *, workers: int = 1, results_root: Path = Path("results")) -> Path:
    commit = git_commit()
    no_failure_gap = fmean(
        solve(instance, shortest_processing_time).makespan / instance.reference_makespan - 1.0  # type: ignore[operator]
        for instance in map(from_benchmark, protocol.split.train_instances)
    )
    rows = []
    for regime in protocol.regime_names:
        cells = [
            Cell(protocol, commit, reference_rule("spt"), regime, "tune", instance)
            for instance in protocol.split.train_instances
        ]
        records = run_cells(cells, workers)
        finished = [r for r in records if not r.truncated]
        rows.append(
            {
                "regime": regime,
                **lifetime_stats(protocol, regime),
                "spt_breakdowns_per_episode": _mean(r.breakdowns for r in finished),
                "spt_breakdown_rate": _mean(r.breakdown_rate for r in finished),
                "spt_lost_work_share": _mean(r.lost_work / r.makespan for r in finished),  # type: ignore[operator]
                "spt_gap": _mean(r.gap for r in finished),
                "spt_gap_no_failure": no_failure_gap,
                "truncated": len(records) - len(finished),
            }
        )
    out = results_dir(protocol, results_root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "calibration.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _mean(values: Iterable[float | int | None]) -> float:
    present = [value for value in values if value is not None]
    return fmean(present) if present else float("nan")


def _durations_by_machine(instance: Instance) -> list[np.ndarray]:
    durations: list[list[int]] = [[] for _ in range(instance.num_machines)]
    for job in instance.jobs:
        for operation in job:
            durations[operation.machine_id].append(operation.duration)
    return [np.array(values) for values in durations]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Report policy-free statistics for each regime.")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    path = calibrate(load_protocol(args.protocol), workers=args.workers, results_root=args.results)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(pd.read_csv(path).to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
