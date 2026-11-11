"""Run contenders over a grid of instances and health seeds, one record per episode."""

import csv
import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

from backend.env.gym_env import JobShopEnv
from backend.env.instance import from_benchmark
from backend.evaluation.actors import Actor, Contender
from backend.evaluation.metrics import EpisodeMetrics, episode_metrics
from backend.training.factory import make_env
from backend.training.protocol import Protocol

EvalPhase = Literal["tune", "test"]


@dataclass(frozen=True, slots=True)
class Outcome:
    events: list[dict[str, Any]]
    decisions: int
    truncated: bool
    illegal_actions: int


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """One row of ``episodes/*.csv``. Metric fields are ``None`` for a truncated episode."""

    protocol: str
    commit: str
    phase: str
    policy: str
    variant: str
    mode: str
    regime: str
    instance: str
    size: str
    seed: int
    decisions: int
    truncated: bool
    illegal_actions: int
    makespan: int | None
    gap: float | None
    breakdowns: int | None
    breakdown_rate: float | None
    maintenances: int | None
    maintenance_downtime: int | None
    maintenance_share: float | None
    repair_downtime: int | None
    lost_work: int | None


@dataclass(frozen=True, slots=True)
class Cell:
    """One contender on one instance in one regime, over every health seed of the phase."""

    protocol: Protocol
    commit: str
    contender: Contender
    regime: str
    phase: EvalPhase
    instance: str


def play_episode(env: JobShopEnv, actor: Actor, *, instance_name: str, seed: int) -> Outcome:
    observation, info = env.reset(seed=seed, options={"instance_name": instance_name})
    events = list(info["events"])
    decisions = illegal = 0
    while True:
        observation, _, terminated, truncated, info = env.step(actor(env, observation))
        events.extend(info["events"])
        decisions += 1
        illegal += int(info["illegal_action"])
        if terminated or truncated:
            return Outcome(events, decisions, truncated, illegal)


def run_cell(cell: Cell) -> list[EpisodeRecord]:
    env = make_env(cell.protocol, cell.contender.mode, cell.regime, cell.phase)
    actor = cell.contender.build()
    instance = from_benchmark(cell.instance)
    records = []
    for seed in cell.protocol.health_seeds(cell.phase):
        outcome = play_episode(env, actor, instance_name=cell.instance, seed=seed)
        metrics = None if outcome.truncated else episode_metrics(outcome.events, instance)
        records.append(
            EpisodeRecord(
                protocol=cell.protocol.sha256,
                commit=cell.commit,
                phase=cell.phase,
                policy=cell.contender.policy,
                variant=cell.contender.variant,
                mode=cell.contender.mode,
                regime=cell.regime,
                instance=cell.instance,
                size=f"{instance.num_jobs}x{instance.num_machines}",
                seed=seed,
                decisions=outcome.decisions,
                truncated=outcome.truncated,
                illegal_actions=outcome.illegal_actions,
                **_metric_fields(metrics),
            )
        )
    return records


def run_cells(cells: Sequence[Cell], workers: int = 1) -> list[EpisodeRecord]:
    """Results in the order of ``cells``. Workers start from a fork server, not a fork of this process."""
    if workers <= 1:
        return [record for cell in cells for record in run_cell(cell)]
    context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return [record for records in pool.map(run_cell, cells) for record in records]


def results_dir(protocol: Protocol, root: Path = Path("results")) -> Path:
    return root / protocol.short_hash


def write_records(path: Path, records: Sequence[EpisodeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [field.name for field in fields(EpisodeRecord)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _metric_fields(metrics: EpisodeMetrics | None) -> dict[str, Any]:
    if metrics is None:
        return {field.name: None for field in fields(EpisodeMetrics)}
    return asdict(metrics)
