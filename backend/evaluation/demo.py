"""Baselines on Kaul's real health model: SPT, LPT, FIFO and threshold + SPT.

    python -m backend.evaluation.demo [--protocol configs/protocol.toml] [--out results/demo]

Every policy meets the same engines and failures: one health seed fixes each
machine's rates and the engine and failure point of its k-th lifetime. Engines
come from the calibration pool, the only scored engines that did not train the
detectors. Tau is chosen per regime on the protocol's tuning seeds and
training instances (lowest mean gap), then every policy runs on the test seeds.

Writes, under ``--out``:
    tau_selection.csv   mean gap per regime and tau on the tuning episodes
    episodes.csv        one row per policy, instance, regime and test seed
    summary.csv         means over seeds per policy, instance and regime
    logs/*.json         one event log per policy for LOG_INSTANCE, schema-valid
"""

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import jsonschema
import pandas as pd

from backend.env.events import Event, EventType, log_document
from backend.env.gym_env import JobShopEnv, benchmark_sampler
from backend.env.instance import from_benchmark
from backend.env.risk_adapter import PRODUCTION_CONFIG, sample_engine
from backend.evaluation.actors import Contender, reference_rule, threshold
from backend.evaluation.episodes import play_episode
from backend.evaluation.metrics import episode_metrics
from backend.training.factory import build_regime, rate_sampler
from backend.training.protocol import REPO_ROOT, Protocol, git_commit, load_protocol

POOL = "calibration"
HEALTH_MODEL = f"cmapss/ocsvm/{PRODUCTION_CONFIG}"
INSTANCES = ("ta06", "ta11", "ta21")
LOG_INSTANCE, LOG_REGIME = "ta06", "medium"
RULES = ("spt", "lpt", "fifo")
SCHEMA_PATH = REPO_ROOT / "shared" / "schemas" / "event_log.schema.json"
MAX_JOBS = MAX_MACHINES = 20
METRIC_KEYS = (
    "makespan",
    "gap",
    "breakdowns",
    "maintenances",
    "repair_downtime",
    "maintenance_downtime",
    "lost_work",
)


@dataclass(frozen=True, slots=True)
class Row:
    protocol: str
    commit: str
    health_model: str
    pool: str
    policy: str
    tau: float | None
    mode: str
    regime: str
    instance: str
    size: str
    seed: int
    makespan: int
    gap: float | None
    breakdowns: int
    maintenances: int
    repair_downtime: int
    maintenance_downtime: int
    lost_work: int


def make_env(protocol: Protocol, contender: Contender, regime: str) -> JobShopEnv:
    return JobShopEnv(
        instance_sampler=benchmark_sampler(INSTANCES),
        regime=build_regime(protocol.regime(regime), POOL),
        mode=contender.mode,
        sample_engine=sample_engine,
        rates=rate_sampler(protocol.rates),
        max_steps=protocol.max_steps,
        max_jobs=MAX_JOBS,
        max_machines=MAX_MACHINES,
    )


def run(
    protocol: Protocol, contender: Contender, regime: str, instance_name: str, seed: int, commit: str
) -> tuple[Row, list[dict[str, Any]]]:
    env = make_env(protocol, contender, regime)
    outcome = play_episode(env, contender.build(), instance_name=instance_name, seed=seed)
    if outcome.truncated or outcome.illegal_actions:
        raise RuntimeError(f"{contender.label} {regime} {instance_name} seed {seed}: unusable episode")
    instance = from_benchmark(instance_name)
    metrics = episode_metrics(outcome.events, instance)
    row = Row(
        protocol=protocol.short_hash,
        commit=commit,
        health_model=HEALTH_MODEL,
        pool=POOL,
        policy=contender.policy,
        tau=contender.tau,
        mode=contender.mode,
        regime=regime,
        instance=instance_name,
        size=f"{instance.num_jobs}x{instance.num_machines}",
        seed=seed,
        makespan=metrics.makespan,
        gap=metrics.gap,
        breakdowns=metrics.breakdowns,
        maintenances=metrics.maintenances,
        repair_downtime=metrics.repair_downtime,
        maintenance_downtime=metrics.maintenance_downtime,
        lost_work=metrics.lost_work,
    )
    return row, outcome.events


def select_tau(protocol: Protocol, commit: str) -> tuple[dict[str, float], pd.DataFrame]:
    """Lowest mean gap on training instances x tuning seeds; ties go to the smaller tau."""
    scores = []
    for regime in protocol.regime_names:
        for tau in protocol.tau_grid:
            gaps = [
                run(protocol, threshold(tau), regime, instance, seed, commit)[0].gap
                for instance in protocol.split.train_instances
                for seed in protocol.seeds.tune
            ]
            scores.append({"regime": regime, "tau": tau, "mean_gap": fmean(g for g in gaps if g is not None)})
    table = pd.DataFrame(scores)
    best = table.loc[table.groupby("regime", sort=False)["mean_gap"].idxmin()]
    return dict(zip(best["regime"], best["tau"], strict=True)), table


def schema_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The simulator's events in the committed schema's vocabulary.

    ``repair_end`` is not in the schema yet; the ``risk_update`` emitted at the
    same instant marks the repair's end. Risk updates for a machine that is
    offline are dropped too: its risk is frozen, and a reader that closes a
    downtime on the machine's next risk update would end it early.
    """
    offline: set[int] = set()
    kept = []
    for event in events:
        kind, machine = event["type"], event.get("machine")
        if kind in (EventType.BREAKDOWN, EventType.MAINTENANCE_START):
            offline.add(machine)
        elif kind in (EventType.REPAIR_END, EventType.MAINTENANCE_END):
            offline.discard(machine)
            if kind == EventType.REPAIR_END:
                continue
        elif kind == EventType.RISK_UPDATE and machine in offline:
            continue
        kept.append(event)
    return kept


def write_log(path: Path, row: Row, events: list[dict[str, Any]], schema: dict[str, Any]) -> None:
    instance = from_benchmark(row.instance)
    document = log_document(
        [Event(**{**e, "type": EventType(e["type"])}) for e in schema_events(events)],
        instance,
        policy=row.policy,
        regime=row.regime,
        seed=row.seed,
    )
    document["meta"] |= {
        "health_model": HEALTH_MODEL,
        "pool": POOL,
        "tau": row.tau,
        "metrics": {key: getattr(row, key) for key in METRIC_KEYS},
    }
    jsonschema.validate(document, schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, separators=(",", ":")))


def summarise(rows: Sequence[Row]) -> pd.DataFrame:
    frame = pd.DataFrame(asdict(row) for row in rows)
    keys = ["instance", "size", "regime", "policy", "tau"]
    summary = frame.groupby(keys, dropna=False, sort=False).agg(
        seeds=("seed", "count"),
        makespan_mean=("makespan", "mean"),
        makespan_std=("makespan", "std"),
        gap_mean=("gap", "mean"),
        breakdowns_mean=("breakdowns", "mean"),
        maintenances_mean=("maintenances", "mean"),
        repair_downtime_mean=("repair_downtime", "mean"),
        lost_work_mean=("lost_work", "mean"),
    )
    return summary.reset_index()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Baselines on the real C-MAPSS health model.")
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / "configs" / "protocol.toml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "demo")
    args = parser.parse_args(argv)

    protocol = load_protocol(args.protocol)
    commit = git_commit()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    taus, tuning = select_tau(protocol, commit)
    tuning.to_csv(args.out / "tau_selection.csv", index=False)
    print("selected tau:", taus)

    rows = []
    for instance in INSTANCES:
        for regime in protocol.regime_names:
            contenders = [*map(reference_rule, RULES), threshold(taus[regime])]
            for contender in contenders:
                for seed in protocol.seeds.test:
                    row, events = run(protocol, contender, regime, instance, seed, commit)
                    rows.append(row)
                    if (instance, regime, seed) == (LOG_INSTANCE, LOG_REGIME, protocol.seeds.test[0]):
                        write_log(args.out / "logs" / f"{instance}__{regime}__s{seed}__{row.policy}.json", row, events, schema)

    pd.DataFrame(asdict(row) for row in rows).to_csv(args.out / "episodes.csv", index=False)
    summary = summarise(rows)
    summary.to_csv(args.out / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.float_format", "{:.3f}".format):
        print(summary.drop(columns=["size"]).to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
