"""Run the selected contenders over the test grid: test instances x regimes x test seeds.

    python -m backend.evaluation.evaluate --protocol configs/protocol.toml [--policies spt full_ppo] [--workers 8] [--save-logs]

Writes, under ``results/<hash>/``:
    episodes/<policy>__<regime>[__<variant>].csv   one per contender and regime;
                                                   skipped if present, so a stopped run resumes
    reference.csv                                  best-known makespans and no-failure rules
    logs/*.json                                    with --save-logs: one instance and seed,
                                                   every contender and regime, for the dashboard
"""

import argparse
import csv
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from backend.baselines.dispatch import RULES, solve
from backend.env.events import log_document
from backend.env.instance import from_benchmark
from backend.evaluation.actors import REFERENCE_RULES, Contender, ppo, reference_rule, threshold
from backend.evaluation.episodes import Cell, play_episode, results_dir, run_cells, write_records
from backend.training.factory import make_env
from backend.training.protocol import MODES, Protocol, git_commit, load_protocol

POLICIES = ("blind_ppo", "full_ppo", "threshold_spt", *REFERENCE_RULES)


def contenders(
    protocol: Protocol, selection: dict[str, Any], policies: Sequence[str]
) -> Iterator[tuple[str, Contender]]:
    """``(regime, contender)`` pairs; tau and PPO checkpoints come from ``selection.json``."""
    for regime in protocol.regime_names:
        for name in REFERENCE_RULES:
            if name in policies:
                yield regime, reference_rule(name)
        if "threshold_spt" in policies:
            yield regime, threshold(_selected(selection, "threshold", regime)["tau"])
        for mode in MODES:
            if f"{mode}_ppo" in policies:
                for seed, entry in _selected(selection, "ppo", mode, regime).items():
                    yield regime, ppo(mode, Path(entry["checkpoint"]), seed)


def episodes_file(out: Path, regime: str, contender: Contender) -> Path:
    suffix = f"__{contender.variant}" if contender.variant else ""
    return out / "episodes" / f"{contender.policy}__{regime}{suffix}.csv"


def evaluate(
    protocol: Protocol,
    policies: Sequence[str] = POLICIES,
    *,
    workers: int = 1,
    results_root: Path = Path("results"),
) -> list[Path]:
    out = results_dir(protocol, results_root)
    selection = _load_selection(out, protocol)
    commit = git_commit()
    write_reference(out / "reference.csv", protocol)

    written = []
    for regime, contender in contenders(protocol, selection, policies):
        path = episodes_file(out, regime, contender)
        if path.exists():
            continue
        cells = [
            Cell(protocol, commit, contender, regime, "test", instance)
            for instance in protocol.split.test_instances
        ]
        write_records(path, run_cells(cells, workers))
        print(f"wrote {path}")
        written.append(path)
    return written


def save_logs(
    protocol: Protocol,
    instance_name: str,
    seed: int,
    policies: Sequence[str] = POLICIES,
    *,
    results_root: Path = Path("results"),
) -> list[Path]:
    """The same instance and failures under every contender, as event-log documents."""
    out = results_dir(protocol, results_root)
    selection = _load_selection(out, protocol)
    instance = from_benchmark(instance_name)
    written = []
    for regime, contender in contenders(protocol, selection, policies):
        env = make_env(protocol, contender.mode, regime, "test")
        outcome = play_episode(env, contender.build(), instance_name=instance_name, seed=seed)
        document = log_document((), instance, policy=contender.policy, regime=regime, seed=seed)
        document["events"] = outcome.events
        suffix = f"_{contender.variant}" if contender.variant else ""
        path = out / "logs" / f"{instance_name}_{regime}_s{seed}_{contender.policy}{suffix}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))
        written.append(path)
    return written


def write_reference(path: Path, protocol: Protocol) -> None:
    """Best-known makespans, and the classical rules without failures (the Day 2 setting)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["instance", "size", "reference", "optimum_known"]
    columns += [f"{rule}_{column}" for rule in REFERENCE_RULES for column in ("makespan", "gap")]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name in protocol.split.test_instances:
            instance = from_benchmark(name)
            reference = instance.reference_makespan
            row: dict[str, Any] = {
                "instance": name,
                "size": f"{instance.num_jobs}x{instance.num_machines}",
                "reference": reference,
                "optimum_known": instance.optimum is not None,
            }
            for rule in REFERENCE_RULES:
                makespan = solve(instance, RULES[rule]).makespan
                row[f"{rule}_makespan"] = makespan
                row[f"{rule}_gap"] = None if reference is None else makespan / reference - 1.0
            writer.writerow(row)


def _load_selection(out: Path, protocol: Protocol) -> dict[str, Any]:
    path = out / "selection.json"
    if not path.exists():
        return {"protocol": protocol.sha256}
    selection = json.loads(path.read_text())
    if selection["protocol"] != protocol.sha256:
        raise ValueError(f"{path} belongs to protocol {selection['protocol']}")
    return selection


def _selected(selection: dict[str, Any], *keys: str) -> Any:
    node: Any = selection
    for key in keys:
        if key not in node:
            raise KeyError(
                f"selection.json has no {'/'.join(keys)}; run backend.evaluation.selection first, "
                "or leave that policy out with --policies"
            )
        node = node[key]
    return node


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the selected policies on the test grid.")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--save-logs", action="store_true", help="also write dashboard event logs")
    parser.add_argument("--log-instance", help="defaults to the first test instance")
    parser.add_argument("--log-seed", type=int, help="defaults to the first test seed")
    args = parser.parse_args(argv)

    protocol = load_protocol(args.protocol)
    evaluate(protocol, args.policies, workers=args.workers, results_root=args.results)
    if args.save_logs:
        instance = args.log_instance or protocol.split.test_instances[0]
        seed = args.log_seed if args.log_seed is not None else protocol.seeds.test[0]
        for path in save_logs(protocol, instance, seed, args.policies, results_root=args.results):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
