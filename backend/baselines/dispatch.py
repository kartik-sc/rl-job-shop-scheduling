"""Classical dispatching rules over the simulator's per-machine decisions.

Each rule picks one candidate from a :class:`Decision`. Ties break on the
lowest job id, so every run is reproducible.

    python -m backend.baselines.dispatch ta01 spt
    python -m backend.baselines.dispatch ta01 fifo --log ta01_fifo.json --regime low
"""

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from backend.env.events import log_document
from backend.env.instance import Instance, from_benchmark
from backend.env.simulator import Candidate, Decision, Dispatch, EpisodeResult, run_episode

Rule = Callable[[Decision], Dispatch]

LOGGABLE_POLICIES = ("spt", "lpt", "fifo")
REGIMES = ("low", "medium", "high")


def shortest_processing_time(decision: Decision) -> Dispatch:
    return _choose(decision, lambda candidate: candidate.duration)


def longest_processing_time(decision: Decision) -> Dispatch:
    return _choose(decision, lambda candidate: -candidate.duration)


def first_in_first_out(decision: Decision) -> Dispatch:
    """The job that has waited longest in this machine's queue."""
    return _choose(decision, lambda candidate: candidate.arrived_at)


def first_come_first_served(decision: Decision) -> Dispatch:
    """The job whose next operation is earliest in its own route.

    This is ``job_shop_lib``'s ``first_come_first_served`` rule, kept so the
    gate can reproduce all three library rules. It ignores waiting time, so it
    is not FIFO in the shop-floor sense; report :func:`first_in_first_out`.
    """
    return _choose(decision, lambda candidate: candidate.position)


RULES: dict[str, Rule] = {
    "spt": shortest_processing_time,
    "lpt": longest_processing_time,
    "fifo": first_in_first_out,
    "fcfs": first_come_first_served,
}


def solve(instance: Instance, rule: Rule) -> EpisodeResult:
    return run_episode(instance, rule)


def _choose(decision: Decision, key: Callable[[Candidate], int]) -> Dispatch:
    best = min(decision.candidates, key=lambda candidate: (key(candidate), candidate.job_id))
    return Dispatch(best.job_id)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a dispatching rule on a benchmark instance.")
    parser.add_argument("instance", help="benchmark name, e.g. ta01")
    parser.add_argument("rule", choices=sorted(RULES))
    parser.add_argument("--log", type=Path, help="write the episode event log as JSON")
    parser.add_argument(
        "--regime",
        choices=REGIMES,
        help="required with --log, since the schema has no no-failure regime yet",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.log and args.regime is None:
        parser.error("--log requires --regime")
    if args.log and args.rule not in LOGGABLE_POLICIES:
        parser.error(f"{args.rule} is not a policy in the event-log schema")

    instance = from_benchmark(args.instance)
    result = solve(instance, RULES[args.rule])
    print(f"{instance.name} {args.rule} makespan={result.makespan}")

    if args.log:
        document = log_document(
            result.events, instance, policy=args.rule, regime=args.regime, seed=args.seed
        )
        args.log.write_text(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
