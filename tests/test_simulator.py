"""Day 2 gate: the no-failure simulator must reproduce known dispatching makespans."""

import json
from collections import defaultdict
from functools import cache
from pathlib import Path

import jsonschema
import pytest
from job_shop_lib import Operation as LibraryOperation
from job_shop_lib import Schedule
from job_shop_lib.benchmarking import load_benchmark_instance
from job_shop_lib.dispatching import Dispatcher
from job_shop_lib.dispatching.rules import DispatchingRuleSolver

from backend.baselines import dispatch
from backend.env.events import Event, EventType, log_document
from backend.env.instance import Instance, Operation, from_benchmark
from backend.env.simulator import Decision, Dispatch, EpisodeResult, Simulation, run_episode

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "shared" / "schemas" / "event_log.schema.json"

# Measured with job-shop-lib 1.7.2. Literal on purpose: this test must not move
# if the library's tie-breaking ever changes.
PINNED_MAKESPANS = {
    "ta01": {"spt": 1462, "lpt": 1701, "fcfs": 1438, "fifo": 1486},
    "ta11": {"spt": 1865, "lpt": 2046, "fcfs": 1665, "fifo": 1701},
    "ta21": {"spt": 2175, "lpt": 2266, "fcfs": 1964, "fifo": 2089},
    "ta31": {"spt": 2335, "lpt": 2417, "fcfs": 2143, "fifo": 2277},
    "ta41": {"spt": 2499, "lpt": 2925, "fcfs": 2538, "fifo": 2543},
}
GATE_INSTANCES = sorted({f"ta{i:02d}" for i in range(1, 11)} | set(PINNED_MAKESPANS))
RULE_NAMES = sorted(dispatch.RULES)
Window = tuple[int, int, int]


def library_fifo(dispatcher: Dispatcher) -> LibraryOperation:
    """Queue-arrival FIFO on the library's own dispatcher, as an independent reference."""
    ready_at = dispatcher.job_next_available_time
    return min(
        dispatcher.available_operations(),
        key=lambda operation: (ready_at[operation.job_id], operation.job_id),
    )


LIBRARY_RULES = {
    "spt": "shortest_processing_time",
    "lpt": "largest_processing_time",
    "fcfs": "first_come_first_served",
    "fifo": library_fifo,
}


@cache
def benchmark(name: str) -> Instance:
    return from_benchmark(name)


@cache
def ours(name: str, rule: str) -> EpisodeResult:
    return dispatch.solve(benchmark(name), dispatch.RULES[rule])


@cache
def library(name: str, rule: str) -> Schedule:
    return DispatchingRuleSolver(LIBRARY_RULES[rule]).solve(load_benchmark_instance(name))


def operation_windows(events: tuple[Event, ...]) -> dict[tuple[int, int], Window]:
    """(job, op) -> (machine, start, end), asserting each operation runs exactly once."""
    starts: dict[tuple[int, int], tuple[int, int]] = {}
    windows: dict[tuple[int, int], Window] = {}
    for event in events:
        key = (event.job, event.op)
        if event.type is EventType.OP_START:
            assert key not in starts, f"{key} started twice"
            starts[key] = (event.machine, event.t)
        elif event.type is EventType.OP_END:
            machine, start = starts[key]
            assert machine == event.machine
            assert key not in windows, f"{key} ended twice"
            windows[key] = (machine, start, event.t)
    assert set(starts) == set(windows), "an operation started but never ended"
    return windows


def make_instance(routes: list[list[tuple[int, int]]], num_machines: int) -> Instance:
    """Routes are lists of (machine, duration) per job."""
    jobs = tuple(
        tuple(
            Operation(job_id=job_id, position=position, machine_id=machine, duration=duration)
            for position, (machine, duration) in enumerate(route)
        )
        for job_id, route in enumerate(routes)
    )
    return Instance(name="hand", jobs=jobs, num_machines=num_machines)


def op_start(t: int, machine: int, job: int, op: int) -> Event:
    return Event(t, EventType.OP_START, machine, job, op)


def op_end(t: int, machine: int, job: int, op: int) -> Event:
    return Event(t, EventType.OP_END, machine, job, op)


def episode_end(t: int) -> Event:
    return Event(t, EventType.EPISODE_END)


# --- Reference makespans -----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "rule"),
    [(name, rule) for name, rules in PINNED_MAKESPANS.items() for rule in rules],
)
def test_makespan_matches_pinned_reference(name: str, rule: str) -> None:
    assert ours(name, rule).makespan == PINNED_MAKESPANS[name][rule]


@pytest.mark.parametrize("rule", RULE_NAMES)
@pytest.mark.parametrize("name", GATE_INSTANCES)
def test_makespan_matches_library(name: str, rule: str) -> None:
    assert ours(name, rule).makespan == library(name, rule).makespan()


@pytest.mark.parametrize("rule", RULE_NAMES)
@pytest.mark.parametrize("name", GATE_INSTANCES)
def test_replaying_library_sequences_reproduces_every_operation(name: str, rule: str) -> None:
    """Mechanics gate, independent of our rules and their tie-breaking.

    Each machine follows the library's job sequence; every operation must then
    start and end exactly when the library placed it.
    """
    schedule = library(name, rule)
    sequences = [
        [op.job_id for op in sorted(machine_ops, key=lambda op: op.start_time)]
        for machine_ops in schedule.schedule
    ]
    progress = [0] * len(sequences)

    def follow_library(decision: Decision) -> Dispatch:
        job_id = sequences[decision.machine_id][progress[decision.machine_id]]
        assert job_id in decision.job_ids, (
            f"library runs job {job_id} next on machine {decision.machine_id}, "
            f"but at t={decision.time} the candidates are {decision.job_ids}"
        )
        progress[decision.machine_id] += 1
        return Dispatch(job_id)

    result = run_episode(benchmark(name), follow_library)
    expected = {
        (op.job_id, op.operation.position_in_job): (op.machine_id, op.start_time, op.end_time)
        for machine_ops in schedule.schedule
        for op in machine_ops
    }
    assert operation_windows(result.events) == expected


@pytest.mark.parametrize("rule", RULE_NAMES)
@pytest.mark.parametrize("name", GATE_INSTANCES)
def test_makespan_respects_bounds(name: str, rule: str) -> None:
    instance = benchmark(name)
    makespan = ours(name, rule).makespan

    assert makespan >= instance.trivial_lower_bound
    assert instance.reference_makespan is not None
    assert makespan >= instance.reference_makespan


# --- Validity from the event log alone ----------------------------------------


@pytest.mark.parametrize("rule", RULE_NAMES)
@pytest.mark.parametrize("name", GATE_INSTANCES)
def test_event_log_describes_a_valid_schedule(name: str, rule: str) -> None:
    instance = benchmark(name)
    result = ours(name, rule)
    windows = operation_windows(result.events)

    assert set(windows) == {(op.job_id, op.position) for job in instance.jobs for op in job}
    for job in instance.jobs:
        for op in job:
            machine, start, end = windows[(op.job_id, op.position)]
            assert machine == op.machine_id
            assert end - start == op.duration
        for earlier, later in zip(job, job[1:]):
            assert windows[(later.job_id, later.position)][1] >= windows[(earlier.job_id, earlier.position)][2]

    assert_no_machine_overlap(windows)
    assert_event_order(result)


def assert_no_machine_overlap(windows: dict[tuple[int, int], Window]) -> None:
    spans_by_machine: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for machine, start, end in windows.values():
        spans_by_machine[machine].append((start, end))
    for spans in spans_by_machine.values():
        spans.sort()
        for (_, earlier_end), (later_start, _) in zip(spans, spans[1:]):
            assert later_start >= earlier_end


def assert_event_order(result: EpisodeResult) -> None:
    events = result.events
    times = [event.t for event in events]
    assert times == sorted(times)
    for previous, current in zip(events, events[1:]):
        if previous.t == current.t:
            assert not (previous.type is EventType.OP_START and current.type is EventType.OP_END)

    assert [event.type for event in events].count(EventType.EPISODE_END) == 1
    assert events[-1] == episode_end(result.makespan)
    assert result.makespan == max(event.t for event in events if event.type is EventType.OP_END)


@pytest.mark.parametrize("name", ["ta01", "ta41"])
def test_decisions_follow_the_contract(name: str) -> None:
    instance = benchmark(name)
    decisions: list[Decision] = []

    def recording_spt(decision: Decision) -> Dispatch:
        decisions.append(decision)
        return dispatch.shortest_processing_time(decision)

    run_episode(instance, recording_spt)

    assert len(decisions) == instance.num_operations
    for previous, current in zip(decisions, decisions[1:]):
        assert (previous.time, previous.machine_id) < (current.time, current.machine_id)
    for decision in decisions:
        assert list(decision.job_ids) == sorted(set(decision.job_ids))
        for candidate in decision.candidates:
            operation = instance.jobs[candidate.job_id][candidate.position]
            assert operation.machine_id == decision.machine_id
            assert operation.duration == candidate.duration
            assert candidate.arrived_at <= decision.time


# --- Hand-worked instances -----------------------------------------------------

# Machine 2 frees at t=6 exactly as job 1 arrives there, while job 0 has been
# waiting since t=3: fifo and fcfs disagree, and completions must come first.
HAND = make_instance(
    [[(0, 1), (1, 2), (2, 3)], [(0, 5), (2, 2)], [(2, 6)]],
    num_machines=3,
)
HAND_PREFIX = [
    op_start(0, 0, 0, 0),  # jobs 0 and 1 tie on machine 0; lowest job id wins
    op_start(0, 2, 2, 0),
    op_end(1, 0, 0, 0),
    op_start(1, 0, 1, 0),
    op_start(1, 1, 0, 1),
    op_end(3, 1, 0, 1),
    op_end(6, 0, 1, 0),
    op_end(6, 2, 2, 0),
]


def test_hand_instance_fifo_takes_the_job_that_waited_longest() -> None:
    result = dispatch.solve(HAND, dispatch.first_in_first_out)
    assert list(result.events) == HAND_PREFIX + [
        op_start(6, 2, 0, 2),
        op_end(9, 2, 0, 2),
        op_start(9, 2, 1, 1),
        op_end(11, 2, 1, 1),
        episode_end(11),
    ]


def test_hand_instance_fcfs_takes_the_lowest_position() -> None:
    result = dispatch.solve(HAND, dispatch.first_come_first_served)
    assert list(result.events) == HAND_PREFIX + [
        op_start(6, 2, 1, 1),
        op_end(8, 2, 1, 1),
        op_start(8, 2, 0, 2),
        op_end(11, 2, 0, 2),
        episode_end(11),
    ]


def test_completions_are_applied_before_machines_decide() -> None:
    decisions: list[Decision] = []

    def recording_fifo(decision: Decision) -> Dispatch:
        decisions.append(decision)
        return dispatch.first_in_first_out(decision)

    run_episode(HAND, recording_fifo)
    at_six = next(d for d in decisions if d.time == 6)

    assert at_six.machine_id == 2
    assert [(c.job_id, c.position, c.arrived_at) for c in at_six.candidates] == [(0, 2, 3), (1, 1, 6)]


def test_job_revisiting_a_machine() -> None:
    instance = make_instance([[(0, 2), (1, 1), (0, 2)], [(0, 1), (1, 3)]], num_machines=2)
    result = dispatch.solve(instance, dispatch.shortest_processing_time)

    assert list(result.events) == [
        op_start(0, 0, 1, 0),
        op_end(1, 0, 1, 0),
        op_start(1, 0, 0, 0),
        op_start(1, 1, 1, 1),
        op_end(3, 0, 0, 0),
        op_end(4, 1, 1, 1),
        op_start(4, 1, 0, 1),
        op_end(5, 1, 0, 1),
        op_start(5, 0, 0, 2),
        op_end(7, 0, 0, 2),
        episode_end(7),
    ]


def test_instance_without_operations_ends_at_zero() -> None:
    result = run_episode(make_instance([[]], num_machines=1), dispatch.shortest_processing_time)
    assert result.makespan == 0
    assert result.events == (episode_end(0),)


# --- Kernel invariants and error paths -------------------------------------------


def test_clock_equals_makespan_when_episode_ends() -> None:
    simulation = Simulation(benchmark("ta01"))
    while (decision := simulation.next_decision()) is not None:
        simulation.apply(dispatch.shortest_processing_time(decision))

    assert simulation.is_complete
    assert simulation.clock == simulation.makespan == PINNED_MAKESPANS["ta01"]["spt"]

    logged = len(simulation.events)
    assert simulation.next_decision() is None
    assert len(simulation.events) == logged
    assert simulation.events_since(logged - 1) == (episode_end(simulation.makespan),)


def test_next_decision_is_idempotent_until_applied() -> None:
    simulation = Simulation(benchmark("ta01"))
    assert simulation.next_decision() is simulation.next_decision()


@pytest.mark.parametrize("rule", RULE_NAMES)
def test_runs_are_deterministic(rule: str) -> None:
    first = dispatch.solve(benchmark("ta01"), dispatch.RULES[rule])
    second = dispatch.solve(benchmark("ta01"), dispatch.RULES[rule])
    assert first == second


def test_apply_without_pending_decision_raises() -> None:
    with pytest.raises(RuntimeError):
        Simulation(benchmark("ta01")).apply(Dispatch(0))


def test_apply_rejects_a_job_that_is_not_a_candidate() -> None:
    simulation = Simulation(benchmark("ta01"))
    decision = simulation.next_decision()
    assert decision is not None
    outsider = next(job_id for job_id in range(15) if job_id not in decision.job_ids)

    with pytest.raises(ValueError):
        simulation.apply(Dispatch(outsider))
    assert simulation.next_decision() == decision


@pytest.mark.parametrize(
    "routes",
    [[[(0, 0)]], [[(0, -1)]], [[(1, 3)]]],
    ids=["zero-duration", "negative-duration", "unknown-machine"],
)
def test_invalid_instances_are_rejected(routes: list[list[tuple[int, int]]]) -> None:
    with pytest.raises(ValueError):
        Simulation(make_instance(routes, num_machines=1))


# --- Event-log schema ---------------------------------------------------------------


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def test_schema_is_valid(schema: dict) -> None:
    jsonschema.Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("rule", dispatch.LOGGABLE_POLICIES)
def test_log_document_matches_schema(schema: dict, rule: str) -> None:
    instance = benchmark("ta01")
    document = log_document(ours("ta01", rule).events, instance, policy=rule, regime="low", seed=0)

    jsonschema.validate(document, schema)
    assert document["meta"] == {
        "instance": "ta01",
        "regime": "low",
        "seed": 0,
        "policy": rule,
        "n_jobs": 15,
        "n_machines": 15,
    }


def test_event_dict_drops_missing_fields() -> None:
    assert episode_end(3).to_dict() == {"t": 3, "type": "episode_end"}
    assert op_start(0, 1, 2, 3).to_dict() == {"t": 0, "type": "op_start", "machine": 1, "job": 2, "op": 3}


# --- Command line ---------------------------------------------------------------------


def test_cli_prints_makespan(capsys: pytest.CaptureFixture[str]) -> None:
    dispatch.main(["ta01", "spt"])
    assert capsys.readouterr().out.strip() == "ta01 spt makespan=1462"


def test_cli_writes_a_schema_valid_log(tmp_path: Path, schema: dict) -> None:
    path = tmp_path / "log.json"
    dispatch.main(["ta01", "fifo", "--log", str(path), "--regime", "low"])
    jsonschema.validate(json.loads(path.read_text()), schema)


@pytest.mark.parametrize(
    "argv",
    [
        ["ta01", "spt", "--log", "unused.json"],
        ["ta01", "fcfs", "--log", "unused.json", "--regime", "low"],
    ],
    ids=["log-without-regime", "fcfs-is-not-a-schema-policy"],
)
def test_cli_refuses_logs_the_schema_cannot_hold(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        dispatch.main(argv)
