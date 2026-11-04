"""Day 2 gate: the simulator must be trustworthy before health logic is added."""

import pytest
from job_shop_lib.benchmarking import load_benchmark_instance
from job_shop_lib.dispatching.rules import DispatchingRuleSolver

from backend.baselines import dispatch
from backend.env import simulator as sim
from backend.env.instance import Instance, from_benchmark

GATE_INSTANCES = [f"ta{i:02d}" for i in range(1, 11)]
LIBRARY_RULES = {
    "spt": "shortest_processing_time",
    "lpt": "largest_processing_time",
    "fifo": "first_come_first_served",
}
OWN_RULES = {
    "spt": dispatch.shortest_processing_time,
    "lpt": dispatch.longest_processing_time,
}


@pytest.fixture(scope="module", params=GATE_INSTANCES)
def instance(request: pytest.FixtureRequest) -> Instance:
    return from_benchmark(request.param)


def from_library(name: str):
    return load_benchmark_instance(name)


@pytest.mark.parametrize("rule", sorted(LIBRARY_RULES))
def test_replay_matches_library_makespan(instance: Instance, rule: str) -> None:
    """Mechanics gate: the library's own dispatch order must reproduce its makespan.

    Independent of tie-breaking and of which machine decides, so a failure here
    means the start-time or bookkeeping logic is wrong.
    """
    library_schedule = DispatchingRuleSolver(LIBRARY_RULES[rule]).solve(
        from_library(instance.name)
    )
    state = sim.initial_state(instance)
    for job_id in _replay_order(library_schedule):
        state = sim.dispatch(state, instance, job_id)

    assert sim.is_complete(state, instance)
    assert sim.makespan(state) == library_schedule.makespan()


def _replay_order(schedule) -> list[int]:
    operations = [op for machine in schedule.schedule for op in machine]
    operations.sort(key=lambda op: (op.start_time, op.machine_id))
    return [op.job_id for op in operations]


@pytest.mark.parametrize("rule", sorted(OWN_RULES))
def test_schedule_is_valid(instance: Instance, rule: str) -> None:
    state = dispatch.solve(instance, OWN_RULES[rule])

    assert sim.is_complete(state, instance)
    assert len(state.scheduled) == instance.num_operations
    _assert_no_machine_overlap(state, instance)
    _assert_job_order_respected(state, instance)


def _assert_no_machine_overlap(state: sim.State, instance: Instance) -> None:
    for machine_id in range(instance.num_machines):
        windows = sorted(
            (op.start, op.end) for op in state.scheduled if op.machine_id == machine_id
        )
        for (_, earlier_end), (later_start, _) in zip(windows, windows[1:]):
            assert later_start >= earlier_end


def _assert_job_order_respected(state: sim.State, instance: Instance) -> None:
    for job_id in range(instance.num_jobs):
        operations = sorted(
            (op for op in state.scheduled if op.job_id == job_id),
            key=lambda op: op.position,
        )
        assert [op.position for op in operations] == list(range(len(operations)))
        for earlier, later in zip(operations, operations[1:]):
            assert later.start >= earlier.end
        for op in operations:
            assert op.end - op.start == instance.jobs[job_id][op.position].duration


@pytest.mark.parametrize("rule", sorted(OWN_RULES))
def test_makespan_respects_bounds(instance: Instance, rule: str) -> None:
    result = sim.makespan(dispatch.solve(instance, OWN_RULES[rule]))

    assert result >= instance.trivial_lower_bound
    assert instance.reference_makespan is not None
    assert result >= instance.reference_makespan


@pytest.mark.parametrize("rule", sorted(OWN_RULES))
def test_own_rules_match_library_makespan(instance: Instance, rule: str) -> None:
    """Our per-machine dispatching reproduces the library's global dispatching.

    The library picks one operation globally per iteration; we fill every
    machine that can start now, lowest id first. These coincide because
    candidate operations belong to distinct jobs, so at a fixed time the choice
    on one machine cannot change the choice on another, and SPT/LPT score each
    operation on its own duration.
    """
    ours = sim.makespan(dispatch.solve(instance, OWN_RULES[rule]))
    theirs = (
        DispatchingRuleSolver(LIBRARY_RULES[rule]).solve(from_library(instance.name)).makespan()
    )
    assert ours == theirs


def test_empty_state_has_zero_makespan() -> None:
    instance = from_benchmark("ta01")
    assert sim.makespan(sim.initial_state(instance)) == 0
    assert not sim.is_complete(sim.initial_state(instance), instance)


def test_dispatch_is_immutable() -> None:
    instance = from_benchmark("ta01")
    state = sim.initial_state(instance)
    after = sim.dispatch(state, instance, 0)

    assert state.next_position[0] == 0
    assert after.next_position[0] == 1
    assert after is not state
