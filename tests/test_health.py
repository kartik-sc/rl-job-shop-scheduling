"""Health layer on the kernel: breakdowns, maintenance, waits and risk logging."""

import copy
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from backend.baselines import dispatch
from backend.env.events import Event, EventType, log_document
from backend.env.health import (
    EngineHealth,
    HealthModel,
    Regime,
    Trajectory,
    exponential_hazard,
    failure_cycle,
)
from backend.env.instance import Instance, from_benchmark
from backend.env.simulator import (
    Action,
    Decision,
    Dispatch,
    Maintain,
    Policy,
    Simulation,
    Wait,
    run_episode,
)
from tests.fakes import (
    ScriptedHealth,
    make_instance,
    random_legal,
    scripted,
    synthetic_engine,
    threshold_spt,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "shared" / "schemas" / "event_log.schema.json"
# Proposed by the health-layer design; the committed schema gains it once the team agrees.
PENDING_EVENT_TYPES = ["repair_end"]


def recording(policy: Policy, decisions: list[Decision]) -> Policy:
    def record(decision: Decision) -> Action:
        decisions.append(decision)
        return policy(decision)

    return record


def op_start(t: int, machine: int, job: int, op: int) -> Event:
    return Event(t, EventType.OP_START, machine, job, op)


def op_end(t: int, machine: int, job: int, op: int) -> Event:
    return Event(t, EventType.OP_END, machine, job, op)


def breakdown(t: int, machine: int, job: int, op: int) -> Event:
    return Event(t, EventType.BREAKDOWN, machine, job, op)


def risk_update(t: int, machine: int, risk: float) -> Event:
    return Event(t, EventType.RISK_UPDATE, machine, risk=risk)


def machine_event(t: int, event_type: EventType, machine: int) -> Event:
    return Event(t, event_type, machine)


def episode_end(t: int) -> Event:
    return Event(t, EventType.EPISODE_END)


def operation_events(events: tuple[Event, ...]) -> list[Event]:
    return [e for e in events if e.type in (EventType.OP_START, EventType.OP_END, EventType.EPISODE_END)]


# --- Hand-worked episode ------------------------------------------------------------

# Machine 0 fails after 4 units of processing in its first lifetime; nothing else
# ever fails. Under threshold + SPT (tau = 0.3) the episode has one breakdown,
# one maintenance, and three moments where a machine is skipped: machine 1 at
# t=0 (fresh, empty queue), machine 1 at t=8 (just reset) and machine 0 at t=12
# (no operations left).
HAND = make_instance([[(0, 3), (1, 2)], [(0, 2), (1, 4)]], num_machines=2)


def hand_health(**overrides: object) -> ScriptedHealth:
    return ScriptedHealth(num_machines=2, fail_at=scripted({(0, 0): 4}), **overrides)  # type: ignore[arg-type]


def test_hand_episode_event_log() -> None:
    result = run_episode(HAND, threshold_spt(0.3), hand_health())

    assert list(result.events) == [
        risk_update(0, 0, 0.0),
        risk_update(0, 1, 0.0),
        op_start(0, 0, 1, 0),
        op_end(2, 0, 1, 0),
        risk_update(2, 0, 0.2),
        risk_update(2, 1, 0.0),
        op_start(2, 0, 0, 0),  # would end at 5, machine 0 fails at age 4
        op_start(2, 1, 1, 1),
        breakdown(4, 0, 0, 0),
        op_end(6, 1, 1, 1),
        risk_update(6, 0, 0.4),  # offline machines report their pre-reset risk
        risk_update(6, 1, 0.4),
        machine_event(6, EventType.MAINTENANCE_START, 1),
        machine_event(8, EventType.MAINTENANCE_END, 1),
        risk_update(8, 1, 0.0),
        machine_event(9, EventType.REPAIR_END, 0),
        risk_update(9, 0, 0.0),
        risk_update(9, 0, 0.0),
        risk_update(9, 1, 0.0),
        op_start(9, 0, 0, 0),  # restarted at full duration
        op_end(12, 0, 0, 0),
        risk_update(12, 0, 0.3),
        risk_update(12, 1, 0.0),
        op_start(12, 1, 0, 1),
        op_end(14, 1, 0, 1),
        episode_end(14),
    ]
    assert result.makespan == 14


def test_hand_episode_decisions() -> None:
    decisions: list[Decision] = []
    run_episode(HAND, recording(threshold_spt(0.3), decisions), hand_health())

    assert [
        (d.time, d.machine_id, d.job_ids, d.can_maintain, d.can_wait, d.risk) for d in decisions
    ] == [
        (0, 0, (0, 1), False, False, 0.0),
        (2, 0, (0,), True, False, 0.2),
        (2, 1, (1,), False, False, 0.0),
        (6, 1, (), True, True, 0.4),
        (9, 0, (0,), False, False, 0.0),
        (12, 1, (0,), False, False, 0.0),
    ]


def test_breakdown_requeues_the_job_with_its_original_arrival() -> None:
    # Job 0 breaks down on machine 0 at t=2. Job 1 reaches machine 0 at t=1,
    # so after the repair fifo must still pick job 0, which arrived at t=0.
    instance = make_instance([[(0, 3)], [(1, 1), (0, 1)]], num_machines=2)
    health = ScriptedHealth(
        num_machines=2, fail_at=scripted({(0, 0): 2}), maintenance_enabled=False, t_repair=10
    )
    decisions: list[Decision] = []
    result = run_episode(instance, recording(dispatch.first_in_first_out, decisions), health)

    after_repair = next(d for d in decisions if d.time == 12)
    assert [(c.job_id, c.position, c.arrived_at) for c in after_repair.candidates] == [(0, 0, 0), (1, 1, 1)]
    assert operation_events(result.events) == [
        op_start(0, 0, 0, 0),
        op_start(0, 1, 1, 0),
        op_end(1, 1, 1, 0),
        op_start(12, 0, 0, 0),
        op_end(15, 0, 0, 0),
        op_start(15, 0, 1, 1),
        op_end(16, 0, 1, 1),
        episode_end(16),
    ]
    assert breakdown(2, 0, 0, 0) in result.events


@pytest.mark.parametrize(
    ("fail_at", "first_close"),
    [(3, breakdown(3, 0, 0, 0)), (4, op_end(3, 0, 0, 0))],
    ids=["duration-equals-time-to-failure", "duration-one-short"],
)
def test_operation_breaks_down_when_it_reaches_the_failure_point(fail_at: int, first_close: Event) -> None:
    instance = make_instance([[(0, 3)]], num_machines=1)
    health = ScriptedHealth(num_machines=1, fail_at=scripted({(0, 0): fail_at}), maintenance_enabled=False)
    result = run_episode(instance, dispatch.shortest_processing_time, health)

    closes = [e for e in result.events if e.type in (EventType.OP_END, EventType.BREAKDOWN)]
    assert closes[0] == first_close


def test_maintenance_resets_the_machine_when_it_returns() -> None:
    # Machine 1 is serviced from t=2 to t=5. Machine 0 decides at t=3, so the
    # risk batch then must still show machine 1's pre-reset risk.
    instance = make_instance([[(1, 2), (1, 1)], [(0, 3), (0, 1)]], num_machines=2)
    health = ScriptedHealth(num_machines=2, t_pm=3)

    def maintain_machine_1(decision: Decision) -> Action:
        if decision.machine_id == 1 and decision.can_maintain:
            return Maintain()
        return dispatch.shortest_processing_time(decision)

    result = run_episode(instance, maintain_machine_1, health)

    assert machine_event(2, EventType.MAINTENANCE_START, 1) in result.events
    assert risk_update(3, 1, 0.2) in result.events
    assert machine_event(5, EventType.MAINTENANCE_END, 1) in result.events
    assert health.lifetimes == [0, 1]


def test_risk_of_a_busy_machine_counts_elapsed_processing() -> None:
    instance = make_instance([[(0, 5)], [(1, 2), (1, 2)]], num_machines=2)
    result = run_episode(instance, dispatch.shortest_processing_time, ScriptedHealth(num_machines=2))

    assert risk_update(2, 0, 0.2) in result.events


# --- Legality ---------------------------------------------------------------------------


def test_maintain_is_illegal_on_a_fresh_machine() -> None:
    simulation = Simulation(HAND, hand_health())
    decision = simulation.next_decision()
    assert decision is not None and not decision.can_maintain

    with pytest.raises(ValueError):
        simulation.apply(Maintain())
    assert simulation.next_decision() == decision


def test_wait_is_illegal_with_a_non_empty_queue() -> None:
    simulation = Simulation(HAND, hand_health())
    decision = simulation.next_decision()
    assert decision is not None and decision.candidates and not decision.can_wait

    with pytest.raises(ValueError):
        simulation.apply(Wait())


def test_dispatch_is_illegal_when_there_are_no_candidates() -> None:
    simulation = Simulation(HAND, hand_health())
    while (decision := simulation.next_decision()) is not None and decision.candidates:
        simulation.apply(dispatch.shortest_processing_time(decision))
    assert decision is not None and decision.can_maintain

    with pytest.raises(ValueError):
        simulation.apply(Dispatch(0))


def test_blind_mode_never_offers_maintain_or_wait() -> None:
    simulation = Simulation(HAND, hand_health(maintenance_enabled=False))
    simulation.apply(dispatch.shortest_processing_time(simulation.next_decision()))  # type: ignore[arg-type]
    decision = simulation.next_decision()
    assert decision is not None and decision.time == 2 and decision.machine_id == 0

    for action in (Maintain(), Wait()):
        with pytest.raises(ValueError):
            simulation.apply(action)


def test_a_wait_is_not_asked_again_before_the_clock_moves() -> None:
    simulation = Simulation(HAND, hand_health())
    while (decision := simulation.next_decision()) is not None and decision.candidates:
        simulation.apply(dispatch.shortest_processing_time(decision))
    assert decision is not None and decision.can_wait

    simulation.apply(Wait())
    following = simulation.next_decision()
    assert following is None or following.time > decision.time


@pytest.mark.parametrize(
    "health",
    [ScriptedHealth(num_machines=2, t_pm=0), ScriptedHealth(num_machines=2, t_repair=0)],
    ids=["zero-t_pm", "zero-t_repair"],
)
def test_zero_downtimes_are_rejected(health: ScriptedHealth) -> None:
    with pytest.raises(ValueError):
        Simulation(HAND, health)


def test_time_to_failure_below_one_is_an_error() -> None:
    health = ScriptedHealth(num_machines=2, fail_at=scripted({(0, 0): 0}))
    simulation = Simulation(HAND, health)

    with pytest.raises(RuntimeError):
        simulation.apply(dispatch.shortest_processing_time(simulation.next_decision()))  # type: ignore[arg-type]


# --- Equivalence with the no-failure kernel ---------------------------------------------------


@pytest.mark.parametrize("maintenance_enabled", [False, True], ids=["blind", "full"])
def test_health_that_never_fails_reproduces_the_no_failure_schedule(maintenance_enabled: bool) -> None:
    instance = from_benchmark("ta01")
    health = ScriptedHealth(num_machines=15, maintenance_enabled=maintenance_enabled)
    never_maintain = threshold_spt(tau=2.0)

    with_health = run_episode(instance, never_maintain, health)
    without = run_episode(instance, dispatch.shortest_processing_time)

    assert operation_events(with_health.events) == list(without.events)
    assert with_health.makespan == without.makespan == 1462


# --- Whole episodes on ta01 ------------------------------------------------------------------


def ta01_health(maintenance_enabled: bool) -> ScriptedHealth:
    """Failure points of 100-238 always exceed Taillard's longest operation (99)."""
    return ScriptedHealth(
        num_machines=15,
        fail_at=lambda machine_id, lifetime: 100 + 23 * ((3 * machine_id + lifetime) % 7),
        maintenance_enabled=maintenance_enabled,
        t_pm=20,
        t_repair=150,
        risk_scale=250,
    )


EPISODES = {
    "full-threshold": (True, lambda: threshold_spt(0.4)),
    "full-random": (True, lambda: random_legal(seed=7)),
    "blind-spt": (False, lambda: dispatch.shortest_processing_time),
    "blind-fifo": (False, lambda: dispatch.first_in_first_out),
}


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


@pytest.fixture(scope="module")
def proposed_schema(schema: dict) -> dict:
    extended = copy.deepcopy(schema)
    extended["properties"]["events"]["items"]["properties"]["type"]["enum"].extend(PENDING_EVENT_TYPES)
    return extended


@pytest.mark.parametrize("episode", sorted(EPISODES))
def test_ta01_episode_is_valid(episode: str, proposed_schema: dict) -> None:
    maintenance_enabled, make_policy = EPISODES[episode]
    instance = from_benchmark("ta01")
    health = ta01_health(maintenance_enabled)
    simulation = Simulation(instance, health)
    policy = make_policy()

    while (decision := simulation.next_decision()) is not None:
        assert decision.candidates or decision.can_maintain, "forced wait was asked"
        assert maintenance_enabled or not (decision.can_maintain or decision.can_wait)
        assert remaining_on(instance, simulation.events, decision.machine_id) > 0
        simulation.apply(policy(decision))

    events = simulation.events
    assert simulation.is_complete
    assert simulation.clock == simulation.makespan
    assert events[-1] == episode_end(simulation.makespan)
    assert [e.t for e in events] == sorted(e.t for e in events)

    assert_runs_are_well_formed(instance, events)
    assert_downtimes(events, t_pm=health.t_pm, t_repair=health.t_repair)
    assert_no_machine_overlap(events)

    types = {e.type for e in events}
    assert EventType.BREAKDOWN in types, "the episode should exercise breakdowns"
    if episode == "full-threshold":
        assert EventType.MAINTENANCE_START in types

    document = log_document(events, instance, policy="threshold_spt", regime="medium", seed=0)
    jsonschema.validate(document, proposed_schema)


def remaining_on(instance: Instance, events: tuple[Event, ...], machine_id: int) -> int:
    total = sum(1 for job in instance.jobs for op in job if op.machine_id == machine_id)
    done = sum(1 for e in events if e.type is EventType.OP_END and e.machine == machine_id)
    return total - done


def assert_runs_are_well_formed(instance: Instance, events: tuple[Event, ...]) -> None:
    """Every op_start is closed by exactly one op_end or breakdown; each operation completes once."""
    open_runs: dict[int, Event] = {}
    completed: dict[tuple[int, int], tuple[int, int]] = {}
    for event in events:
        if event.type is EventType.OP_START:
            assert event.machine not in open_runs
            open_runs[event.machine] = event  # type: ignore[index]
        elif event.type in (EventType.OP_END, EventType.BREAKDOWN):
            start = open_runs.pop(event.machine)  # type: ignore[arg-type]
            assert (start.job, start.op) == (event.job, event.op)
            if event.type is EventType.OP_END:
                key = (event.job, event.op)
                assert key not in completed
                completed[key] = (start.t, event.t)  # type: ignore[assignment]
    assert not open_runs

    for job in instance.jobs:
        for op in job:
            start, end = completed[(op.job_id, op.position)]
            assert end - start == op.duration
        for earlier, later in zip(job, job[1:]):
            assert completed[(later.job_id, later.position)][0] >= completed[(earlier.job_id, earlier.position)][1]


def assert_downtimes(events: tuple[Event, ...], *, t_pm: int, t_repair: int) -> None:
    pairs = {EventType.BREAKDOWN: (EventType.REPAIR_END, t_repair), EventType.MAINTENANCE_START: (EventType.MAINTENANCE_END, t_pm)}
    for event in events:
        if event.type in pairs:
            closing, downtime = pairs[event.type]
            assert machine_event(event.t + downtime, closing, event.machine) in events  # type: ignore[arg-type]


def assert_no_machine_overlap(events: tuple[Event, ...]) -> None:
    """Runs (including broken ones) and offline windows never overlap on a machine."""
    opening = {EventType.OP_START, EventType.MAINTENANCE_START, EventType.BREAKDOWN}
    closing = {EventType.OP_END, EventType.BREAKDOWN, EventType.MAINTENANCE_END, EventType.REPAIR_END}
    occupied: dict[int, bool] = {}
    for event in events:
        if event.machine is None:
            continue
        if event.type in closing:
            assert occupied.get(event.machine), f"{event} closes nothing"
            occupied[event.machine] = False
        if event.type in opening:
            assert not occupied.get(event.machine), f"{event} opens on a busy machine"
            occupied[event.machine] = True
    assert not any(occupied.values())


def test_episodes_are_deterministic() -> None:
    instance = from_benchmark("ta01")
    first = run_episode(instance, random_legal(seed=3), ta01_health(True))
    second = run_episode(instance, random_legal(seed=3), ta01_health(True))
    assert first == second


def test_health_logs_fit_the_schema_once_repair_end_is_added(proposed_schema: dict) -> None:
    result = run_episode(HAND, threshold_spt(0.3), hand_health())
    jsonschema.validate(log_document(result.events, HAND, policy="threshold_spt", regime="low", seed=0), proposed_schema)


@pytest.mark.xfail(strict=True, reason="repair_end awaits team agreement; remove this mark when the schema adds it")
def test_committed_schema_accepts_repair_end(schema: dict) -> None:
    result = run_episode(HAND, threshold_spt(0.3), hand_health())
    jsonschema.validate(log_document(result.events, HAND, policy="threshold_spt", regime="low", seed=0), schema)


# --- EngineHealth ----------------------------------------------------------------------------


def fixed_engine(pool: str, rng: np.random.Generator) -> Trajectory:
    return Trajectory(risk=np.linspace(0.0, 1.0, 101), rul=np.arange(100, -1, -1, dtype=float))


MEDIUM = Regime(
    name="medium",
    hazard=exponential_hazard(base=0.05, scale=20.0),
    pm_factor=1.0,
    repair_factor=6.0,
    cycles_per_mean_op=2.0,
    pool="fit",
)


def engine_health(seed: int = 0, **overrides: object) -> EngineHealth:
    arguments: dict[str, object] = {
        "regime": MEDIUM,
        "rates": (1.0, 1.5, 0.5),
        "mean_duration": 50.0,
        "seed": seed,
        "sample_engine": synthetic_engine,
        "maintenance_enabled": True,
    }
    arguments.update(overrides)
    return EngineHealth(**arguments)  # type: ignore[arg-type]


def test_engine_health_satisfies_the_protocol() -> None:
    health: HealthModel = engine_health()
    assert health.time_to_failure(0) >= 1


def test_failure_cycle_inverts_the_cumulative_hazard() -> None:
    rul = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    half = lambda remaining: 0.5  # noqa: E731

    assert failure_cycle(rul, half, 0.4) == 0  # F(0) = 0.5
    assert failure_cycle(rul, half, 0.6) == 1  # F(1) = 0.75
    assert failure_cycle(rul, half, 0.8) == 2  # F(2) = 0.875
    assert failure_cycle(rul, lambda remaining: 0.0, 0.0) == len(rul) - 1
    assert failure_cycle(rul, lambda remaining: 1.0, 0.999) == 0


def test_a_lifetime_does_not_depend_on_how_work_is_split() -> None:
    def failure_points(health: EngineHealth, chunks: list[int]) -> list[int]:
        points = []
        for _ in range(4):
            points.append(health.time_to_failure(0) + health.age(0))
            for chunk in chunks:
                health.wear(0, chunk)
                health.wear(1, chunk * 2)
            health.reset(0)
            health.reset(1)
        return points

    one_chunk = failure_points(engine_health(seed=11), [10])
    many_chunks = failure_points(engine_health(seed=11), [1, 2, 3, 4])
    assert one_chunk == many_chunks


def test_lifetimes_match_across_policies_on_the_same_seed() -> None:
    instance = from_benchmark("ta01")
    regime = Regime("high", exponential_hazard(0.05, 20.0), 1.0, 6.0, cycles_per_mean_op=20.0, pool="fit")

    def failure_points(policy: Policy, maintenance_enabled: bool) -> dict[tuple[int, int], int]:
        inner = engine_health(
            seed=5,
            regime=regime,
            rates=[1.0] * 15,
            mean_duration=instance.mean_duration,
            maintenance_enabled=maintenance_enabled,
        )
        probe = FailureProbe(inner)
        run_episode(instance, policy, probe)
        return probe.points

    blind = failure_points(dispatch.shortest_processing_time, maintenance_enabled=False)
    full = failure_points(threshold_spt(0.3), maintenance_enabled=True)

    shared = blind.keys() & full.keys()
    assert any(lifetime >= 1 for _, lifetime in shared), "no shared later lifetime to compare"
    assert {key: blind[key] for key in shared} == {key: full[key] for key in shared}


class FailureProbe:
    """Wraps a HealthModel and records each lifetime's failure point."""

    def __init__(self, inner: EngineHealth) -> None:
        self._inner = inner
        self._lifetimes = [0] * len(inner._age)
        self.points: dict[tuple[int, int], int] = {}
        self.maintenance_enabled = inner.maintenance_enabled
        self.t_pm = inner.t_pm
        self.t_repair = inner.t_repair

    def time_to_failure(self, machine_id: int) -> int:
        remaining = self._inner.time_to_failure(machine_id)
        self.points[(machine_id, self._lifetimes[machine_id])] = remaining + self._inner.age(machine_id)
        return remaining

    def wear(self, machine_id: int, processed: int) -> None:
        self._inner.wear(machine_id, processed)

    def reset(self, machine_id: int) -> None:
        self._lifetimes[machine_id] += 1
        self._inner.reset(machine_id)

    def age(self, machine_id: int) -> int:
        return self._inner.age(machine_id)

    def risk(self, machine_id: int, in_progress: int = 0) -> float:
        return self._inner.risk(machine_id, in_progress)


def test_each_lifetime_draws_a_new_engine() -> None:
    health = engine_health(seed=2)
    points = []
    for _ in range(5):
        points.append(health.time_to_failure(0))
        health.reset(0)
    assert len(set(points)) > 1


def test_different_seeds_draw_different_lifetimes() -> None:
    draws = {
        tuple(health.time_to_failure(m) for m in range(3))
        for health in (engine_health(seed=seed) for seed in range(5))
    }
    assert len(draws) > 1


def test_time_to_failure_is_at_least_one_even_for_certain_failure() -> None:
    certain = Regime("high", lambda remaining: 1.0, 1.0, 6.0, cycles_per_mean_op=2.0, pool="fit")
    health = engine_health(regime=certain)
    assert [health.time_to_failure(m) for m in range(3)] == [1, 1, 1]


def test_downtimes_scale_with_mean_duration() -> None:
    health = engine_health(mean_duration=50.3)
    assert (health.t_pm, health.t_repair) == (50, 302)


def test_risk_follows_the_trajectory_at_the_machine_rate() -> None:
    # 2 cycles per mean operation of 50 time units: 0.04 cycles per unit at rate 1.
    health = engine_health(rates=(1.0, 2.0), sample_engine=fixed_engine)
    health.wear(0, 100)
    health.wear(1, 100)

    assert health.risk(0) == pytest.approx(0.04)
    assert health.risk(0, in_progress=25) == pytest.approx(0.05)
    assert health.risk(1) == pytest.approx(0.08)
    health.wear(0, 10**6)
    assert health.risk(0) == 1.0


@pytest.mark.parametrize("repair_factor", [4.9, 10.1])
def test_regime_requires_repair_to_be_5_to_10_times_maintenance(repair_factor: float) -> None:
    with pytest.raises(ValueError):
        Regime("low", exponential_hazard(0.05, 20.0), 1.0, repair_factor, 2.0, "fit")


@pytest.mark.parametrize(
    ("risk", "rul"),
    [([], []), ([0.1, 0.2], [1.0]), ([0.5, 1.2], [1.0, 0.0])],
    ids=["empty", "length-mismatch", "risk-above-one"],
)
def test_trajectory_rejects_bad_input(risk: list[float], rul: list[float]) -> None:
    with pytest.raises(ValueError):
        Trajectory(risk=risk, rul=rul)
