"""Gymnasium wrapper, observation builder and snapshot."""

import copy
import dataclasses
import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from backend.baselines import dispatch
from backend.env.events import EventType
from backend.env.gym_env import JobShopEnv, Mode, benchmark_sampler
from backend.env.health import EngineHealth, Regime, exponential_hazard
from backend.env.instance import Instance, from_benchmark
from backend.env.observation import ObservationLayout, build_observation
from backend.env.simulator import (
    JobView,
    MachineView,
    Simulation,
    Snapshot,
    run_episode,
)
from tests.fakes import NEVER, ScriptedHealth, make_instance, synthetic_engine, threshold_spt

# Lifetimes of a few mean operations, so episodes see breakdowns and maintenance.
HIGH = Regime(
    name="high",
    hazard=exponential_hazard(base=0.05, scale=20.0),
    pm_factor=1.0,
    repair_factor=6.0,
    cycles_per_mean_op=40.0,
    pool="fit",
)
SMALL = make_instance([[(0, 3), (1, 2), (2, 4)], [(1, 2), (0, 4)], [(2, 3), (0, 1), (1, 2)]], num_machines=3)


def unit_rates(instance: Instance, rng: np.random.Generator) -> list[float]:
    return [1.0] * instance.num_machines


def make_env(mode: Mode, sampler=benchmark_sampler(["ta01", "ta02"]), **overrides) -> JobShopEnv:  # type: ignore[no-untyped-def]
    arguments = {
        "instance_sampler": sampler,
        "regime": HIGH,
        "mode": mode,
        "sample_engine": synthetic_engine,
        "rates": unit_rates,
    }
    arguments.update(overrides)
    return JobShopEnv(**arguments)  # type: ignore[arg-type]


def play(env: JobShopEnv, seed: int, policy_seed: int, options: dict | None = None):  # type: ignore[no-untyped-def]
    """A random masked episode. Returns rewards, infos, masks and the final info."""
    rng = np.random.default_rng(policy_seed)
    observation, info = env.reset(seed=seed, options=options)
    rewards, infos, masks = [], [info], []
    while True:
        mask = env.action_masks()
        masks.append(mask)
        observation, reward, terminated, truncated, info = env.step(rng.choice(np.flatnonzero(mask)))
        assert observation in env.observation_space
        rewards.append(reward)
        infos.append(info)
        if terminated or truncated:
            return rewards, infos, masks, terminated


# --- check_env ---------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["full", "blind"])
def test_check_env_passes_without_warnings(mode: Mode) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(make_env(mode), skip_render_check=True)


# --- Masks match kernel legality ---------------------------------------------------------------


@pytest.mark.parametrize("mode", ["full", "blind"])
def test_mask_matches_kernel_legality_for_every_action(mode: Mode) -> None:
    env = make_env(mode, sampler=lambda rng: SMALL)
    rng = np.random.default_rng(0)
    for seed in range(20):
        env.reset(seed=seed)
        while env._decision is not None:
            mask = env.action_masks()
            for index in range(env.action_space.n):
                probe = copy.deepcopy(env._simulation)
                try:
                    probe.apply(env._to_kernel(index))  # type: ignore[union-attr]
                    legal = True
                except ValueError:
                    legal = False
                assert mask[index] == legal, (seed, index, env._decision)
            env.step(rng.choice(np.flatnonzero(mask)))


def test_masks_on_ta01_reject_exactly_the_unmasked_actions() -> None:
    env = make_env("full")
    env.reset(seed=3, options={"instance_name": "ta01"})
    rng = np.random.default_rng(1)
    seen = np.zeros(env.action_space.n, dtype=bool)
    while env._decision is not None:
        mask = env.action_masks()
        seen |= mask
        decision = env._decision
        for index in np.flatnonzero(~mask):
            with pytest.raises(ValueError):
                env._simulation.apply(env._to_kernel(int(index)))  # type: ignore[union-attr]
            assert env._decision == decision
        env.step(rng.choice(np.flatnonzero(mask)))
    assert seen[env._maintain_index] and seen[env._wait_index]


def test_blind_mode_never_offers_maintain_or_wait() -> None:
    env = make_env("blind")
    _, infos, masks, terminated = play(env, seed=4, policy_seed=4, options={"instance_name": "ta01"})

    assert terminated
    assert not any(mask[env._maintain_index] or mask[env._wait_index] for mask in masks)
    types = {event["type"] for info in infos for event in info["events"]}
    assert EventType.BREAKDOWN.value in types
    assert EventType.MAINTENANCE_START.value not in types


# --- Episodes ------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["full", "blind"])
def test_return_equals_negative_makespan(mode: Mode) -> None:
    env = make_env(mode)
    rewards, infos, _, terminated = play(env, seed=11, policy_seed=2, options={"instance_name": "ta01"})

    assert terminated
    makespan = infos[-1]["makespan"]
    assert sum(rewards) * from_benchmark("ta01").mean_duration == pytest.approx(-makespan, rel=1e-12)


@pytest.mark.parametrize("mode", ["full", "blind"])
def test_info_delivers_every_event_exactly_once(mode: Mode) -> None:
    env = make_env(mode)
    _, infos, _, _ = play(env, seed=5, policy_seed=5)

    delivered = [event for info in infos for event in info["events"]]
    assert delivered == [event.to_dict() for event in env._simulation.events]  # type: ignore[union-attr]
    assert delivered[-1]["type"] == EventType.EPISODE_END.value


def test_reset_seed_is_the_health_seed() -> None:
    """So the paired-comparison guarantee tested in test_health carries over to the env."""
    env = make_env("blind")
    env.reset(seed=21, options={"instance_name": "ta01"})
    instance = from_benchmark("ta01")
    direct = EngineHealth(
        regime=HIGH,
        rates=[1.0] * 15,
        mean_duration=instance.mean_duration,
        seed=21,
        sample_engine=synthetic_engine,
        maintenance_enabled=False,
    )
    in_env = env._simulation._health  # type: ignore[union-attr]
    assert [in_env.time_to_failure(m) for m in range(15)] == [direct.time_to_failure(m) for m in range(15)]


def test_options_select_instance_and_regime() -> None:
    env = make_env("full")
    low = dataclasses.replace(HIGH, name="low", cycles_per_mean_op=1.0)
    _, info = env.reset(seed=0, options={"instance_name": "ta11", "regime": low})

    assert info["instance"] == "ta11"
    assert info["episode_seed"] == 0
    assert env._simulation._health._regime is low  # type: ignore[union-attr]


def test_unseeded_resets_draw_fresh_health_seeds() -> None:
    env = make_env("full")
    env.reset(seed=0)
    seeds = {env.reset()[1]["episode_seed"] for _ in range(5)}
    assert len(seeds) == 5


def test_illegal_action_falls_back_to_the_lowest_legal_action() -> None:
    first, second = make_env("full"), make_env("full")
    first.reset(seed=8)
    second.reset(seed=8)
    mask = first.action_masks()
    illegal = int(np.flatnonzero(~mask)[0])
    lowest = int(np.flatnonzero(mask)[0])

    observation_a, reward_a, *_, info_a = first.step(illegal)
    observation_b, reward_b, *_, info_b = second.step(lowest)

    assert info_a["illegal_action"] and not info_b["illegal_action"]
    assert np.array_equal(observation_a, observation_b) and reward_a == reward_b


def test_truncates_at_max_steps() -> None:
    env = make_env("full", max_steps=10)
    _, _, masks, terminated = play(env, seed=0, policy_seed=0)
    assert not terminated and len(masks) == 10


def test_step_before_reset_raises() -> None:
    with pytest.raises(RuntimeError):
        make_env("full").step(0)


def test_instance_larger_than_the_layout_is_rejected() -> None:
    env = make_env("full", sampler=benchmark_sampler(["ta31"]))  # 30 x 15
    with pytest.raises(ValueError):
        env.reset(seed=0)


# --- Observation -----------------------------------------------------------------------------


def test_observation_size_matches_layout() -> None:
    full, blind = make_env("full"), make_env("blind")
    assert full.reset(seed=0)[0].shape == (ObservationLayout(include_health=True).size,)
    assert blind.reset(seed=0)[0].shape == (ObservationLayout(include_health=False).size,)
    assert blind.observation_space.shape[0] == full.observation_space.shape[0] - 3 * 15


def test_observations_do_not_depend_on_future_failures() -> None:
    """Two worlds that differ only in when machine 3 fails look identical until it does."""
    instance = from_benchmark("ta01")
    layout = ObservationLayout(include_health=True)

    def trace(fail_at):  # type: ignore[no-untyped-def]
        health = ScriptedHealth(15, fail_at=fail_at, t_pm=20, t_repair=150, risk_scale=250)
        simulation = Simulation(instance, health)
        policy = threshold_spt(tau=2.0)
        observations = []
        while (decision := simulation.next_decision()) is not None:
            observations.append((decision.time, build_observation(simulation.snapshot(), decision, layout)))
            simulation.apply(policy(decision))
        return observations, simulation.events

    never, _ = trace(lambda m, k: NEVER)
    doomed, events = trace(lambda m, k: 300 if (m, k) == (3, 0) else NEVER)

    failed_at = next(e.t for e in events if e.type is EventType.BREAKDOWN)
    doomed_start = max(e.t for e in events if e.type is EventType.OP_START and e.machine == 3 and e.t < failed_at)
    before = [(t, o) for t, o in doomed if t < failed_at]

    assert any(doomed_start <= t < failed_at for t, _ in before), "no decision while the failure was pending"
    for (time_a, observation_a), (time_b, observation_b) in zip(never, before, strict=False):
        assert time_a == time_b
        assert np.array_equal(observation_a, observation_b), f"observation differs at t={time_a}"


def test_snapshot_exposes_only_observable_state() -> None:
    def names(cls: type) -> set[str]:
        return {f.name for f in dataclasses.fields(cls)}

    assert names(Snapshot) == {"time", "instance", "machines", "jobs"}
    assert names(JobView) == {"job_id", "position", "arrived_at"}
    assert names(MachineView) == {
        "machine_id", "status", "job_id", "queue", "free_at",
        "remaining_ops", "last_reset_at", "age", "risk",
    }


def test_busy_machine_shows_its_nominal_end_not_its_breakdown() -> None:
    instance = make_instance([[(0, 10)]], num_machines=1)
    simulation = Simulation(instance, ScriptedHealth(1, fail_at=lambda m, k: 4 if k == 0 else NEVER))
    simulation.apply(dispatch.shortest_processing_time(simulation.next_decision()))  # type: ignore[arg-type]

    view = simulation.snapshot().machines[0]
    assert view.free_at == 10


def test_snapshot_is_frozen() -> None:
    snapshot = Simulation(SMALL).snapshot()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.time = 5  # type: ignore[misc]


def test_snapshot_without_health_has_no_age_or_risk() -> None:
    simulation = Simulation(SMALL)
    decision = simulation.next_decision()
    snapshot = simulation.snapshot()

    assert all(m.age is None and m.risk is None for m in snapshot.machines)
    observation = build_observation(snapshot, decision, ObservationLayout(include_health=False))
    assert observation.min() >= 0.0 and observation.max() <= 1.0


# --- run_episode guard -------------------------------------------------------------------------


def test_run_episode_stops_an_endless_episode() -> None:
    # Every lifetime fails after 2 units, so a 3-unit operation never completes.
    instance = make_instance([[(0, 3)]], num_machines=1)
    health = ScriptedHealth(1, fail_at=lambda m, k: 2, maintenance_enabled=False)

    with pytest.raises(RuntimeError, match="no completion"):
        run_episode(instance, dispatch.shortest_processing_time, health, max_decisions=50)


def test_guard_does_not_trip_on_a_normal_episode() -> None:
    result = run_episode(SMALL, dispatch.shortest_processing_time, max_decisions=SMALL.num_operations)
    assert result.makespan > 0
