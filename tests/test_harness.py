"""Training and evaluation harness: protocol, factory, threshold rule, metrics, pairing, aggregation."""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.baselines.dispatch import shortest_processing_time
from backend.baselines.threshold import threshold_spt
from backend.env.events import EventType
from backend.env.instance import from_benchmark, random_instance
from backend.env.simulator import Candidate, Decision, Dispatch, Maintain, Simulation, Wait, run_episode
from backend.evaluation.actors import reference_rule, rule_actor, threshold
from backend.evaluation.aggregate import check, signed_rank_p
from backend.evaluation.episodes import Cell, play_episode, run_cell
from backend.evaluation.metrics import episode_metrics
from backend.risk import synthetic
from backend.training.factory import make_env, rate_sampler, training_sampler
from backend.training.protocol import MAX_JOBS, MAX_MACHINES, Protocol, load_protocol
from tests import fakes

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PROTOCOL = CONFIGS / "protocol.toml"
SMOKE = CONFIGS / "smoke.toml"


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    return load_protocol(PROTOCOL)


@pytest.fixture(scope="module")
def smoke() -> Protocol:
    return load_protocol(SMOKE)


# --- Protocol ----------------------------------------------------------------------------


def test_protocol_meets_the_locked_requirements(protocol: Protocol) -> None:
    assert protocol.regime_names == ("low", "medium", "high")
    assert len(protocol.seeds.test) >= 10
    assert not set(protocol.split.train_instances) & set(protocol.split.test_instances)
    assert protocol.pool("train") == protocol.pool("tune") == "fit"
    assert protocol.pool("test") == "test"
    assert 1.0 in protocol.tau_grid


@pytest.mark.parametrize("path", [PROTOCOL, SMOKE])
def test_every_named_instance_loads_and_fits_the_layout(path: Path) -> None:
    split = load_protocol(path).split
    for name in split.train_instances + split.test_instances:
        instance = from_benchmark(name)
        assert instance.num_jobs <= MAX_JOBS and instance.num_machines <= MAX_MACHINES
        assert instance.reference_makespan is not None


def test_protocol_is_frozen(protocol: Protocol) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        protocol.max_steps = 1  # type: ignore[misc]


def test_hash_changes_with_the_file(tmp_path: Path, smoke: Protocol) -> None:
    edited = tmp_path / "edited.toml"
    edited.write_text(SMOKE.read_text() + "\n# a comment still changes the protocol\n")
    assert load_protocol(edited).sha256 != smoke.sha256
    assert load_protocol(SMOKE).sha256 == smoke.sha256


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('test_instances = ["la01", "la02"]', 'test_instances = ["la01", "ft06"]', "both train and test"),
        ("test = [200, 201]", "test = [101, 200]", "both tune and test"),
        ("tau_grid = [0.5, 1.0]", "tau_grid = [0.5, 0.9]", "must include 1.0"),
        ("tau_grid = [0.5, 1.0]", "tau_grid = [1.5, 1.0]", r"every tau in \[0, 1\]"),
        ("repair_factor = 6.0", "repair_factor = 20.0", "5-10x"),
        ("random_sizes = [[6, 6]]", "random_sizes = [[25, 6]]", "exceeds"),
        ("net_arch = [32, 32]", "net_arch = [32, 32]\ngamma = 0.99", "gamma is fixed"),
        ("n_epochs = 1", "n_epochs = 1\nepochs = 3", "malformed"),
        ('engine = "synthetic"', 'engine = "synthetic"\nextra = 1', "unknown top-level"),
    ],
)
def test_invalid_protocols_are_rejected(tmp_path: Path, old: str, new: str, message: str) -> None:
    text = SMOKE.read_text()
    assert old in text
    path = tmp_path / "bad.toml"
    path.write_text(text.replace(old, new, 1))
    with pytest.raises(ValueError, match=message):
        load_protocol(path)


# --- Instances, engines and samplers ---------------------------------------------------


def test_random_instance_is_a_valid_taillard_style_instance() -> None:
    instance = random_instance(np.random.default_rng(3), 20, 15, (1, 99))

    assert (instance.num_jobs, instance.num_machines) == (20, 15)
    for job in instance.jobs:
        assert sorted(op.machine_id for op in job) == list(range(15))
        assert all(1 <= op.duration <= 99 for op in job)
    assert instance.reference_makespan is None
    assert run_episode(instance, shortest_processing_time).makespan >= instance.trivial_lower_bound
    assert random_instance(np.random.default_rng(3), 20, 15).jobs == instance.jobs


def test_synthetic_engine_is_valid_and_deterministic() -> None:
    first = synthetic.sample_engine("fit", np.random.default_rng(7))
    again = synthetic.sample_engine("fit", np.random.default_rng(7))

    assert np.array_equal(first.risk, again.risk) and np.array_equal(first.rul, again.rul)
    assert 128 <= len(first.rul) <= 362 and first.rul[-1] == 0
    with pytest.raises(ValueError):
        synthetic.sample_engine("train", np.random.default_rng(0))


def test_training_sampler_mixes_random_and_benchmark_instances(protocol: Protocol) -> None:
    sample = training_sampler(protocol.split)
    rng = np.random.default_rng(0)
    names = {sample(rng).name for _ in range(200)}

    assert names >= set(protocol.split.train_instances)
    assert names >= {f"random_{jobs}x{machines}" for jobs, machines in protocol.split.random_sizes}
    assert not names & set(protocol.split.test_instances)


def test_rates_are_log_uniform_in_range(protocol: Protocol) -> None:
    rates = rate_sampler(protocol.rates)(from_benchmark("ta01"), np.random.default_rng(0))
    assert len(rates) == 15
    assert all(protocol.rates.low <= rate <= protocol.rates.high for rate in rates)


def test_phases_use_their_own_pool(protocol: Protocol) -> None:
    pools = {phase: make_env(protocol, "full", "high", phase)._regime.pool for phase in ("train", "tune", "test")}
    assert pools == {"train": "fit", "tune": "fit", "test": "test"}


# --- Threshold + SPT ---------------------------------------------------------------------------


def decision(risk: float, *, candidates: bool = True) -> Decision:
    jobs = (Candidate(job_id=0, position=0, duration=5, arrived_at=0), Candidate(1, 0, 3, 0))
    return Decision(
        time=0,
        machine_id=0,
        candidates=jobs if candidates else (),
        can_maintain=True,
        can_wait=not candidates,
        risk=risk,
    )


def test_threshold_maintains_only_strictly_above_tau() -> None:
    policy = threshold_spt(0.5)
    assert policy(decision(0.51)) == Maintain()
    assert policy(decision(0.5)) == Dispatch(1)
    assert policy(decision(0.2, candidates=False)) == Wait()


def test_threshold_respects_legality() -> None:
    blocked = dataclasses.replace(decision(0.9), can_maintain=False)
    assert threshold_spt(0.1)(blocked) == Dispatch(1)


@pytest.mark.parametrize("tau", [-0.1, 1.01])
def test_threshold_rejects_tau_outside_unit_interval(tau: float) -> None:
    with pytest.raises(ValueError):
        threshold_spt(tau)


# --- Metrics -------------------------------------------------------------------------------------

HAND = fakes.make_instance([[(0, 3), (1, 2)], [(0, 2), (1, 4)]], num_machines=2)


def test_metrics_on_the_hand_worked_episode() -> None:
    """One breakdown at 4 (started at 2, repaired 4-9), one maintenance 6-8, makespan 14."""
    health = fakes.ScriptedHealth(num_machines=2, fail_at=fakes.scripted({(0, 0): 4}))
    result = run_episode(HAND, fakes.threshold_spt(0.3), health)

    metrics = episode_metrics([event.to_dict() for event in result.events], HAND)

    assert metrics.makespan == 14
    assert metrics.gap is None
    assert (metrics.breakdowns, metrics.lost_work, metrics.repair_downtime) == (1, 2, 5)
    assert (metrics.maintenances, metrics.maintenance_downtime) == (1, 2)
    assert metrics.breakdown_rate == 25.0
    assert metrics.maintenance_share == pytest.approx(2 / 28)


def test_gap_uses_the_reference_makespan() -> None:
    instance = from_benchmark("ta01")
    result = run_episode(instance, shortest_processing_time)
    metrics = episode_metrics([event.to_dict() for event in result.events], instance)
    assert metrics.gap == pytest.approx(1462 / 1231 - 1)


def test_metrics_reject_an_incomplete_log() -> None:
    events = [{"t": 0, "type": EventType.OP_START.value, "machine": 0, "job": 0, "op": 0}]
    with pytest.raises(ValueError, match="incomplete"):
        episode_metrics(events, HAND)


# --- Pairing ---------------------------------------------------------------------------------------


def health_at_reset(protocol: Protocol, mode: str, seed: int):  # type: ignore[no-untyped-def]
    env = make_env(protocol, mode, "high", "test")  # type: ignore[arg-type]
    env.reset(seed=seed, options={"instance_name": "ta06"})
    health = env._simulation._health  # type: ignore[union-attr]
    return health._cycles_per_time, [health.time_to_failure(m) for m in range(15)]


def test_blind_and_full_envs_draw_the_same_rates_and_failures(protocol: Protocol) -> None:
    assert health_at_reset(protocol, "blind", 2000) == health_at_reset(protocol, "full", 2000)
    assert health_at_reset(protocol, "blind", 2000) != health_at_reset(protocol, "blind", 2001)


def test_never_maintaining_threshold_matches_blind_spt(protocol: Protocol) -> None:
    """tau = 1 in full mode differs from blind SPT only by waiting, which changes nothing."""
    for seed in protocol.seeds.test[:3]:
        blind = play_episode(
            make_env(protocol, "blind", "high", "test"), rule_actor(shortest_processing_time),
            instance_name="ta06", seed=seed,
        )
        full = play_episode(
            make_env(protocol, "full", "high", "test"), threshold(1.0).build(), instance_name="ta06", seed=seed
        )
        instance = from_benchmark("ta06")
        blind_metrics, full_metrics = episode_metrics(blind.events, instance), episode_metrics(full.events, instance)
        assert blind_metrics.breakdowns > 0
        assert full_metrics == blind_metrics


def test_rule_actor_reproduces_the_kernel_policy(smoke: Protocol) -> None:
    record = run_cell(Cell(smoke, "test", reference_rule("spt"), "low", "test", "la01"))[0]
    env = make_env(smoke, "blind", "low", "test")
    env.reset(seed=smoke.seeds.test[0], options={"instance_name": "la01"})
    simulation = env._simulation
    while (pending := simulation.next_decision()) is not None:  # type: ignore[union-attr]
        simulation.apply(shortest_processing_time(pending))  # type: ignore[union-attr]
    assert record.makespan == simulation.makespan  # type: ignore[union-attr]


def test_encode_inverts_the_env_decoding(smoke: Protocol) -> None:
    env = make_env(smoke, "full", "high", "test")
    env.reset(seed=0, options={"instance_name": "la01"})
    rng = np.random.default_rng(0)
    while env.decision is not None:
        legal = np.flatnonzero(env.action_masks())
        assert all(env.encode(env._to_kernel(int(index))) == index for index in legal)
        env.step(rng.choice(legal))


# --- Aggregation checks -----------------------------------------------------------------------------


def episode_rows(sha256: str, **overrides: object) -> pd.DataFrame:
    rows = [
        {
            "protocol": sha256, "policy": policy, "variant": "", "regime": "low", "instance": instance,
            "size": "10x5", "seed": seed, "truncated": False, "illegal_actions": 0, "gap": 0.1,
        }
        for policy in ("spt", "threshold_spt")
        for instance in ("la01", "la02")
        for seed in (200, 201)
    ]
    frame = pd.DataFrame(rows)
    for column, value in overrides.items():
        frame.loc[0, column] = value
    return frame


def test_aggregate_accepts_a_consistent_grid(smoke: Protocol) -> None:
    check(episode_rows(smoke.sha256), smoke)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"protocol": "0" * 64}, "other protocols"),
        ({"truncated": True}, "truncated"),
        ({"illegal_actions": 2}, "illegal"),
        ({"seed": 999}, "different"),
    ],
)
def test_aggregate_rejects_inconsistent_episodes(smoke: Protocol, override: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check(episode_rows(smoke.sha256, **override), smoke)


def test_signed_rank_handles_identical_policies() -> None:
    assert np.isnan(signed_rank_p(np.zeros(5)))
    assert 0 < signed_rank_p(np.array([-0.1, -0.2, -0.05, -0.3, 0.02])) < 1


def test_simulation_accepts_random_instances_with_health() -> None:
    instance = random_instance(np.random.default_rng(1), 6, 6)
    simulation = Simulation(instance, fakes.ScriptedHealth(6))
    assert simulation.next_decision() is not None
