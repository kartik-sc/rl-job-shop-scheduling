"""The C-MAPSS adapter: Kaul's real calibration output, served to the simulator."""

from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from backend.baselines.threshold import threshold_spt
from backend.env import risk_adapter
from backend.env.gym_env import JobShopEnv, benchmark_sampler
from backend.env.health import (
    EngineHealth,
    Hazard,
    Regime,
    SampleEngine,
    Trajectory,
    exponential_hazard,
)
from backend.env.risk_adapter import (
    CmapssEngines,
    cmapss_engines,
    kaul_split,
    load_engines,
)
from backend.env.simulator import Decision, MachineView, Snapshot
from backend.evaluation.actors import rule_actor
from backend.training.factory import rate_sampler
from backend.training.protocol import RateSpec

DETECTORS = ("ocsvm", "iforest")
RATES = rate_sampler(RateSpec(low=0.5, high=2.0))


MEDIUM_HAZARD = exponential_hazard(0.05, 20.0)


def regime(hazard: Hazard = MEDIUM_HAZARD) -> Regime:
    return Regime(
        name="medium", hazard=hazard, pm_factor=1.0, repair_factor=6.0, cycles_per_mean_op=14.0, pool="calibration"
    )


def env(sample_engine: SampleEngine, hazard: Hazard = MEDIUM_HAZARD) -> JobShopEnv:
    return JobShopEnv(
        instance_sampler=benchmark_sampler(["ta01"]),
        regime=regime(hazard),
        mode="full",
        sample_engine=sample_engine,
        rates=RATES,
    )


@pytest.mark.parametrize("detector", DETECTORS)
def test_serves_the_forty_calibration_engines_twenty_per_dataset(detector: str) -> None:
    engines = load_engines(detector)  # type: ignore[arg-type]
    assert {e.engine_id for e in engines} == set(kaul_split()["calibration"])
    assert sum(e.dataset == "FD001" for e in engines) == sum(e.dataset == "FD003" for e in engines) == 20


def test_split_matches_kauls_saved_scores() -> None:
    split = kaul_split()
    fit_scored = set(pd.read_csv(risk_adapter.RESULTS_DIR / "all_fit_scores.csv")["engine_id"])
    assert set(split["fit"]) == fit_scored
    assert not set(split["fit"]) & set(split["calibration"])
    assert not set(split["test"]) & (set(split["fit"]) | set(split["calibration"]))


@pytest.mark.parametrize("detector", DETECTORS)
def test_risk_lies_in_unit_interval_and_rul_runs_to_zero(detector: str) -> None:
    for engine in load_engines(detector):  # type: ignore[arg-type]
        risk, rul = np.asarray(engine.trajectory.risk), np.asarray(engine.trajectory.rul)
        assert 0.0 <= risk.min() and risk.max() <= 1.0
        assert np.all(np.diff(rul) == -1) and rul[-1] == 0


def test_each_engine_carries_its_own_datasets_calibration() -> None:
    """Risk is replayed per engine, so an FD003 engine keeps the FD003 calibrator's output."""
    predictions = pd.read_csv(risk_adapter.predictions_path("ocsvm"))
    for engine in load_engines("ocsvm"):
        rows = predictions[predictions["engine_id"] == engine.engine_id].sort_values("cycle")
        assert engine.dataset == engine.engine_id[:5]
        assert np.array_equal(engine.trajectory.risk, rows["risk_score"].to_numpy())


def test_rows_relabelled_to_the_other_dataset_are_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    predictions = pd.read_csv(risk_adapter.predictions_path("ocsvm"))
    predictions.loc[predictions["engine_id"] == "FD003_006", "dataset"] = "FD001"
    (tmp_path / "calibration_experiments").mkdir()
    predictions.to_csv(risk_adapter.predictions_path("ocsvm", tmp_path), index=False)
    (tmp_path / "all_fit_scores.csv").write_bytes((risk_adapter.RESULTS_DIR / "all_fit_scores.csv").read_bytes())
    with pytest.raises(ValueError):
        load_engines("ocsvm", tmp_path)


def test_fit_pool_is_refused_and_test_pool_is_empty() -> None:
    engines = cmapss_engines()
    with pytest.raises(ValueError, match="fit"):
        engines("fit", np.random.default_rng(0))
    with pytest.raises(ValueError, match="test"):
        engines("test", np.random.default_rng(0))
    with pytest.raises(ValueError):
        engines("train", np.random.default_rng(0))


def test_fit_engines_cannot_be_registered() -> None:
    split = kaul_split()
    fit_engine = risk_adapter.Engine(split["fit"][0], "FD001", Trajectory(risk=(0.5,), rul=(0.0,)))
    with pytest.raises(ValueError, match="fit"):
        CmapssEngines((fit_engine,), split)


def test_same_seed_gives_the_same_engine_draws() -> None:
    engines = cmapss_engines()
    first = [engines.draw("calibration", np.random.default_rng([7, m, k])).engine_id for m in range(15) for k in range(5)]
    again = [engines.draw("calibration", np.random.default_rng([7, m, k])).engine_id for m in range(15) for k in range(5)]
    assert first == again
    assert len(set(first)) > 1


def test_lifetimes_do_not_depend_on_the_order_machines_reset() -> None:
    """Two policies reset machines in different orders; the k-th lifetime must still match."""

    def health() -> EngineHealth:
        return EngineHealth(
            regime=regime(),
            rates=[1.0, 0.7, 1.6],
            mean_duration=50.0,
            seed=2000,
            sample_engine=risk_adapter.sample_engine,
            maintenance_enabled=True,
        )

    def lifetimes(model: EngineHealth, order: list[int]) -> dict[tuple[int, int], tuple[int, float]]:
        seen: dict[tuple[int, int], tuple[int, float]] = {}
        counts = [0, 0, 0]
        for machine in order:
            seen[machine, counts[machine]] = (model.time_to_failure(machine), model.risk(machine))
            model.reset(machine)
            counts[machine] += 1
        return seen

    assert lifetimes(health(), [0, 1, 2, 0, 1, 2, 0]) == lifetimes(health(), [2, 2, 0, 1, 0, 0, 1])


def test_simulator_state_exposes_no_remaining_life() -> None:
    for cls in (Snapshot, MachineView, Decision):
        names = {field.name for field in fields(cls)}
        assert not {"rul", "fail_at", "time_to_failure", "trajectory"} & names


def test_observations_are_identical_when_only_true_rul_differs() -> None:
    """With failure pinned to each engine's last cycle, RUL values reach nothing a policy sees.

    A zero hazard makes the failure cycle depend only on trajectory length, so
    replacing every RUL value leaves the episode unchanged unless RUL leaks
    into the observation.
    """
    engines = cmapss_engines()

    def scrambled(pool: str, rng: np.random.Generator) -> Trajectory:
        trajectory = engines(pool, rng)
        return Trajectory(risk=trajectory.risk, rul=tuple(7.0 * r + 1000.0 for r in trajectory.rul))

    real, fake = env(engines, hazard=lambda rul: 0.0), env(scrambled, hazard=lambda rul: 0.0)
    actor = rule_actor(threshold_spt(0.8))
    obs_real, _ = real.reset(seed=3)
    obs_fake, _ = fake.reset(seed=3)
    for _ in range(400):
        assert np.array_equal(obs_real, obs_fake)
        action = actor(real, obs_real)
        assert action == actor(fake, obs_fake)
        obs_real, _, done, _, _ = real.step(action)
        obs_fake, _, _, _, _ = fake.step(action)
        if done:
            break


def test_true_rul_still_drives_breakdowns() -> None:
    engines = cmapss_engines()

    def shifted(pool: str, rng: np.random.Generator) -> Trajectory:
        trajectory = engines(pool, rng)
        return Trajectory(risk=trajectory.risk, rul=tuple(r + 300.0 for r in trajectory.rul))

    def failures(sample_engine: SampleEngine) -> list[int]:
        model = EngineHealth(
            regime=regime(), rates=[1.0] * 10, mean_duration=50.0, seed=1,
            sample_engine=sample_engine, maintenance_enabled=False,
        )
        return [model.time_to_failure(machine) for machine in range(10)]

    assert failures(engines) != failures(shifted)


def test_observed_risk_is_the_engines_risk_score() -> None:
    environment = env(cmapss_engines())
    observation, info = environment.reset(seed=11)
    risks = {e["machine"]: e["risk"] for e in info["events"] if e["type"] == "risk_update"}
    layout_offset = 20 * (8 + 15)
    machine_width = 8 + 3
    for machine, risk in risks.items():
        assert observation[layout_offset + machine * machine_width + 8] == pytest.approx(risk, abs=1e-6)
        assert 0.0 <= risk <= 1.0
