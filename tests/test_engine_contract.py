"""Every registered sample_engine must honour the risk interface (INTERFACES.md section 2).

When the C-MAPSS engine is registered in ``backend.training.factory.ENGINES``,
these tests run against it with no further change.
"""

import time

import numpy as np
import pytest

from backend.env.health import Trajectory
from backend.training.factory import ENGINES, sample_engine

POOLS = ("fit", "calibration", "test")


@pytest.fixture(params=sorted(ENGINES))
def engine(request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
    return sample_engine(request.param)


@pytest.mark.parametrize("pool", POOLS)
def test_trajectories_are_valid(engine, pool: str) -> None:  # type: ignore[no-untyped-def]
    for seed in range(20):
        trajectory = engine(pool, np.random.default_rng(seed))
        assert isinstance(trajectory, Trajectory)
        risk, rul = np.asarray(trajectory.risk), np.asarray(trajectory.rul)
        assert risk.min() >= 0.0 and risk.max() <= 1.0
        assert np.all(np.diff(rul) <= 0) and rul[-1] == 0


def test_same_generator_state_gives_the_same_engine(engine) -> None:  # type: ignore[no-untyped-def]
    first = engine("test", np.random.default_rng([5, 3, 1]))
    again = engine("test", np.random.default_rng([5, 3, 1]))
    assert np.array_equal(first.risk, again.risk) and np.array_equal(first.rul, again.rul)


def test_unknown_pool_is_rejected(engine) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises((ValueError, KeyError)):
        engine("train", np.random.default_rng(0))


def test_sampling_is_fast_enough_for_resets(engine) -> None:  # type: ignore[no-untyped-def]
    """Every maintenance and repair samples an engine, so this runs inside PPO training."""
    rng = np.random.default_rng(0)
    started = time.perf_counter()
    for _ in range(200):
        engine("fit", rng)
    assert (time.perf_counter() - started) / 200 < 0.005
