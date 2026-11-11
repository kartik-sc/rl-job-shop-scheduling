"""Synthetic engine trajectories, a stand-in until the C-MAPSS ``sample_engine`` lands.

Lifetimes span 128-362 cycles, the range of the FD001 training engines. Risk
rises with wear along a per-engine curve plus noise, so it says something
about remaining life without being a perfect proxy for it. The pool is
validated but does not change the distribution.
"""

import numpy as np

from backend.env.health import Trajectory

POOLS = ("fit", "calibration", "test")
_SHORTEST, _LONGEST = 128, 362
_NOISE = 0.05


def sample_engine(pool: str, rng: np.random.Generator) -> Trajectory:
    if pool not in POOLS:
        raise ValueError(f"unknown pool {pool!r}; expected one of {POOLS}")
    length = int(rng.integers(_SHORTEST, _LONGEST + 1))
    wear = np.arange(length) / (length - 1)
    curve = wear ** rng.uniform(1.0, 3.0)
    risk = np.clip(curve + rng.normal(0.0, _NOISE, length), 0.0, 1.0)
    rul = np.arange(length - 1, -1, -1, dtype=float)
    return Trajectory(risk=risk, rul=rul)
