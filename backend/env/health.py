"""Machine health: wear, failure and the observable risk score.

The kernel only sees :class:`HealthModel`. Wear accrues only while a machine
processes, so each lifetime's failure point is fixed when the lifetime starts,
and every policy meets the same failures on the same seed.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class HealthModel(Protocol):
    """Per-episode health of every machine, in processing-time units."""

    @property
    def maintenance_enabled(self) -> bool: ...

    @property
    def t_pm(self) -> int: ...

    @property
    def t_repair(self) -> int: ...

    def time_to_failure(self, machine_id: int) -> int:
        """Processing time left in this lifetime before the machine fails. At least 1."""
        ...

    def wear(self, machine_id: int, processed: int) -> None: ...

    def reset(self, machine_id: int) -> None:
        """Start the machine's next lifetime."""
        ...

    def age(self, machine_id: int) -> int:
        """Processing time accumulated in the current lifetime."""
        ...

    def risk(self, machine_id: int, in_progress: int = 0) -> float:
        """Observable score in [0, 1]; ``in_progress`` is a running operation's elapsed time."""
        ...


@dataclass(frozen=True)
class Trajectory:
    """One engine's life, per cycle: observable risk and true remaining useful life."""

    risk: Sequence[float]
    rul: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.risk) == 0 or len(self.risk) != len(self.rul):
            raise ValueError("risk and rul must be non-empty and the same length")
        if min(self.risk) < 0.0 or max(self.risk) > 1.0:
            raise ValueError("risk must lie in [0, 1]")


SampleEngine = Callable[[str, np.random.Generator], Trajectory]
Hazard = Callable[[float], float]


def exponential_hazard(base: float, scale: float) -> Hazard:
    """Per-cycle failure probability ``base * exp(-rul / scale)``, capped at 1.

    A placeholder shape: the hazard form and its parameters freeze on Day 4.
    """

    def hazard(rul: float) -> float:
        return min(1.0, base * math.exp(-rul / scale))

    return hazard


def failure_cycle(rul: Sequence[float], hazard: Hazard, draw: float) -> int:
    """The cycle at which an engine fails, by inverting its cumulative hazard.

    ``draw`` is one uniform in [0, 1). Failure is certain by the last cycle.
    """
    survival = 1.0
    for cycle, remaining in enumerate(rul):
        survival *= 1.0 - hazard(remaining)
        if 1.0 - survival > draw:
            return cycle
    return len(rul) - 1


@dataclass(frozen=True, slots=True)
class Regime:
    """A hazard setting. Downtimes are multiples of the instance's mean operation duration."""

    name: str
    hazard: Hazard
    pm_factor: float
    repair_factor: float
    cycles_per_mean_op: float
    pool: str

    def __post_init__(self) -> None:
        if self.pm_factor <= 0 or not 5 <= self.repair_factor / self.pm_factor <= 10:
            raise ValueError("t_repair must be 5-10x t_pm, both positive")
        if self.cycles_per_mean_op <= 0:
            raise ValueError("cycles_per_mean_op must be positive")


class EngineHealth:
    """Health driven by engine trajectories.

    Each lifetime of each machine draws its engine and failure cycle from an
    RNG seeded by ``(seed, machine_id, lifetime_index)``, so the k-th lifetime
    of a machine is the same under every policy. Processing time maps to
    engine cycles at a per-machine rate, the heterogeneity lever. The machine
    fails when its age reaches the start of the failure cycle.
    """

    def __init__(
        self,
        *,
        regime: Regime,
        rates: Sequence[float],
        mean_duration: float,
        seed: int,
        sample_engine: SampleEngine,
        maintenance_enabled: bool,
    ) -> None:
        if not rates or min(rates) <= 0:
            raise ValueError("every machine needs a positive degradation rate")
        if mean_duration <= 0:
            raise ValueError("mean_duration must be positive")

        self.maintenance_enabled = maintenance_enabled
        self.t_pm = max(1, round(regime.pm_factor * mean_duration))
        self.t_repair = max(1, round(regime.repair_factor * mean_duration))
        self._regime = regime
        self._seed = seed
        self._sample_engine = sample_engine
        self._cycles_per_time = [rate * regime.cycles_per_mean_op / mean_duration for rate in rates]
        self._lifetime = [0] * len(rates)
        self._age = [0] * len(rates)

        lifetimes = [self._draw_lifetime(machine_id) for machine_id in range(len(rates))]
        self._trajectory = [trajectory for trajectory, _ in lifetimes]
        self._fail_at = [fail_at for _, fail_at in lifetimes]

    def time_to_failure(self, machine_id: int) -> int:
        return self._fail_at[machine_id] - self._age[machine_id]

    def wear(self, machine_id: int, processed: int) -> None:
        self._age[machine_id] += processed

    def reset(self, machine_id: int) -> None:
        self._lifetime[machine_id] += 1
        self._age[machine_id] = 0
        self._trajectory[machine_id], self._fail_at[machine_id] = self._draw_lifetime(machine_id)

    def age(self, machine_id: int) -> int:
        return self._age[machine_id]

    def risk(self, machine_id: int, in_progress: int = 0) -> float:
        trajectory = self._trajectory[machine_id]
        cycle = math.floor((self._age[machine_id] + in_progress) * self._cycles_per_time[machine_id])
        return float(trajectory.risk[min(cycle, len(trajectory.risk) - 1)])

    def _draw_lifetime(self, machine_id: int) -> tuple[Trajectory, int]:
        rng = np.random.default_rng([self._seed, machine_id, self._lifetime[machine_id]])
        trajectory = self._sample_engine(self._regime.pool, rng)
        cycle = failure_cycle(trajectory.rul, self._regime.hazard, rng.random())
        return trajectory, max(1, math.ceil(cycle / self._cycles_per_time[machine_id]))
