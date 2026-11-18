"""Test doubles shared by the health and environment tests."""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from backend.baselines import dispatch
from backend.env.health import Trajectory
from backend.env.instance import Instance, Operation
from backend.env.simulator import Action, Decision, Dispatch, Maintain, Policy, Wait

NEVER = 10**9


@dataclass
class ScriptedHealth:
    """A HealthModel whose failure points are scripted per (machine, lifetime)."""

    num_machines: int
    fail_at: Callable[[int, int], int] = lambda machine_id, lifetime: NEVER
    maintenance_enabled: bool = True
    t_pm: int = 2
    t_repair: int = 5
    risk_scale: int = 10
    ages: list[int] = field(init=False)
    lifetimes: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.ages = [0] * self.num_machines
        self.lifetimes = [0] * self.num_machines

    def time_to_failure(self, machine_id: int) -> int:
        return self.fail_at(machine_id, self.lifetimes[machine_id]) - self.ages[machine_id]

    def wear(self, machine_id: int, processed: int) -> None:
        self.ages[machine_id] += processed

    def reset(self, machine_id: int) -> None:
        self.lifetimes[machine_id] += 1
        self.ages[machine_id] = 0

    def age(self, machine_id: int) -> int:
        return self.ages[machine_id]

    def risk(self, machine_id: int, in_progress: int = 0) -> float:
        return min(1.0, (self.ages[machine_id] + in_progress) / self.risk_scale)


def scripted(table: dict[tuple[int, int], int]) -> Callable[[int, int], int]:
    return lambda machine_id, lifetime: table.get((machine_id, lifetime), NEVER)


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


def threshold_spt(tau: float) -> Policy:
    def policy(decision: Decision) -> Action:
        if decision.can_maintain and decision.risk is not None and decision.risk >= tau:
            return Maintain()
        if decision.candidates:
            return dispatch.shortest_processing_time(decision)
        return Wait()

    return policy


def random_legal(seed: int) -> Policy:
    rng = np.random.default_rng(seed)

    def policy(decision: Decision) -> Action:
        actions: list[Action] = [Dispatch(job_id) for job_id in decision.job_ids]
        if decision.can_maintain:
            actions.append(Maintain())
        if decision.can_wait:
            actions.append(Wait())
        return actions[int(rng.integers(len(actions)))]

    return policy


def synthetic_engine(pool: str, rng: np.random.Generator) -> Trajectory:
    """Stand-in for Kaul's sample_engine until backend/risk lands."""
    length = int(rng.integers(150, 300))
    return Trajectory(risk=np.linspace(0.0, 1.0, length), rul=np.arange(length - 1, -1, -1, dtype=float))
