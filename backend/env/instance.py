"""Typed, immutable view of a job-shop instance."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from job_shop_lib.benchmarking import load_benchmark_instance


@dataclass(frozen=True, slots=True)
class Operation:
    job_id: int
    position: int
    machine_id: int
    duration: int


@dataclass(frozen=True)
class Instance:
    name: str
    jobs: tuple[tuple[Operation, ...], ...]
    num_machines: int
    optimum: int | None = None
    upper_bound: int | None = None

    @property
    def num_jobs(self) -> int:
        return len(self.jobs)

    @property
    def num_operations(self) -> int:
        return sum(len(job) for job in self.jobs)

    @cached_property
    def job_durations(self) -> tuple[int, ...]:
        return tuple(sum(op.duration for op in job) for job in self.jobs)

    @cached_property
    def machine_loads(self) -> tuple[int, ...]:
        loads = [0] * self.num_machines
        for job in self.jobs:
            for op in job:
                loads[op.machine_id] += op.duration
        return tuple(loads)

    @cached_property
    def mean_duration(self) -> float:
        return sum(self.job_durations) / self.num_operations

    @cached_property
    def trivial_lower_bound(self) -> int:
        return max(max(self.job_durations), max(self.machine_loads))

    @property
    def reference_makespan(self) -> int | None:
        return self.optimum if self.optimum is not None else self.upper_bound


def from_benchmark(name: str) -> Instance:
    """Build an Instance from a job_shop_lib benchmark such as 'ta01'."""
    source = load_benchmark_instance(name)
    jobs = tuple(
        tuple(
            Operation(
                job_id=op.job_id,
                position=op.position_in_job,
                machine_id=op.machine_id,
                duration=op.duration,
            )
            for op in job
        )
        for job in source.jobs
    )
    return Instance(
        name=source.name,
        jobs=jobs,
        num_machines=source.num_machines,
        optimum=source.metadata.get("optimum"),
        upper_bound=source.metadata.get("upper_bound"),
    )


def random_instance(
    rng: np.random.Generator,
    num_jobs: int,
    num_machines: int,
    durations: tuple[int, int] = (1, 99),
) -> Instance:
    """A Taillard-style instance: every job visits every machine once, in random
    order, with durations uniform on the inclusive range. It has no known optimum."""
    low, high = durations
    if low < 1 or high < low:
        raise ValueError(f"durations must satisfy 1 <= low <= high, got {durations}")
    jobs = tuple(
        tuple(
            Operation(
                job_id=job_id,
                position=position,
                machine_id=int(machine_id),
                duration=int(rng.integers(low, high + 1)),
            )
            for position, machine_id in enumerate(rng.permutation(num_machines))
        )
        for job_id in range(num_jobs)
    )
    return Instance(name=f"random_{num_jobs}x{num_machines}", jobs=jobs, num_machines=num_machines)
