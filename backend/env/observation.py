"""Fixed-size observation vector built from a :class:`Snapshot`.

Every feature lies in [0, 1]: flags, fractions, risk, and non-negative times in
units of the instance's mean operation duration squashed by ``x / (1 + x)``.
Slots beyond the instance's jobs and machines are zero, with ``present = 0``.

Layout, in order:
    per job slot      present, done, candidate, running, next duration,
                      remaining work, remaining-operations fraction, waiting
                      time, one-hot of the next operation's machine
    per machine slot  present, idle, busy, offline, time until free, queue
                      length, queued work, remaining-operations fraction,
                      and with health: risk, age, time since reset
    decision          one-hot of the deciding machine, can_maintain, can_wait
    global            time, fraction of operations completed

The blind layout drops the three health features.
"""

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from backend.env.simulator import Decision, MachineStatus, Snapshot

_JOB_FEATURES = 8
_MACHINE_FEATURES = 8
_HEALTH_FEATURES = 3
_GLOBAL_FEATURES = 2


@dataclass(frozen=True, slots=True)
class ObservationLayout:
    max_jobs: int = 20
    max_machines: int = 15
    include_health: bool = True

    @property
    def job_width(self) -> int:
        return _JOB_FEATURES + self.max_machines

    @property
    def machine_width(self) -> int:
        return _MACHINE_FEATURES + (_HEALTH_FEATURES if self.include_health else 0)

    @property
    def size(self) -> int:
        return (
            self.max_jobs * self.job_width
            + self.max_machines * self.machine_width
            + self.max_machines
            + 2
            + _GLOBAL_FEATURES
        )

    def space(self) -> spaces.Box:
        return spaces.Box(low=0.0, high=1.0, shape=(self.size,), dtype=np.float32)


def build_observation(
    snapshot: Snapshot, decision: Decision | None, layout: ObservationLayout
) -> np.ndarray:
    instance = snapshot.instance
    if instance.num_jobs > layout.max_jobs or instance.num_machines > layout.max_machines:
        raise ValueError(
            f"{instance.name} is {instance.num_jobs}x{instance.num_machines}; "
            f"layout holds {layout.max_jobs}x{layout.max_machines}"
        )

    jobs = np.zeros((layout.max_jobs, layout.job_width), dtype=np.float32)
    machines = np.zeros((layout.max_machines, layout.machine_width), dtype=np.float32)
    deciding = np.zeros(layout.max_machines, dtype=np.float32)
    flags = np.zeros(2, dtype=np.float32)

    scale = instance.mean_duration
    now = snapshot.time
    candidates = set(decision.job_ids) if decision is not None else set()
    running = {m.job_id for m in snapshot.machines if m.job_id is not None}

    for job in snapshot.jobs:
        route = instance.jobs[job.job_id]
        row = jobs[job.job_id]
        row[0] = 1.0
        if job.position >= len(route):
            row[1] = 1.0
            continue
        is_running = job.job_id in running
        row[2] = job.job_id in candidates
        row[3] = is_running
        row[4] = _squash(route[job.position].duration / scale)
        row[5] = _squash(sum(op.duration for op in route[job.position :]) / scale)
        row[6] = (len(route) - job.position) / len(route)
        row[7] = 0.0 if is_running else _squash((now - job.arrived_at) / scale)
        row[_JOB_FEATURES + route[job.position].machine_id] = 1.0

    loads = _operations_per_machine(snapshot)
    for machine in snapshot.machines:
        row = machines[machine.machine_id]
        row[0] = 1.0
        row[1] = machine.status is MachineStatus.IDLE
        row[2] = machine.status is MachineStatus.BUSY
        row[3] = machine.status is MachineStatus.OFFLINE
        row[4] = 0.0 if machine.free_at is None else _squash((machine.free_at - now) / scale)
        row[5] = len(machine.queue) / layout.max_jobs
        row[6] = _squash(
            sum(instance.jobs[j][snapshot.jobs[j].position].duration for j in machine.queue) / scale
        )
        row[7] = machine.remaining_ops / loads[machine.machine_id] if loads[machine.machine_id] else 0.0
        if layout.include_health:
            row[8] = machine.risk or 0.0
            row[9] = _squash((machine.age or 0) / scale)
            row[10] = _squash((now - machine.last_reset_at) / scale)

    if decision is not None:
        deciding[decision.machine_id] = 1.0
        flags[:] = (decision.can_maintain, decision.can_wait)

    completed = sum(job.position for job in snapshot.jobs) / instance.num_operations
    globals_ = np.array([_squash(now / scale), completed], dtype=np.float32)

    return np.concatenate([jobs.ravel(), machines.ravel(), deciding, flags, globals_])


def _operations_per_machine(snapshot: Snapshot) -> list[int]:
    counts = [0] * snapshot.instance.num_machines
    for route in snapshot.instance.jobs:
        for operation in route:
            counts[operation.machine_id] += 1
    return counts


def _squash(value: float) -> float:
    return value / (1.0 + value)
