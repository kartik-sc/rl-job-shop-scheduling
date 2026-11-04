"""Job-shop simulator.

Two layers, deliberately separate:

- Mechanics: :func:`dispatch` places an operation at
  ``max(machine_free_at, job_free_at)``, matching ``job_shop_lib``'s start-time
  rule. Replay-testable against the library.
- Decision policy: :func:`decision_machine` and :func:`legal_jobs` define the
  per-machine decision point the RL environment needs. The library has no
  equivalent, since it selects operations globally.

Machine health and maintenance arrive on Day 3 and belong in the decision layer.
"""

from dataclasses import dataclass, replace

from backend.env.instance import Instance, Operation


@dataclass(frozen=True, slots=True)
class ScheduledOperation:
    job_id: int
    position: int
    machine_id: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class State:
    next_position: tuple[int, ...]
    machine_free_at: tuple[int, ...]
    job_free_at: tuple[int, ...]
    scheduled: tuple[ScheduledOperation, ...]


def initial_state(instance: Instance) -> State:
    return State(
        next_position=(0,) * instance.num_jobs,
        machine_free_at=(0,) * instance.num_machines,
        job_free_at=(0,) * instance.num_jobs,
        scheduled=(),
    )


def is_complete(state: State, instance: Instance) -> bool:
    return all(
        position == len(job)
        for position, job in zip(state.next_position, instance.jobs)
    )


def makespan(state: State) -> int:
    return max((op.end for op in state.scheduled), default=0)


def next_operation(state: State, instance: Instance, job_id: int) -> Operation | None:
    position = state.next_position[job_id]
    job = instance.jobs[job_id]
    return job[position] if position < len(job) else None


def ready_operations(state: State, instance: Instance) -> tuple[Operation, ...]:
    """The next unscheduled operation of every unfinished job."""
    candidates = (
        next_operation(state, instance, job_id)
        for job_id in range(instance.num_jobs)
    )
    return tuple(op for op in candidates if op is not None)


def start_time(state: State, operation: Operation) -> int:
    return max(
        state.machine_free_at[operation.machine_id],
        state.job_free_at[operation.job_id],
    )


def dispatch(state: State, instance: Instance, job_id: int) -> State:
    """Place the job's next operation on its machine. Mechanics layer."""
    operation = next_operation(state, instance, job_id)
    if operation is None:
        raise ValueError(f"job {job_id} has no remaining operations")

    start = start_time(state, operation)
    end = start + operation.duration
    scheduled = ScheduledOperation(
        job_id=job_id,
        position=operation.position,
        machine_id=operation.machine_id,
        start=start,
        end=end,
    )

    return replace(
        state,
        next_position=_with(state.next_position, job_id, operation.position + 1),
        machine_free_at=_with(state.machine_free_at, operation.machine_id, end),
        job_free_at=_with(state.job_free_at, job_id, end),
        scheduled=state.scheduled + (scheduled,),
    )


def current_time(state: State, instance: Instance) -> int:
    """Earliest time any ready operation could start."""
    ready = ready_operations(state, instance)
    if not ready:
        return makespan(state)
    return min(start_time(state, op) for op in ready)


def decision_machine(state: State, instance: Instance) -> int | None:
    """The machine that decides next: lowest id that is idle with a ready job.

    Decision policy layer. Non-delay: only machines that can start work now.
    """
    now = current_time(state, instance)
    machines = {
        op.machine_id
        for op in ready_operations(state, instance)
        if start_time(state, op) == now
    }
    return min(machines, default=None)


def legal_jobs(state: State, instance: Instance, machine_id: int) -> tuple[int, ...]:
    """Jobs whose next operation can start now on this machine."""
    now = current_time(state, instance)
    return tuple(
        op.job_id
        for op in ready_operations(state, instance)
        if op.machine_id == machine_id and start_time(state, op) == now
    )


def _with(values: tuple[int, ...], index: int, value: int) -> tuple[int, ...]:
    return values[:index] + (value,) + values[index + 1 :]
