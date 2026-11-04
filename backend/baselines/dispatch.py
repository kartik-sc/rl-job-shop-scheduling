"""Classical dispatching rules over the per-machine decision point.

Each rule picks one job from the legal jobs of the deciding machine. Ties break
on the lowest job id, which keeps every run reproducible.
"""

from collections.abc import Callable

from backend.env import simulator as sim
from backend.env.instance import Instance

Rule = Callable[[sim.State, Instance, tuple[int, ...], int], int]


def shortest_processing_time(
    state: sim.State, instance: Instance, jobs: tuple[int, ...], machine_id: int
) -> int:
    return min(jobs, key=lambda job_id: (_duration(state, instance, job_id), job_id))


def longest_processing_time(
    state: sim.State, instance: Instance, jobs: tuple[int, ...], machine_id: int
) -> int:
    return min(jobs, key=lambda job_id: (-_duration(state, instance, job_id), job_id))


def first_in_first_out(
    state: sim.State, instance: Instance, jobs: tuple[int, ...], machine_id: int
) -> int:
    """TODO(kartik): choose and implement the FIFO definition.

    Two defensible readings, and they schedule differently:

    a) Queue arrival order: the job that has been waiting longest at this
       machine, i.e. the smallest ``state.job_free_at[job_id]``. This is what
       FIFO means on a real shop floor queue.
    b) Earliest in its job: the smallest ``position`` of the next operation,
       which is what ``job_shop_lib``'s ``first_come_first_served_rule`` does.
       It favours jobs that have barely started, regardless of waiting time.

    (a) is the honest baseline and reads as FIFO to anyone reviewing the
    report; (b) makes our numbers directly comparable to the library's. Break
    ties on the lowest job id either way.
    """
    raise NotImplementedError


def run(state: sim.State, instance: Instance, rule: Rule) -> sim.State:
    """Dispatch to completion under the given rule."""
    while not sim.is_complete(state, instance):
        machine_id = sim.decision_machine(state, instance)
        if machine_id is None:
            raise RuntimeError("no machine can act but the schedule is incomplete")
        jobs = sim.legal_jobs(state, instance, machine_id)
        state = sim.dispatch(state, instance, rule(state, instance, jobs, machine_id))
    return state


def solve(instance: Instance, rule: Rule) -> sim.State:
    return run(sim.initial_state(instance), instance, rule)


def _duration(state: sim.State, instance: Instance, job_id: int) -> int:
    operation = sim.next_operation(state, instance, job_id)
    if operation is None:
        raise ValueError(f"job {job_id} has no remaining operations")
    return operation.duration
