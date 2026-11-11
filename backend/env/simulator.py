"""Event-driven job-shop simulator.

Mechanics: an integer clock, a heap of timed events and a waiting queue per
machine. The clock only moves by popping the heap, and every timed event at a
timestamp is applied before any machine decides, so completions come first.

Decisions: an idle machine with a non-empty queue needs a decision. Machines
decide in ascending id at each timestamp, and the caller answers each
:class:`Decision` through :meth:`Simulation.apply`. The clock never moves while
a decision is pending, which makes every schedule non-delay.

Machine health extends four seams: ``_TimedKind`` (new timed events),
``_handle`` (what they do), ``_needs_decision`` (idle machines that may
maintain or wait) and the ``Action`` union.
"""

import heapq
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum

from backend.env.events import Event, EventType
from backend.env.instance import Instance, Operation


class MachineStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"


class _TimedKind(IntEnum):
    """Kinds of timed event. Lower values are handled first at equal times."""

    OP_END = 0


@dataclass(slots=True)
class Machine:
    machine_id: int
    status: MachineStatus = MachineStatus.IDLE
    job_id: int | None = None
    queue: list[int] = field(default_factory=list)


@dataclass(slots=True)
class Job:
    """``position`` is the job's current operation, whether queued or running.

    ``arrived_at`` is when the job joined its current machine's queue: the end
    of its previous operation, or 0 for the first one.
    """

    job_id: int
    position: int = 0
    arrived_at: int = 0


@dataclass(frozen=True, slots=True)
class Candidate:
    job_id: int
    position: int
    duration: int
    arrived_at: int


@dataclass(frozen=True, slots=True)
class Decision:
    """A machine that must choose now. Candidates are in ascending job id."""

    time: int
    machine_id: int
    candidates: tuple[Candidate, ...]

    @property
    def job_ids(self) -> tuple[int, ...]:
        return tuple(candidate.job_id for candidate in self.candidates)


@dataclass(frozen=True, slots=True)
class Dispatch:
    job_id: int


Action = Dispatch
Policy = Callable[[Decision], Action]


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    makespan: int
    events: tuple[Event, ...]


class Simulation:
    """One episode. Drive it with :meth:`next_decision` and :meth:`apply`."""

    def __init__(self, instance: Instance) -> None:
        _validate(instance)
        self._instance = instance
        self._clock = 0
        self._makespan = 0
        self._remaining = instance.num_operations
        self._timed: list[tuple[int, _TimedKind, int]] = []
        self._machines = [Machine(machine_id) for machine_id in range(instance.num_machines)]
        self._jobs = [Job(job_id) for job_id in range(instance.num_jobs)]
        self._events: list[Event] = []
        self._pending: Decision | None = None
        self._ended = False
        for job in self._jobs:
            self._enqueue(job)

    @property
    def clock(self) -> int:
        return self._clock

    @property
    def makespan(self) -> int:
        return self._makespan

    @property
    def is_complete(self) -> bool:
        return self._remaining == 0

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    def events_since(self, cursor: int) -> tuple[Event, ...]:
        return tuple(self._events[cursor:])

    def next_decision(self) -> Decision | None:
        """The pending decision, advancing the clock until there is one.

        Returns ``None`` once every operation has finished, at which point
        ``clock == makespan``.
        """
        while self._pending is None:
            machine = self._first_deciding_machine()
            if machine is not None:
                self._pending = self._decision_for(machine)
            elif self._timed:
                self._advance()
            else:
                self._end_episode()
                return None
        return self._pending

    def apply(self, action: Action) -> None:
        decision = self._pending
        if decision is None:
            raise RuntimeError("no pending decision; call next_decision() first")
        if action.job_id not in decision.job_ids:
            raise ValueError(
                f"job {action.job_id} cannot start on machine {decision.machine_id} "
                f"at t={decision.time}; candidates are {decision.job_ids}"
            )
        self._pending = None
        self._start(self._machines[decision.machine_id], self._jobs[action.job_id])

    def _first_deciding_machine(self) -> Machine | None:
        return next((m for m in self._machines if self._needs_decision(m)), None)

    def _needs_decision(self, machine: Machine) -> bool:
        return machine.status is MachineStatus.IDLE and bool(machine.queue)

    def _decision_for(self, machine: Machine) -> Decision:
        candidates = tuple(
            self._candidate(self._jobs[job_id]) for job_id in sorted(machine.queue)
        )
        return Decision(time=self._clock, machine_id=machine.machine_id, candidates=candidates)

    def _candidate(self, job: Job) -> Candidate:
        return Candidate(
            job_id=job.job_id,
            position=job.position,
            duration=self._operation(job).duration,
            arrived_at=job.arrived_at,
        )

    def _advance(self) -> None:
        time = self._timed[0][0]
        self._clock = time
        while self._timed and self._timed[0][0] == time:
            _, kind, machine_id = heapq.heappop(self._timed)
            self._handle(kind, machine_id)

    def _handle(self, kind: _TimedKind, machine_id: int) -> None:
        match kind:
            case _TimedKind.OP_END:
                self._finish_operation(self._machines[machine_id])

    def _start(self, machine: Machine, job: Job) -> None:
        machine.queue.remove(job.job_id)
        machine.status = MachineStatus.BUSY
        machine.job_id = job.job_id
        self._emit(EventType.OP_START, machine=machine.machine_id, job=job.job_id, op=job.position)
        end = self._clock + self._operation(job).duration
        self._schedule(end, _TimedKind.OP_END, machine.machine_id)

    def _finish_operation(self, machine: Machine) -> None:
        if machine.job_id is None:
            raise RuntimeError(f"machine {machine.machine_id} finished with no job running")
        job = self._jobs[machine.job_id]
        self._emit(EventType.OP_END, machine=machine.machine_id, job=job.job_id, op=job.position)

        machine.status = MachineStatus.IDLE
        machine.job_id = None
        self._remaining -= 1
        self._makespan = self._clock

        job.position += 1
        job.arrived_at = self._clock
        self._enqueue(job)

    def _enqueue(self, job: Job) -> None:
        if job.position < len(self._instance.jobs[job.job_id]):
            self._machines[self._operation(job).machine_id].queue.append(job.job_id)

    def _end_episode(self) -> None:
        if self._remaining:
            raise RuntimeError(f"no machine can act but {self._remaining} operations remain")
        if not self._ended:
            self._emit(EventType.EPISODE_END)
            self._ended = True

    def _operation(self, job: Job) -> Operation:
        return self._instance.jobs[job.job_id][job.position]

    def _schedule(self, time: int, kind: _TimedKind, machine_id: int) -> None:
        heapq.heappush(self._timed, (time, kind, machine_id))

    def _emit(
        self,
        event_type: EventType,
        *,
        machine: int | None = None,
        job: int | None = None,
        op: int | None = None,
    ) -> None:
        self._events.append(Event(t=self._clock, type=event_type, machine=machine, job=job, op=op))


def run_episode(instance: Instance, policy: Policy) -> EpisodeResult:
    simulation = Simulation(instance)
    while (decision := simulation.next_decision()) is not None:
        simulation.apply(policy(decision))
    return EpisodeResult(makespan=simulation.makespan, events=simulation.events)


def _validate(instance: Instance) -> None:
    """Reject instances the kernel's ordering guarantees do not cover.

    A zero-length operation would finish at the timestamp it started, after
    other machines had already decided, breaking "completions first".
    """
    for job_id, job in enumerate(instance.jobs):
        for position, operation in enumerate(job):
            if (operation.job_id, operation.position) != (job_id, position):
                raise ValueError(f"operation {operation} is stored at job {job_id}, position {position}")
            if not 0 <= operation.machine_id < instance.num_machines:
                raise ValueError(f"operation {operation} uses an unknown machine")
            if operation.duration <= 0:
                raise ValueError(f"operation {operation} has non-positive duration")
