"""Event-driven job-shop simulator.

Mechanics: an integer clock, a heap of timed events and a waiting queue per
machine. The clock only moves by popping the heap, and every timed event at a
timestamp is applied before any machine decides, so completions come first.

Decisions: an idle machine with a non-empty queue needs a decision. Machines
decide in ascending id at each timestamp, and the caller answers each
:class:`Decision` through :meth:`Simulation.apply`. The clock never moves while
a decision is pending, which makes every schedule non-delay.

Health is optional. Without a :class:`HealthModel` this is the no-failure
kernel the Day 2 gate checks. With one, machines wear while processing, an
operation breaks down if it would outlast the machine's current lifetime, and
both a breakdown and maintenance take the machine offline and reset it when it
returns. With maintenance enabled, an idle machine that has worked since its
last reset may also maintain, or wait when its queue is empty. A machine whose
only legal action would be waiting is not asked.
"""

import heapq
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum

from backend.env.events import Event, EventType
from backend.env.health import HealthModel
from backend.env.instance import Instance, Operation


class MachineStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"


class _TimedKind(IntEnum):
    """Kinds of timed event. Lower values are handled first at equal times."""

    OP_END = 0
    BREAKDOWN = 1
    MAINTENANCE_END = 2
    REPAIR_END = 3


@dataclass(slots=True)
class Machine:
    """``remaining_ops`` counts this machine's operations not yet completed."""

    machine_id: int
    status: MachineStatus = MachineStatus.IDLE
    job_id: int | None = None
    queue: list[int] = field(default_factory=list)
    busy_since: int | None = None
    offline_until: int | None = None
    waited_at: int | None = None
    last_reset_at: int = 0
    remaining_ops: int = 0


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
    """A machine that must choose now. Candidates are in ascending job id.

    ``candidates`` is empty only when the machine may maintain or wait.
    ``risk`` is the deciding machine's score, or ``None`` without health.
    """

    time: int
    machine_id: int
    candidates: tuple[Candidate, ...]
    can_maintain: bool = False
    can_wait: bool = False
    risk: float | None = None

    @property
    def job_ids(self) -> tuple[int, ...]:
        return tuple(candidate.job_id for candidate in self.candidates)


@dataclass(frozen=True, slots=True)
class Dispatch:
    job_id: int


@dataclass(frozen=True, slots=True)
class Maintain:
    """Take the idle machine offline for ``t_pm``, then reset it."""


@dataclass(frozen=True, slots=True)
class Wait:
    """Leave the idle machine idle until the clock next moves."""


Action = Dispatch | Maintain | Wait
Policy = Callable[[Decision], Action]


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    makespan: int
    events: tuple[Event, ...]


@dataclass(frozen=True, slots=True)
class MachineView:
    """What an observer may know about a machine.

    ``free_at`` is nominal: the scheduled end of the running operation, or the
    end of the current downtime, never a pending breakdown. ``age`` and
    ``risk`` are ``None`` without health.
    """

    machine_id: int
    status: MachineStatus
    job_id: int | None
    queue: tuple[int, ...]
    free_at: int | None
    remaining_ops: int
    last_reset_at: int
    age: int | None
    risk: float | None


@dataclass(frozen=True, slots=True)
class JobView:
    job_id: int
    position: int
    arrived_at: int


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Read-only state for building observations. Holds no failure point, RUL or timed event."""

    time: int
    instance: Instance
    machines: tuple[MachineView, ...]
    jobs: tuple[JobView, ...]


class Simulation:
    """One episode. Drive it with :meth:`next_decision` and :meth:`apply`."""

    def __init__(self, instance: Instance, health: HealthModel | None = None) -> None:
        _validate(instance)
        if health is not None:
            _validate_health(health)
        self._instance = instance
        self._health = health
        self._risk_logged_at: int | None = None
        self._clock = 0
        self._makespan = 0
        self._remaining = instance.num_operations
        self._timed: list[tuple[int, _TimedKind, int]] = []
        self._machines = [Machine(machine_id) for machine_id in range(instance.num_machines)]
        self._jobs = [Job(job_id) for job_id in range(instance.num_jobs)]
        self._events: list[Event] = []
        self._pending: Decision | None = None
        self._ended = False
        for operation in (op for job in instance.jobs for op in job):
            self._machines[operation.machine_id].remaining_ops += 1
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

    def snapshot(self) -> Snapshot:
        return Snapshot(
            time=self._clock,
            instance=self._instance,
            machines=tuple(self._machine_view(machine) for machine in self._machines),
            jobs=tuple(JobView(job.job_id, job.position, job.arrived_at) for job in self._jobs),
        )

    def _machine_view(self, machine: Machine) -> MachineView:
        free_at = machine.offline_until
        in_progress = 0
        if machine.busy_since is not None:
            free_at = machine.busy_since + self._operation(self._running_job(machine)).duration
            in_progress = self._clock - machine.busy_since
        health = self._health
        return MachineView(
            machine_id=machine.machine_id,
            status=machine.status,
            job_id=machine.job_id,
            queue=tuple(sorted(machine.queue)),
            free_at=free_at,
            remaining_ops=machine.remaining_ops,
            last_reset_at=machine.last_reset_at,
            age=None if health is None else health.age(machine.machine_id),
            risk=None if health is None else health.risk(machine.machine_id, in_progress),
        )

    def next_decision(self) -> Decision | None:
        """The pending decision, advancing the clock until there is one.

        Returns ``None`` once every operation has finished, at which point
        ``clock == makespan``.
        """
        while self._pending is None:
            machine = self._first_deciding_machine()
            if machine is not None:
                self._log_risk_once()
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
        machine = self._machines[decision.machine_id]
        match action:
            case Dispatch():
                if action.job_id not in decision.job_ids:
                    raise ValueError(
                        f"job {action.job_id} cannot start on machine {decision.machine_id} "
                        f"at t={decision.time}; candidates are {decision.job_ids}"
                    )
                self._pending = None
                self._start(machine, self._jobs[action.job_id])
            case Maintain():
                if not decision.can_maintain:
                    raise ValueError(f"machine {decision.machine_id} cannot maintain at t={decision.time}")
                self._pending = None
                self._start_maintenance(machine)
            case Wait():
                if not decision.can_wait:
                    raise ValueError(f"machine {decision.machine_id} cannot wait at t={decision.time}")
                self._pending = None
                machine.waited_at = self._clock

    def _first_deciding_machine(self) -> Machine | None:
        return next((m for m in self._machines if self._needs_decision(m)), None)

    def _needs_decision(self, machine: Machine) -> bool:
        if machine.status is not MachineStatus.IDLE or machine.waited_at == self._clock:
            return False
        return bool(machine.queue) or self._can_maintain(machine)

    def _can_maintain(self, machine: Machine) -> bool:
        """Idle machines only: maintenance is on, it has worked since its reset, and work remains."""
        health = self._health
        return (
            health is not None
            and health.maintenance_enabled
            and machine.remaining_ops > 0
            and health.age(machine.machine_id) > 0
        )

    def _decision_for(self, machine: Machine) -> Decision:
        candidates = tuple(
            self._candidate(self._jobs[job_id]) for job_id in sorted(machine.queue)
        )
        health = self._health
        return Decision(
            time=self._clock,
            machine_id=machine.machine_id,
            candidates=candidates,
            can_maintain=self._can_maintain(machine),
            can_wait=health is not None and health.maintenance_enabled and not machine.queue,
            risk=None if health is None else health.risk(machine.machine_id),
        )

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
        machine = self._machines[machine_id]
        match kind:
            case _TimedKind.OP_END:
                self._finish_operation(machine)
            case _TimedKind.BREAKDOWN:
                self._break_down(machine)
            case _TimedKind.MAINTENANCE_END:
                self._return_to_service(machine, EventType.MAINTENANCE_END)
            case _TimedKind.REPAIR_END:
                self._return_to_service(machine, EventType.REPAIR_END)

    def _start(self, machine: Machine, job: Job) -> None:
        machine.queue.remove(job.job_id)
        machine.status = MachineStatus.BUSY
        machine.job_id = job.job_id
        machine.busy_since = self._clock
        self._emit(EventType.OP_START, machine=machine.machine_id, job=job.job_id, op=job.position)
        end = self._clock + self._operation(job).duration
        failure = self._failure_time(machine)
        if failure is not None and failure <= end:
            self._schedule(failure, _TimedKind.BREAKDOWN, machine.machine_id)
        else:
            self._schedule(end, _TimedKind.OP_END, machine.machine_id)

    def _failure_time(self, machine: Machine) -> int | None:
        if self._health is None:
            return None
        time_to_failure = self._health.time_to_failure(machine.machine_id)
        if time_to_failure < 1:
            raise RuntimeError(
                f"machine {machine.machine_id} has time_to_failure {time_to_failure}; must be >= 1"
            )
        return self._clock + time_to_failure

    def _finish_operation(self, machine: Machine) -> None:
        job = self._running_job(machine)
        duration = self._operation(job).duration
        self._emit(EventType.OP_END, machine=machine.machine_id, job=job.job_id, op=job.position)

        machine.status = MachineStatus.IDLE
        machine.job_id = None
        machine.busy_since = None
        machine.remaining_ops -= 1
        if self._health is not None:
            self._health.wear(machine.machine_id, duration)
        self._remaining -= 1
        self._makespan = self._clock

        job.position += 1
        job.arrived_at = self._clock
        self._enqueue(job)

    def _break_down(self, machine: Machine) -> None:
        """The running job goes back to this machine's queue; its ``arrived_at`` is kept."""
        health = self._require_health()
        job = self._running_job(machine)
        self._emit(EventType.BREAKDOWN, machine=machine.machine_id, job=job.job_id, op=job.position)
        health.wear(machine.machine_id, self._clock - self._busy_since(machine))
        machine.queue.append(job.job_id)
        self._take_offline(machine, health.t_repair, _TimedKind.REPAIR_END)

    def _start_maintenance(self, machine: Machine) -> None:
        health = self._require_health()
        self._emit(EventType.MAINTENANCE_START, machine=machine.machine_id)
        self._take_offline(machine, health.t_pm, _TimedKind.MAINTENANCE_END)

    def _take_offline(self, machine: Machine, duration: int, kind: _TimedKind) -> None:
        machine.status = MachineStatus.OFFLINE
        machine.job_id = None
        machine.busy_since = None
        machine.offline_until = self._clock + duration
        self._schedule(machine.offline_until, kind, machine.machine_id)

    def _return_to_service(self, machine: Machine, event_type: EventType) -> None:
        health = self._require_health()
        self._emit(event_type, machine=machine.machine_id)
        health.reset(machine.machine_id)
        machine.status = MachineStatus.IDLE
        machine.offline_until = None
        machine.last_reset_at = self._clock
        self._emit(EventType.RISK_UPDATE, machine=machine.machine_id, risk=health.risk(machine.machine_id))

    def _log_risk_once(self) -> None:
        """Every machine's risk, once per timestamp that has a decision."""
        health = self._health
        if health is None or self._risk_logged_at == self._clock:
            return
        self._risk_logged_at = self._clock
        for machine in self._machines:
            in_progress = 0 if machine.busy_since is None else self._clock - machine.busy_since
            self._emit(
                EventType.RISK_UPDATE,
                machine=machine.machine_id,
                risk=health.risk(machine.machine_id, in_progress),
            )

    def _running_job(self, machine: Machine) -> Job:
        if machine.job_id is None:
            raise RuntimeError(f"machine {machine.machine_id} has no job running")
        return self._jobs[machine.job_id]

    def _busy_since(self, machine: Machine) -> int:
        if machine.busy_since is None:
            raise RuntimeError(f"machine {machine.machine_id} is not busy")
        return machine.busy_since

    def _require_health(self) -> HealthModel:
        if self._health is None:
            raise RuntimeError("this event needs a health model")
        return self._health

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
        risk: float | None = None,
    ) -> None:
        self._events.append(
            Event(t=self._clock, type=event_type, machine=machine, job=job, op=op, risk=risk)
        )


def run_episode(
    instance: Instance,
    policy: Policy,
    health: HealthModel | None = None,
    max_decisions: int = 1_000_000,
) -> EpisodeResult:
    """Run to completion. Raises if the episode needs more than ``max_decisions``.

    The guard catches regimes where an operation can never fit in a machine
    lifetime and so breaks down forever.
    """
    simulation = Simulation(instance, health)
    decisions = 0
    while (decision := simulation.next_decision()) is not None:
        if decisions == max_decisions:
            raise RuntimeError(
                f"{instance.name}: no completion after {max_decisions} decisions at t={simulation.clock}"
            )
        simulation.apply(policy(decision))
        decisions += 1
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


def _validate_health(health: HealthModel) -> None:
    """Downtimes must be positive for the same reason operation durations must."""
    if health.t_pm < 1 or health.t_repair < 1:
        raise ValueError(f"t_pm ({health.t_pm}) and t_repair ({health.t_repair}) must be >= 1")
