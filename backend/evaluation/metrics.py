"""Episode metrics computed from the event log alone.

Works on the event dicts from ``info["events"]`` and on saved log documents,
so any episode can be scored after the fact.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from backend.env.events import EventType
from backend.env.instance import Instance


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """``gap`` is ``None`` when the instance has no reference makespan.

    ``breakdown_rate`` is breakdowns per 100 operations. ``maintenance_share``
    is maintenance downtime over total machine time (machines x makespan).
    ``lost_work`` is processing time discarded by interrupted operations.
    """

    makespan: int
    gap: float | None
    breakdowns: int
    breakdown_rate: float
    maintenances: int
    maintenance_downtime: int
    maintenance_share: float
    repair_downtime: int
    lost_work: int


def episode_metrics(events: Iterable[Mapping[str, Any]], instance: Instance) -> EpisodeMetrics:
    running: dict[int, int] = {}
    offline: dict[int, tuple[EventType, int]] = {}
    makespan: int | None = None
    breakdowns = maintenances = maintenance_downtime = repair_downtime = lost_work = 0

    for event in events:
        t, kind, machine = event["t"], EventType(event["type"]), event.get("machine")
        match kind:
            case EventType.OP_START:
                running[machine] = t
            case EventType.OP_END:
                running.pop(machine)
            case EventType.BREAKDOWN:
                breakdowns += 1
                lost_work += t - running.pop(machine)
                offline[machine] = (EventType.BREAKDOWN, t)
            case EventType.MAINTENANCE_START:
                maintenances += 1
                offline[machine] = (EventType.MAINTENANCE_START, t)
            case EventType.MAINTENANCE_END:
                maintenance_downtime += t - _close(offline, machine, EventType.MAINTENANCE_START)
            case EventType.REPAIR_END:
                repair_downtime += t - _close(offline, machine, EventType.BREAKDOWN)
            case EventType.EPISODE_END:
                makespan = t

    if makespan is None or running or offline:
        raise ValueError("incomplete episode: no episode_end, or an operation or downtime left open")
    reference = instance.reference_makespan
    return EpisodeMetrics(
        makespan=makespan,
        gap=None if reference is None else makespan / reference - 1.0,
        breakdowns=breakdowns,
        breakdown_rate=100.0 * breakdowns / instance.num_operations,
        maintenances=maintenances,
        maintenance_downtime=maintenance_downtime,
        maintenance_share=maintenance_downtime / (instance.num_machines * makespan),
        repair_downtime=repair_downtime,
        lost_work=lost_work,
    )


def _close(offline: dict[int, tuple[EventType, int]], machine: int, opened_by: EventType) -> int:
    kind, started = offline.pop(machine)
    if kind is not opened_by:
        raise ValueError(f"machine {machine}: downtime opened by {kind.value} closed as if by {opened_by.value}")
    return started
