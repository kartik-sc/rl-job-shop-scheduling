"""Event-log records, mirroring ``shared/schemas/event_log.schema.json``."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from backend.env.instance import Instance


class EventType(StrEnum):
    OP_START = "op_start"
    OP_END = "op_end"
    RISK_UPDATE = "risk_update"
    MAINTENANCE_START = "maintenance_start"
    MAINTENANCE_END = "maintenance_end"
    BREAKDOWN = "breakdown"
    REPAIR_END = "repair_end"
    EPISODE_END = "episode_end"


@dataclass(frozen=True, slots=True)
class Event:
    """One log entry. ``op`` is the operation's position within its job."""

    t: int
    type: EventType
    machine: int | None = None
    job: int | None = None
    op: int | None = None
    risk: float | None = None

    def to_dict(self) -> dict[str, Any]:
        fields = {
            "t": self.t,
            "type": self.type.value,
            "machine": self.machine,
            "job": self.job,
            "op": self.op,
            "risk": self.risk,
        }
        return {key: value for key, value in fields.items() if value is not None}


def log_document(
    events: Iterable[Event],
    instance: Instance,
    *,
    policy: str,
    regime: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "meta": {
            "instance": instance.name,
            "regime": regime,
            "seed": seed,
            "policy": policy,
            "n_jobs": instance.num_jobs,
            "n_machines": instance.num_machines,
        },
        "events": [event.to_dict() for event in events],
    }
