"""Episode logs for the dashboard: real simulator logs first, the hand-written mock last.

Real logs are the JSON files ``backend.evaluation.demo`` writes; set
``RAJS_LOG_DIR`` to serve another directory. Every mock episode's label starts
with ``MOCK_LABEL`` so the dashboard can never show it as a result.
"""

import json
import os
from functools import cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from backend.api.mock_data import get_mock_episode
from backend.training.protocol import REPO_ROOT

MOCK_ID = "mock"
MOCK_LABEL = "MOCK DATA"
DEFAULT_LOG_DIR = REPO_ROOT / "results" / "demo" / "logs"
POLICY_NAMES = {
    "threshold_spt": "Threshold+SPT",
    "spt": "SPT",
    "lpt": "LPT",
    "fifo": "FIFO",
    "full_ppo": "Full PPO",
    "blind_ppo": "Blind PPO",
}


class EpisodeMeta(BaseModel):
    model_config = ConfigDict(extra="allow")

    instance: str
    regime: Literal["low", "medium", "high"]
    seed: int
    policy: str
    n_jobs: int
    n_machines: int


class ScheduleEvent(BaseModel):
    t: float
    type: str
    machine: int | None = None
    job: int | None = None
    op: int | None = None
    risk: float | None = None


class EpisodeLog(BaseModel):
    meta: EpisodeMeta
    events: list[ScheduleEvent]


class EpisodeSummary(BaseModel):
    """One selectable episode. Downtimes are ``None`` where the log cannot give them."""

    id: str
    source: Literal["real", "mock"]
    label: str
    detail: str
    meta: EpisodeMeta
    makespan: float
    maintenances: int
    breakdowns: int
    maintenance_downtime: float | None
    repair_downtime: float | None


def log_dir() -> Path:
    return Path(os.environ.get("RAJS_LOG_DIR", DEFAULT_LOG_DIR))


def list_episodes() -> list[EpisodeSummary]:
    return [*(_real_summary(path.stem, _read(path)) for path in _real_logs()), _mock_summary()]


def get_episode(episode_id: str) -> dict[str, Any] | None:
    if episode_id == MOCK_ID:
        return get_mock_episode()
    path = log_dir() / f"{episode_id}.json"
    if path.parent != log_dir() or not path.is_file():
        return None
    return _read(path)


def _real_logs() -> list[Path]:
    order = list(POLICY_NAMES)
    paths = sorted(log_dir().glob("*.json")) if log_dir().is_dir() else []
    return sorted(paths, key=lambda p: (order.index(_read(p)["meta"]["policy"]), p.stem))


@cache
def _read_cached(path: Path, mtime: float) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read(path: Path) -> dict[str, Any]:
    return _read_cached(path, path.stat().st_mtime)


def _policy_name(policy: str) -> str:
    return POLICY_NAMES.get(policy, policy)


def _real_summary(episode_id: str, log: dict[str, Any]) -> EpisodeSummary:
    meta = log["meta"]
    metrics = meta.get("metrics", {})
    tau = f" · τ={meta['tau']:g}" if meta.get("tau") is not None else ""
    counts = _counts(log["events"])
    return EpisodeSummary(
        id=episode_id,
        source="real",
        label=f"{_policy_name(meta['policy'])}, real health model",
        detail=f"{meta['instance']} · {meta['regime']} hazard · seed {meta['seed']}{tau}",
        meta=EpisodeMeta(**meta),
        makespan=metrics.get("makespan", counts["makespan"]),
        maintenances=counts["maintenance_start"],
        breakdowns=counts["breakdown"],
        maintenance_downtime=metrics.get("maintenance_downtime"),
        repair_downtime=metrics.get("repair_downtime"),
    )


def _mock_summary() -> EpisodeSummary:
    log = get_mock_episode()
    counts = _counts(log["events"])
    return EpisodeSummary(
        id=MOCK_ID,
        source="mock",
        label=f"{MOCK_LABEL}: {_policy_name(log['meta']['policy'])} (hand-written)",
        detail=f"{log['meta']['instance']} · not a simulation result",
        meta=EpisodeMeta(**log["meta"]),
        makespan=counts["makespan"],
        maintenances=counts["maintenance_start"],
        breakdowns=counts["breakdown"],
        maintenance_downtime=None,
        repair_downtime=None,
    )


def _counts(events: list[dict[str, Any]]) -> dict[str, float]:
    counts: dict[str, float] = {"maintenance_start": 0, "breakdown": 0, "makespan": 0}
    for event in events:
        if event["type"] in counts:
            counts[event["type"]] += 1
        if event["type"] == "episode_end":
            counts["makespan"] = event["t"]
    return counts
