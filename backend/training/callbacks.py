"""Per-episode scheduling metrics for the training logger, read from the env's event stream.

Rates use the same definitions as ``backend/evaluation/metrics.py``: events per
100 completed operations. Values are averaged over the episodes that finish
within each rollout, so they line up with ``rollout/ep_rew_mean``.
"""

from collections import Counter

from stable_baselines3.common.callbacks import BaseCallback

from backend.env.events import EventType

_OPERATION = EventType.OP_END.value
_BREAKDOWN = EventType.BREAKDOWN.value
_MAINTENANCE = EventType.MAINTENANCE_START.value
_COUNTED = frozenset((_OPERATION, _BREAKDOWN, _MAINTENANCE))


def per_100_operations(counts: Counter[str], event: str) -> float:
    return 100.0 * counts[event] / max(1, counts[_OPERATION])


class EpisodeEventsCallback(BaseCallback):
    def _on_training_start(self) -> None:
        self._counts: list[Counter[str]] = [Counter() for _ in range(self.training_env.num_envs)]

    def _on_step(self) -> bool:
        for counts, info, done in zip(self._counts, self.locals["infos"], self.locals["dones"], strict=True):
            counts.update(event["type"] for event in info.get("events", ()) if event["type"] in _COUNTED)
            if done:
                self.logger.record_mean("episode/breakdowns_per_100_ops", per_100_operations(counts, _BREAKDOWN))
                self.logger.record_mean("episode/maintenances_per_100_ops", per_100_operations(counts, _MAINTENANCE))
                counts.clear()
        return True
