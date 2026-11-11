"""Policies under evaluation, all acting through :class:`JobShopEnv`.

A :class:`Contender` is a picklable description; :meth:`Contender.build` turns
it into an actor inside the worker process that runs it.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.baselines.dispatch import RULES
from backend.baselines.threshold import threshold_spt
from backend.env.gym_env import JobShopEnv, Mode
from backend.env.simulator import Policy

Actor = Callable[[JobShopEnv, np.ndarray], int]
REFERENCE_RULES = ("spt", "lpt", "fifo")


@dataclass(frozen=True, slots=True)
class Contender:
    """``policy`` is the event-log policy name; ``variant`` tells apart a policy's
    instances, such as a tau value or a PPO training seed."""

    policy: str
    variant: str
    mode: Mode
    tau: float | None = None
    checkpoint: Path | None = None

    @property
    def label(self) -> str:
        return f"{self.policy}[{self.variant}]" if self.variant else self.policy

    def build(self) -> Actor:
        if self.checkpoint is not None:
            return ppo_actor(self.checkpoint)
        if self.tau is not None:
            return rule_actor(threshold_spt(self.tau))
        return rule_actor(RULES[self.policy])


def reference_rule(name: str) -> Contender:
    """A classical rule, run blind: breakdowns happen, nothing maintains."""
    if name not in REFERENCE_RULES:
        raise ValueError(f"unknown reference rule {name!r}")
    return Contender(policy=name, variant="", mode="blind")


def threshold(tau: float) -> Contender:
    return Contender(policy="threshold_spt", variant=f"tau={tau:g}", mode="full", tau=tau)


def ppo(mode: Mode, checkpoint: Path, variant: str) -> Contender:
    return Contender(policy=f"{mode}_ppo", variant=variant, mode=mode, checkpoint=checkpoint)


def rule_actor(policy: Policy) -> Actor:
    def act(env: JobShopEnv, observation: np.ndarray) -> int:
        decision = env.decision
        if decision is None:
            raise RuntimeError("no pending decision")
        return env.encode(policy(decision))

    return act


def ppo_actor(checkpoint: Path) -> Actor:
    """Greedy actions from a saved MaskablePPO model, restricted to the mask."""
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    model = MaskablePPO.load(checkpoint, device="cpu")

    def act(env: JobShopEnv, observation: np.ndarray) -> int:
        action, _ = model.predict(observation, action_masks=env.action_masks(), deterministic=True)
        return int(action)

    return act
