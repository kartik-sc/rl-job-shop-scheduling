"""Gymnasium wrapper over the kernel's ``next_decision`` / ``apply`` loop.

One step is one decision. Actions are ``Discrete(max_jobs + 2)``: job indices,
then Maintain, then Wait. :meth:`JobShopEnv.action_masks` is what
``MaskablePPO`` reads.

An action outside the mask is replaced by the lowest legal action and flagged
in ``info["illegal_action"]``. Masked training never sends one; the fallback
exists because ``gymnasium``'s ``check_env`` steps with unmasked samples. A
no-op would be exploitable, since stalling until truncation scores 0.
"""

from collections.abc import Callable, Sequence
from functools import cache
from typing import Any, Literal

import gymnasium as gym
import numpy as np

from backend.env.health import EngineHealth, Regime, SampleEngine
from backend.env.instance import Instance, from_benchmark
from backend.env.observation import ObservationLayout, build_observation
from backend.env.simulator import Action, Decision, Dispatch, Maintain, Simulation, Wait

InstanceSampler = Callable[[np.random.Generator], Instance]
RateSampler = Callable[[Instance, np.random.Generator], Sequence[float]]
Mode = Literal["full", "blind"]

_SEED_BOUND = 2**31 - 1


@cache
def _benchmark(name: str) -> Instance:
    return from_benchmark(name)


def benchmark_sampler(names: Sequence[str]) -> InstanceSampler:
    """Pick one of the named benchmark instances uniformly."""
    choices = tuple(names)

    def sample(rng: np.random.Generator) -> Instance:
        return _benchmark(choices[int(rng.integers(len(choices)))])

    return sample


class JobShopEnv(gym.Env[np.ndarray, np.int64]):
    """Job shop with machine health, in full or blind mode.

    ``reset(seed=s)`` uses ``s`` as the health seed, so every policy reset with
    the same seed meets the same failures. Without a seed, one is drawn from
    the env's generator. ``options`` may set ``instance_name`` and ``regime``.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        *,
        instance_sampler: InstanceSampler,
        regime: Regime,
        mode: Mode,
        sample_engine: SampleEngine,
        rates: RateSampler,
        max_steps: int = 100_000,
        max_jobs: int = 20,
        max_machines: int = 15,
    ) -> None:
        if mode not in ("full", "blind"):
            raise ValueError(f"mode must be 'full' or 'blind', not {mode!r}")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.mode = mode
        self._instance_sampler = instance_sampler
        self._regime = regime
        self._sample_engine = sample_engine
        self._rates = rates
        self._max_steps = max_steps
        self._layout = ObservationLayout(max_jobs, max_machines, include_health=mode == "full")
        self._maintain_index = max_jobs
        self._wait_index = max_jobs + 1

        self.action_space = gym.spaces.Discrete(max_jobs + 2)
        self.observation_space = self._layout.space()

        self._instance: Instance | None = None
        self._simulation: Simulation | None = None
        self._decision: Decision | None = None
        self._cursor = 0
        self._steps = 0
        self._episode_seed = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        instance = (
            _benchmark(options["instance_name"])
            if "instance_name" in options
            else self._instance_sampler(self.np_random)
        )
        regime: Regime = options.get("regime", self._regime)
        self._episode_seed = seed if seed is not None else int(self.np_random.integers(_SEED_BOUND))
        health = EngineHealth(
            regime=regime,
            rates=self._rates(instance, self.np_random),
            mean_duration=instance.mean_duration,
            seed=self._episode_seed,
            sample_engine=self._sample_engine,
            maintenance_enabled=self.mode == "full",
        )
        self._instance = instance
        self._simulation = Simulation(instance, health)
        self._decision = self._simulation.next_decision()
        self._cursor = 0
        self._steps = 0
        observation = self._observation()
        info = self._info()
        info["instance"] = instance.name
        info["episode_seed"] = self._episode_seed
        return observation, info

    def step(self, action: np.int64 | int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        simulation, instance = self._require_episode()
        if self._decision is None:
            raise RuntimeError("the episode has ended; call reset()")

        mask = self.action_masks()
        index = int(action)
        illegal = not (0 <= index < mask.size and mask[index])
        if illegal:
            index = int(np.flatnonzero(mask)[0])

        before = simulation.clock
        simulation.apply(self._to_kernel(index))
        self._decision = simulation.next_decision()
        self._steps += 1

        terminated = self._decision is None
        truncated = not terminated and self._steps >= self._max_steps
        reward = -(simulation.clock - before) / instance.mean_duration
        info = self._info()
        info["illegal_action"] = illegal
        if terminated:
            info["makespan"] = simulation.makespan
        return self._observation(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        decision = self._decision
        if decision is None:
            return mask
        mask[list(decision.job_ids)] = True
        mask[self._maintain_index] = decision.can_maintain
        mask[self._wait_index] = decision.can_wait
        return mask

    def _to_kernel(self, index: int) -> Action:
        if index == self._maintain_index:
            return Maintain()
        if index == self._wait_index:
            return Wait()
        return Dispatch(index)

    def _observation(self) -> np.ndarray:
        simulation, _ = self._require_episode()
        return build_observation(simulation.snapshot(), self._decision, self._layout)

    def _info(self) -> dict[str, Any]:
        simulation, _ = self._require_episode()
        events = simulation.events_since(self._cursor)
        self._cursor += len(events)
        return {"events": [event.to_dict() for event in events]}

    def _require_episode(self) -> tuple[Simulation, Instance]:
        if self._simulation is None or self._instance is None:
            raise RuntimeError("call reset() before step()")
        return self._simulation, self._instance
