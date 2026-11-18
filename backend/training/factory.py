"""Build regimes, samplers and environments from a :class:`Protocol`.

Every policy, trained or not, is run through an env made here, so all of them
share one path from ``(instance, regime, seed)`` to rates and failure draws.
That is what makes the comparisons paired.
"""

from collections.abc import Callable, Sequence

import numpy as np

from backend.env.gym_env import InstanceSampler, JobShopEnv, Mode, RateSampler, benchmark_sampler
from backend.env.health import Regime, SampleEngine, exponential_hazard
from backend.env.instance import Instance, random_instance
from backend.risk import synthetic
from backend.training.protocol import Phase, Protocol, RateSpec, RegimeSpec, SplitSpec

ENGINES: dict[str, Callable[[], SampleEngine]] = {
    "synthetic": lambda: synthetic.sample_engine,
}


def sample_engine(name: str) -> SampleEngine:
    if name not in ENGINES:
        raise ValueError(f"unknown engine {name!r}; registered: {sorted(ENGINES)}")
    return ENGINES[name]()


def build_regime(spec: RegimeSpec, pool: str) -> Regime:
    return Regime(
        name=spec.name,
        hazard=exponential_hazard(spec.hazard_base, spec.hazard_scale),
        pm_factor=spec.pm_factor,
        repair_factor=spec.repair_factor,
        cycles_per_mean_op=spec.cycles_per_mean_op,
        pool=pool,
    )


def rate_sampler(spec: RateSpec) -> RateSampler:
    low, high = np.log(spec.low), np.log(spec.high)

    def sample(instance: Instance, rng: np.random.Generator) -> Sequence[float]:
        return np.exp(rng.uniform(low, high, instance.num_machines)).tolist()

    return sample


def training_sampler(split: SplitSpec) -> InstanceSampler:
    """Random instances with probability ``random_share``, otherwise a training benchmark."""
    benchmarks = benchmark_sampler(split.train_instances)
    durations = (split.duration_low, split.duration_high)

    def sample(rng: np.random.Generator) -> Instance:
        if rng.random() < split.random_share:
            jobs, machines = split.random_sizes[int(rng.integers(len(split.random_sizes)))]
            return random_instance(rng, jobs, machines, durations)
        return benchmarks(rng)

    return sample


def make_env(protocol: Protocol, mode: Mode, regime: str, phase: Phase) -> JobShopEnv:
    """Training envs sample instances; tune and test envs are reset with an explicit instance."""
    sampler = training_sampler(protocol.split) if phase == "train" else benchmark_sampler(protocol.instances(phase))
    return JobShopEnv(
        instance_sampler=sampler,
        regime=build_regime(protocol.regime(regime), protocol.pool(phase)),
        mode=mode,
        sample_engine=sample_engine(protocol.engine),
        rates=rate_sampler(protocol.rates),
        max_steps=protocol.max_steps,
    )
