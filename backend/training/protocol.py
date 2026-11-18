"""The experimental protocol: one TOML file, loaded into frozen dataclasses.

Every training run and every result row carries the file's sha256, so results
produced under different protocols can never be combined. Editing the file
after the Day 4 freeze shows up as a new hash.
"""

import hashlib
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from backend.env.gym_env import Mode

Phase = Literal["train", "tune", "test"]
MODES: tuple[Mode, ...] = ("blind", "full")
MAX_JOBS, MAX_MACHINES = 20, 15
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class RegimeSpec:
    """Exponential hazard ``base * exp(-rul / scale)``; downtimes in mean-operation units."""

    name: str
    hazard_base: float
    hazard_scale: float
    pm_factor: float
    repair_factor: float
    cycles_per_mean_op: float


@dataclass(frozen=True, slots=True)
class RateSpec:
    """Per-machine degradation multipliers, log-uniform on [low, high], drawn per episode."""

    low: float
    high: float


@dataclass(frozen=True, slots=True)
class SplitSpec:
    train_instances: tuple[str, ...]
    test_instances: tuple[str, ...]
    random_sizes: tuple[tuple[int, int], ...]
    random_share: float
    duration_low: int
    duration_high: int
    train_pool: str
    test_pool: str


@dataclass(frozen=True, slots=True)
class SeedSpec:
    """``train`` seeds PPO; ``tune`` and ``test`` are health seeds passed to ``reset``."""

    train: tuple[int, ...]
    tune: tuple[int, ...]
    test: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PPOSpec:
    """MaskablePPO hyperparameters, shared by blind and full PPO. ``gamma`` is fixed at 1."""

    n_envs: int
    n_steps: int
    batch_size: int
    n_epochs: int
    learning_rate: float
    ent_coef: float
    gae_lambda: float
    clip_range: float
    net_arch: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    blind_steps: int
    full_steps: int
    checkpoints: int

    def steps(self, mode: Mode) -> int:
        return self.full_steps if mode == "full" else self.blind_steps


@dataclass(frozen=True, slots=True)
class Protocol:
    path: Path
    sha256: str
    engine: str
    max_steps: int
    regimes: tuple[RegimeSpec, ...]
    rates: RateSpec
    split: SplitSpec
    seeds: SeedSpec
    ppo: PPOSpec
    budget: BudgetSpec
    tau_grid: tuple[float, ...]

    @property
    def short_hash(self) -> str:
        return self.sha256[:8]

    @property
    def regime_names(self) -> tuple[str, ...]:
        return tuple(regime.name for regime in self.regimes)

    def regime(self, name: str) -> RegimeSpec:
        for regime in self.regimes:
            if regime.name == name:
                return regime
        raise KeyError(f"no regime {name!r}; the protocol has {self.regime_names}")

    def pool(self, phase: Phase) -> str:
        return self.split.test_pool if phase == "test" else self.split.train_pool

    def instances(self, phase: Phase) -> tuple[str, ...]:
        return self.split.test_instances if phase == "test" else self.split.train_instances

    def health_seeds(self, phase: Literal["tune", "test"]) -> tuple[int, ...]:
        return self.seeds.test if phase == "test" else self.seeds.tune


def load_protocol(path: Path | str) -> Protocol:
    path = Path(path)
    raw = path.read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    try:
        protocol = Protocol(
            path=path,
            sha256=hashlib.sha256(raw).hexdigest(),
            engine=data.pop("engine"),
            max_steps=data.pop("max_steps"),
            regimes=tuple(RegimeSpec(name=name, **table) for name, table in data.pop("regimes").items()),
            rates=RateSpec(**data.pop("rates")),
            split=_split(data.pop("split")),
            seeds=SeedSpec(**{key: tuple(value) for key, value in data.pop("seeds").items()}),
            ppo=_ppo(data.pop("ppo")),
            budget=BudgetSpec(**data.pop("budget")),
            tau_grid=_tau_grid(**data.pop("threshold")),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path}: malformed protocol ({error})") from error
    if data:
        raise ValueError(f"{path}: unknown top-level keys {sorted(data)}")
    _validate(protocol)
    return protocol


def git_commit() -> str:
    """Short commit hash, with ``+dirty`` when the tree has uncommitted changes."""
    try:
        commit = _git("rev-parse", "--short", "HEAD")
        return commit + ("+dirty" if _git("status", "--porcelain", "--untracked-files=no") else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _split(table: dict[str, Any]) -> SplitSpec:
    table = dict(table)
    table["train_instances"] = tuple(table["train_instances"])
    table["test_instances"] = tuple(table["test_instances"])
    table["random_sizes"] = tuple((int(jobs), int(machines)) for jobs, machines in table["random_sizes"])
    return SplitSpec(**table)


def _ppo(table: dict[str, Any]) -> PPOSpec:
    if "gamma" in table:
        raise ValueError("gamma is fixed at 1.0 so the return equals -makespan; remove it from [ppo]")
    return PPOSpec(**{**table, "net_arch": tuple(table["net_arch"])})


def _tau_grid(*, tau_grid: list[float]) -> tuple[float, ...]:
    return tuple(float(tau) for tau in tau_grid)


def _validate(protocol: Protocol) -> None:
    problems: list[str] = []
    split, seeds = protocol.split, protocol.seeds

    if not protocol.regimes:
        problems.append("at least one regime is required")
    for regime in protocol.regimes:
        if regime.pm_factor <= 0 or not 5 <= regime.repair_factor / regime.pm_factor <= 10:
            problems.append(f"regime {regime.name}: repair_factor must be 5-10x a positive pm_factor")
    if not 0 < protocol.rates.low <= protocol.rates.high:
        problems.append("rates need 0 < low <= high")

    if overlap := set(split.train_instances) & set(split.test_instances):
        problems.append(f"instances in both train and test: {sorted(overlap)}")
    if not split.train_instances or not split.test_instances:
        problems.append("train_instances and test_instances must be non-empty")
    if not 0 <= split.random_share <= 1 or (split.random_share > 0 and not split.random_sizes):
        problems.append("random_share must lie in [0, 1], and needs random_sizes when positive")
    for jobs, machines in split.random_sizes:
        if not (1 <= jobs <= MAX_JOBS and 1 <= machines <= MAX_MACHINES):
            problems.append(f"random size {jobs}x{machines} exceeds the {MAX_JOBS}x{MAX_MACHINES} layout")
    if not 1 <= split.duration_low <= split.duration_high:
        problems.append("durations need 1 <= duration_low <= duration_high")

    if overlap := set(seeds.tune) & set(seeds.test):
        problems.append(f"health seeds in both tune and test: {sorted(overlap)}")
    if not (seeds.train and seeds.tune and seeds.test):
        problems.append("train, tune and test seeds must be non-empty")

    if not protocol.tau_grid or not all(0.0 <= tau <= 1.0 for tau in protocol.tau_grid):
        problems.append("tau_grid must be non-empty, with every tau in [0, 1]")
    elif 1.0 not in protocol.tau_grid:
        problems.append("tau_grid must include 1.0 (never maintain), so tuning can fall back to SPT")

    if min(protocol.budget.blind_steps, protocol.budget.full_steps, protocol.budget.checkpoints) < 1:
        problems.append("budget steps and checkpoints must be positive")
    if protocol.max_steps < 1:
        problems.append("max_steps must be positive")

    if problems:
        raise ValueError(f"{protocol.path}: " + "; ".join(problems))
