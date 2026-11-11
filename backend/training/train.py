"""Train one MaskablePPO model for one mode, regime and training seed.

    python -m backend.training.train --protocol configs/protocol.toml --mode full --regime high --train-seed 0

Writes ``runs/<protocol hash>/<mode>_<regime>_s<seed>/``: ``config.json``,
``progress.csv``, ``monitor.csv``, ``checkpoints/step_<n>_steps.zip`` and
``final.zip``. Training resets are unseeded, so each episode draws fresh
failures; only the first reset of each worker is derived from the seed.
"""

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import asdict
from functools import partial
from importlib import metadata
from pathlib import Path
from typing import Literal

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from backend.training.factory import make_env
from backend.training.protocol import MODES, Mode, Protocol, git_commit, load_protocol

Vectorize = Literal["subproc", "dummy"]
_PACKAGES = ("stable-baselines3", "sb3-contrib", "gymnasium", "torch", "numpy", "job-shop-lib")
_CHECKPOINT = re.compile(r"step_(\d+)_steps\.zip")


def run_dir(runs_root: Path, protocol: Protocol, mode: Mode, regime: str, train_seed: int) -> Path:
    return runs_root / protocol.short_hash / f"{mode}_{regime}_s{train_seed}"


def checkpoints(directory: Path) -> list[tuple[str, Path]]:
    """Saved models of one run in training order, as ``(label, path)``; ``final`` last."""
    saved = [
        (int(match.group(1)), path)
        for path in (directory / "checkpoints").glob("*.zip")
        if (match := _CHECKPOINT.fullmatch(path.name))
    ]
    found = [(f"step_{step}", path) for step, path in sorted(saved)]
    final = directory / "final.zip"
    if final.exists():
        found.append(("final", final))
    return found


def train(
    protocol: Protocol,
    mode: Mode,
    regime: str,
    train_seed: int,
    *,
    runs_root: Path = Path("runs"),
    steps: int | None = None,
    vectorize: Vectorize = "subproc",
    quiet: bool = False,
) -> Path:
    directory = run_dir(runs_root, protocol, mode, regime, train_seed)
    if (directory / "final.zip").exists():
        raise FileExistsError(f"{directory} already holds a finished run")
    protocol.regime(regime)
    total = steps if steps is not None else protocol.budget.steps(mode)
    ppo = protocol.ppo

    directory.mkdir(parents=True, exist_ok=True)
    _write_config(directory / "config.json", protocol, mode, regime, train_seed, total)
    torch.set_num_threads(1)

    factories = [partial(make_env, protocol, mode, regime, "train")] * ppo.n_envs
    vector = SubprocVecEnv(factories) if vectorize == "subproc" else DummyVecEnv(factories)
    env = VecMonitor(vector, filename=str(directory / "monitor.csv"))
    model = MaskablePPO(
        "MlpPolicy",
        env,
        gamma=1.0,
        n_steps=ppo.n_steps,
        batch_size=ppo.batch_size,
        n_epochs=ppo.n_epochs,
        learning_rate=ppo.learning_rate,
        ent_coef=ppo.ent_coef,
        gae_lambda=ppo.gae_lambda,
        clip_range=ppo.clip_range,
        policy_kwargs={"net_arch": list(ppo.net_arch)},
        seed=train_seed,
        device="cpu",
        verbose=0,
    )
    model.set_logger(configure(str(directory), ["csv"] if quiet else ["csv", "stdout"]))
    every = max(1, total // protocol.budget.checkpoints // ppo.n_envs)
    saver = CheckpointCallback(save_freq=every, save_path=str(directory / "checkpoints"), name_prefix="step")
    try:
        model.learn(total_timesteps=total, callback=saver)
    finally:
        env.close()
    model.save(directory / "final.zip")
    return directory


def _write_config(
    path: Path, protocol: Protocol, mode: Mode, regime: str, train_seed: int, total: int
) -> None:
    config = {
        "protocol_sha256": protocol.sha256,
        "commit": git_commit(),
        "mode": mode,
        "regime": regime,
        "train_seed": train_seed,
        "total_steps": total,
        "versions": {package: metadata.version(package) for package in _PACKAGES},
        "protocol": asdict(protocol),
    }
    path.write_text(json.dumps(config, indent=2, default=str))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train MaskablePPO for one mode, regime and seed.")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--steps", type=int, help="override the protocol budget (pilot runs only)")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--vectorize", choices=("subproc", "dummy"), default="subproc")
    args = parser.parse_args(argv)

    protocol = load_protocol(args.protocol)
    if args.train_seed not in protocol.seeds.train:
        parser.error(f"--train-seed must be one of the protocol's train seeds {protocol.seeds.train}")
    directory = train(
        protocol,
        args.mode,
        args.regime,
        args.train_seed,
        runs_root=args.runs,
        steps=args.steps,
        vectorize=args.vectorize,
    )
    print(f"saved {directory}")


if __name__ == "__main__":
    main()
