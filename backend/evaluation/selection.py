"""Choose tau for threshold + SPT and one checkpoint per PPO run, on the tuning split.

    python -m backend.evaluation.selection --protocol configs/protocol.toml [--only threshold] [--workers 8]

Both choices use the same episodes (training instances x tune seeds, training
pool) and the same criterion: the lowest mean optimality gap. Ties go to the
earlier candidate, so the first tau in the grid or the earlier checkpoint.
Writes ``results/<hash>/selection.json`` and the tuning episodes under
``tune/``; a later run with ``--only`` keeps the other half of the file.
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from backend.evaluation.actors import ppo, threshold
from backend.evaluation.episodes import Cell, EpisodeRecord, results_dir, run_cells, write_records
from backend.training.protocol import MODES, Mode, Protocol, git_commit, load_protocol
from backend.training.train import checkpoints, run_dir

Part = Literal["threshold", "ppo"]
Scores = dict[tuple[str, str, str], float]


def select_threshold(protocol: Protocol, commit: str, workers: int, out: Path) -> dict[str, Any]:
    contenders = [threshold(tau) for tau in protocol.tau_grid]
    cells = [
        Cell(protocol, commit, contender, regime, "tune", instance)
        for regime in protocol.regime_names
        for contender in contenders
        for instance in protocol.split.train_instances
    ]
    records = run_cells(cells, workers)
    write_records(out / "tune" / "threshold_spt.csv", records)
    scores = _mean_gaps(records)

    chosen = {}
    for regime in protocol.regime_names:
        by_tau = {c.variant: scores["threshold_spt", regime, c.variant] for c in contenders}
        best = min(contenders, key=lambda c: by_tau[c.variant])
        chosen[regime] = {"tau": best.tau, "at_grid_edge": best.tau == min(protocol.tau_grid), "scores": by_tau}
    return chosen


def select_ppo(
    protocol: Protocol, commit: str, workers: int, out: Path, runs_root: Path
) -> dict[str, Any]:
    runs: list[tuple[Mode, str, str, str, Path]] = []
    missing = []
    for mode in MODES:
        for regime in protocol.regime_names:
            for seed in protocol.seeds.train:
                directory = run_dir(runs_root, protocol, mode, regime, seed)
                saved = checkpoints(directory)
                if not saved or saved[-1][0] != "final":
                    missing.append(str(directory))
                runs += [(mode, regime, f"s{seed}", label, path) for label, path in saved]
    if missing:
        raise FileNotFoundError("unfinished or missing training runs:\n  " + "\n  ".join(missing))

    cells = [
        Cell(protocol, commit, ppo(mode, path, f"{seed}/{label}"), regime, "tune", instance)
        for mode, regime, seed, label, path in runs
        for instance in protocol.split.train_instances
    ]
    records = run_cells(cells, workers)
    write_records(out / "tune" / "ppo.csv", records)
    scores = _mean_gaps(records)

    chosen: dict[str, Any] = {}
    for mode, regime, seed, label, path in runs:
        entry = chosen.setdefault(mode, {}).setdefault(regime, {}).setdefault(seed, {"scores": {}})
        score = entry["scores"][label] = scores[f"{mode}_ppo", regime, f"{seed}/{label}"]
        if "label" not in entry or score < entry["scores"][entry["label"]]:
            entry.update(label=label, checkpoint=str(path))
    return chosen


def _mean_gaps(records: Sequence[EpisodeRecord]) -> Scores:
    gaps: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in records:
        if record.truncated or record.illegal_actions or record.gap is None:
            raise RuntimeError(f"unusable tuning episode: {record}")
        gaps[record.policy, record.regime, record.variant].append(record.gap)
    return {key: fmean(values) for key, values in gaps.items()}


def select(
    protocol: Protocol,
    parts: Sequence[Part] = ("threshold", "ppo"),
    *,
    workers: int = 1,
    results_root: Path = Path("results"),
    runs_root: Path = Path("runs"),
) -> Path:
    out = results_dir(protocol, results_root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "selection.json"
    selection = json.loads(path.read_text()) if path.exists() else {"protocol": protocol.sha256}
    if selection["protocol"] != protocol.sha256:
        raise ValueError(f"{path} belongs to protocol {selection['protocol']}")

    commit = git_commit()
    if "threshold" in parts:
        selection["threshold"] = select_threshold(protocol, commit, workers, out)
    if "ppo" in parts:
        selection["ppo"] = select_ppo(protocol, commit, workers, out, runs_root)
    selection["commit"] = commit
    path.write_text(json.dumps(selection, indent=2))
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Select tau and PPO checkpoints on the tuning split.")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--only", choices=("threshold", "ppo"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args(argv)

    parts: tuple[Part, ...] = (args.only,) if args.only else ("threshold", "ppo")
    path = select(
        load_protocol(args.protocol), parts, workers=args.workers, results_root=args.results, runs_root=args.runs
    )
    selection = json.loads(path.read_text())
    for regime, entry in selection.get("threshold", {}).items():
        warning = "  WARNING: lowest tau in the grid; extend tau_grid downward" if entry["at_grid_edge"] else ""
        print(f"threshold_spt {regime}: tau={entry['tau']:g}{warning}")
    for mode, regimes in selection.get("ppo", {}).items():
        for regime, seeds in regimes.items():
            for seed, entry in seeds.items():
                print(f"{mode}_ppo {regime} {seed}: {entry['label']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
