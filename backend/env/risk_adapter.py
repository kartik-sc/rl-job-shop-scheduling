"""Kaul's C-MAPSS risk model, served to the simulator as a ``SampleEngine``.

The trained detectors and calibrators are pickles under ``models/``, which is
git-ignored, so the adapter replays what they produced instead of loading
them: the per-cycle output of the production configuration (D_temporal
detector, dataset-specific isotonic calibration, "Exp A") that
``backend.risk.calibration_experiments`` wrote for the calibration engines.
Nothing is refit here.

Kaul's columns map onto :class:`Trajectory` as follows:

    risk_score    -> Trajectory.risk   observable, what policies see
    rul_true      -> Trajectory.rul    drives failure draws inside EngineHealth only
    rul_estimate  -> dropped           the simulator has no slot for it
    cycle 1..N    -> index 0..N-1

Two meanings of "pool" meet here. The simulator's pool is a split (``fit``,
``calibration``, ``test``); Kaul's FD001/FD003 are datasets, each with its own
detector and calibrator. Every engine keeps the dataset it was scored with,
and loading checks that its risk came from that dataset's calibrator.

Only the calibration split can be served. Fit engines trained the detectors
and are refused outright; test engines were never scored, by design, so the
``test`` pool is empty until Kaul runs the held-out evaluation.
"""

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from backend.env.health import Trajectory
from backend.risk.cmapss_pipeline import split_engines

Detector = Literal["ocsvm", "iforest"]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
DATASETS = ("FD001", "FD003")
UNITS_PER_DATASET = 100
POOLS = ("fit", "calibration", "test")
PRODUCTION_CONFIG = "A_ds_iso"
DEFAULT_DETECTOR: Detector = "ocsvm"

_RISK_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Engine:
    engine_id: str
    dataset: str
    trajectory: Trajectory


def predictions_path(detector: Detector, results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / "calibration_experiments" / f"predictions_{PRODUCTION_CONFIG}_{detector}.csv"


def kaul_split() -> dict[str, list[str]]:
    """The engine split every risk script uses: ``split_engines`` at ``SPLIT_SEED``.

    It depends only on the sorted engine IDs, so it is rebuilt here without the
    raw C-MAPSS files. No ``splits.json`` was ever committed.
    """
    engine_ids = [f"{dataset}_{unit:03d}" for dataset in DATASETS for unit in range(1, UNITS_PER_DATASET + 1)]
    return split_engines(pd.DataFrame({"engine_id": engine_ids}))


def load_engines(detector: Detector = DEFAULT_DETECTOR, results_dir: Path = RESULTS_DIR) -> tuple[Engine, ...]:
    predictions = pd.read_csv(predictions_path(detector, results_dir))
    fit_scores = pd.read_csv(results_dir / "all_fit_scores.csv")
    split = kaul_split()

    _check_against_split(predictions, fit_scores, split)
    _check_dataset_calibration(predictions, _fit_lifetimes(fit_scores, split), detector)
    return tuple(_to_engine(engine_id, rows) for engine_id, rows in predictions.groupby("engine_id", sort=True))


class CmapssEngines:
    """A ``SampleEngine`` drawing uniformly from the engines of the requested pool."""

    def __init__(self, engines: tuple[Engine, ...], split: Mapping[str, list[str]]) -> None:
        by_id = {engine.engine_id: engine for engine in engines}
        self._pools = {
            pool: tuple(by_id[engine_id] for engine_id in sorted(split[pool]) if engine_id in by_id)
            for pool in POOLS
        }
        if self._pools["fit"]:
            raise ValueError("fit engines trained the detectors and must not be served")

    def engines(self, pool: str) -> tuple[Engine, ...]:
        if pool == "fit":
            raise ValueError("the fit pool trained Kaul's detectors; evaluate on 'calibration' or 'test'")
        if pool not in POOLS:
            raise ValueError(f"unknown pool {pool!r}; expected one of {POOLS}")
        engines = self._pools[pool]
        if not engines:
            raise ValueError(f"no scored engines in the {pool!r} pool; Kaul's test split has not been evaluated")
        return engines

    def draw(self, pool: str, rng: np.random.Generator) -> Engine:
        engines = self.engines(pool)
        return engines[int(rng.integers(len(engines)))]

    def __call__(self, pool: str, rng: np.random.Generator) -> Trajectory:
        return self.draw(pool, rng).trajectory


@functools.cache
def cmapss_engines(detector: Detector = DEFAULT_DETECTOR) -> CmapssEngines:
    return CmapssEngines(load_engines(detector), kaul_split())


def sample_engine(pool: str, rng: np.random.Generator) -> Trajectory:
    """The production detector (OC-SVM, per the finalisation report) as a ``SampleEngine``."""
    return cmapss_engines()(pool, rng)


def _to_engine(engine_id: str, rows: pd.DataFrame) -> Engine:
    rows = rows.sort_values("cycle")
    datasets = rows["dataset"].unique()
    if len(datasets) != 1 or not engine_id.startswith(f"{datasets[0]}_"):
        raise ValueError(f"{engine_id}: rows carry datasets {list(datasets)}")
    if not np.array_equal(rows["cycle"].to_numpy(), np.arange(1, len(rows) + 1)):
        raise ValueError(f"{engine_id}: cycles are not contiguous from 1")
    return Engine(
        engine_id=engine_id,
        dataset=str(datasets[0]),
        trajectory=Trajectory(
            risk=tuple(rows["risk_score"].astype(float)),
            rul=tuple(rows["rul_true"].astype(float)),
        ),
    )


def _check_against_split(
    predictions: pd.DataFrame, fit_scores: pd.DataFrame, split: Mapping[str, list[str]]
) -> None:
    scored = set(predictions["engine_id"])
    if scored & set(split["fit"]) or scored & set(fit_scores["engine_id"]):
        raise ValueError("predictions include fit-split engines")
    if set(fit_scores["engine_id"]) != set(split["fit"]):
        raise ValueError("Kaul's fit scores do not match split_engines at SPLIT_SEED")
    if not scored <= set(split["calibration"]) | set(split["test"]):
        raise ValueError("predictions include engines outside the split")


def _fit_lifetimes(fit_scores: pd.DataFrame, split: Mapping[str, list[str]]) -> dict[str, float]:
    """Each dataset's longest fit-split lifetime: the denominator its calibrator uses."""
    fit = fit_scores[fit_scores["engine_id"].isin(split["fit"])]
    return {str(dataset): float(cycles.max()) for dataset, cycles in fit.groupby("dataset")["cycle"]}


def _check_dataset_calibration(
    predictions: pd.DataFrame, max_lifetime: Mapping[str, float], detector: Detector
) -> None:
    """Each dataset's rows fit one calibrator of their own.

    Two fingerprints of an isotonic ``CalibrationBundle``: ``rul_estimate`` is
    a single non-increasing function of the detector score, and risk is
    ``1 - clip(rul_estimate / L, 0, 1)`` with that dataset's fit-split ``L``.
    """
    for dataset, rows in predictions.groupby("dataset"):
        if dataset not in max_lifetime:
            raise ValueError(f"no fit-split lifetime for dataset {dataset}")
        by_score = rows.sort_values([f"{detector}_score", "rul_estimate"], ascending=[True, False])
        if np.any(np.diff(by_score["rul_estimate"].to_numpy()) > _RISK_TOLERANCE):
            raise ValueError(f"{dataset} RUL estimates are not one isotonic map of the {detector} score")
        expected = 1.0 - np.clip(rows["rul_estimate"] / max_lifetime[dataset], 0.0, 1.0)
        if not np.allclose(rows["risk_score"], expected, rtol=0.0, atol=_RISK_TOLERANCE):
            raise ValueError(f"{dataset} risk was not produced by the {dataset} calibrator")
