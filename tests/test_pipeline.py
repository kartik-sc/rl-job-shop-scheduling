"""End to end on configs/smoke.toml: train, select, evaluate, aggregate."""

import copy
import json
from pathlib import Path

import jsonschema
import pandas as pd
import pytest

from backend.evaluation.aggregate import COMPARISONS, aggregate
from backend.evaluation.evaluate import POLICIES, evaluate, save_logs
from backend.evaluation.selection import select
from backend.training.protocol import MODES, Protocol, load_protocol
from backend.training.train import checkpoints, train

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> tuple[Protocol, Path, Path]:
    protocol = load_protocol(ROOT / "configs" / "smoke.toml")
    runs, results = tmp_path_factory.mktemp("runs"), tmp_path_factory.mktemp("results")
    for mode in MODES:
        for regime in protocol.regime_names:
            train(protocol, mode, regime, protocol.seeds.train[0], runs_root=runs, vectorize="dummy", quiet=True)
    select(protocol, runs_root=runs, results_root=results)
    evaluate(protocol, results_root=results)
    aggregate(protocol, results_root=results)
    return protocol, runs, results / protocol.short_hash


def test_training_saves_checkpoints_and_config(pipeline: tuple[Protocol, Path, Path]) -> None:
    protocol, runs, _ = pipeline
    run = runs / protocol.short_hash / "full_high_s0"
    labels = [label for label, _ in checkpoints(run)]

    assert labels[-1] == "final" and len(labels) >= 2
    config = json.loads((run / "config.json").read_text())
    assert config["protocol_sha256"] == protocol.sha256 and config["mode"] == "full"
    assert (run / "progress.csv").exists() and (run / "monitor.csv").exists()


def test_training_refuses_to_overwrite_a_finished_run(pipeline: tuple[Protocol, Path, Path]) -> None:
    protocol, runs, _ = pipeline
    with pytest.raises(FileExistsError):
        train(protocol, "blind", "low", 0, runs_root=runs, vectorize="dummy", quiet=True)


def test_selection_covers_every_regime_and_run(pipeline: tuple[Protocol, Path, Path]) -> None:
    protocol, _, out = pipeline
    selection = json.loads((out / "selection.json").read_text())

    assert selection["protocol"] == protocol.sha256
    for regime in protocol.regime_names:
        tuned = selection["threshold"][regime]
        assert set(tuned["scores"]) == {f"tau={tau:g}" for tau in protocol.tau_grid}
        assert tuned["scores"][f"tau={tuned['tau']:g}"] == min(tuned["scores"].values())
        for mode in MODES:
            entry = selection["ppo"][mode][regime]["s0"]
            assert Path(entry["checkpoint"]).exists()
            assert entry["scores"][entry["label"]] == min(entry["scores"].values())


def test_tuning_and_testing_use_their_own_instances_and_seeds(pipeline: tuple[Protocol, Path, Path]) -> None:
    protocol, _, out = pipeline
    tune = pd.concat(pd.read_csv(path) for path in (out / "tune").glob("*.csv"))
    test = pd.concat(pd.read_csv(path) for path in (out / "episodes").glob("*.csv"))

    assert set(tune["instance"]) == set(protocol.split.train_instances)
    assert set(tune["seed"]) == set(protocol.seeds.tune) and set(tune["phase"]) == {"tune"}
    assert set(test["instance"]) == set(protocol.split.test_instances)
    assert set(test["seed"]) == set(protocol.seeds.test) and set(test["phase"]) == {"test"}


def test_summary_has_every_policy_in_every_regime(pipeline: tuple[Protocol, Path, Path]) -> None:
    protocol, _, out = pipeline
    summary = pd.read_csv(out / "summary.csv")
    overall = summary[summary["size"] == "all"]

    assert set(zip(overall["policy"], overall["regime"], strict=True)) == {
        (policy, regime) for policy in POLICIES for regime in protocol.regime_names
    }
    assert (overall["gap_lo"] <= overall["gap_mean"]).all() and (overall["gap_mean"] <= overall["gap_hi"]).all()
    assert (overall["instances"] == len(protocol.split.test_instances)).all()


def test_paired_table_covers_each_comparison(pipeline: tuple[Protocol, Path, Path]) -> None:
    protocol, _, out = pipeline
    paired = pd.read_csv(out / "paired.csv")
    expected_pairs = len(protocol.split.test_instances) * len(protocol.seeds.test)

    assert set(zip(paired["a"], paired["b"], strict=True)) == set(COMPARISONS)
    assert set(paired["regime"]) == set(protocol.regime_names)
    assert (paired["pairs"] == expected_pairs).all()


def test_evaluation_resumes_without_rerunning(pipeline: tuple[Protocol, Path, Path]) -> None:
    protocol, _, out = pipeline
    assert evaluate(protocol, results_root=out.parent) == []


def test_saved_logs_fit_the_schema_once_repair_end_is_added(pipeline: tuple[Protocol, Path, Path]) -> None:
    protocol, _, out = pipeline
    schema = json.loads((ROOT / "shared" / "schemas" / "event_log.schema.json").read_text())
    proposed = copy.deepcopy(schema)
    proposed["properties"]["events"]["items"]["properties"]["type"]["enum"].append("repair_end")

    paths = save_logs(protocol, "la01", protocol.seeds.test[0], results_root=out.parent)

    assert len(paths) == len(POLICIES) * len(protocol.regime_names)
    for path in paths:
        jsonschema.validate(json.loads(path.read_text()), proposed)
