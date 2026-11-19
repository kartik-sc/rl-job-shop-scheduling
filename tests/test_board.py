"""The TensorBoard export reproduces Kaul's headline numbers from the committed CSVs."""

from pathlib import Path

import pandas as pd

from backend.evaluation.board import PER_ENGINE, POLICIES, REGIMES, SUMMARY, baseline_table, load_predictions, risk_table, write_board


def test_risk_table_matches_kauls_phase3_results() -> None:
    table = risk_table(load_predictions(), pd.read_csv(PER_ENGINE))
    assert table.loc["FD001", "near-failure RUL MAE (cycles)"] == 3.726
    assert table.loc["FD003", "near-failure RUL MAE (cycles)"] == 12.049
    assert table.loc["FD001", "median risk Spearman"] == 0.758


def test_baseline_table_covers_every_policy_and_regime() -> None:
    table = baseline_table(pd.read_csv(SUMMARY))
    assert len(table) == len(POLICIES) * len(REGIMES) and not table.isna().any().any()


def test_board_writes_one_run_per_series(tmp_path: Path) -> None:
    write_board(tmp_path)
    runs = {path.name for path in tmp_path.iterdir()}
    assert runs == {"kaul_risk", "report", *(f"baseline_{policy}" for policy in POLICIES)}
    assert all(any(run.glob("events.out.tfevents.*")) for run in tmp_path.iterdir())
