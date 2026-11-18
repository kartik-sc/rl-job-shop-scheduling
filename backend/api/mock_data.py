"""Mock episode event logs conforming to shared/schemas/event_log.schema.json."""

from __future__ import annotations

from typing import Any


def get_mock_episode() -> dict[str, Any]:
    """Return the canonical mock episode (6 machines, 20 operations, 1 PM, 1 breakdown).

    Schedule properties:
    - 6 machines (0..5), 5 jobs (0..4), 4 operations per job = 20 completed operations.
    - All job precedences (op 0 -> op 1 -> op 2 -> op 3) and machine non-overlap
      constraints are strictly satisfied.
    - Machine 2 accumulates degradation across 3 operations (risk 0.11 -> 0.78) and
      undergoes preventive maintenance at t=42..50 (t_pm = 8), resetting risk to 0.08.
    - Machine 4 accumulates high risk (0.14 -> 0.84 -> 0.93) and suffers a mid-operation
      breakdown during (job=3, op=2) at t=38. Repair takes t_repair = 40 (t=38..78),
      resetting risk to 0.09, after which (job=3, op=2) restarts at t=78..94.
    """
    return {
        "meta": {
            "instance": "taillard_mock_5x6",
            "regime": "medium",
            "seed": 42,
            "policy": "full_ppo",
            "n_jobs": 5,
            "n_machines": 6,
        },
        "events": [
            # Initial risk readings at t=0 (healthy engines from C-MAPSS fit/cal distribution)
            {"t": 0, "type": "risk_update", "machine": 0, "risk": 0.10},
            {"t": 0, "type": "risk_update", "machine": 1, "risk": 0.08},
            {"t": 0, "type": "risk_update", "machine": 2, "risk": 0.11},
            {"t": 0, "type": "risk_update", "machine": 3, "risk": 0.09},
            {"t": 0, "type": "risk_update", "machine": 4, "risk": 0.14},
            {"t": 0, "type": "risk_update", "machine": 5, "risk": 0.07},
            # Wave 1 starts at t=0
            {"t": 0, "type": "op_start", "machine": 0, "job": 0, "op": 0},
            {"t": 0, "type": "op_start", "machine": 1, "job": 1, "op": 0},
            {"t": 0, "type": "op_start", "machine": 2, "job": 2, "op": 0},
            {"t": 0, "type": "op_start", "machine": 4, "job": 3, "op": 0},
            {"t": 0, "type": "op_start", "machine": 5, "job": 4, "op": 0},
            # Mid-operation sensor risk updates
            {"t": 8, "type": "risk_update", "machine": 2, "risk": 0.24},
            {"t": 8, "type": "risk_update", "machine": 4, "risk": 0.31},
            # t=14 completions & next dispatches
            {"t": 14, "type": "op_end", "machine": 0, "job": 0, "op": 0},
            {"t": 14, "type": "risk_update", "machine": 0, "risk": 0.18},
            {"t": 14, "type": "op_end", "machine": 1, "job": 1, "op": 0},
            {"t": 14, "type": "risk_update", "machine": 1, "risk": 0.15},
            {"t": 14, "type": "op_end", "machine": 2, "job": 2, "op": 0},
            {"t": 14, "type": "risk_update", "machine": 2, "risk": 0.34},
            {"t": 14, "type": "op_start", "machine": 0, "job": 2, "op": 1},
            {"t": 14, "type": "op_start", "machine": 2, "job": 0, "op": 1},
            # t=16 completions & next dispatches
            {"t": 16, "type": "op_end", "machine": 4, "job": 3, "op": 0},
            {"t": 16, "type": "risk_update", "machine": 4, "risk": 0.48},
            {"t": 16, "type": "op_end", "machine": 5, "job": 4, "op": 0},
            {"t": 16, "type": "risk_update", "machine": 5, "risk": 0.16},
            {"t": 16, "type": "op_start", "machine": 3, "job": 3, "op": 1},
            {"t": 16, "type": "op_start", "machine": 4, "job": 4, "op": 1},
            # Mid-wave risk progression
            {"t": 22, "type": "risk_update", "machine": 2, "risk": 0.46},
            {"t": 24, "type": "risk_update", "machine": 4, "risk": 0.67},
            # t=28 completions & dispatches
            {"t": 28, "type": "op_end", "machine": 2, "job": 0, "op": 1},
            {"t": 28, "type": "risk_update", "machine": 2, "risk": 0.56},
            {"t": 28, "type": "op_start", "machine": 1, "job": 0, "op": 2},
            {"t": 28, "type": "op_start", "machine": 2, "job": 1, "op": 1},
            # t=30 completion & dispatch
            {"t": 30, "type": "op_end", "machine": 0, "job": 2, "op": 1},
            {"t": 30, "type": "risk_update", "machine": 0, "risk": 0.29},
            {"t": 30, "type": "op_start", "machine": 5, "job": 2, "op": 2},
            # t=32 completions & dispatches
            {"t": 32, "type": "op_end", "machine": 3, "job": 3, "op": 1},
            {"t": 32, "type": "risk_update", "machine": 3, "risk": 0.21},
            {"t": 32, "type": "op_end", "machine": 4, "job": 4, "op": 1},
            {"t": 32, "type": "risk_update", "machine": 4, "risk": 0.84},
            {"t": 32, "type": "op_start", "machine": 0, "job": 4, "op": 2},
            # Machine 4 starts (job=3, op=2) at high risk (0.84) and breaks down at t=38
            {"t": 32, "type": "op_start", "machine": 4, "job": 3, "op": 2},
            {"t": 36, "type": "risk_update", "machine": 2, "risk": 0.69},
            {"t": 38, "type": "risk_update", "machine": 4, "risk": 0.93},
            {"t": 38, "type": "breakdown", "machine": 4, "job": 3, "op": 2, "risk": 0.93},
            # t=42 Machine 2 finishes (job=1, op=1) with critical risk 0.78 -> Preventive Maintenance!
            {"t": 42, "type": "op_end", "machine": 2, "job": 1, "op": 1},
            {"t": 42, "type": "risk_update", "machine": 2, "risk": 0.78},
            {"t": 42, "type": "maintenance_start", "machine": 2, "risk": 0.78},
            {"t": 42, "type": "op_start", "machine": 3, "job": 1, "op": 2},
            # t=44 completions & dispatches
            {"t": 44, "type": "op_end", "machine": 1, "job": 0, "op": 2},
            {"t": 44, "type": "risk_update", "machine": 1, "risk": 0.27},
            {"t": 44, "type": "op_end", "machine": 5, "job": 2, "op": 2},
            {"t": 44, "type": "risk_update", "machine": 5, "risk": 0.28},
            {"t": 44, "type": "op_start", "machine": 5, "job": 0, "op": 3},
            # t=48 Machine 0 finishes (job=4, op=2)
            {"t": 48, "type": "op_end", "machine": 0, "job": 4, "op": 2},
            {"t": 48, "type": "risk_update", "machine": 0, "risk": 0.41},
            # t=50 Machine 2 finishes Preventive Maintenance (t_pm = 8) -> risk resets to 0.08
            {"t": 50, "type": "maintenance_end", "machine": 2, "risk": 0.08},
            {"t": 50, "type": "risk_update", "machine": 2, "risk": 0.08},
            {"t": 50, "type": "op_start", "machine": 2, "job": 2, "op": 3},
            # t=58 Machine 3 finishes (job=1, op=2); Job 1 op 3 & Job 4 op 3 start
            {"t": 58, "type": "op_end", "machine": 3, "job": 1, "op": 2},
            {"t": 58, "type": "risk_update", "machine": 3, "risk": 0.35},
            {"t": 58, "type": "op_start", "machine": 0, "job": 1, "op": 3},
            {"t": 58, "type": "op_start", "machine": 3, "job": 4, "op": 3},
            # t=62 Machine 5 finishes (job=0, op=3) -> Job 0 complete
            {"t": 62, "type": "op_end", "machine": 5, "job": 0, "op": 3},
            {"t": 62, "type": "risk_update", "machine": 5, "risk": 0.39},
            # t=66 Machine 2 finishes (job=2, op=3) -> Job 2 complete
            {"t": 66, "type": "op_end", "machine": 2, "job": 2, "op": 3},
            {"t": 66, "type": "risk_update", "machine": 2, "risk": 0.19},
            # t=74 Machine 3 finishes (job=4, op=3) -> Job 4 complete
            {"t": 74, "type": "op_end", "machine": 3, "job": 4, "op": 3},
            {"t": 74, "type": "risk_update", "machine": 3, "risk": 0.49},
            # t=76 Machine 0 finishes (job=1, op=3) -> Job 1 complete
            {"t": 76, "type": "op_end", "machine": 0, "job": 1, "op": 3},
            {"t": 76, "type": "risk_update", "machine": 0, "risk": 0.54},
            # t=78 Machine 4 finishes breakdown repair (t_repair = 40 = 5x t_pm), risk resets to 0.09,
            # and restarts interrupted operation (job=3, op=2)
            {"t": 78, "type": "risk_update", "machine": 4, "risk": 0.09},
            {"t": 78, "type": "op_start", "machine": 4, "job": 3, "op": 2},
            {"t": 86, "type": "risk_update", "machine": 4, "risk": 0.15},
            # t=94 Machine 4 finishes (job=3, op=2); final op (job=3, op=3) starts on Machine 1
            {"t": 94, "type": "op_end", "machine": 4, "job": 3, "op": 2},
            {"t": 94, "type": "risk_update", "machine": 4, "risk": 0.22},
            {"t": 94, "type": "op_start", "machine": 1, "job": 3, "op": 3},
            {"t": 102, "type": "risk_update", "machine": 1, "risk": 0.34},
            # t=110 Machine 1 finishes (job=3, op=3) -> Episode ends at makespan 110
            {"t": 110, "type": "op_end", "machine": 1, "job": 3, "op": 3},
            {"t": 110, "type": "risk_update", "machine": 1, "risk": 0.40},
            {"t": 110, "type": "episode_end"},
        ],
    }
