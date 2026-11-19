# RAJS: Progress Report

Risk-Aware Job-Shop Scheduling: a PPO agent that schedules jobs on machines that wear out, and decides when to do preventive maintenance using a calibrated failure-risk signal learned from NASA C-MAPSS engine data.

## Summary

| Area | Status | Key result |
|---|---|---|
| Risk model (Kaul) | Done on fit + calibration split | Near-failure RUL error **3.7 / 12.0 cycles** (FD001 / FD003), down from 23.8 / 42.5. Risk Spearman with true remaining life **0.76 / 0.77**. |
| Simulator | Done, verified | Matches `job_shop_lib` makespans exactly on Taillard ta01-ta10 + 4 larger instances. |
| Gym env + PPO pipeline | Done | MaskablePPO, fixed seeds, checkpoints, TensorBoard, hashed protocol, evaluation with bootstrap CIs + Wilcoxon. |
| Baselines on Kaul's risk | Done | Risk-threshold maintenance + SPT cuts the gap vs best-known makespan from **1.94 to 1.19** and breakdowns from **35 to 2.4** in the high-failure regime. |
| PPO learning | Demo-scale only | On a 6x6 instance, PPO that sees risk improves makespan **~13% in 75 s**. The risk-blind agent improves ~5%. Full-scale training (2M / 16M steps) not run yet. |
| Dashboard | Done (frontend + API) | Gantt, risk heatmap, factory floor and playback over real simulator logs. PPO panels still use illustrative data. |
| Tests | 487 passing | Covers simulator, env, risk pipeline, calibration, training/evaluation pipeline, API. |

## 1. Problem

Classic job-shop scheduling assumes machines never fail. Here every machine carries a degrading engine. Running a degraded machine risks a breakdown (long repair plus lost work). Preventive maintenance (PM) costs a short, planned stop. At each decision the agent picks one of three actions:

- dispatch a job,
- maintain the machine,
- wait.

The goal is the minimum makespan. The comparison is between two agents:

- **full**: sees calibrated risk and may maintain,
- **blind**: no health information and no maintenance.

The paired difference between them is the value of the risk signal.

## 2. Risk model (Kaul)

Code is in `backend/risk/`. Write-ups are in `results/*/phase*.md`.

- **Data:** C-MAPSS FD001 + FD003, 200 run-to-failure engines, split 120 / 40 / 40 into fit / calibration / test with a fixed seed.
- **Layer 1: anomaly detection.** OC-SVM and Isolation Forest are trained only on healthy cycles (the first 30) of fit engines, using 18 sensors plus temporal features (`D_temporal`). The anomaly score rises as an engine degrades on 100% of fit engines.
- **Layer 2: calibration.** Per-dataset isotonic regression maps the score to remaining-useful-life (RUL) and to a risk in [0, 1]. This is the production config `A_ds_iso`.
- **Results** (calibration split, OC-SVM):

  | | FD001 | FD003 |
  |---|---|---|
  | Near-failure RUL MAE (RUL ≤ 20), Phase 1 baseline | 23.8 | 42.5 |
  | Near-failure RUL MAE, production | **3.7** | **12.0** |
  | Median per-engine risk Spearman | 0.758 | 0.767 |
  | Mean risk, first 30 cycles → last 20 cycles | 0.60 → 0.97 | 0.63 → 0.96 |

- **Into the simulator** (`backend/env/risk_adapter.py`): each machine replays a real calibration engine. The policy sees `risk_score`. Failures are drawn from the true RUL, which the policy never sees. Fit engines are refused (they trained the detectors).

## 3. Simulator and environment

- `backend/env/simulator.py` is an event-driven job shop with machine health. On Taillard benchmarks with no failures it reproduces `job_shop_lib` schedules operation by operation.
- `backend/env/gym_env.py` is a Gymnasium env with:
  - action masking: dispatch / maintain / wait;
  - reward = −(clock advance) / mean operation length, with γ = 1, so the return is exactly −makespan / mean operation length;
  - full and blind observation modes.
- Failure regimes are low / medium / high hazard. PM costs 1× a mean operation; a repair costs 6×.

## 4. Baselines on Kaul's risk model

Source: `backend/evaluation/demo.py` → `results/demo/`. Instances ta06 (15x15), ta11 (20x15) and ta21 (20x20), 10 test seeds each. Every policy meets identical failures. Values are means of the per-instance means.

| Regime | SPT gap | FIFO gap | LPT gap | **Threshold+SPT gap** | SPT breakdowns | Threshold breakdowns |
|---|---|---|---|---|---|---|
| low | 0.64 | 0.51 | 0.79 | **0.39** | 3.9 | 0.4 |
| medium | 1.08 | 1.03 | 1.32 | **0.54** | 14.4 | 2.2 |
| high | 1.94 | 1.85 | 2.27 | **1.19** | 35.3 | 2.4 |

Threshold+SPT maintains a machine once its risk reaches a threshold (τ = 0.85, 0.65 and 0.55 for low, medium and high), with τ tuned on separate seeds and instances. Kaul's risk signal alone roughly halves the gap. This is the bar PPO has to beat.

## 5. Training pipeline

- `configs/protocol.toml` defines the whole experiment: regimes, instance split, seeds, PPO hyperparameters and budgets. Its sha256 is stamped on every run and every result row, so results from different protocols never mix.
- `backend/training/train.py` runs MaskablePPO with fixed seeds, 10 checkpoints per run and TensorBoard logging. TensorBoard also gets breakdowns and maintenances per 100 operations.
- `backend/evaluation/` picks τ and the best checkpoint on tuning seeds, evaluates on held-out instances and seeds, and reports 95% bootstrap CIs plus Wilcoxon signed-rank tests.

## 6. PPO status (honest)

- **Demo (6x6, 150k steps, ~75 s):**
  - full: −18.6 → −16.1 (~13% shorter makespan), with breakdowns at ~1.8 per 100 operations;
  - blind: −22.1 → −20.9 (~5%), with ~6 breakdowns per 100 operations.
- **Pilot (15x15, 200k steps):** no clear improvement yet. Large instances need the full budget.
- **Not done yet:**
  - full-scale runs (2M blind / 16M full steps × 3 regimes × 3 seeds);
  - PPO on the C-MAPSS engine (training currently uses the synthetic engine, because Kaul's test split hasn't been scored yet and fit engines can't be served);
  - real PPO logs in the dashboard.

## 7. Dashboard

Next.js + D3 frontend (`frontend/`) served by a FastAPI backend (`backend/api/`). It shows:

- a factory floor animation,
- a schedule Gantt with PM and repair blocks,
- a per-machine risk heatmap,
- playback controls.

Episodes are real simulator logs of SPT / LPT / FIFO / Threshold+SPT on Kaul's risk model (ta06, medium regime). The *Full PPO* episode, *RL Brain* and *Training Arena* are illustrative and labelled MOCK until trained policies are logged.

## 8. Next steps

1. Run the full protocol on GPU/CPU cluster time; pick the budget from the pilot curve.
2. Score Kaul's test split and register the C-MAPSS engine for training.
3. Report full PPO vs blind PPO vs Threshold+SPT with CIs; feed the PPO logs into the dashboard.
