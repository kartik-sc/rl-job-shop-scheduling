# RAJS Codebase: Complete Documentation

Audit date: 2026-11-11. Repository: `rl-job-shop-scheduling` (remote `origin` = `github.com/kartik-sc/rl-job-shop-scheduling`).

This document describes what is in the repository on every branch, what each part does, how far it has been verified, and what is still missing. Statements are marked as:

- **Verified**: checked by reading the code and/or running its tests in this audit.
- **Reported**: taken from a document or summary in the repository, not reproduced here.
- **Not verified**: not run or built in this audit.

---

## Executive Summary

RAJS (Risk-Aware Job-Shop Scheduling) is a research project. It asks whether a reinforcement-learning scheduler that can see machine-health risk and schedule preventive maintenance beats (a) a maintenance-blind RL scheduler and (b) a tuned threshold + shortest-processing-time rule. The project runs on the Taillard job-shop benchmarks and on NASA C-MAPSS turbofan degradation data.

Where things stand:

| Area | Status | Where it lives |
|---|---|---|
| Event-driven job-shop simulator | Complete and tested. Committed on `js-sim-taillard`. | `backend/env/simulator.py` |
| Machine health layer (wear, breakdowns, maintenance, repair) | Complete and tested. **Not committed.** | `backend/env/health.py`, simulator |
| Gymnasium env (`JobShopEnv`, masked actions) | Complete and tested. Env committed; some changes uncommitted. | `backend/env/gym_env.py`, `observation.py` |
| Training and evaluation harness | Complete and tested. Pilot run finished. **Not committed.** | `backend/training/`, `backend/evaluation/`, `configs/` |
| Risk layer (anomaly detection and calibration on C-MAPSS) | Layer 1 and baseline Layer 2 done, on a separate branch. **Not merged.** | `kartik-risk-model` branch |
| Dashboard (FastAPI + Next.js + D3) | Backend API and frontend code exist on a separate branch. **Not merged.** Uses mock data only. Frontend not built in this audit. | `feat/dashboard` branch |
| Risk layer wired into the simulator | Not done. The simulator uses a synthetic engine stand-in. | `backend/risk/synthetic.py` (uncommitted) |
| PPO training on real risk data, final results | Not done. A 200k-step pilot ran on synthetic health; results are not final. | `results/67b059fc/` (git-ignored) |

The most important fact for the next developer: **three lines of work exist in parallel and none has been merged into `main`**. The simulator and harness are on `js-sim-taillard` (plus uncommitted work). The risk model is on `kartik-risk-model`. The dashboard is on `feat/dashboard`. Their shared contract is `shared/schemas/event_log.schema.json`, and the branches disagree on that file (see Known Issues).

---

## Project Overview

### What the project asks

A factory has several machines. Jobs must visit machines in a fixed order. Each machine's health degrades while it processes work, and a breakdown stops it and forces a repair. A scheduler may also take a machine offline for cheaper preventive maintenance.

The project compares three policies:

1. **Maintenance-blind PPO**: reinforcement learning with no risk input and no maintenance action.
2. **Threshold + SPT**: maintain when risk exceeds a threshold tau, otherwise dispatch the shortest job. tau is tuned on training data.
3. **Full risk-aware PPO**: reinforcement learning with the risk input and a maintenance action.

Classical rules (SPT, LPT, FIFO) and the best-known Taillard makespans are reference points.

### How the pieces fit

```
C-MAPSS engine data ──► risk layer (Layer 1 anomaly score, Layer 2 calibrated risk)
                                  │  sample_engine(pool, rng) -> Trajectory(risk[], rul[])
                                  ▼
Taillard instance ──► simulator kernel + HealthModel ──► JobShopEnv (Gymnasium, masked)
                                  │                              │
                                  │                              ▼
                                  │                     MaskablePPO training
                                  ▼                              │
                       event log (schema) ◄──────────────────────┘
                                  │
                                  ▼
                 evaluation harness (selection, test grid, metrics, aggregation)
                                  │
                                  ▼
                       dashboard (FastAPI + Next.js + D3)  [mock data today]
```

### Key definitions used throughout

- **Makespan**: finish time of the last operation.
- **Gap**: makespan / best-known makespan − 1.
- **Decision**: a moment when a machine is idle and must act (dispatch a job, maintain, or wait).
- **Health model**: the object that tracks machine age, the hidden failure point, and the observable risk score.
- **Regime**: a hazard setting (`low`, `medium`, `high`) that controls how fast machines wear.
- **Common random numbers**: the same seed gives identical failure draws to every policy, so comparisons are paired.

---

## Complete Technology Stack

| Layer | Technology | Where it is used | Status |
|---|---|---|---|
| Language | Python 3.12 (WSL venv `~/venvs/rajs`); project declares `>=3.11` | everything | **Verified** (3.12.3) |
| Scheduling benchmarks | `job-shop-lib` 1.7.2 | `backend/env/instance.py`, notebook | **Verified** |
| Simulation | Pure Python (`heapq`, dataclasses) | `backend/env/simulator.py` | **Verified** |
| RL | Gymnasium 1.4.0, stable-baselines3 2.9.2, sb3-contrib 2.3 (`MaskablePPO`), PyTorch 2.15 CPU | `gym_env.py`, `training/` | **Verified** |
| Numerics and data | numpy 2.5, pandas, scipy 1.18, scikit-learn | evaluation, risk layer | **Verified** (venv) |
| Risk models | scikit-learn `OneClassSVM`, `IsolationForest`, `IsotonicRegression` | `kartik-risk-model` branch | **Reported** (code read; data absent, so not re-run) |
| API | FastAPI, Starlette TestClient, httpx, uvicorn | `backend/api/` on `feat/dashboard` (and a stub on `main`) | **Verified** (tests pass) |
| Frontend | Next.js 15, React 19, TypeScript 5.7, D3 7.9, Tailwind 4, lucide-react | `frontend/` on `feat/dashboard` | **Not verified** (not built or run) |
| Schema validation | `jsonschema` | tests | **Verified** |
| Tests | pytest | `tests/` | **Verified** |
| Config | TOML (`tomllib`) | `configs/` | **Verified** |
| Database | None | none | No database exists in this repository |
| Authentication | None | none | No authentication exists |
| Deployment | None configured | none | No deployment configuration exists |

Dependencies:

- `main`: numpy, pandas, scikit-learn, gymnasium, stable-baselines3, sb3-contrib, job-shop-lib, fastapi, uvicorn, httpx, matplotlib, pytest.
- `js-sim-taillard` (committed): adds `jsonschema>=4.0`.
- Working tree (uncommitted): adds `scipy>=1.11`.

PyTorch is installed separately, as the comment in `requirements.txt` says.

---

## Overall Application Architecture

There is no web application in the usual sense yet. The working system is a Python library with command-line tools. A dashboard prototype sits on a separate branch.

### Layers

1. **Instance layer** (`backend/env/instance.py`). Immutable `Instance` and `Operation` types, the Taillard loader, and `random_instance` for training.
2. **Kernel** (`backend/env/simulator.py`). Discrete-event simulator: an integer clock, a heap of timed events, and one queue per machine. The caller answers each `Decision` with `apply(action)`.
3. **Health** (`backend/env/health.py`). The `HealthModel` protocol and `EngineHealth`, which samples each machine's lifetime from an engine trajectory and a hazard function.
4. **Env** (`backend/env/gym_env.py`, `observation.py`). `JobShopEnv`, a Gymnasium environment that turns decisions into steps, with a fixed-size observation and action masks.
5. **Baselines** (`backend/baselines/`). SPT, LPT, FIFO and FCFS dispatch rules, and the threshold + SPT rule.
6. **Training** (`backend/training/`). Protocol loading, env factory, MaskablePPO training.
7. **Evaluation** (`backend/evaluation/`). Actors, episode runner, metrics from the event log, tau and checkpoint selection, test grid, aggregation, calibration.
8. **Risk** (`backend/risk/`). On `main`: an empty package. Uncommitted: `synthetic.py`, a stand-in engine. On `kartik-risk-model`: the real C-MAPSS pipeline (not merged).
9. **API and dashboard** (`backend/api/`, `frontend/`). A stub on `main`; a mock-data API and Next.js UI on `feat/dashboard`.

### Control flow of one episode

1. `JobShopEnv.reset(seed, options)` builds an `EngineHealth` from the seed, rates, regime and engine sampler, then a `Simulation`.
2. `Simulation.next_decision()` advances the clock to the next timestamp that has a decision, logging `risk_update` events first.
3. The actor (PPO, a rule, or the threshold rule) picks an action.
4. `env.step(action)` applies it, asks for the next decision, and returns reward `-(Δclock)/mean_duration`.
5. The episode ends when every operation is done. `info["makespan"]` is set.

### Determinism

- Failures for a machine's k-th lifetime come from `default_rng([seed, machine_id, lifetime_index])`, so every policy sees the same failures.
- Ties break on the lowest job id.
- Events at one timestamp are ordered: completions first, then breakdowns and repairs, then the risk batch, then decisions in ascending machine id.

---

## Repository and Folder Structure

Working tree on `js-sim-taillard`, as it is on disk now. Items marked `[uncommitted]` are not in git yet.

```
rl-job-shop-scheduling/
├── .gitignore                    (modified, uncommitted: ignores results/)
├── README.md                     (committed simulator section; uncommitted training and evaluation section)
├── pyproject.toml                pytest config (pythonpath ".", testpaths "tests")
├── requirements.txt              dependencies
├── CODEBASE_COMPLETE_DOCUMENTATION.md   this file [uncommitted]
├── backend/
│   ├── env/
│   │   ├── events.py             event types, Event record, log_document()
│   │   ├── instance.py           Instance/Operation, from_benchmark(), random_instance() [random_instance uncommitted]
│   │   ├── simulator.py          kernel: Simulation, decisions, run_episode()
│   │   ├── health.py             HealthModel, EngineHealth, Regime, Trajectory, failure_cycle()
│   │   ├── observation.py        ObservationLayout, build_observation()
│   │   └── gym_env.py            JobShopEnv, benchmark_sampler() [decision and encode uncommitted]
│   ├── baselines/
│   │   ├── dispatch.py           SPT, LPT, FIFO, FCFS; CLI
│   │   └── threshold.py          threshold_spt(tau) [uncommitted]
│   ├── risk/
│   │   └── synthetic.py          stand-in sample_engine [uncommitted]
│   ├── training/                 [uncommitted]
│   │   ├── protocol.py           TOML protocol loader, sha256, validation
│   │   ├── factory.py            builds Regime, samplers and env from a protocol
│   │   └── train.py              MaskablePPO training CLI
│   ├── evaluation/               [uncommitted]
│   │   ├── actors.py             Contender and actors (PPO, rules, threshold)
│   │   ├── episodes.py           Cell, play_episode, run_cells, EpisodeRecord, CSV writer
│   │   ├── metrics.py            episode_metrics() from event logs
│   │   ├── selection.py          tau and checkpoint selection on the tuning split
│   │   ├── evaluate.py           test grid, reference table, event-log export
│   │   ├── aggregate.py          summary and paired tables, bootstrap, Wilcoxon
│   │   └── calibrate.py          regime statistics under blind SPT
│   └── api/
│       └── main.py               /health only (on main)
├── configs/                      [uncommitted]
│   ├── protocol.toml             experimental protocol (placeholder values, pre-freeze)
│   ├── smoke.toml                tiny protocol for tests
│   └── pilot.toml                pilot protocol (medium regime, 200k steps)
├── tests/
│   ├── fakes.py                  test doubles (ScriptedHealth, helpers)
│   ├── test_simulator.py         Day 2 gate
│   ├── test_health.py            health layer
│   ├── test_gym_env.py           Gymnasium wrapper
│   ├── test_smoke.py             API health (main)
│   ├── test_harness.py           protocol, factory, metrics, pairing, aggregation [uncommitted]
│   ├── test_pipeline.py          end-to-end on the smoke protocol [uncommitted]
│   └── test_engine_contract.py   engine interface contract [uncommitted]
├── shared/schemas/event_log.schema.json   event-log contract (draft)
├── notebooks/
│   ├── 01_taillard_exploration.ipynb      (js-sim-taillard)
│   └── .ipynb_checkpoints/...             (committed by accident; see Known Issues)
├── data/README.md                describes git-ignored data folders
├── frontend/README.md            placeholder on main; full guide on feat/dashboard
└── results/, runs/, .venv/       git-ignored outputs and environment
```

Project material outside the repository, in the parent `aiml-el` folder:

- `aiml-el/DECISIONS.md`: decision log. Updated this session with the health layer, the Full-mode note, and harness decisions.
- `aiml-el/docs/PLAN.md`, `INTERFACES.md`, `RAJS Project Report.md`: planning and interface documents. `INTERFACES.md` is still marked DRAFT.

---

## Detailed Feature-by-Feature Breakdown

Each feature gives its status, the files involved, and what it does.

### F1. Taillard instance loading

- **Status**: Complete. **Verified**.
- **Files**: `backend/env/instance.py`.
- **What it does**: `from_benchmark("ta01")` loads a `job_shop_lib` benchmark and converts it into frozen dataclasses. Each `Instance` exposes `num_jobs`, `num_machines`, `mean_duration`, `trivial_lower_bound` and `reference_makespan` (the optimum if known, otherwise the upper bound).
- **Verified values**: ft06 optimum 55; la01 666; ta06 1238; ta10 1241; ta11 1357; ta20 1348.

### F2. Random training instances

- **Status**: Complete for the protocol. **Verified** by test.
- **Files**: `backend/env/instance.py` (`random_instance`, uncommitted); `backend/training/factory.py` (`training_sampler`).
- **What it does**: Builds Taillard-style instances. Each job visits every machine once in random order, with durations uniform on `[duration_low, duration_high]`. They have no known optimum. Training mixes them with the training Taillard instances, in the proportion `random_share` and at the sizes `random_sizes`.

### F3. Event-driven simulator kernel (no failures)

- **Status**: Complete. Committed on `js-sim-taillard` in `7d8a6bc`. **Verified**: the Day 2 gate passes.
- **Files**: `backend/env/simulator.py`, `backend/env/events.py`, `backend/baselines/dispatch.py`, `tests/test_simulator.py`, `tests/fakes.py`.
- **What it does**:
  - Holds an integer clock, a heap of `(time, kind, machine)` timed events, and a queue per machine.
  - The clock moves only by popping the heap. All timed events at one timestamp are applied before any machine decides, so completions come first.
  - An idle machine with a non-empty queue needs a decision. Decisions go to the caller in ascending machine id.
  - Non-delay: the clock never moves while a decision is pending.
  - Rejects zero-length operations, since one would finish at the timestamp it started and break the ordering.
- **Gate (`tests/test_simulator.py`)**: makespans pinned as literals for ta01, ta11, ta21, ta31 and ta41, and equal to `job_shop_lib` on ta01–ta10. It also replays the library's own schedules operation by operation.
- **Pinned makespans**:

| Instance | SPT | LPT | FCFS (library rule) | FIFO (queue wait) |
|---|---|---|---|---|
| ta01 | 1462 | 1701 | 1438 | 1486 |
| ta11 | 1865 | 2046 | 1665 | 1701 |
| ta21 | 2175 | 2266 | 1964 | 2089 |
| ta31 | 2335 | 2417 | 2143 | 2277 |
| ta41 | 2499 | 2925 | 2538 | 2543 |

- **Naming**: `fifo` means "longest wait in this machine's queue" (the shop-floor meaning, used in reports). `fcfs` means "lowest position in the job", the library's rule of that name, kept only so the gate can reproduce the library. Do not report `fcfs` as FIFO.

### F4. Event log

- **Status**: Writer complete; schema is a draft. **Verified** in tests.
- **Files**: `backend/env/events.py`, `shared/schemas/event_log.schema.json`.
- **What it does**: `Event(t, type, machine, job, op, risk)`; `to_dict()` drops `None` fields. `log_document()` wraps events with `meta` (instance, regime, seed, policy, n_jobs, n_machines).
- **Event types in code**: `op_start`, `op_end`, `risk_update`, `maintenance_start`, `maintenance_end`, `breakdown`, `repair_end`, `episode_end`.
- **Event types in the committed schema**: all except `repair_end`. `repair_end` awaits team agreement. A strict `xfail` test in `tests/test_health.py` flags this.

### F5. Machine health layer

- **Status**: Complete and tested. **Uncommitted**. The design was approved; the code was built this session. **Verified**: the health tests pass in the working tree.
- **Files**: `backend/env/health.py`, plus changes in `backend/env/simulator.py`.
- **What it does**:
  - Wear accumulates only while a machine processes an operation.
  - Each machine lifetime has one failure point, drawn when the lifetime starts from `default_rng([seed, machine, lifetime])`. The engine trajectory provides risk and true remaining life per cycle. A hazard function `exp(-rul/scale)` converts that into a failure cycle by inverting the cumulative hazard. The cycle becomes an integer processing time of at least 1.
  - When an operation starts, the kernel schedules `OP_END` if it finishes first, or `BREAKDOWN` if the failure comes first. No cancellation is needed.
  - A breakdown requeues the running job with its original `arrived_at`, wears the machine for the elapsed time, and takes the machine offline for `t_repair`.
  - `Maintain` takes the machine offline for `t_pm`. It is illegal on a machine with zero age (no work since its last reset), so a policy cannot maintain forever.
  - On return, the machine resets: a new lifetime, a new engine, and `last_reset_at` is updated.
  - `Wait` is legal only in full mode, on an idle machine with an empty queue. A forced wait (where Wait would be the only legal action) is not asked.
  - Regime validation: `t_repair` must be 5–10× `t_pm`.
- **Downtime defaults**: `t_pm = round(pm_factor × mean_duration)` and `t_repair = round(repair_factor × mean_duration)`, each at least 1.

### F6. Observation builder

- **Status**: Complete. **Verified**: size tests, bounded-range tests, and a no-leak test.
- **Files**: `backend/env/observation.py`; `Simulation.snapshot()` in `simulator.py`.
- **What it does**:
  - `snapshot()` returns frozen views of jobs and machines. It contains no failure point, no true remaining life, and no pending timed events.
  - A busy machine's `free_at` is its nominal end (`busy_since + duration`), never the pending breakdown time.
  - `build_observation(snapshot, decision, layout)` builds a flat vector padded to 20 jobs × 15 machines. Every feature lies in [0, 1]. Times are squashed with `x/(1+x)` in mean-duration units.
  - Full layout: 644 features. Blind layout: 599 (drops risk, age and time since reset).
- **No-leak test**: two worlds that differ only in when one machine fails produce identical observations until that failure happens, including while the failure is pending.

### F7. Gymnasium environment `JobShopEnv`

- **Status**: Complete and tested. The env is committed on `js-sim-taillard` (`17e158a`). The `decision` property and `encode()` method are **uncommitted**.
- **Files**: `backend/env/gym_env.py`, `tests/test_gym_env.py`.
- **What it does**:
  - Action space `Discrete(22)`: indices 0–19 dispatch a job slot, 20 is Maintain, 21 is Wait.
  - `action_masks()` returns the legal-action mask that `MaskablePPO` reads.
  - `reset(seed=s, options={"instance_name", "regime"})`: `s` is used directly as the health seed. Without a seed, a fresh one is drawn and reported in `info["episode_seed"]`.
  - `step(action)`: reward `-(Δclock)/mean_duration`, so the undiscounted return equals `-makespan`. `info["events"]` delivers every event exactly once. `info["illegal_action"]` flags a fallback.
  - An illegal action is replaced by the lowest legal action. A no-op was rejected because stalling would score zero, an exploit.
  - Two modes: `full` (maintenance and risk) and `blind` (no maintenance, no risk features).
  - `truncated` at `max_steps`.
- **Verified**: `check_env` passes with warnings treated as errors in both modes. The mask matches kernel legality on every action of a small instance and on ta01. Five deliberate breakages were all caught.
- **Measured throughput**: about 3,800 env steps/s (full) and 4,100 (blind) in one process. A full-mode ta01 episode is about 2,079 steps; blind is about 263.

### F8. Baseline dispatch rules and threshold rule

- **Status**: Complete. **Verified**.
- **Files**: `backend/baselines/dispatch.py` (committed); `backend/baselines/threshold.py` (uncommitted).
- **What it does**:
  - SPT, LPT, FIFO and FCFS each pick one candidate per decision, with ties to the lowest job id. CLI: `python -m backend.baselines.dispatch ta01 spt --log out.json --regime low`. `fcfs` cannot be logged.
  - `threshold_spt(tau)`: maintain if allowed and `risk > tau` (strictly greater); otherwise SPT; otherwise wait. `tau = 1.0` never maintains.

### F9. Training harness

- **Status**: Complete and tested. Pilot run completed. **Uncommitted**.
- **Files**: `backend/training/protocol.py`, `factory.py`, `train.py`; `configs/*.toml`.
- **What it does**:
  - `protocol.toml` defines regimes, the degradation-rate range, the instance split, seeds, PPO hyperparameters, budgets and the tau grid.
  - The file's sha256 is stamped on every run and every result row. Results from different protocols cannot be combined.
  - `train.py` trains one MaskablePPO model for one mode, regime and training seed, using `SubprocVecEnv`. Training resets are unseeded, so each episode draws fresh failures. It saves `config.json`, `progress.csv`, `monitor.csv`, checkpoints and `final.zip` under `runs/<hash>/`.
  - It refuses to overwrite a finished run.
- **Validation (`_validate`)**: rejects overlapping train and test instances or seeds, repair factors outside 5–10×, a tau grid without 1.0, oversized random instances, and a `gamma` key (gamma is fixed at 1.0 in code).

### F10. Evaluation harness

- **Status**: Complete and tested. **Uncommitted**.
- **Files**: `backend/evaluation/` (`actors.py`, `episodes.py`, `metrics.py`, `selection.py`, `evaluate.py`, `aggregate.py`, `calibrate.py`).
- **What it does**:
  - **Actors**: a contender is a policy plus a variant (a tau value or a PPO checkpoint). Rules and the threshold rule act through `env.encode(...)`, so they share the PPO path.
  - **Metrics** (`metrics.py`) come from the event log only: makespan, gap, breakdowns, breakdown rate per 100 operations, maintenance count and downtime, maintenance share of machine time, repair downtime, lost work. A hand-worked episode gives known values.
  - **Selection** (`selection.py`) chooses tau per regime and the best checkpoint per PPO run, using the tuning split only, by lowest mean gap.
  - **Evaluate** (`evaluate.py`) runs each chosen contender on the test grid. It skips files that already exist, so runs resume. It can export event logs for the dashboard and writes `reference.csv` (best-known makespans, and SPT, LPT and FIFO with no failures).
  - **Aggregate** (`aggregate.py`) refuses mixed protocol hashes, truncated episodes, illegal actions, and contenders that did not see the same (instance, seed) pairs. It averages within each instance first, gives 95% bootstrap intervals over instances, and runs paired Wilcoxon tests.
  - **Calibrate** (`calibrate.py`) reports regime statistics under blind SPT only (lifetime quantiles, the chance an operation outlasts a lifetime, breakdowns per episode). Regimes can then be fixed before any PPO result exists.
- **Verified**: 51 new tests across `test_harness.py`, `test_pipeline.py` and `test_engine_contract.py`. Seven deliberate breakages were all caught: threshold comparison, unseeded rates, swapped downtime metrics, wrong pool, wrong seeds, missing pairing check, and worst-candidate selection.

### F11. Synthetic engine (stand-in for the risk layer)

- **Status**: Placeholder. **Uncommitted**.
- **Files**: `backend/risk/synthetic.py`.
- **What it does**: Samples a lifetime of 128–362 cycles. Risk rises as `(t/T)^p` with `p` in [1, 3], plus Gaussian noise, clipped to [0, 1]. RUL counts down to zero. It uses no real data and does not model real degradation.
- **Limitation**: this is not the real C-MAPSS risk score. Results from it describe the scheduler under synthetic health only.

### F12. Risk layer (C-MAPSS anomaly detection and calibration)

- **Status**: Layer 1 done; baseline Layer 2 done; experiments done. **On branch `kartik-risk-model`, not merged.** Figures are **Reported** by the branch's document. The code was read in this audit. Tests were run in this audit with the C-MAPSS data absent: 79 passed, 187 skipped.
- **Files (on `kartik-risk-model`)**:
  - `backend/risk/cmapss_pipeline.py`: loads FD001 and FD003 (`train_FD00X.txt`), names the 26 columns, drops 6 constant columns (`op3`, `s1`, `s5`, `s16`, `s18`, `s19`) to leave 18 features, computes RUL as the last cycle minus the current cycle, splits engines 120 / 40 / 40 with `SPLIT_SEED = 42`, and scales using fit-split statistics only.
  - `backend/risk/anomaly_models.py`: One-Class SVM (RBF, ν = 0.05, gamma "scale") and Isolation Forest (200 trees, seed 42), both trained only on healthy early cycles (1–30) of fit engines. Scores are negated so higher means more anomalous.
  - `backend/risk/calibration.py`: isotonic regression (decreasing) from anomaly score to expected RUL, fitted on the 40 calibration engines only. Risk = `1 − clamp(RUL_estimate / max_lifetime, 0, 1)`, where `max_lifetime` comes from fit-split engines.
  - `train_anomaly.py` and `train_calibration.py`: entry points that save bundles to `models/` (git-ignored).
  - `diagnostic.py`, `feature_analysis.py`, `rul_error_analysis.py`, `anomaly_experiments.py`, `calibration_experiments.py`, `calibration_extended.py`: analysis and experiment scripts. `_phase3_inspect.py` is a scratch script committed by accident.
- **Reported results (fit split, 120 engines)**:

| Model | Normalised separation | Mean Spearman ρ with RUL | Median early-warning lead (cycles) | Engines warned |
|---|---|---|---|---|
| OC-SVM | 7.95 | 0.75 | 62.5 | 100% |
| Isolation Forest | 8.65 | 0.77 | 56.0 | 100% |

  Both models give 100% directional success on the fit split.
- **Reported calibration results (calibration split, 40 engines)**: RUL error at true RUL 0–20 is 16.1 cycles MAE for OC-SVM and 52.1 for IForest. The document names this the critical open problem (P1): IForest scores are too compressed to separate near-failure from mid-life cycles.
- **Not yet done**: the test split has not been used. Nothing in this layer feeds the scheduler yet, because `sample_engine` is not implemented.

### F13. Dashboard

- **Status**: Prototype. Backend API and frontend code exist on `feat/dashboard`, **not merged**. The frontend was not built or run in this audit.
- **Files (on `feat/dashboard`)**:
  - `backend/api/main.py`: FastAPI app with CORS, `/health`, `/episodes` (list), `/episodes/mock`, and `/episodes/{id}`, each also under `/api/`. It serves only the mock episode.
  - `backend/api/mock_data.py`: a hand-written episode (6 machines, 5 jobs, 20 operations, one maintenance on machine 2, one breakdown on machine 4) with roughly 80 event records. It conforms to the schema.
  - `frontend/src/app/page.tsx`: main page. Fetches `/episodes/mock` from `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`) and falls back to a built-in mock if the API is unreachable. It has three policy presets with hard-coded labels and numbers.
  - `frontend/src/components/GanttChart.tsx`: D3 schedule chart with operation, maintenance and breakdown blocks.
  - `frontend/src/components/RiskHeatmap.tsx`: D3 per-machine risk heatmap with health bands (Healthy < 0.30, Degrading 0.30–0.70, Critical > 0.70).
  - `frontend/src/components/PlaybackControls.tsx`: play and pause, step, speeds 0.5×–4×, scrubbing.
  - `frontend/src/lib/parseEpisode.ts`: converts an event log into operation, maintenance and breakdown intervals, and per-machine risk samples.
  - `frontend/src/lib/mockEpisode.ts`: the built-in fallback episode.
  - `frontend/src/types/eventLog.ts`: TypeScript types for the event log.
- **Verified**: 2 API tests pass (`/health` and the mock episode's schema and plan constraints).
- **Not verified**: the frontend build, type-checking, and rendering.
- **Limitations**: the only data source is the mock. The three policy presets carry numbers (`110u`, `116u`, `132u`, τ = 0.70) that come from the hand-written mock, not from a simulator run. Real episodes written by the harness are not loaded.

### F14. Exploratory notebook

- **Status**: Complete as an exploration. **Reported** (content read).
- **File**: `notebooks/01_taillard_exploration.ipynb` (on `js-sim-taillard`).
- **What it does**: loads the Taillard instances, tabulates sizes and bounds, compares trivial lower bounds with published optima for 15×15 and 20×15, runs the library's dispatching baselines, and checks whether the filter choice changes the answer for ta01. It covers the questions the Day 2 gate later settled.

---

## Frontend Implementation

Only on `feat/dashboard`. The `main` branch has a placeholder README in `frontend/`.

| Item | Detail | Status |
|---|---|---|
| Framework | Next.js 15 (App Router), React 19, TypeScript | Declared in `package.json`; **not built** in this audit |
| Styling | Tailwind CSS 4 via PostCSS, plus `globals.css` | Declared |
| Charts | D3 7.9 (Gantt and heatmap) | Code present |
| Icons | lucide-react | Used in `page.tsx` |
| Data | `fetch` to the API, with fallback to the built-in mock | Code present |
| Tests | None on the frontend | Missing |
| Lockfile | `package-lock.json` (2,519 lines) committed | Present |

Known frontend gaps: no tests, no error handling beyond the fallback, no loading of real episode logs, and presets with hard-coded numbers.

---

## Backend Implementation

### On `main`

- `backend/api/main.py`: a single `/health` endpoint.
- `backend/risk/`, `backend/env/`, `backend/baselines/`, `backend/evaluation/`: empty packages (`__init__.py` only).

### On `js-sim-taillard` (committed)

- **Simulator and health**: `simulator.py` (509 lines), `health.py` (170), `events.py` (63), `observation.py` (134), `gym_env.py` (179), `instance.py` (79), `dispatch.py` (95).
- **Tests**: `test_simulator.py` (434 lines), `test_health.py` (612), `test_gym_env.py` (308), `fakes.py` (94).
- **Run on the committed tree**: **334 passed, 1 xfailed**. The xfail is the `repair_end` schema test.

### Uncommitted in the working tree

- **Modified**: `backend/env/gym_env.py` (adds `decision` and `encode`), `backend/env/instance.py` (adds `random_instance`), `README.md`, `requirements.txt` (adds scipy), `.gitignore` (adds `results/`).
- **New**: `backend/baselines/threshold.py` (22 lines), `backend/risk/synthetic.py` (26), `backend/training/` (446 lines across three files), `backend/evaluation/` (about 900 lines across seven files), `configs/` (three TOML files), and three test files (`test_harness.py` 303 lines, `test_pipeline.py` 113, `test_engine_contract.py` 50).
- **Run in the working tree**: **385 passed, 1 xfailed** (last run this session, before this audit).

### On `kartik-risk-model`

- `backend/risk/`: 12 modules, about 7,100 lines, including the scratch script `_phase3_inspect.py`.
- `tests/`: six files, about 2,300 lines.
- `results/`: experiment outputs (see F12 and Known Issues).

### On `feat/dashboard`

- `backend/api/main.py` (42 lines changed) and `backend/api/mock_data.py` (136 lines). Described in F13.

---

## Database and Data Flow

There is **no database**. Persistence is files only:

| Data | Written by | Location | Committed? |
|---|---|---|---|
| Taillard instances | `job_shop_lib` (downloaded by the library) | library cache | No (not in repo) |
| C-MAPSS raw data | placed manually by each developer | `data/raw/cmapss/` | No (git-ignored) |
| Engine split | risk pipeline (on its branch) | `data/splits.json` (planned) | Not on any branch yet |
| Trained risk bundles | `train_anomaly.py`, `train_calibration.py` | `models/*.pkl` | No (git-ignored) |
| Training runs | `train.py` | `runs/<hash>/` | No (git-ignored) |
| Evaluation results | `evaluate.py`, `aggregate.py`, `selection.py` | `results/<hash>/` | No (git-ignored) |
| Event logs | `log_document()`, `save_logs()` | `results/<hash>/logs/`, or the `--log` flag | No |
| Mock episode for the dashboard | written by hand | `backend/api/mock_data.py` (feat/dashboard) | Yes, on that branch |
| Risk experiment results | risk scripts | `results/` on `kartik-risk-model` | **Yes** (about 20 MB of CSV; see Known Issues) |

Data flow for a real experiment, as built:

1. The developer places C-MAPSS files in `data/raw/cmapss/` (not in git).
2. The risk pipeline (on its branch) builds the engine split and trains the anomaly and calibration models. These are not yet connected to the simulator.
3. The simulator's `sample_engine` is currently the synthetic stand-in.
4. `train.py` and the evaluation scripts read `configs/protocol.toml` and write to `runs/` and `results/`.
5. The dashboard reads only the mock episode today. Connecting it to `results/<hash>/logs/` is not done.

---

## APIs and External Integrations

### HTTP API (on `feat/dashboard`; `main` has `/health` only)

| Method | Path | Returns | Status |
|---|---|---|---|
| GET | `/health` | `{"status": "ok"}` | Verified |
| GET | `/episodes` and `/api/episodes` | list with one mock entry | Verified (by test) |
| GET | `/episodes/mock` and `/api/episodes/mock` | mock event log | Verified |
| GET | `/episodes/{episode_id}` | 404 for any id except `mock` | Code present |

There is no authentication. CORS allows all origins with credentials enabled (`allow_origins=["*"]`, `allow_credentials=True`). This is acceptable for a local prototype but must be narrowed before any deployment.

### External services and libraries

- **Taillard benchmarks** via `job_shop_lib`. The library downloads them on first use.
- **C-MAPSS** (NASA): downloaded manually. No code fetches it.
- **PyTorch CPU wheel**: installed from the PyTorch index, per `requirements.txt`.
- **GitHub**: the `origin` remote.

No external API, cloud service or third-party web integration is used.

---

## Authentication and Security

- **Authentication**: none. The API has no user model and no protected routes.
- **Authorization**: none.
- **Secrets**: none in the repository. `.gitignore` excludes `.env` and `.env.*`, except `.env.example`.
- **CORS**: permissive (see APIs section).
- **Input handling**: the API accepts only `mock` as an episode id. The Python tools read files the developer names on the command line.
- **Data sensitivity**: all data is public (Taillard, C-MAPSS). The repository holds no personal data.
- **Pickle loading**: the risk layer loads model bundles with `pickle`. Only load bundles you created, because a bundle from an untrusted source can run code.

The user's global CLAUDE.md describes a different project (a habit tracker with Clerk or JWT authentication, PostgreSQL and Vercel). **None of that applies to this repository.** No part of this codebase implements it.

---

## Git Branch-by-Branch Analysis

Every branch was fetched from `origin` in this audit. All four remote branches are accessible; none is inaccessible. There are no stashes and no tags.

| Branch | Tip | Tip author | Tip date | Commits beyond `main` | Merged into `main`? |
|---|---|---|---|---|---|
| `main` (local and `origin/main`) | `f7ffcf5` | kartik-sc | 2026-10-29 | 0 | (it is `main`) |
| `js-sim-taillard` (local and `origin/js-sim-taillard`) | `17e158a` | kartik-sc | 2026-11-11 | 4 | **No** |
| `feat/dashboard` (`origin` only) | `92c9dc4` | kevinmmath | 2026-11-04 | 1 | **No** |
| `kartik-risk-model` (`origin` only) | `b15b3de` | Kartikkaul123 | 2026-10-31 | 2 | **No** |

The local `main` is at `f7ffcf5`, the same commit as `origin/main`. The working tree is on `js-sim-taillard` and has uncommitted changes (listed below).

**Note on the fetch**: this audit ran `git fetch origin`. That added two remote-tracking refs (`origin/feat/dashboard` and `origin/kartik-risk-model`). It changed no local branch, no working-tree file, and no history.

### `js-sim-taillard` (the main development branch)

- Commits beyond `main`: `2900225`, `a152458`, `7d8a6bc`, `17e158a` (see the history section).
- Files changed against `main`: 16 files, about 2,876 lines added.
- Committed tests: 334 passed, 1 xfailed (verified).
- Notable: a notebook checkpoint file is committed (see Known Issues).

### `feat/dashboard`

- Single commit `92c9dc4` by kevinmmath.
- Files changed against `main`: 19 files, about 5,451 lines added, including a 2,519-line lockfile.
- Also changes `shared/schemas/event_log.schema.json` (drops `machine` from the required fields of event items) and rewrites `tests/test_smoke.py`.
- Tests: 2 passed (verified).
- Frontend: not built in this audit.

### `kartik-risk-model`

- Commits: `b8f6cb0` and `b15b3de` (see the history section).
- Files changed against `main`: 121 files, 193,230 lines added. About 190 of those are result CSVs and Markdown reports under `results/`.
- Source: 12 modules in `backend/risk/`, and 6 test files in `tests/`.
- Tests: 79 passed, 187 skipped (verified). The skips occur because `data/raw/cmapss` is absent from the repository. The branch's document reports 114 passing tests; that number presumably needs the C-MAPSS data present. Not reproduced here.

---

## Complete Development and Commit History

Commits in chronological order, with their branch. Dates are commit dates.

| Date | Hash | Branch | Author (handle) | Message | What it changed |
|---|---|---|---|---|---|
| 2026-10-29 | `f7ffcf5` | `main` (base of all branches) | kartik-sc | chore: initial commit, project scaffold | Repository skeleton: `backend/` packages, `shared/schemas/event_log.schema.json` (draft), `frontend/README.md`, `data/README.md`, `pyproject.toml`, `requirements.txt`, `tests/test_smoke.py`, a `/health` API stub |
| 2026-10-29 | `2900225` | `js-sim-taillard` | kartik-sc | feat: added notebook for taillard dataset exploration | `notebooks/01_taillard_exploration.ipynb` (also a checkpoint file, see Known Issues) |
| 2026-10-30 | `b8f6cb0` | `kartik-risk-model` | Kartikkaul123 | feat(risk): Phases 1-3 risk layer - anomaly detection, calibration and experiments | Risk modules, tests, and experiment outputs |
| 2026-10-31 | `b15b3de` | `kartik-risk-model` | Kartikkaul123 | Add RAJS risk detection and calibration pipeline | Further risk pipeline work and result files; the branch tip |
| 2026-11-04 | `a152458` | `js-sim-taillard` | kartik-sc | add baseline tests for spt lpt, create environment file for all instances and a test simulator. | Dispatch rules, Taillard instance loader, `tests/test_simulator.py`, and the first (derived-clock) simulator |
| 2026-11-04 | `92c9dc4` | `feat/dashboard` | kevinmmath | feat(dashboard): add mock episode API and Next.js + D3 scheduling dashboard | Mock API, mock data, the Next.js and D3 frontend, and the schema change |
| 2026-11-11 | `7d8a6bc` | `js-sim-taillard` | kartik-sc | Replace derived-clock simulator with event-queue kernel | Event-queue kernel (`simulator.py`), `events.py`, rewritten dispatch, rewritten simulator tests, requirements (`jsonschema`) |
| 2026-11-11 | `17e158a` | `js-sim-taillard` | kartik-sc | gymnasium env and tests | `gym_env.py`, `observation.py`, `health.py`, health tests, `fakes.py`, `test_gym_env.py` |

Commit `7d8a6bc` also changed `shared/schemas/event_log.schema.json` (drops `machine` from required fields). Commit `17e158a` is the tip of `js-sim-taillard`.

What happened in between, but **is not in any commit**:

- The health layer's milestone 1 and milestone 2 pieces were committed together in `17e158a`. Later health-layer edits are uncommitted.
- The harness, threshold rule, synthetic engine, configs and the three new test files were written after `17e158a`. None is committed.
- The `decision` property and `encode()` method in `gym_env.py`, and `random_instance` in `instance.py`, are uncommitted.
- The pilot run (`results/67b059fc/`, `runs/67b059fc/`) is generated output, git-ignored.

Authors are the GitHub handles recorded in the commits. Git-user identity and e-mail addresses are not reproduced here.

---

## Changes Unique to Individual Branches

### Unique to `js-sim-taillard`

- An event-queue kernel that replaced the earlier derived-clock simulator (the earlier version is in git history at `a152458`).
- The machine-health layer, the observation builder, and the Gymnasium environment (health and env are in `17e158a`).
- The dispatch baselines and the Day 2 gate tests.
- The exploratory notebook.
- Uncommitted: the harness, threshold rule, synthetic engine, configs and tests (see below).

### Unique to `feat/dashboard`

- FastAPI endpoints for episodes, and the mock data module.
- The Next.js frontend, D3 components, and the TypeScript types and parsing for event logs.
- A schema change: `machine` is no longer required on event items.

### Unique to `kartik-risk-model`

- The full risk pipeline: C-MAPSS loader, anomaly models, calibration, diagnostics, feature analysis, RUL error analysis, and experiment runners.
- Conventions for trained artifacts (`models/`, git-ignored).
- Large result CSVs and reports under `results/`.

### Uncommitted in the working tree (on `js-sim-taillard`)

Not in any commit; not on any remote:

- `backend/training/` (protocol, factory, train)
- `backend/evaluation/` (actors, episodes, metrics, selection, evaluate, aggregate, calibrate)
- `backend/baselines/threshold.py`
- `backend/risk/synthetic.py`
- `configs/protocol.toml`, `configs/smoke.toml`, `configs/pilot.toml`
- `tests/test_harness.py`, `tests/test_pipeline.py`, `tests/test_engine_contract.py`
- Modified: `backend/env/gym_env.py`, `backend/env/instance.py`, `README.md`, `requirements.txt`, `.gitignore`
- Outside the repository: `aiml-el/DECISIONS.md` (health and harness decisions appended)
- Generated and git-ignored: `runs/67b059fc/`, `results/67b059fc/` (pilot), `.pytest_cache/`, `__pycache__/`, `.venv/`

**Risk**: this work exists only on the local disk. Committing it is the most urgent housekeeping step.

---

## Merged vs Unmerged Work

| Work | State | Risk if it stays this way |
|---|---|---|
| Simulator kernel and gate | Committed on `js-sim-taillard`, not merged | Low: tested and pushed |
| Health layer, env, observation | Committed (`17e158a`), not merged; later health edits uncommitted | Medium |
| Harness, configs, threshold, synthetic engine | **Uncommitted** | High: exists only locally |
| Risk layer | On `kartik-risk-model`, not merged | High: `main` has no risk code; the scheduler cannot use real risk until merged |
| Dashboard | On `feat/dashboard`, not merged | Medium: `main` API has only `/health` |
| Schema changes | Two different edits on two branches, neither merged | High: see Known Issues |

Nothing has been merged into `main`. It contains only the scaffold commit.

---

## Completed Features

Implemented and verified by tests or by running them in this audit.

| Feature | Evidence |
|---|---|
| Taillard instance loading | Verified |
| Event-driven kernel without failures | Verified: 334 tests pass on the committed tree; Day 2 gate pinned |
| SPT, LPT, FIFO and FCFS dispatch | Verified: pinned makespans |
| Event log writer and CLI | Verified |
| Health layer (wear, breakdowns, maintenance, repair, failure draws) | Verified (working tree) |
| Observation builder with no-leak property | Verified |
| `JobShopEnv` (Gymnasium, masked, both modes), passes `check_env` | Verified |
| Threshold + SPT rule | Verified |
| Protocol loader with sha256 and validation | Verified |
| Training pipeline (MaskablePPO, checkpoints, logs) | Verified: pilot ran end to end |
| Evaluation pipeline (selection, test grid, metrics, aggregation, calibration, event-log export) | Verified: pilot ran end to end; smoke pipeline tested |
| Deliberate-breakage checks on the harness (7 of 7 caught) | Verified this session |
| Risk layer Layer 1 and baseline Layer 2 (fit-split diagnostics) | Reported; partly verified (79 tests pass without data) |
| Dashboard backend API for the mock episode | Verified (2 tests) |

---

## Partially Completed Features

| Feature | What is done | What is missing |
|---|---|---|
| Risk layer | Anomaly models, calibration, diagnostics, experiments (on branch) | Not merged. Test split unused. Calibration problem P1 (IForest near-failure overestimation) unresolved. Problems P2 and P3 (score saturation, joint FD001+FD003 model) open. `sample_engine` not implemented. |
| Dashboard | Mock API, Gantt, heatmap, playback, parsing | Not merged. Frontend not built or tested here. Mock data only. No real episode loading. No frontend tests. |
| Training | Pilot run | Full-budget runs not done. Pilot full PPO selected early checkpoints, so it is undertrained. |
| Threshold + SPT | Rule implemented; tau tuned on synthetic health in the pilot | Tuning on real risk not done. |
| Protocol | Placeholder values in `configs/protocol.toml` | Day 4 freeze not done; hazard, splits, budgets and regime values are provisional. |
| Event log schema | Draft, used by all code | `repair_end` not in the schema; `machine` requirement conflicts between branches. |
| Documentation | README sections, DECISIONS.md entries | `INTERFACES.md` still DRAFT; README not updated on the dashboard and risk branches. |

---

## Experimental and Abandoned Work

### Experimental

- **Pilot training run** (`results/67b059fc/`): medium regime only, 200k steps, 2 seeds, synthetic health. A budget test, not a result.
- **Risk experiments** on `kartik-risk-model`: scoring variants A–D (`ds_iso`, `anchored`, `piecewise`, `quantile_25`), feature rankings, temporal features, and per-engine early warning. Results are reported and kept in `results/`.
- **Notebook**: exploratory, not part of the pipeline.

### Abandoned or superseded

- **Derived-clock simulator**: the first simulator design, replaced by the event-queue kernel (`7d8a6bc`). It is preserved in history at `a152458`. Its design could not express breakdowns that interrupt an operation, or maintenance decisions on idle machines.
- **`fcfs` as "FIFO"**: an earlier naming. Now `fcfs` is only the library's rule for the gate, and `fifo` is the shop-floor rule.
- **Immutable simulator state**: an early design copied state on every dispatch, which was quadratic per episode. Replaced by mutable state with seeded RNG for paired comparisons.
- **Risk-layer branch name**: the risk branch's document calls it `feature/kartik-risk-layer`. The branch is actually `kartik-risk-model`. The name in the document is stale.
- **No-op action in the env**: considered and rejected, because stalling would be an exploit. The lowest-legal-action fallback replaced it.

---

## Known Issues, TODOs, and Limitations

### Critical

1. **The harness and health layer are not committed.** The health-layer edits, `JobShopEnv` changes, harness, configs and tests exist only in the working tree. Action: review and commit on `js-sim-taillard`.
2. **Three branches are unmerged.** `main` contains none of the simulator, risk layer or dashboard. Action: decide the merge order (see Recommendations).
3. **The event-log schema disagrees across branches.**
   - `main`: `machine` is required on every event item. That is wrong for `episode_end`, which has no machine. The code's tests and DECISIONS.md say `machine` is optional.
   - `js-sim-taillard` and `feat/dashboard` both make `machine` optional, through two separate edits to the same line. They agree with each other.
   - Neither branch adds `repair_end`, which the code emits.
   - Action: agree one schema, then merge.
4. **The risk layer does not reach the simulator.** `sample_engine` is not implemented; the synthetic stand-in drives all current results. Any claim about real risk-aware performance is currently unsupported.

### High

5. **Risk Layer 1 and calibration have a known failure.** The reported IForest RUL overestimate at RUL 0–20 is 52 cycles MAE, because its scores compress near failure (problem P1). This matters for maintenance, since it is exactly the region where maintenance matters.
6. **Inconsistent MAE figures in the risk document.** The document's summary table gives calibration-split MAE of 49.0 for OC-SVM and 55.5 for IForest. The branch's calibration summary CSV, `Baseline_P1` row, gives 51.1 for OC-SVM and 49.0 for IForest, averaged across seeds. Either the table, the CSV, or the configuration differs. Check before quoting any figure.
7. **The risk document's test count does not reproduce without data.** 79 tests pass and 187 skip without `data/raw/cmapss`. The document's 114 is unverified here.
8. **The frontend has never been built here.** `npm install`, `next build` and TypeScript checks have not been run on `feat/dashboard`. Whether they pass is unknown.
9. **Dashboard presets are hand-written.** The three policy labels and numbers (`110u`, `116u`, `132u`, `τ=0.70`) in `page.tsx` are not generated by the simulator. The threshold preset's τ = 0.70 is not the tau the harness selects. Shown as results, they would mislead.

### Medium

10. **The risk branch commits about 20 MB of generated results**, for example `results/all_fit_scores.csv` (about 2 MB) and many predictions files of about 0.9 MB each. They are outputs, not source, and make the repository heavy.
11. **A notebook checkpoint is committed** on `js-sim-taillard`: `notebooks/.ipynb_checkpoints/01_taillard_exploration-checkpoint.ipynb`. Remove it and ignore the folder.
12. **A scratch script is committed**: `backend/risk/_phase3_inspect.py` on `kartik-risk-model`.
13. **CORS is wide open** with credentials enabled. Acceptable locally; must change before deployment.
14. **Pickle model bundles** are loaded without integrity checks.
15. **Full-mode PPO is undertrained** at the pilot budget. Selected checkpoints were early, and reward was still improving at 200k steps. Full-mode budgets of about one million steps or more are needed before judging it.
16. **Placeholder protocol values.** Regime hazards, wear rates (`cycles_per_mean_op` 8 / 14 / 22), the degradation-rate range, PPO hyperparameters, budgets and the tau grid were set for the pilot. None is frozen.
17. **Synthetic engine only.** Any result from it describes the scheduler against a synthetic degradation model.
18. **`repair_end` is not in the committed schema**, so logs containing repairs fail the committed schema. An xfail test marks this.
19. **The threshold rule compares with `>`**, while the test helper in `tests/fakes.py` uses `>=`. The kernel tests depend on that boundary and are unchanged. The two rules are intentionally different, so be careful when reading both.
20. **The `main` README is outdated.** It describes the layout but not the health layer, the env or the harness. The harness section exists only in the uncommitted working tree.

### Low

21. The notebook lists all 80 Taillard instances, but only a subset is used.
22. `data/README.md` describes the data layout. `data/splits.json` is planned but not created on any branch.
23. The `results/` folder is git-ignored, so pilot outputs will not be committed by accident.
24. `docs/` and `DECISIONS.md` live in the parent `aiml-el` folder, outside the repository, so they are not versioned with the code.

### Planned, not built

- Connecting the real risk layer to the simulator (implementing `sample_engine`).
- The Day 4 protocol freeze.
- Full training budgets, and final evaluation on the test split.
- Loading real evaluation episodes into the dashboard.
- Updating `INTERFACES.md` and the committed schema for the health layer.
- Dashboard deployment and authentication (not in the current plan).

---

## Important Files and Their Responsibilities

| File | Branch | Responsibility |
|---|---|---|
| `backend/env/simulator.py` | js-sim (health edits uncommitted) | Discrete-event kernel, decisions, health hooks, snapshot, `run_episode` |
| `backend/env/health.py` | js-sim (uncommitted edits) | `HealthModel`, `EngineHealth`, `Regime`, `Trajectory`, failure-cycle inversion |
| `backend/env/gym_env.py` | js-sim | `JobShopEnv`: Gymnasium wrapper, masks, reward, events |
| `backend/env/observation.py` | js-sim | Fixed-size observation builder |
| `backend/env/events.py` | js-sim | Event types and log records |
| `backend/env/instance.py` | js-sim | Instance types, Taillard loader, random instances |
| `backend/baselines/dispatch.py` | js-sim | SPT, LPT, FIFO, FCFS rules and CLI |
| `backend/baselines/threshold.py` | working tree | Threshold + SPT rule |
| `backend/risk/synthetic.py` | working tree | Stand-in engine until the real risk layer is merged |
| `backend/training/protocol.py` | working tree | Protocol file loader and validation |
| `backend/training/factory.py` | working tree | Builds regimes, samplers and envs from a protocol |
| `backend/training/train.py` | working tree | MaskablePPO training CLI |
| `backend/evaluation/selection.py` | working tree | Chooses tau and checkpoints on the tuning split |
| `backend/evaluation/evaluate.py` | working tree | Runs the test grid; writes episodes and logs |
| `backend/evaluation/aggregate.py` | working tree | Summary and paired tables |
| `backend/evaluation/metrics.py` | working tree | Metrics from event logs |
| `configs/protocol.toml` | working tree | Experimental protocol (placeholder values) |
| `shared/schemas/event_log.schema.json` | main (changed on two branches) | Event-log contract |
| `tests/test_simulator.py` | js-sim | Day 2 gate |
| `tests/test_health.py` | js-sim | Health-layer tests and the schema xfail |
| `tests/test_gym_env.py` | js-sim | Env tests (check_env, masks, no-leak) |
| `tests/test_harness.py`, `test_pipeline.py`, `test_engine_contract.py` | working tree | Harness tests |
| `backend/risk/cmapss_pipeline.py` | kartik-risk-model | C-MAPSS loading, split, scaling |
| `backend/risk/anomaly_models.py` | kartik-risk-model | One-Class SVM and Isolation Forest |
| `backend/risk/calibration.py` | kartik-risk-model | Isotonic calibration to risk and RUL |
| `backend/api/main.py` | feat/dashboard | FastAPI app with mock episode endpoints |
| `backend/api/mock_data.py` | feat/dashboard | Hand-written mock episode |
| `frontend/src/app/page.tsx` | feat/dashboard | Dashboard page |
| `frontend/src/components/GanttChart.tsx` | feat/dashboard | D3 schedule chart |
| `frontend/src/components/RiskHeatmap.tsx` | feat/dashboard | D3 risk heatmap |
| `frontend/src/lib/parseEpisode.ts` | feat/dashboard | Event log to intervals and risk samples |

---

## Overall Current Development Status

**Done and verified**
- The scheduling core: instances, the event-driven kernel, dispatch baselines, and the Day 2 gate.
- The health model and observation, then the Gymnasium environment with action masks, passing `check_env` in both modes.
- The training and evaluation pipeline, run end to end on a pilot.
- The risk layer's first two layers and diagnostics, on a separate branch.
- A dashboard prototype that shows a mock episode.

**Done but not committed or merged**
- Health-layer edits, harness, threshold rule, synthetic engine, configs and tests.
- Risk layer (on its branch).
- Dashboard (on its branch).

**Not done**
- Real risk data feeding the simulator.
- A frozen protocol and final budgets.
- Full PPO and baseline training runs, and final test-split results.
- Merging the three branches and agreeing one event-log schema.
- Real episodes in the dashboard.
- Frontend build verification.

**Current state in one sentence**: the simulator, health model, environment and evaluation pipeline work end to end on synthetic health; the real risk layer and the dashboard sit on unmerged branches; and the newest work has not been committed.

---

## Final Summary of Everything That Has Been Built

1. **A discrete-event job-shop simulator** that reproduces SPT, LPT and FIFO makespans on Taillard benchmarks exactly, checked against pinned values and the library.
2. **A machine-health model** with wear during processing, one failure draw per machine lifetime, breakdowns that requeue jobs, preventive maintenance, and repair with full reset, all under a fixed event ordering.
3. **An observation builder** that gives the agent a fixed-size, bounded view and provably excludes future breakdowns.
4. **A Gymnasium environment** with 22 actions, exact action masks, a reward equal to the negative makespan, full and blind modes, and an event stream.
5. **A baseline set**: SPT, LPT, FIFO, FCFS, and a tuned threshold + SPT rule.
6. **A reproducible training pipeline** with a hashed protocol file, paired seeds, checkpointing and logs.
7. **An evaluation pipeline** that tunes on a separate split, tests on held-out instances, computes metrics from event logs, and produces summary and paired statistics with bootstrap intervals and Wilcoxon tests.
8. **A risk layer** for C-MAPSS on a separate branch: data pipeline, two anomaly detectors, isotonic calibration, and extensive diagnostics and experiments. Layer 1 separates healthy from degraded engines well on the fit split. Near-failure RUL is the open problem.
9. **A dashboard prototype** with a FastAPI mock API and a Next.js + D3 interface showing a Gantt chart, a risk heatmap, and playback controls.
10. **Tests**: 385 passing and 1 expected failure on the working tree; 334 and 1 on the committed js-sim tree; 79 passing and 187 skipped on the risk branch without its data; 2 on the dashboard branch.

### Recommendations, in order

1. Review and commit the working-tree work on `js-sim-taillard` in logical commits (health-layer edits; env changes; harness; configs; tests). Do not include `results/` or `runs/`.
2. Remove the committed notebook checkpoint and add `.ipynb_checkpoints/` to `.gitignore`.
3. Agree one event-log schema (`machine` optional; add `repair_end`) and update all branches to it.
4. Merge the risk branch and the dashboard branch into `main` once the schema is agreed. Remove the `_phase3_inspect.py` scratch script.
5. Check the risk document's MAE figures against the summary CSV.
6. Build and test the frontend on `feat/dashboard` (`npm ci`, `npm run build`).
7. Implement `sample_engine` on top of the merged risk layer and rerun the pilot with it.
