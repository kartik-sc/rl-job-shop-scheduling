# RAJS: Risk-Aware Job-Shop Scheduling

Deep RL job-shop scheduler that reads machine-health risk scores and decides when to schedule an operation or take a machine offline for preventive maintenance. AIML EL 2026-27, RV College of Engineering.

Compared policies: maintenance-blind PPO, threshold + SPT, and full risk-aware PPO.

## Layout

```
backend/
  env/          simulator and Gymnasium environment
  risk/         Layer 1 anomaly detection and calibration (C-MAPSS)
  baselines/    SPT, LPT, FIFO, threshold + SPT
  training/     protocol loading, env factory, PPO training
  evaluation/   multi-seed evaluation and statistics
  api/          FastAPI service for the dashboard
configs/        experimental protocol (TOML)
frontend/       Next.js + D3 dashboard
shared/schemas/ event-log schema shared by backend and frontend
data/           local datasets (git-ignored) and committed splits
tests/
```

## Setup

Python 3.11+.

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest
```

Run the API:

```
uvicorn backend.api.main:app --reload
```

## Simulator

`backend/env/simulator.py` is a discrete-event job-shop kernel: an integer clock, a heap of timed events and a queue per machine. At each timestamp every completion is applied first, then each idle machine with waiting jobs decides, in ascending machine id. The caller answers each decision, so a dispatching rule, the Gymnasium wrapper and PPO all drive the same loop:

```python
from backend.env.instance import from_benchmark
from backend.env.simulator import Simulation
from backend.baselines import dispatch

simulation = Simulation(from_benchmark("ta01"))
while (decision := simulation.next_decision()) is not None:
    simulation.apply(dispatch.shortest_processing_time(decision))
print(simulation.makespan)  # 1462
```

Every run produces an event log (`op_start`, `op_end`, `episode_end`) that follows `shared/schemas/event_log.schema.json`.

Dispatching rules, from the command line:

```
python -m backend.baselines.dispatch ta01 spt
python -m backend.baselines.dispatch ta01 fifo --log ta01_fifo.json --regime low
```

| Rule | Picks | Use |
|---|---|---|
| `spt` / `lpt` | shortest / longest next operation | baselines |
| `fifo` | job that has waited longest in this machine's queue | baseline reported as FIFO |
| `fcfs` | job whose next operation is earliest in its own route | `job_shop_lib`'s rule of that name, kept only for the gate |

Ties break on the lowest job id. `tests/test_simulator.py` is the Day 2 gate: makespans pinned on ta01, ta11, ta21, ta31 and ta41, exact equality with `job_shop_lib` on ta01-ta10 plus those four, and an operation-by-operation replay of the library's schedules.

Run code in WSL: Windows Smart App Control blocks compiled extensions in Windows Python.

## Training and evaluation

The experiment is defined by one file, `configs/protocol.toml`: regimes, per-machine degradation rates, the instance split, seeds, PPO hyperparameters, step budgets and the tau grid. Every run directory and every result row records the file's sha256, and results from different hashes are never combined. `configs/smoke.toml` has the same shape with tiny budgets.

Every policy, trained or not, acts through `JobShopEnv.reset(seed=...)`, so all of them meet the same instance, rates and failures for a given `(instance, regime, seed)`.

| Split | Instances | Health seeds | Engine pool | Used for |
|---|---|---|---|---|
| train | training Taillard + random instances | fresh each episode | fit | PPO learning |
| tune | training Taillard | `seeds.tune` | fit | choosing tau and PPO checkpoints |
| test | held-out Taillard 15x15 and 20x15 | `seeds.test` | test | every reported number |

```bash
P=configs/protocol.toml
python -m backend.evaluation.calibrate --protocol $P --workers 16            # regime statistics under blind SPT
python -m backend.training.train --protocol $P --mode full --regime high --train-seed 0   # one per mode x regime x seed
python -m backend.evaluation.selection --protocol $P --workers 16            # tau and best checkpoint per run
python -m backend.evaluation.evaluate --protocol $P --workers 16 --save-logs
python -m backend.evaluation.aggregate --protocol $P
```

Outputs:
- `runs/<hash>/<mode>_<regime>_s<seed>/`: `config.json`, `progress.csv`, `monitor.csv`, checkpoints and `final.zip`.
- `results/<hash>/`:
  - `selection.json` and `tune/`
  - `episodes/` (one CSV per contender and regime; reruns skip finished files)
  - `reference.csv` (best-known makespans and no-failure rules)
  - `summary.csv` and `paired.csv`
  - `logs/` (dashboard event logs)

Metrics come from the event log alone (`backend/evaluation/metrics.py`):

| Metric | Definition |
|---|---|
| gap | makespan / best-known makespan - 1 |
| breakdown rate | breakdowns per 100 operations |
| maintenance share | maintenance downtime / (machines x makespan) |

Tables average seeds within each instance first and give 95% bootstrap intervals over instances. Paired comparisons (`full_ppo` vs `blind_ppo`, `full_ppo` vs `threshold_spt`) add a Wilcoxon signed-rank test over instances.

Health is currently driven by `backend/risk/synthetic.py`. A real engine plugs in by registering its `sample_engine` in `backend/training/factory.py` (`ENGINES`), then setting `engine` in the protocol. `tests/test_engine_contract.py` runs against every registered engine.

## Conventions

- Typed Python, small pure functions, logic in modules rather than route handlers.
- No data, models or run outputs in git (see `.gitignore`).
- Experimental protocol freezes on Day 4: hazard grid, threshold search, splits, metrics.
