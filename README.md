# RAJS: Risk-Aware Job-Shop Scheduling

Deep RL job-shop scheduler that reads machine-health risk scores and decides when to schedule an operation or take a machine offline for preventive maintenance. AIML EL 2026-27, RV College of Engineering.

Compared policies: maintenance-blind PPO, threshold + SPT, and full risk-aware PPO.

## Layout

```
backend/
  env/          simulator and Gymnasium environment
  risk/         Layer 1 anomaly detection and calibration (C-MAPSS)
  baselines/    SPT, LPT, FIFO, threshold + SPT
  evaluation/   multi-seed evaluation and statistics
  api/          FastAPI service for the dashboard
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

## Conventions

- Typed Python, small pure functions, logic in modules rather than route handlers.
- No data, models or run outputs in git (see `.gitignore`).
- Experimental protocol freezes on Day 4: hazard grid, threshold search, splits, metrics.
