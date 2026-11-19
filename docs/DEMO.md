# Demo Runbook (~10 min)

Python runs in WSL (`~/venvs/rajs`). The frontend runs in Windows PowerShell, because its `node_modules` has the Windows build. WSL2 forwards `localhost`, so everything opens in the Windows browser.

## Before the evaluator arrives

Open three terminals:

```bash
# T1 (WSL): API for the dashboard
cd /mnt/c/Users/kartik/aiml-el/rl-job-shop-scheduling && source ~/venvs/rajs/bin/activate
uvicorn backend.api.main:app --port 8000

# T2 (WSL): TensorBoard
cd /mnt/c/Users/kartik/aiml-el/rl-job-shop-scheduling && source ~/venvs/rajs/bin/activate
mkdir -p runs/demo && tensorboard --logdir runs/demo
```

```powershell
# T3 (PowerShell): dashboard
cd C:\Users\kartik\aiml-el\rl-job-shop-scheduling\frontend
npm run dev
```

Browser tabs: http://localhost:3000 (dashboard) and http://localhost:6006 (TensorBoard).

## 1. Dashboard (3 min): http://localhost:3000

- **Factory Floor**: machines run jobs. Risk colours rise; maintenance and breakdowns animate.
- **Episode dropdown (header)**: pick *SPT*, then *Threshold+SPT*. On the same failures (ta06, medium, seed 2000), SPT has 8 breakdowns and makespan 2320. Threshold+SPT has 0 breakdowns, 14 PMs and makespan 1609.
- **Gantt + risk heatmap**: scrub to a breakdown and show the risk was already high before it.
- Say it upfront: *Full PPO*, *RL Brain* and *Training Arena* are labelled MOCK. Real PPO logs come after full training.

## 2. Kaul's risk model + baselines (2 min): TensorBoard

In a fourth WSL terminal (venv active, repo root):

```bash
bash scripts/rl_demo.sh
```

The script first writes Kaul's results and the baselines (instant), then starts training. Refresh TensorBoard:

- **Text → report**: Kaul's metrics table (near-failure RUL MAE 3.7 / 12.0, Spearman 0.76) and the baseline table.
- **Images**: risk vs true remaining life (risk climbs to ~1 as the engine nears failure) and baseline gap / breakdowns by regime.
- **Scalars → `kaul_risk/*`**: risk per cycle for 4 real engines.
- **Scalars → `baselines/*`**: x-axis 0, 1, 2 = low, medium, high failure regime. Threshold+SPT stays lowest as failures get worse.

## 3. Live RL (≤ 2 min): TensorBoard Scalars

The script trains **full** (sees risk, can maintain) and **blind** PPO side by side for 150k steps (~75-90 s). Its terminal prints the PPO table every rollout. Set TensorBoard auto-reload (gear icon, 30 s) or press refresh.

| Chart | What to point at |
|---|---|
| `rollout/ep_rew_mean` | = −makespan / mean operation length. Full rises from ~−18.6 to ~−16; blind stays near −21. |
| `episode/breakdowns_per_100_ops` | full ~1.8 vs blind ~6. Maintaining on risk avoids breakdowns. |
| `episode/maintenances_per_100_ops` | full only (blind cannot maintain). |
| `train/approx_kl`, `train/entropy_loss` | stable updates; the policy is becoming more decisive. |

To rerun: the script wipes `runs/demo` itself.

## If something breaks

- Port in use: stop the old process, or use `--port 8001` and set `NEXT_PUBLIC_API_URL=http://localhost:8001` for the frontend.
- No real episodes in the dashboard: run `python -m backend.evaluation.demo` (regenerates `results/demo/`).
- TensorBoard is empty: check that `--logdir` is `runs/demo` and that the script printed `wrote runs/demo`.

What was done and why: see [PROGRESS_REPORT.md](PROGRESS_REPORT.md).
