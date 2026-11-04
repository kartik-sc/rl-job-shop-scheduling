# Frontend (RAJS Dashboard)

Next.js (TypeScript) + D3 dashboard consuming episode event logs matching `shared/schemas/event_log.schema.json`.

## Features

- **D3 Schedule Gantt Chart**: Renders operations per machine, preventive maintenance blocks (`t_pm = 8`), and breakdown + repair blocks (`t_repair = 40`, followed by operation restart).
- **Synchronized D3 Risk Heatmap**: Per-machine calibrated C-MAPSS anomaly risk trajectory (`[0, 1]`) with live health band classification (`Healthy < 0.30`, `Degrading 0.30–0.70`, `Critical > 0.70`).
- **Interactive Scrubber & Playback**: Play/pause, step forward/backward, playback speed (`0.5×`, `1×`, `2×`, `4×`), milestone bookmarks, and click-to-scrub on both charts.

## Run Locally

1. Start the FastAPI backend from the repo root:
   ```bash
   uvicorn backend.api.main:app --reload --port 8000
   ```
2. Install and start the Next.js frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
3. Open [http://localhost:3000](http://localhost:3000).
