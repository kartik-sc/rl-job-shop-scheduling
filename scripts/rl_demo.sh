#!/usr/bin/env bash
# Live demo, ~90 s: writes Kaul's risk model and the baselines to TensorBoard, then trains
# full and blind PPO side by side on configs/demo.toml. Start TensorBoard first:
#   tensorboard --logdir runs/demo
# Run from the repo root in WSL with the venv active. Each run starts from a clean runs/demo.
set -euo pipefail

OUT=runs/demo
rm -rf "$OUT"
python -m backend.evaluation.board --out "$OUT"

python -m backend.training.train --protocol configs/demo.toml --mode blind --regime high --train-seed 0 \
  --runs "$OUT" > "$OUT/blind.log" 2>&1 &
python -m backend.training.train --protocol configs/demo.toml --mode full --regime high --train-seed 0 \
  --runs "$OUT"
wait
