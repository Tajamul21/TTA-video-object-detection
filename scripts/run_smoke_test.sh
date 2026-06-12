#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./XS-VID}"
OUT_DIR="${2:-./runs/qwen3vl_xsvid_smoke}"

python tools/inspect_dataset.py --data-root "${DATA_ROOT}" --split test --max-images 50
python tools/infer_zero_shot.py \
  --data-root "${DATA_ROOT}" \
  --split test \
  --output-dir "${OUT_DIR}" \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --max-images 50 \
  --tile-size 0 \
  --evaluate \
  --visualize-samples 5
