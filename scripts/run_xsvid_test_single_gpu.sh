#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./XS-VID}"
OUT_DIR="${2:-./runs/qwen3vl_xsvid_test}"

python tools/infer_zero_shot.py \
  --data-root "${DATA_ROOT}" \
  --split test \
  --output-dir "${OUT_DIR}" \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --attn auto \
  --device-map auto \
  --tile-size 1024 \
  --tile-overlap 0.20 \
  --nms-iou 0.50 \
  --evaluate \
  --visualize-samples 50 \
  --resume
