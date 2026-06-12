#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./XS-VID}"
OUT_DIR="${2:-./runs/qwen3vl_xsvid_test_parallel}"
GPU_IDS="${3:-0,1,2,3}"

python tools/run_parallel.py \
  --data-root "${DATA_ROOT}" \
  --split test \
  --output-dir "${OUT_DIR}" \
  --gpu-ids "${GPU_IDS}" \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --attn auto \
  --device-map single \
  --tile-size 1024 \
  --tile-overlap 0.20 \
  --nms-iou 0.50 \
  --evaluate \
  --resume
