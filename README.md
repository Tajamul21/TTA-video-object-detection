# Qwen3-VL zero-shot evaluation on XS-VID

This repository runs **zero-shot object detection** with `Qwen/Qwen3-VL-8B-Instruct` on the XS-VID test split and reports detection metrics. It does not train or fine-tune Qwen3-VL.

The code supports the expected XS-VID layout:

```text
XS-VID/
  annotations/
    test.json
    train.json
    yolo/
      <video_or_sequence_id>/000000.txt
  images/
    <video_or_sequence_id>/000000.jpg
```

COCO `test.json` is preferred because it contains category names and image ids. YOLO `.txt` labels are also supported, but then you must pass the exact class names with `--class-names` unless you only want placeholder `class_0`, `class_1`, etc.

## What the code produces

For every test image, the inference script prompts Qwen3-VL to return JSON detections with class, box, and score. It writes:

```text
runs/qwen3vl_xsvid_test/
  predictions_coco.json        # COCO result JSON: image_id, category_id, bbox, score
  predictions_verbose.jsonl    # readable per-detection output
  raw_outputs.jsonl            # raw Qwen replies per image or tile
  metrics_summary.json         # mAP/AP50/AP75 and XS-VID size buckets
  metrics_by_class.csv
  metrics_by_size.csv
  visualizations/              # optional prediction + GT drawings
  cache/                       # per-image cache for resume
```

The metric implementation reports:

- `mAP_50_95`, `AP50`, `AP75`
- XS-VID tiny-object buckets: `0-12`, `12-20`, `20-32`
- COCO-style size buckets: `small`, `medium`, `large`
- per-class and per-size CSV files

All AP values in the output JSON and CSV are percentages from 0 to 100.

## 1. Create environment

Use a recent CUDA PyTorch build that matches your driver. Example:

```bash
cd qwen3vl_xsvid_zeroshot
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your cluster already has PyTorch installed, install the remaining packages after activating that environment:

```bash
pip install accelerate pillow numpy tqdm pandas pycocotools huggingface_hub
pip install git+https://github.com/huggingface/transformers
```

## 2. Download XS-VID from scratch

Hugging Face download:

```bash
bash scripts/download_xsvid_hf.sh ./XS-VID
```

ModelScope download, useful in China:

```bash
bash scripts/download_xsvid_modelscope.sh ./XS-VID
```

If you already downloaded XS-VID, skip this step and pass your existing path to `--data-root`.

## 3. Inspect the dataset

```bash
python tools/inspect_dataset.py --data-root ./XS-VID --split test
```

Check that it prints the correct number of test images, ground-truth boxes, and category names. If it falls back to YOLO and the categories are `class_0`, `class_1`, etc., run again with a class-name file:

```bash
python tools/inspect_dataset.py \
  --data-root ./XS-VID \
  --annotation-format yolo \
  --class-names configs/my_xsvid_class_names.txt
```

## 4. Smoke test on a few images

Run this before launching the full test set:

```bash
bash scripts/run_smoke_test.sh ./XS-VID ./runs/qwen3vl_xsvid_smoke
```

Or manually:

```bash
python tools/infer_zero_shot.py \
  --data-root ./XS-VID \
  --split test \
  --output-dir ./runs/qwen3vl_xsvid_smoke \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --max-images 5 \
  --tile-size 0 \
  --evaluate \
  --visualize-samples 5
```

## 5. Full test run, single GPU or automatic device map

For XS-VID, tiling is recommended because objects are very small. This runs zero-shot inference on the test split with 1024-pixel tiles and 20 percent overlap:

```bash
python tools/infer_zero_shot.py \
  --data-root ./XS-VID \
  --split test \
  --output-dir ./runs/qwen3vl_xsvid_test \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --attn auto \
  --device-map auto \
  --tile-size 1024 \
  --tile-overlap 0.20 \
  --nms-iou 0.50 \
  --evaluate \
  --visualize-samples 50 \
  --resume
```

Equivalent helper script:

```bash
bash scripts/run_xsvid_test_single_gpu.sh ./XS-VID ./runs/qwen3vl_xsvid_test
```

## 6. Full test run, multiple GPUs

This starts one worker process per GPU. Each worker loads one copy of Qwen3-VL and processes a shard of the test images. Use GPU ids from your machine, for example `0,4,6,7`:

```bash
python tools/run_parallel.py \
  --data-root ./XS-VID \
  --split test \
  --output-dir ./runs/qwen3vl_xsvid_test_parallel \
  --gpu-ids 0,4,6,7 \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --attn auto \
  --device-map single \
  --tile-size 1024 \
  --tile-overlap 0.20 \
  --nms-iou 0.50 \
  --evaluate \
  --resume
```

Equivalent helper script:

```bash
bash scripts/run_xsvid_test_multi_gpu.sh ./XS-VID ./runs/qwen3vl_xsvid_test_parallel 0,4,6,7
```

## 7. Evaluate existing predictions only

If you already have a COCO result JSON:

```bash
python tools/evaluate_predictions.py \
  --data-root ./XS-VID \
  --split test \
  --pred-json ./runs/qwen3vl_xsvid_test/predictions_coco.json \
  --output-dir ./runs/qwen3vl_xsvid_test
```

## Important options

### Tiling

- `--tile-size 0`: full image only.
- `--tile-size 1024 --tile-overlap 0.20`: tiled inference, recommended for XS-VID.
- `--include-full-image`: adds one full-image prompt in addition to tiles. This costs more time but can recover large context.

### Resume

Use `--resume` to reuse per-image JSON files in `output_dir/cache`. This is useful if a long cluster job stops.

### YOLO-only annotations

If there is no COCO `test.json`, pass exact names:

```bash
python tools/infer_zero_shot.py \
  --data-root ./XS-VID \
  --annotation-format yolo \
  --class-names configs/my_xsvid_class_names.txt \
  --output-dir ./runs/qwen3vl_xsvid_yolo \
  --evaluate
```

The class-name file must have one class per line. Line 0 is YOLO class id 0, line 1 is YOLO class id 1, and so on.

### Run a subset

```bash
python tools/infer_zero_shot.py \
  --data-root ./XS-VID \
  --output-dir ./runs/subset \
  --start-index 0 \
  --end-index 100 \
  --evaluate
```

Or use an image-id file:

```bash
python tools/infer_zero_shot.py \
  --data-root ./XS-VID \
  --output-dir ./runs/subset_ids \
  --image-ids-file image_ids.txt \
  --evaluate
```

## Notes on zero-shot results

This is a pure prompt-based zero-shot detector. Qwen3-VL was not trained here on XS-VID labels. The model may miss extremely small objects, hallucinate boxes, or output class names that need alias matching. The code saves raw outputs so you can audit parsing problems.

For leaderboard-style reporting, keep the same prompt, tiling settings, NMS settings, and metric code across all runs.

## Troubleshooting

### `Qwen3VLForConditionalGeneration` import error

Install a newer Transformers build:

```bash
pip install -U git+https://github.com/huggingface/transformers
```

### CUDA out of memory

Try one of these:

```bash
# Use SDPA instead of flash attention.
--attn sdpa

# Do not include the full image when tiling.
# Remove --include-full-image

# Use a smaller tile size.
--tile-size 768
```

### Metrics are all zero

Common causes:

1. Category names in the prompt do not match the dataset labels. Prefer COCO `test.json` or pass an exact `--class-names` file for YOLO.
2. The model returns invalid JSON. Check `raw_outputs.jsonl`.
3. The boxes are too coarse for tiny objects. Try tiled inference.

## Repository layout

```text
configs/
  default_class_aliases.json
  example_yolo_class_names.txt
scripts/
  download_xsvid_hf.sh
  download_xsvid_modelscope.sh
  run_smoke_test.sh
  run_xsvid_test_single_gpu.sh
  run_xsvid_test_multi_gpu.sh
tools/
  inspect_dataset.py
  infer_zero_shot.py
  evaluate_predictions.py
  run_parallel.py
xsvid_qwen3vl/
  dataset.py
  metrics.py
  nms.py
  parsing.py
  prompts.py
  qwen.py
  tiling.py
  visualize.py
```
