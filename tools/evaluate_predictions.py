#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xsvid_qwen3vl.dataset import load_dataset_auto, load_image_ids_file
from xsvid_qwen3vl.metrics import evaluate_map, print_summary, write_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate COCO-format predictions on XS-VID.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--pred-json", required=True, help="COCO result JSON produced by infer_zero_shot.py")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--annotation-format", default="auto", choices=["auto", "coco", "yolo"])
    p.add_argument("--image-root", default=None)
    p.add_argument("--coco-json", default=None)
    p.add_argument("--yolo-label-root", default=None)
    p.add_argument("--class-names", default="auto")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--image-ids-file", default=None)
    p.add_argument("--max-dets-per-image", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image_ids = load_image_ids_file(args.image_ids_file)
    ds = load_dataset_auto(
        data_root=args.data_root,
        split=args.split,
        annotation_format=args.annotation_format,
        image_root=args.image_root,
        coco_json=args.coco_json,
        yolo_label_root=args.yolo_label_root,
        class_names=args.class_names,
        max_images=args.max_images,
        image_ids=image_ids,
    )
    pred_path = Path(args.pred_json).expanduser().resolve()
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    result = evaluate_map(ds.ground_truths, predictions, ds.categories, max_dets_per_image=args.max_dets_per_image)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else pred_path.parent
    write_metrics(result, output_dir)
    print_summary(result)
    print(f"\nSaved metrics: {output_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
