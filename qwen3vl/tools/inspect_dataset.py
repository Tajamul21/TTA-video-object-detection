#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xsvid_qwen3vl.dataset import load_dataset_auto, load_image_ids_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect XS-VID annotations and categories.")
    p.add_argument("--data-root", required=True, help="Path to XS-VID root folder.")
    p.add_argument("--split", default="test", help="Split name, usually test.")
    p.add_argument("--annotation-format", default="auto", choices=["auto", "coco", "yolo"])
    p.add_argument("--image-root", default=None)
    p.add_argument("--coco-json", default=None)
    p.add_argument("--yolo-label-root", default=None)
    p.add_argument("--class-names", default="auto", help="For YOLO-only labels: comma list or text file.")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--image-ids-file", default=None)
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
    print("Dataset inspection")
    print("------------------")
    print(f"data_root: {ds.data_root}")
    print(f"split: {ds.split}")
    print(f"annotation_format: {ds.annotation_format}")
    print(f"image_root: {ds.image_root}")
    if ds.coco_json:
        print(f"coco_json: {ds.coco_json}")
    if ds.yolo_label_root:
        print(f"yolo_label_root: {ds.yolo_label_root}")
    print(f"num_images: {len(ds.images)}")
    print(f"num_ground_truths: {len(ds.ground_truths)}")
    print("categories:")
    for cat in ds.categories:
        n = sum(1 for gt in ds.ground_truths if int(gt.category_id) == int(cat.id))
        print(f"  {cat.id}: {cat.name} ({n} gt)")
    if ds.warnings:
        print("warnings:")
        for w in ds.warnings[:20]:
            print(f"  - {w}")
        if len(ds.warnings) > 20:
            print(f"  ... {len(ds.warnings) - 20} more")


if __name__ == "__main__":
    main()
