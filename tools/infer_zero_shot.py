#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xsvid_qwen3vl.dataset import ImageRecord, XSVidDataset, load_dataset_auto, load_image_ids_file, write_coco_results
from xsvid_qwen3vl.metrics import evaluate_map, print_summary, write_metrics
from xsvid_qwen3vl.nms import per_class_nms
from xsvid_qwen3vl.parsing import load_aliases, normalize_detections
from xsvid_qwen3vl.prompts import build_detection_prompt
from xsvid_qwen3vl.qwen import Qwen3VLDetector
from xsvid_qwen3vl.tiling import Tile, crop_tile, detection_to_global, make_tiles
from xsvid_qwen3vl.visualize import draw_predictions


def parse_args() -> argparse.Namespace:
    default_aliases = REPO_ROOT / "configs" / "default_class_aliases.json"
    p = argparse.ArgumentParser(description="Zero-shot Qwen3-VL inference on XS-VID.")

    # Dataset.
    p.add_argument("--data-root", required=True, help="Path to XS-VID root folder.")
    p.add_argument("--split", default="test")
    p.add_argument("--annotation-format", default="auto", choices=["auto", "coco", "yolo"])
    p.add_argument("--image-root", default=None)
    p.add_argument("--coco-json", default=None)
    p.add_argument("--yolo-label-root", default=None)
    p.add_argument("--class-names", default="auto", help="For YOLO-only labels: comma list or text file. COCO categories are automatic.")
    p.add_argument("--class-aliases", default=str(default_aliases), help="JSON alias map for parsing VLM class names.")
    p.add_argument("--image-ids-file", default=None, help="Optional newline file of image ids to process.")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--start-index", type=int, default=None)
    p.add_argument("--end-index", type=int, default=None)

    # Model.
    p.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--attn", default="auto", help="auto, sdpa, flash_attention_2, eager, or none.")
    p.add_argument("--device-map", default="auto", help="auto, single, cpu, balanced, or any HF device_map string.")
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--allow-tf32", action="store_true")

    # Inference behavior.
    p.add_argument("--tile-size", type=int, default=0, help="0 disables tiling. For XS-VID try 1024.")
    p.add_argument("--tile-overlap", type=float, default=0.20)
    p.add_argument("--include-full-image", action="store_true", help="Also run one full-image prompt in addition to tiles.")
    p.add_argument("--nms-iou", type=float, default=0.50)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--max-detections-per-image", type=int, default=300)

    # Output.
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume", action="store_true", help="Reuse per-image cache files if present.")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--save-raw", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--visualize-samples", type=int, default=0)
    return p.parse_args()


def safe_name(value: object) -> str:
    s = str(value)
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s[:180] if len(s) > 180 else s


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def subset_by_index(ds: XSVidDataset, start: int | None, end: int | None) -> XSVidDataset:
    if start is None and end is None:
        return ds
    images = ds.images[slice(start, end)]
    keep = {str(img.id) for img in images}
    ds.images = images
    ds.ground_truths = [gt for gt in ds.ground_truths if str(gt.image_id) in keep]
    return ds


def get_tiles(width: int, height: int, tile_size: int, overlap: float, include_full: bool) -> List[Tile]:
    tiles = make_tiles(width, height, tile_size=tile_size, overlap=overlap)
    if include_full and tile_size > 0:
        full = Tile(index=-1, x0=0, y0=0, x1=width, y1=height)
        if not (len(tiles) == 1 and tiles[0].x0 == 0 and tiles[0].y0 == 0 and tiles[0].x1 == width and tiles[0].y1 == height):
            tiles = [full] + tiles
    return tiles


def run_one_image(
    runner: Qwen3VLDetector,
    img: ImageRecord,
    prompt: str,
    ds: XSVidDataset,
    aliases: Dict[str, List[str]],
    args: argparse.Namespace,
) -> dict:
    with Image.open(img.path).convert("RGB") as image:
        width, height = image.size
        tiles = get_tiles(width, height, args.tile_size, args.tile_overlap, args.include_full_image)
        all_dets: List[dict] = []
        raw_records: List[dict] = []
        for tile in tiles:
            crop = crop_tile(image, tile)
            t0 = time.time()
            raw_text = runner.generate(crop, prompt)
            elapsed = time.time() - t0
            dets, parsed_json = normalize_detections(
                raw_text=raw_text,
                categories=ds.categories,
                image_width=crop.size[0],
                image_height=crop.size[1],
                aliases=aliases,
                min_score=args.min_score,
            )
            global_dets = [detection_to_global(det, tile) for det in dets]
            all_dets.extend(global_dets)
            if args.save_raw:
                raw_records.append(
                    {
                        "image_id": img.id,
                        "file_name": img.file_name,
                        "tile_index": tile.index,
                        "tile_xyxy": [tile.x0, tile.y0, tile.x1, tile.y1],
                        "elapsed_sec": round(elapsed, 4),
                        "raw_text": raw_text,
                        "parsed_json": parsed_json,
                        "num_detections_after_parse": len(global_dets),
                    }
                )
            crop.close()

    all_dets = per_class_nms(all_dets, iou_threshold=args.nms_iou, max_detections=args.max_detections_per_image)
    predictions = []
    verbose = []
    cats = ds.categories_by_id
    for det in all_dets:
        pred = {
            "image_id": img.id,
            "category_id": int(det["category_id"]),
            "bbox": [float(x) for x in det["bbox"]],
            "score": float(det.get("score", 0.0)),
        }
        predictions.append(pred)
        verbose.append(
            {
                **pred,
                "file_name": img.file_name,
                "image_path": str(img.path),
                "category_name": cats[int(det["category_id"])].name if int(det["category_id"]) in cats else str(det["category_id"]),
                "bbox_xyxy": [float(x) for x in det.get("bbox_xyxy", [])],
                "tile_index": det.get("tile_index"),
                "tile_xyxy": det.get("tile_xyxy"),
            }
        )
    return {
        "image_id": img.id,
        "file_name": img.file_name,
        "image_path": str(img.path),
        "predictions": predictions,
        "verbose_predictions": verbose,
        "raw_records": raw_records,
    }


def save_partial(output_dir: Path, predictions: List[dict], verbose_rows: List[dict]) -> None:
    write_coco_results(predictions, output_dir / "predictions_coco.json")
    with (output_dir / "predictions_verbose.jsonl").open("w", encoding="utf-8") as f:
        for row in verbose_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def visualize_samples(ds: XSVidDataset, predictions: List[dict], output_dir: Path, n: int) -> None:
    if n <= 0:
        return
    pred_by_img: Dict[str, List[dict]] = defaultdict(list)
    for p in predictions:
        pred_by_img[str(p["image_id"])].append(p)
    gt_by_img: Dict[str, List[object]] = defaultdict(list)
    for gt in ds.ground_truths:
        gt_by_img[str(gt.image_id)].append(gt)
    vis_dir = output_dir / "visualizations"
    count = 0
    for img in ds.images:
        preds = pred_by_img.get(str(img.id), [])
        if not preds and count >= n:
            continue
        out_path = vis_dir / f"{count:05d}_{safe_name(img.id)}.jpg"
        draw_predictions(img.path, preds, ds.categories, out_path, gt_by_img.get(str(img.id), []))
        count += 1
        if count >= n:
            break


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

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
    ds = subset_by_index(ds, args.start_index, args.end_index)
    if not ds.categories:
        raise RuntimeError("No categories were found. Use COCO test.json or pass --class-names for YOLO labels.")
    if not ds.images:
        raise RuntimeError("No images were found for this run.")

    aliases = load_aliases(args.class_aliases) if args.class_aliases else {}
    prompt = build_detection_prompt(ds.categories)
    write_json(
        output_dir / "run_config.json",
        {
            "args": vars(args),
            "dataset": {
                "data_root": str(ds.data_root),
                "split": ds.split,
                "annotation_format": ds.annotation_format,
                "image_root": str(ds.image_root),
                "coco_json": str(ds.coco_json) if ds.coco_json else None,
                "num_images": len(ds.images),
                "num_ground_truths": len(ds.ground_truths),
                "categories": [{"id": c.id, "name": c.name} for c in ds.categories],
                "warnings": ds.warnings,
            },
            "prompt": prompt,
        },
    )

    print(f"Images: {len(ds.images)}")
    print(f"GT boxes: {len(ds.ground_truths)}")
    print("Categories: " + ", ".join([f"{c.id}:{c.name}" for c in ds.categories]))
    print(f"Output: {output_dir}")

    runner = Qwen3VLDetector(
        model_id=args.model_id,
        attn=args.attn,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        allow_tf32=args.allow_tf32,
    )

    all_predictions: List[dict] = []
    verbose_rows: List[dict] = []
    raw_rows: List[dict] = []

    for idx, img in enumerate(tqdm(ds.images, desc="Qwen3-VL zero-shot"), start=1):
        cache_file = cache_dir / f"{safe_name(img.id)}.json"
        if args.resume and cache_file.exists():
            record = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            record = run_one_image(runner, img, prompt, ds, aliases, args)
            write_json(cache_file, record)

        all_predictions.extend(record.get("predictions", []))
        verbose_rows.extend(record.get("verbose_predictions", []))
        raw_rows.extend(record.get("raw_records", []))

        if args.save_every > 0 and idx % args.save_every == 0:
            save_partial(output_dir, all_predictions, verbose_rows)

    save_partial(output_dir, all_predictions, verbose_rows)
    if args.save_raw:
        with (output_dir / "raw_outputs.jsonl").open("w", encoding="utf-8") as f:
            for row in raw_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.evaluate:
        result = evaluate_map(ds.ground_truths, all_predictions, ds.categories, max_dets_per_image=100)
        write_metrics(result, output_dir)
        print_summary(result)

    visualize_samples(ds, all_predictions, output_dir, args.visualize_samples)
    print(f"\nSaved predictions: {output_dir / 'predictions_coco.json'}")
    if args.evaluate:
        print(f"Saved metrics: {output_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
