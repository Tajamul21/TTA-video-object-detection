#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xsvid_qwen3vl.dataset import load_dataset_auto, write_coco_results
from xsvid_qwen3vl.metrics import evaluate_map, print_summary, write_metrics


def parse_args() -> argparse.Namespace:
    default_aliases = REPO_ROOT / "configs" / "default_class_aliases.json"
    p = argparse.ArgumentParser(description="Run Qwen3-VL zero-shot XS-VID inference in one process per GPU.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--annotation-format", default="auto", choices=["auto", "coco", "yolo"])
    p.add_argument("--image-root", default=None)
    p.add_argument("--coco-json", default=None)
    p.add_argument("--yolo-label-root", default=None)
    p.add_argument("--class-names", default="auto")
    p.add_argument("--class-aliases", default=str(default_aliases))
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--gpu-ids", required=True, help="Comma-separated physical GPU ids, for example 0,4,6,7.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--attn", default="auto")
    p.add_argument("--device-map", default="single", help="Use single for one full model per worker.")
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--tile-size", type=int, default=0)
    p.add_argument("--tile-overlap", type=float, default=0.20)
    p.add_argument("--include-full-image", action="store_true")
    p.add_argument("--nms-iou", type=float, default=0.50)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--max-detections-per-image", type=int, default=300)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-raw", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--evaluate", action="store_true")
    return p.parse_args()


def parse_gpu_ids(text: str) -> List[str]:
    ids = [x.strip() for x in text.split(",") if x.strip()]
    if not ids:
        raise ValueError("No GPU ids were provided.")
    return ids


def add_arg(cmd: List[str], name: str, value) -> None:
    if value is not None:
        cmd.extend([name, str(value)])


def add_flag(cmd: List[str], name: str, enabled: bool) -> None:
    if enabled:
        cmd.append(name)


def main() -> None:
    args = parse_args()
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    output_dir = Path(args.output_dir).expanduser().resolve()
    shard_dir = output_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset_auto(
        data_root=args.data_root,
        split=args.split,
        annotation_format=args.annotation_format,
        image_root=args.image_root,
        coco_json=args.coco_json,
        yolo_label_root=args.yolo_label_root,
        class_names=args.class_names,
        max_images=args.max_images,
    )
    images = ds.images
    shards = [[] for _ in gpu_ids]
    for i, img in enumerate(images):
        shards[i % len(gpu_ids)].append(str(img.id))

    infer_script = REPO_ROOT / "tools" / "infer_zero_shot.py"
    procs = []
    for gpu, image_ids in zip(gpu_ids, shards):
        if not image_ids:
            continue
        shard_file = shard_dir / f"gpu_{gpu}_image_ids.txt"
        shard_file.write_text("\n".join(image_ids) + "\n", encoding="utf-8")
        child_out = shard_dir / f"gpu_{gpu}_run"
        cmd = [
            sys.executable,
            str(infer_script),
            "--data-root",
            args.data_root,
            "--split",
            args.split,
            "--annotation-format",
            args.annotation_format,
            "--image-ids-file",
            str(shard_file),
            "--output-dir",
            str(child_out),
            "--model-id",
            args.model_id,
            "--attn",
            args.attn,
            "--device-map",
            args.device_map,
            "--torch-dtype",
            args.torch_dtype,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--tile-size",
            str(args.tile_size),
            "--tile-overlap",
            str(args.tile_overlap),
            "--nms-iou",
            str(args.nms_iou),
            "--min-score",
            str(args.min_score),
            "--max-detections-per-image",
            str(args.max_detections_per_image),
        ]
        add_arg(cmd, "--image-root", args.image_root)
        add_arg(cmd, "--coco-json", args.coco_json)
        add_arg(cmd, "--yolo-label-root", args.yolo_label_root)
        add_arg(cmd, "--class-names", args.class_names)
        add_arg(cmd, "--class-aliases", args.class_aliases)
        add_flag(cmd, "--include-full-image", args.include_full_image)
        add_flag(cmd, "--allow-tf32", args.allow_tf32)
        add_flag(cmd, "--resume", args.resume)
        if not args.save_raw:
            cmd.append("--no-save-raw")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print("Launching GPU", gpu, "with", len(image_ids), "images")
        procs.append((gpu, child_out, subprocess.Popen(cmd, env=env)))

    failures = []
    for gpu, child_out, proc in procs:
        rc = proc.wait()
        if rc != 0:
            failures.append({"gpu": gpu, "returncode": rc})
    if failures:
        raise RuntimeError(f"Some worker processes failed: {failures}")

    merged = []
    verbose_lines = []
    for _, child_out, _ in procs:
        pred_path = child_out / "predictions_coco.json"
        if pred_path.exists():
            merged.extend(json.loads(pred_path.read_text(encoding="utf-8")))
        verbose_path = child_out / "predictions_verbose.jsonl"
        if verbose_path.exists():
            verbose_lines.extend(verbose_path.read_text(encoding="utf-8").splitlines())

    write_coco_results(merged, output_dir / "predictions_coco.json")
    if verbose_lines:
        (output_dir / "predictions_verbose.jsonl").write_text("\n".join(verbose_lines) + "\n", encoding="utf-8")

    run_config = {
        "args": vars(args),
        "num_images": len(images),
        "gpu_ids": gpu_ids,
        "num_predictions": len(merged),
        "shard_dir": str(shard_dir),
    }
    (output_dir / "parallel_run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    if args.evaluate:
        result = evaluate_map(ds.ground_truths, merged, ds.categories, max_dets_per_image=100)
        write_metrics(result, output_dir)
        print_summary(result)

    print(f"\nMerged predictions: {output_dir / 'predictions_coco.json'}")
    if args.evaluate:
        print(f"Merged metrics: {output_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
