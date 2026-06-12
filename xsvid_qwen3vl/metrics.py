from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .dataset import Category, GroundTruth

IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)
AREA_RANGES = OrderedDict(
    [
        ("all", (0.0, 1.0e10)),
        ("0-12", (0.0, 12.0**2)),
        ("12-20", (12.0**2, 20.0**2)),
        ("20-32", (20.0**2, 32.0**2)),
        ("small", (0.0, 32.0**2)),
        ("medium", (32.0**2, 96.0**2)),
        ("large", (96.0**2, 1.0e10)),
    ]
)


def _as_gt_dict(gt: GroundTruth | dict, idx: int) -> dict:
    if isinstance(gt, GroundTruth):
        return {
            "id": gt.id if gt.id is not None else idx,
            "image_id": gt.image_id,
            "category_id": int(gt.category_id),
            "bbox": [float(x) for x in gt.bbox],
            "area": float(gt.area),
            "iscrowd": int(gt.iscrowd),
        }
    bbox = [float(x) for x in gt.get("bbox", [0, 0, 0, 0])]
    return {
        "id": gt.get("id", idx),
        "image_id": gt.get("image_id"),
        "category_id": int(gt.get("category_id")),
        "bbox": bbox,
        "area": float(gt.get("area", max(0.0, bbox[2]) * max(0.0, bbox[3]))),
        "iscrowd": int(gt.get("iscrowd", 0)),
    }


def _as_pred_dict(pred: dict) -> dict:
    bbox = [float(x) for x in pred.get("bbox", [0, 0, 0, 0])]
    return {
        "image_id": pred.get("image_id"),
        "category_id": int(pred.get("category_id")),
        "bbox": bbox,
        "score": float(pred.get("score", 0.0)),
        "area": max(0.0, bbox[2]) * max(0.0, bbox[3]),
    }


def bbox_iou_xywh(a: List[float], b: List[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _in_area(area: float, area_range: Tuple[float, float]) -> bool:
    return float(area) >= area_range[0] and float(area) < area_range[1]


def _limit_predictions(preds: List[dict], max_dets_per_image: int) -> List[dict]:
    grouped: Dict[Tuple[object, int], List[dict]] = defaultdict(list)
    for pred in preds:
        grouped[(pred["image_id"], int(pred["category_id"]))].append(pred)
    out: List[dict] = []
    for group in grouped.values():
        out.extend(sorted(group, key=lambda p: p["score"], reverse=True)[:max_dets_per_image])
    return out


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    # Monotonic precision envelope.
    precision = precision.copy()
    for i in range(precision.size - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    values = []
    for thr in RECALL_THRESHOLDS:
        inds = np.where(recall >= thr)[0]
        values.append(float(np.max(precision[inds])) if inds.size else 0.0)
    return float(np.mean(values))


def _evaluate_one(
    gt_all: List[dict],
    pred_all: List[dict],
    category_id: int,
    iou_thr: float,
    area_range: Tuple[float, float],
    max_dets_per_image: int,
) -> dict:
    gts_cat = [g for g in gt_all if int(g["category_id"]) == int(category_id)]
    preds_cat = [p for p in pred_all if int(p["category_id"]) == int(category_id)]
    preds_cat = _limit_predictions(preds_cat, max_dets_per_image=max_dets_per_image)
    preds_cat = sorted(preds_cat, key=lambda p: p["score"], reverse=True)

    active_by_img: Dict[object, List[dict]] = defaultdict(list)
    ignore_by_img: Dict[object, List[dict]] = defaultdict(list)
    npos = 0
    for idx, gt in enumerate(gts_cat):
        gt = dict(gt)
        gt["_match_id"] = f"{gt['image_id']}::{gt.get('id', idx)}::{idx}"
        if int(gt.get("iscrowd", 0)) == 1 or not _in_area(gt["area"], area_range):
            ignore_by_img[gt["image_id"]].append(gt)
        else:
            active_by_img[gt["image_id"]].append(gt)
            npos += 1

    if npos == 0:
        return {"ap": None, "recall": None, "num_gt": 0, "num_dt": len(preds_cat)}

    matched = set()
    tp: List[int] = []
    fp: List[int] = []

    for pred in preds_cat:
        image_id = pred["image_id"]
        best_iou = 0.0
        best_gt = None
        for gt in active_by_img.get(image_id, []):
            if gt["_match_id"] in matched:
                continue
            iou = bbox_iou_xywh(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt
        if best_gt is not None and best_iou >= iou_thr:
            matched.add(best_gt["_match_id"])
            tp.append(1)
            fp.append(0)
            continue

        # Ignore detections that hit ignored GT or are outside the current area range.
        ignore_det = False
        for gt in ignore_by_img.get(image_id, []):
            if bbox_iou_xywh(pred["bbox"], gt["bbox"]) >= iou_thr:
                ignore_det = True
                break
        if not ignore_det and not _in_area(pred.get("area", pred["bbox"][2] * pred["bbox"][3]), area_range):
            ignore_det = True
        if ignore_det:
            continue
        tp.append(0)
        fp.append(1)

    if not tp:
        return {"ap": 0.0, "recall": 0.0, "num_gt": npos, "num_dt": len(preds_cat)}

    tp_arr = np.cumsum(np.asarray(tp, dtype=np.float64))
    fp_arr = np.cumsum(np.asarray(fp, dtype=np.float64))
    recall = tp_arr / float(npos)
    precision = tp_arr / np.maximum(tp_arr + fp_arr, np.finfo(np.float64).eps)
    ap = _compute_ap(recall, precision)
    return {"ap": ap, "recall": float(recall[-1]) if recall.size else 0.0, "num_gt": npos, "num_dt": len(preds_cat)}


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not vals:
        return None
    return float(np.mean(vals))


def _pct(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value) * 100.0


def evaluate_map(
    ground_truths: Iterable[GroundTruth | dict],
    predictions: Iterable[dict],
    categories: Iterable[Category],
    max_dets_per_image: int = 100,
) -> dict:
    cats = list(categories)
    gt_all = [_as_gt_dict(gt, idx) for idx, gt in enumerate(ground_truths)]
    pred_all = [_as_pred_dict(pred) for pred in predictions]

    raw: Dict[str, Dict[int, Dict[str, List[dict]]]] = defaultdict(lambda: defaultdict(dict))
    for area_name, area_range in AREA_RANGES.items():
        for cat in cats:
            per_thr = []
            for thr in IOU_THRESHOLDS:
                per_thr.append(
                    _evaluate_one(
                        gt_all,
                        pred_all,
                        category_id=cat.id,
                        iou_thr=float(thr),
                        area_range=area_range,
                        max_dets_per_image=max_dets_per_image,
                    )
                )
            raw[area_name][cat.id] = {"thresholds": per_thr}

    def ap(area: str, cat_id: int, thr_index: Optional[int] = None) -> Optional[float]:
        rows = raw[area][cat_id]["thresholds"]
        if thr_index is not None:
            return rows[thr_index]["ap"]
        return _mean([r["ap"] for r in rows])

    def recall(area: str, cat_id: int, thr_index: Optional[int] = None) -> Optional[float]:
        rows = raw[area][cat_id]["thresholds"]
        if thr_index is not None:
            return rows[thr_index]["recall"]
        return _mean([r["recall"] for r in rows])

    def gt_count(area: str, cat_id: Optional[int] = None) -> int:
        lo, hi = AREA_RANGES[area]
        count = 0
        for gt in gt_all:
            if cat_id is not None and int(gt["category_id"]) != int(cat_id):
                continue
            if int(gt.get("iscrowd", 0)) != 1 and _in_area(gt["area"], (lo, hi)):
                count += 1
        return count

    def pred_count(cat_id: Optional[int] = None) -> int:
        count = 0
        for pred in pred_all:
            if cat_id is None or int(pred["category_id"]) == int(cat_id):
                count += 1
        return count

    # IoU threshold index 0 is AP50, index 5 is AP75 for [.50:.05:.95].
    all_cat_ids = [c.id for c in cats]
    summary = {
        "mAP_50_95": _pct(_mean([ap("all", cid) for cid in all_cat_ids])),
        "AP50": _pct(_mean([ap("all", cid, 0) for cid in all_cat_ids])),
        "AP75": _pct(_mean([ap("all", cid, 5) for cid in all_cat_ids])),
        "AR_50_95": _pct(_mean([recall("all", cid) for cid in all_cat_ids])),
        "num_images_with_gt_or_pred": len(set([g["image_id"] for g in gt_all] + [p["image_id"] for p in pred_all])),
        "num_gt": len(gt_all),
        "num_predictions": len(pred_all),
        "max_dets_per_image": max_dets_per_image,
    }
    for area_name in AREA_RANGES:
        summary[f"AP_{area_name}"] = _pct(_mean([ap(area_name, cid) for cid in all_cat_ids]))
        summary[f"AP50_{area_name}"] = _pct(_mean([ap(area_name, cid, 0) for cid in all_cat_ids]))

    by_class = []
    for cat in cats:
        row = {
            "category_id": cat.id,
            "category_name": cat.name,
            "num_gt": gt_count("all", cat.id),
            "num_predictions": pred_count(cat.id),
            "mAP_50_95": _pct(ap("all", cat.id)),
            "AP50": _pct(ap("all", cat.id, 0)),
            "AP75": _pct(ap("all", cat.id, 5)),
            "AR_50_95": _pct(recall("all", cat.id)),
        }
        by_class.append(row)

    by_size = []
    for area_name in AREA_RANGES:
        by_size.append(
            {
                "area": area_name,
                "num_gt": gt_count(area_name),
                "mAP_50_95": _pct(_mean([ap(area_name, cid) for cid in all_cat_ids])),
                "AP50": _pct(_mean([ap(area_name, cid, 0) for cid in all_cat_ids])),
                "AP75": _pct(_mean([ap(area_name, cid, 5) for cid in all_cat_ids])),
                "AR_50_95": _pct(_mean([recall(area_name, cid) for cid in all_cat_ids])),
            }
        )

    return {"summary": summary, "by_class": by_class, "by_size": by_size}


def write_metrics(result: dict, output_dir: str | Path) -> None:
    import csv

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics_summary.json").write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")

    for key, filename in [("by_class", "metrics_by_class.csv"), ("by_size", "metrics_by_size.csv")]:
        rows = result.get(key, [])
        if not rows:
            continue
        fieldnames = list(rows[0].keys())
        with (out / filename).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def print_summary(result: dict) -> None:
    summary = result["summary"]
    keys = ["mAP_50_95", "AP50", "AP75", "AP_0-12", "AP_12-20", "AP_20-32", "AP_small", "AP_medium", "AP_large"]
    print("\nXS-VID zero-shot detection metrics")
    print("----------------------------------")
    for key in keys:
        value = summary.get(key)
        if value is None:
            text = "nan"
        else:
            text = f"{value:.3f}"
        print(f"{key:>12}: {text}")
    print(f"{'num_gt':>12}: {summary.get('num_gt', 0)}")
    print(f"{'num_predictions':>12}: {summary.get('num_predictions', 0)}")
