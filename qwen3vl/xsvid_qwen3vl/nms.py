from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List


def iou_xywh(a: List[float], b: List[float]) -> float:
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


def per_class_nms(detections: Iterable[dict], iou_threshold: float = 0.50, max_detections: int = 300) -> List[dict]:
    by_class: Dict[int, List[dict]] = defaultdict(list)
    for det in detections:
        by_class[int(det["category_id"])].append(det)

    kept: List[dict] = []
    for _, dets in by_class.items():
        ordered = sorted(dets, key=lambda d: float(d.get("score", 0.0)), reverse=True)
        class_kept: List[dict] = []
        for det in ordered:
            if all(iou_xywh(det["bbox"], old["bbox"]) < iou_threshold for old in class_kept):
                class_kept.append(det)
            if len(class_kept) >= max_detections:
                break
        kept.extend(class_kept)
    return sorted(kept, key=lambda d: float(d.get("score", 0.0)), reverse=True)
