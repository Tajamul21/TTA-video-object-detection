from __future__ import annotations

import difflib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .dataset import Category, normalize_name


def load_aliases(path: Optional[str]) -> Dict[str, List[str]]:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Class alias file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[str, List[str]] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                out[str(key)] = [value]
            elif isinstance(value, list):
                out[str(key)] = [str(x) for x in value]
    return out


def strip_code_fence(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    return s


def extract_json(text: str) -> Any:
    s = strip_code_fence(text)
    if not s:
        return {"detections": []}
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try the largest JSON object.
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        candidate = s[start : end + 1]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Try a JSON list of detections.
    start = s.find("[")
    end = s.rfind("]")
    if start >= 0 and end > start:
        candidate = s[start : end + 1]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return {"detections": json.loads(candidate)}
        except Exception:
            pass
    return {"detections": []}


def get_detection_items(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        for key in ["detections", "objects", "bboxes", "boxes", "results"]:
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        # Sometimes the model returns a single object dictionary.
        if "bbox_2d" in data or "bbox" in data:
            return [data]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def build_label_lookup(categories: Iterable[Category], aliases: Optional[Dict[str, List[str]]] = None) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    cats = list(categories)
    for cat in cats:
        keys = {cat.name, cat.name.replace("-", " "), cat.name.replace("_", " ")}
        for key in keys:
            lookup[normalize_name(key)] = cat.id
    aliases = aliases or {}
    name_to_id = {normalize_name(c.name): c.id for c in cats}
    for canonical, alias_list in aliases.items():
        cid = name_to_id.get(normalize_name(canonical))
        if cid is None:
            continue
        for alias in alias_list:
            lookup[normalize_name(alias)] = cid
    return lookup


def canonicalize_label(
    raw_label: Any,
    categories: Iterable[Category],
    aliases: Optional[Dict[str, List[str]]] = None,
    fuzzy_cutoff: float = 0.72,
) -> Optional[int]:
    label = normalize_name(str(raw_label or ""))
    if not label:
        return None
    lookup = build_label_lookup(categories, aliases)
    if label in lookup:
        return lookup[label]

    # Remove common surrounding words from VLM outputs.
    cleaned = label
    for token in ["a ", "an ", "the "]:
        if cleaned.startswith(token):
            cleaned = cleaned[len(token) :]
    cleaned = cleaned.replace(" object", "").replace(" target", "")
    if cleaned in lookup:
        return lookup[cleaned]

    # Last resort: fuzzy match against class names and aliases.
    matches = difflib.get_close_matches(cleaned, list(lookup.keys()), n=1, cutoff=fuzzy_cutoff)
    if matches:
        return lookup[matches[0]]
    return None


def parse_score(det: Dict[str, Any]) -> float:
    for key in ["score", "confidence", "conf", "probability"]:
        if key not in det:
            continue
        value = det.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return max(0.0, min(1.0, float(value)))
        text = str(value).strip().lower()
        try:
            number = float(text.rstrip("%"))
            if number > 1.0:
                number = number / 100.0
            return max(0.0, min(1.0, number))
        except Exception:
            pass
        mapping = {"very high": 0.95, "high": 0.85, "medium": 0.60, "moderate": 0.60, "low": 0.35, "very low": 0.20}
        if text in mapping:
            return mapping[text]
    return 0.50


def parse_bbox_value(value: Any) -> Optional[List[float]]:
    if isinstance(value, dict):
        keys_xyxy = ["x1", "y1", "x2", "y2"]
        if all(k in value for k in keys_xyxy):
            return [float(value[k]) for k in keys_xyxy]
        keys_xywh = ["x", "y", "w", "h"]
        if all(k in value for k in keys_xywh):
            x, y, w, h = [float(value[k]) for k in keys_xywh]
            return [x, y, x + w, y + h]
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return [float(value[i]) for i in range(4)]
        except Exception:
            return None
    return None


def get_raw_bbox(det: Dict[str, Any]) -> Optional[List[float]]:
    for key in ["bbox_2d", "bbox", "box", "bounding_box", "coordinates"]:
        if key in det:
            parsed = parse_bbox_value(det.get(key))
            if parsed is not None:
                return parsed
    return None


def xyxy_to_xywh(box: List[float]) -> List[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def clip_xyxy(box: List[float], width: int, height: int) -> Optional[List[float]]:
    x1, y1, x2, y2 = box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def qwen_box_to_pixels(box: List[float], width: int, height: int) -> Optional[List[float]]:
    vals = [float(x) for x in box]
    max_abs = max(abs(x) for x in vals) if vals else 0.0
    if max_abs <= 1.5:
        # Normalized 0..1 coordinates.
        x1, y1, x2, y2 = vals
        pix = [x1 * width, y1 * height, x2 * width, y2 * height]
    elif max_abs <= 1000.0:
        # Qwen coordinate grid.
        x1, y1, x2, y2 = vals
        pix = [x1 / 1000.0 * width, y1 / 1000.0 * height, x2 / 1000.0 * width, y2 / 1000.0 * height]
    else:
        # Already pixel coordinates.
        pix = vals
    return clip_xyxy(pix, width, height)


def normalize_detections(
    raw_text: str,
    categories: Iterable[Category],
    image_width: int,
    image_height: int,
    aliases: Optional[Dict[str, List[str]]] = None,
    min_score: float = 0.0,
) -> Tuple[List[Dict[str, Any]], Any]:
    data = extract_json(raw_text)
    items = get_detection_items(data)
    out: List[Dict[str, Any]] = []
    cats_by_id = {c.id: c for c in categories}
    for det in items:
        label = det.get("label", det.get("class", det.get("category", det.get("name", ""))))
        cid = canonicalize_label(label, cats_by_id.values(), aliases=aliases)
        if cid is None:
            continue
        raw_box = get_raw_bbox(det)
        if raw_box is None:
            continue
        xyxy = qwen_box_to_pixels(raw_box, image_width, image_height)
        if xyxy is None:
            continue
        score = parse_score(det)
        if score < min_score:
            continue
        out.append(
            {
                "category_id": int(cid),
                "label": cats_by_id[cid].name,
                "raw_label": str(label),
                "bbox_xyxy": xyxy,
                "bbox": xyxy_to_xywh(xyxy),
                "score": score,
            }
        )
    return out, data
