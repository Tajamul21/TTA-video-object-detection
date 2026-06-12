from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image, ImageDraw, ImageFont

from .dataset import Category, GroundTruth


def _font(size: int = 18):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_box(draw: ImageDraw.ImageDraw, xywh: List[float], text: str, color: str, width: int, font) -> None:
    x, y, w, h = xywh
    x2, y2 = x + w, y + h
    draw.rectangle([x, y, x2, y2], outline=color, width=width)
    if text:
        pad = 2
        try:
            tb = draw.textbbox((x, y), text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = len(text) * 8, 14
        ty = max(0, y - th - 2 * pad)
        draw.rectangle([x, ty, x + tw + 2 * pad, ty + th + 2 * pad], fill=color)
        draw.text((x + pad, ty + pad), text, fill="white", font=font)


def draw_predictions(
    image_path: str | Path,
    predictions: Iterable[dict],
    categories: Iterable[Category],
    output_path: str | Path,
    ground_truths: Optional[Iterable[GroundTruth | dict]] = None,
) -> None:
    cats: Dict[int, str] = {c.id: c.name for c in categories}
    with Image.open(image_path).convert("RGB") as img:
        draw = ImageDraw.Draw(img)
        font = _font(max(12, min(img.size) // 45))
        if ground_truths is not None:
            for gt in ground_truths:
                if isinstance(gt, GroundTruth):
                    bbox = gt.bbox
                    name = cats.get(gt.category_id, str(gt.category_id))
                else:
                    bbox = [float(x) for x in gt.get("bbox", [0, 0, 0, 0])]
                    name = cats.get(int(gt.get("category_id")), str(gt.get("category_id")))
                _draw_box(draw, bbox, f"GT {name}", "green", 2, font)
        for pred in predictions:
            bbox = [float(x) for x in pred.get("bbox", [0, 0, 0, 0])]
            name = cats.get(int(pred.get("category_id")), str(pred.get("category_id")))
            score = float(pred.get("score", 0.0))
            _draw_box(draw, bbox, f"{name} {score:.2f}", "red", 2, font)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
