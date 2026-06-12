from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Category:
    id: int
    name: str


@dataclass
class ImageRecord:
    id: int | str
    file_name: str
    path: Path
    width: int
    height: int
    video_id: Optional[str] = None


@dataclass
class GroundTruth:
    image_id: int | str
    category_id: int
    bbox: List[float]  # COCO xywh in pixels
    area: float
    iscrowd: int = 0
    id: Optional[int | str] = None


@dataclass
class XSVidDataset:
    data_root: Path
    split: str
    image_root: Path
    annotation_format: str
    images: List[ImageRecord]
    categories: List[Category]
    ground_truths: List[GroundTruth]
    coco_json: Optional[Path] = None
    yolo_label_root: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def categories_by_id(self) -> Dict[int, Category]:
        return {c.id: c for c in self.categories}

    @property
    def category_id_by_name(self) -> Dict[str, int]:
        return {normalize_name(c.name): c.id for c in self.categories}

    @property
    def images_by_id(self) -> Dict[int | str, ImageRecord]:
        return {img.id: img for img in self.images}


def normalize_name(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("_", "-").split())


def read_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return sorted(out)


def find_coco_json(data_root: Path, split: str, explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"COCO json not found: {p}")
        return p

    candidates = [
        data_root / "annotations" / f"{split}.json",
        data_root / "annotations" / "coco" / f"{split}.json",
        data_root / f"{split}.json",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def resolve_image_root(data_root: Path, explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (data_root / "images").resolve()


def build_basename_index(image_root: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for p in list_images(image_root):
        index.setdefault(p.name, []).append(p)
    return index


def resolve_image_path(
    file_name: str,
    data_root: Path,
    image_root: Path,
    basename_index: Optional[Dict[str, List[Path]]] = None,
) -> Optional[Path]:
    raw = Path(file_name)
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend(
        [
            image_root / raw,
            data_root / raw,
            data_root / "images" / raw,
            image_root / raw.name,
        ]
    )
    for p in candidates:
        if p.exists():
            return p.resolve()

    if basename_index is not None:
        matches = basename_index.get(raw.name, [])
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            # Prefer a match that ends with the original relative path.
            wanted = str(raw).replace("\\", "/")
            for m in matches:
                if str(m).replace("\\", "/").endswith(wanted):
                    return m.resolve()
    return None


def parse_class_names(class_names: Optional[str]) -> Optional[List[str]]:
    if not class_names or class_names.strip().lower() in {"auto", "none"}:
        return None
    p = Path(class_names).expanduser()
    if p.exists():
        names: List[str] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            names.append(s)
        return names
    return [x.strip() for x in class_names.split(",") if x.strip()]


def load_coco_dataset(
    data_root: str | Path,
    split: str = "test",
    image_root: Optional[str] = None,
    coco_json: Optional[str] = None,
    max_images: Optional[int] = None,
    image_ids: Optional[Sequence[str | int]] = None,
) -> XSVidDataset:
    root = Path(data_root).expanduser().resolve()
    img_root = resolve_image_root(root, image_root)
    json_path = find_coco_json(root, split, coco_json)
    if json_path is None:
        raise FileNotFoundError(
            "Could not find COCO json. Tried annotations/{split}.json and annotations/coco/{split}.json."
        )

    with json_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = [Category(int(c["id"]), str(c.get("name", c["id"]))) for c in coco.get("categories", [])]
    categories = sorted(categories, key=lambda c: c.id)

    requested_ids = None
    if image_ids is not None:
        requested_ids = {str(x) for x in image_ids}

    basename_index: Optional[Dict[str, List[Path]]] = None
    images: List[ImageRecord] = []
    warnings: List[str] = []
    for item in coco.get("images", []):
        image_id = item.get("id")
        if requested_ids is not None and str(image_id) not in requested_ids:
            continue
        file_name = str(item.get("file_name", item.get("name", "")))
        path = resolve_image_path(file_name, root, img_root, basename_index)
        if path is None:
            if basename_index is None:
                basename_index = build_basename_index(img_root)
                path = resolve_image_path(file_name, root, img_root, basename_index)
        if path is None:
            warnings.append(f"Image not found for COCO file_name={file_name!r}; skipping image_id={image_id}")
            continue
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        if width <= 0 or height <= 0:
            width, height = read_image_size(path)
        video_id = item.get("video_id")
        images.append(
            ImageRecord(
                id=image_id,
                file_name=file_name,
                path=path,
                width=width,
                height=height,
                video_id=str(video_id) if video_id is not None else None,
            )
        )
        if max_images is not None and len(images) >= max_images:
            break

    keep_image_ids = {str(img.id) for img in images}
    gts: List[GroundTruth] = []
    for ann in coco.get("annotations", []):
        if str(ann.get("image_id")) not in keep_image_ids:
            continue
        bbox = [float(x) for x in ann.get("bbox", [0, 0, 0, 0])]
        if len(bbox) != 4:
            continue
        area = float(ann.get("area", bbox[2] * bbox[3]))
        gts.append(
            GroundTruth(
                image_id=ann.get("image_id"),
                category_id=int(ann.get("category_id")),
                bbox=bbox,
                area=area,
                iscrowd=int(ann.get("iscrowd", 0)),
                id=ann.get("id"),
            )
        )

    return XSVidDataset(
        data_root=root,
        split=split,
        image_root=img_root,
        annotation_format="coco",
        images=images,
        categories=categories,
        ground_truths=gts,
        coco_json=json_path,
        warnings=warnings,
    )


def load_yolo_dataset(
    data_root: str | Path,
    split: str = "test",
    image_root: Optional[str] = None,
    yolo_label_root: Optional[str] = None,
    class_names: Optional[str] = None,
    max_images: Optional[int] = None,
    image_ids: Optional[Sequence[str | int]] = None,
) -> XSVidDataset:
    root = Path(data_root).expanduser().resolve()
    img_root = resolve_image_root(root, image_root)
    label_root = Path(yolo_label_root).expanduser().resolve() if yolo_label_root else (root / "annotations" / "yolo").resolve()
    if not img_root.exists():
        raise FileNotFoundError(f"Image root not found: {img_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"YOLO label root not found: {label_root}")

    names = parse_class_names(class_names)
    requested_ids = None
    if image_ids is not None:
        requested_ids = {str(x) for x in image_ids}

    image_paths = list_images(img_root)
    images: List[ImageRecord] = []
    gts: List[GroundTruth] = []
    max_class_id = -1
    ann_id = 1

    for p in image_paths:
        rel = p.relative_to(img_root)
        img_id = str(rel.with_suffix(""))
        if requested_ids is not None and str(img_id) not in requested_ids and str(rel) not in requested_ids:
            continue
        width, height = read_image_size(p)
        images.append(ImageRecord(id=img_id, file_name=str(rel), path=p, width=width, height=height))

        label_path = label_root / rel.with_suffix(".txt")
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                x = (xc - bw / 2.0) * width
                y = (yc - bh / 2.0) * height
                w = bw * width
                h = bh * height
                max_class_id = max(max_class_id, cls)
                gts.append(
                    GroundTruth(
                        image_id=img_id,
                        category_id=cls,
                        bbox=[x, y, w, h],
                        area=max(0.0, w) * max(0.0, h),
                        iscrowd=0,
                        id=ann_id,
                    )
                )
                ann_id += 1
        if max_images is not None and len(images) >= max_images:
            break

    if names is None:
        n = max_class_id + 1 if max_class_id >= 0 else 0
        names = [f"class_{i}" for i in range(n)]
    categories = [Category(i, name) for i, name in enumerate(names)]

    return XSVidDataset(
        data_root=root,
        split=split,
        image_root=img_root,
        annotation_format="yolo",
        images=images,
        categories=categories,
        ground_truths=gts,
        coco_json=None,
        yolo_label_root=label_root,
        warnings=[] if names else ["No class names were provided for YOLO labels."],
    )


def load_image_ids_file(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    ids: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            ids.append(s)
    return ids


def load_dataset_auto(
    data_root: str | Path,
    split: str = "test",
    annotation_format: str = "auto",
    image_root: Optional[str] = None,
    coco_json: Optional[str] = None,
    yolo_label_root: Optional[str] = None,
    class_names: Optional[str] = None,
    max_images: Optional[int] = None,
    image_ids: Optional[Sequence[str | int]] = None,
) -> XSVidDataset:
    fmt = annotation_format.strip().lower()
    root = Path(data_root).expanduser().resolve()
    if fmt not in {"auto", "coco", "yolo"}:
        raise ValueError("annotation_format must be auto, coco, or yolo")
    if fmt in {"auto", "coco"}:
        try:
            return load_coco_dataset(
                root,
                split=split,
                image_root=image_root,
                coco_json=coco_json,
                max_images=max_images,
                image_ids=image_ids,
            )
        except FileNotFoundError:
            if fmt == "coco":
                raise
    return load_yolo_dataset(
        root,
        split=split,
        image_root=image_root,
        yolo_label_root=yolo_label_root,
        class_names=class_names,
        max_images=max_images,
        image_ids=image_ids,
    )


def write_coco_results(predictions: Iterable[dict], output_path: str | Path) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for pred in predictions:
        serializable.append(
            {
                "image_id": pred["image_id"],
                "category_id": int(pred["category_id"]),
                "bbox": [float(x) for x in pred["bbox"]],
                "score": float(pred.get("score", 0.0)),
            }
        )
    p.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def dataset_to_coco_dict(ds: XSVidDataset) -> dict:
    return {
        "images": [
            {"id": img.id, "file_name": img.file_name, "width": img.width, "height": img.height}
            for img in ds.images
        ],
        "categories": [{"id": c.id, "name": c.name} for c in ds.categories],
        "annotations": [
            {
                "id": gt.id if gt.id is not None else i + 1,
                "image_id": gt.image_id,
                "category_id": gt.category_id,
                "bbox": gt.bbox,
                "area": gt.area,
                "iscrowd": gt.iscrowd,
            }
            for i, gt in enumerate(ds.ground_truths)
        ],
    }
