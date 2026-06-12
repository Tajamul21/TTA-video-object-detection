from __future__ import annotations

from dataclasses import dataclass
from typing import List

from PIL import Image


@dataclass(frozen=True)
class Tile:
    index: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def make_tiles(width: int, height: int, tile_size: int, overlap: float) -> List[Tile]:
    if tile_size <= 0 or (width <= tile_size and height <= tile_size):
        return [Tile(index=0, x0=0, y0=0, x1=width, y1=height)]
    overlap = max(0.0, min(0.9, float(overlap)))
    stride = max(1, int(round(tile_size * (1.0 - overlap))))

    xs = list(range(0, max(1, width - tile_size + 1), stride))
    ys = list(range(0, max(1, height - tile_size + 1), stride))
    if not xs or xs[-1] + tile_size < width:
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] + tile_size < height:
        ys.append(max(0, height - tile_size))

    tiles: List[Tile] = []
    seen = set()
    idx = 0
    for y in ys:
        for x in xs:
            x0 = int(max(0, x))
            y0 = int(max(0, y))
            x1 = int(min(width, x0 + tile_size))
            y1 = int(min(height, y0 + tile_size))
            key = (x0, y0, x1, y1)
            if key in seen:
                continue
            seen.add(key)
            tiles.append(Tile(index=idx, x0=x0, y0=y0, x1=x1, y1=y1))
            idx += 1
    return tiles


def crop_tile(image: Image.Image, tile: Tile) -> Image.Image:
    return image.crop((tile.x0, tile.y0, tile.x1, tile.y1))


def detection_to_global(det: dict, tile: Tile) -> dict:
    out = dict(det)
    x1, y1, x2, y2 = out["bbox_xyxy"]
    x1 += tile.x0
    x2 += tile.x0
    y1 += tile.y0
    y2 += tile.y0
    out["bbox_xyxy"] = [x1, y1, x2, y2]
    out["bbox"] = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
    out["tile_index"] = tile.index
    out["tile_xyxy"] = [tile.x0, tile.y0, tile.x1, tile.y1]
    return out
