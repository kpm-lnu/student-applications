"""Utilities for slicing (tiling) YOLO datasets.

This module is intentionally framework-agnostic: it contains only geometry and
label conversion helpers so it can be unit-tested without GPU/Ultralytics.

Coordinate conventions:
- YOLO labels use normalized (cx, cy, w, h) in [0, 1] relative to image width/height.
- Pixel boxes use XYXY: (x1, y1, x2, y2) with x2 > x1 and y2 > y1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence


@dataclass(frozen=True, slots=True)
class BoxXYXY:
    """Pixel-space box in XYXY format."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        w = self.width
        h = self.height
        if w <= 0 or h <= 0:
            return 0.0
        return w * h


@dataclass(frozen=True, slots=True)
class YoloLabel:
    """A single YOLO-format label."""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float


def clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def clip_xyxy_to_image(box: BoxXYXY, image_w: int, image_h: int) -> BoxXYXY | None:
    """Clip XYXY box to image bounds.

    Returns None if the clipped box is degenerate.
    """

    if image_w <= 0 or image_h <= 0:
        return None

    x1 = clamp(box.x1, 0.0, float(image_w))
    y1 = clamp(box.y1, 0.0, float(image_h))
    x2 = clamp(box.x2, 0.0, float(image_w))
    y2 = clamp(box.y2, 0.0, float(image_h))

    if x2 <= x1 or y2 <= y1:
        return None

    return BoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2)


def yolo_to_xyxy_px(label: YoloLabel, image_w: int, image_h: int) -> BoxXYXY | None:
    """Convert a YOLO normalized label to pixel XYXY."""

    if image_w <= 0 or image_h <= 0:
        return None

    cx_px = label.cx * image_w
    cy_px = label.cy * image_h
    w_px = label.w * image_w
    h_px = label.h * image_h

    x1 = cx_px - w_px / 2.0
    y1 = cy_px - h_px / 2.0
    x2 = cx_px + w_px / 2.0
    y2 = cy_px + h_px / 2.0

    return clip_xyxy_to_image(BoxXYXY(x1, y1, x2, y2), image_w=image_w, image_h=image_h)


def xyxy_to_yolo_norm(box: BoxXYXY, image_w: int, image_h: int) -> YoloLabel | None:
    """Convert pixel XYXY box to normalized YOLO (without class_id)."""

    if image_w <= 0 or image_h <= 0:
        return None

    if box.x2 <= box.x1 or box.y2 <= box.y1:
        return None

    cx = ((box.x1 + box.x2) / 2.0) / image_w
    cy = ((box.y1 + box.y2) / 2.0) / image_h
    w = (box.x2 - box.x1) / image_w
    h = (box.y2 - box.y1) / image_h

    # Clip to [0, 1] defensively.
    cx = clamp(cx, 0.0, 1.0)
    cy = clamp(cy, 0.0, 1.0)
    w = clamp(w, 0.0, 1.0)
    h = clamp(h, 0.0, 1.0)

    return YoloLabel(class_id=-1, cx=cx, cy=cy, w=w, h=h)


def intersection(a: BoxXYXY, b: BoxXYXY) -> BoxXYXY | None:
    """Compute intersection of two XYXY boxes."""

    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return BoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2)


def generate_tile_starts(full: int, tile: int, stride: int) -> list[int]:
    """Generate deterministic tile start coordinates.

    Includes a final tile that touches the image boundary if needed.
    """

    if full <= 0:
        return [0]
    if tile <= 0:
        raise ValueError("tile must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    if full <= tile:
        return [0]

    starts = list(range(0, full - tile + 1, stride))
    last = full - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def iter_tiles(
    image_w: int, image_h: int, tile: int, stride: int
) -> Iterator[tuple[int, int, int, int]]:
    """Yield tiles as (x0, y0, x1, y1) in pixel coordinates."""

    x_starts = generate_tile_starts(image_w, tile, stride)
    y_starts = generate_tile_starts(image_h, tile, stride)

    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(x0 + tile, image_w)
            y1 = min(y0 + tile, image_h)
            yield (x0, y0, x1, y1)


def remap_label_to_tile(
    label: YoloLabel,
    image_w: int,
    image_h: int,
    tile_xyxy: BoxXYXY,
    min_intersection_ratio: float,
    min_box_size_px: float = 2.0,
) -> YoloLabel | None:
    """Project a YOLO label from full image into a tile.

    Keeps a box if its intersection area with the tile is at least
    `min_intersection_ratio` of the original box area.
    """

    if not (0.0 <= min_intersection_ratio <= 1.0):
        raise ValueError("min_intersection_ratio must be in [0, 1]")

    full_box = yolo_to_xyxy_px(label, image_w=image_w, image_h=image_h)
    if full_box is None:
        return None

    inter = intersection(full_box, tile_xyxy)
    if inter is None:
        return None

    full_area = full_box.area
    if full_area <= 0:
        return None

    if (inter.area / full_area) < min_intersection_ratio:
        return None

    # Convert to tile-local coordinates.
    local = BoxXYXY(
        x1=inter.x1 - tile_xyxy.x1,
        y1=inter.y1 - tile_xyxy.y1,
        x2=inter.x2 - tile_xyxy.x1,
        y2=inter.y2 - tile_xyxy.y1,
    )

    if local.width < min_box_size_px or local.height < min_box_size_px:
        return None

    tile_w = int(tile_xyxy.x2 - tile_xyxy.x1)
    tile_h = int(tile_xyxy.y2 - tile_xyxy.y1)

    norm = xyxy_to_yolo_norm(local, image_w=tile_w, image_h=tile_h)
    if norm is None:
        return None

    return YoloLabel(class_id=label.class_id, cx=norm.cx, cy=norm.cy, w=norm.w, h=norm.h)


def parse_yolo_labels(lines: Sequence[str]) -> list[YoloLabel]:
    """Parse YOLO txt lines into labels.

    Ignores malformed lines.
    """

    labels: list[YoloLabel] = []
    for raw in lines:
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        labels.append(YoloLabel(class_id=class_id, cx=cx, cy=cy, w=w, h=h))
    return labels


def format_yolo_labels(labels: Iterable[YoloLabel]) -> str:
    """Format labels as YOLO txt content."""

    return "\n".join(
        f"{lab.class_id} {lab.cx:.6f} {lab.cy:.6f} {lab.w:.6f} {lab.h:.6f}" for lab in labels
    )
