from __future__ import annotations

from src.dataset import _visdrone_row_to_yolo_label
from src.tiling import BoxXYXY, YoloLabel, remap_label_to_tile


def test_visdrone_row_to_yolo_clips_and_maps_category() -> None:
    # VisDrone row: x,y,w,h,score,category,truncation,occlusion
    # Put box partially outside the top-left of a 100x100 image.
    row = "-10,-10,20,20,1,4,0,0"  # category=4 -> class_id=3

    label = _visdrone_row_to_yolo_label(row, image_w=100, image_h=100, class_map={4: 3})
    assert label is not None
    assert label.class_id == 3

    # After clipping, box should become [0,0]..[10,10] => center at (5,5), size (10,10).
    assert abs(label.cx - 0.05) < 1e-6
    assert abs(label.cy - 0.05) < 1e-6
    assert abs(label.w - 0.10) < 1e-6
    assert abs(label.h - 0.10) < 1e-6


def test_remap_label_to_tile_keeps_when_intersection_sufficient() -> None:
    # Full image 100x100; box is 20x20 centered at (50,50).
    lab = YoloLabel(class_id=1, cx=0.5, cy=0.5, w=0.2, h=0.2)
    tile = BoxXYXY(40, 40, 80, 80)  # 40x40 tile fully contains the box.

    mapped = remap_label_to_tile(
        lab,
        image_w=100,
        image_h=100,
        tile_xyxy=tile,
        min_intersection_ratio=0.30,
    )

    assert mapped is not None
    assert mapped.class_id == 1

    # In tile coordinates, the box is centered at (10,10) in a 40x40 tile => 0.25.
    assert abs(mapped.cx - 0.25) < 1e-6
    assert abs(mapped.cy - 0.25) < 1e-6
    assert abs(mapped.w - 0.5) < 1e-6
    assert abs(mapped.h - 0.5) < 1e-6
