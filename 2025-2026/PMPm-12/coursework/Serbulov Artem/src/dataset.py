from __future__ import annotations

import os
from pathlib import Path
import random
import shutil

from loguru import logger
from PIL import Image
from tqdm import tqdm
import typer
import yaml

from src.config import PROCESSED_DATA_DIR
from src.tiling import BoxXYXY, YoloLabel, format_yolo_labels, iter_tiles, remap_label_to_tile

app = typer.Typer()


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_ultralytics_dataset_yaml(
    dataset_root: Path,
    yaml_path: Path,
    train_images: str,
    val_images: str,
    test_images: str,
    class_names: list[str],
) -> None:
    """Write an Ultralytics dataset YAML.

    Ultralytics is least error-prone when `path` is absolute and `train/val/test`
    are relative.
    """

    config = {
        "path": str(dataset_root.absolute()),
        "train": train_images,
        "val": val_images,
        "test": test_images,
        "names": {i: name for i, name in enumerate(class_names)},
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _visdrone_row_to_yolo_label(
    row: str,
    image_w: int,
    image_h: int,
    class_map: dict[int, int],
    *,
    min_box_size_px: float = 2.0,
) -> YoloLabel | None:
    """Parse and convert one VisDrone annotation row to a clipped YOLO label.

    VisDrone DET annotations are CSV-like lines:
    x, y, w, h, score, category, truncation, occlusion

    We clip in pixel XYXY space first, then recompute normalized YOLO.
    """

    parts = row.strip().split(",")
    if len(parts) < 6:
        return None

    try:
        category = int(parts[5])
    except ValueError:
        return None

    if category not in class_map:
        return None

    try:
        x = float(parts[0])
        y = float(parts[1])
        w = float(parts[2])
        h = float(parts[3])
    except ValueError:
        return None

    if image_w <= 0 or image_h <= 0:
        return None
    if w <= 0 or h <= 0:
        return None

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    # Clip to image bounds.
    x1 = max(0.0, min(float(image_w), x1))
    y1 = max(0.0, min(float(image_h), y1))
    x2 = max(0.0, min(float(image_w), x2))
    y2 = max(0.0, min(float(image_h), y2))

    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) < min_box_size_px or (y2 - y1) < min_box_size_px:
        return None

    cx = ((x1 + x2) / 2.0) / image_w
    cy = ((y1 + y2) / 2.0) / image_h
    wn = (x2 - x1) / image_w
    hn = (y2 - y1) / image_h

    class_id = class_map[category]
    return YoloLabel(class_id=class_id, cx=cx, cy=cy, w=wn, h=hn)


def convert_visdrone_to_yolo(data_path: str, output_path: str):
    """Convert a VisDrone-style dataset (images + comma-separated annotations) to YOLO format."""
    os.makedirs(f"{output_path}/images", exist_ok=True)
    os.makedirs(f"{output_path}/labels", exist_ok=True)

    class_map = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        7: 6,
        8: 7,
        9: 8,
        10: 9,
    }

    images_dir = os.path.join(data_path, "images")
    annotations_dir = os.path.join(data_path, "annotations")

    if not os.path.exists(images_dir):
        logger.error(f"Folder {images_dir} was not found!")
        return

    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    logger.info(f"Found {len(image_files)} images in {data_path}. Starting conversion...")

    for img_file in tqdm(image_files, desc="Converting"):
        # 1. Image handling
        src_img_path = os.path.join(images_dir, img_file)
        dst_img_path = os.path.join(output_path, "images", img_file)

        # Open to get image dimensions
        with Image.open(src_img_path) as img:
            img_w, img_h = img.size

        # COPY the image (important step!)
        # If you want to save space, replace copy with os.symlink (Linux/Mac only)
        if not os.path.exists(dst_img_path):
            shutil.copy(src_img_path, dst_img_path)

        # 2. Annotation handling
        txt_name = img_file.replace(".jpg", ".txt")
        ann_path = os.path.join(annotations_dir, txt_name)

        yolo_labels: list[YoloLabel] = []
        if os.path.exists(ann_path):
            with open(ann_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    label = _visdrone_row_to_yolo_label(
                        line,
                        image_w=img_w,
                        image_h=img_h,
                        class_map=class_map,
                    )
                    if label is not None:
                        yolo_labels.append(label)

        # 3. Write result
        with open(f"{output_path}/labels/{txt_name}", "w", encoding="utf-8") as out_f:
            out_f.write(format_yolo_labels(yolo_labels))


@app.command()
def tile_train_split(
    source_dataset: Path = typer.Option(  # noqa: B008
        PROCESSED_DATA_DIR / "visdrone_yolo",
        help="Root of full-frame YOLO dataset with train/val/test_dev splits.",
    ),
    output_dataset: Path = typer.Option(  # noqa: B008
        PROCESSED_DATA_DIR / "visdrone_yolo_tiled",
        help="Where to write tiled dataset (train tiles + symlinked val/test_dev).",
    ),
    tile_size: int = typer.Option(1024, help="Square tile size in pixels."),  # noqa: B008
    overlap: float = typer.Option(0.25, help="Tile overlap ratio in [0, 1)."),  # noqa: B008
    min_intersection_ratio: float = typer.Option(  # noqa: B008
        0.30,
        help="Keep a GT box in a tile if (intersection area / original box area) >= this.",
    ),
    keep_empty_prob: float = typer.Option(  # noqa: B008
        0.10,
        help="Probability to keep a tile that contains no labels (background tiles).",
    ),
    seed: int = typer.Option(1337, help="Random seed used for empty-tile sampling."),  # noqa: B008
):
    """Create an offline tiled training split.

    - Generates `output_dataset/train/{images,labels}` from `source_dataset/train/...`.
    - Reuses full-frame `val` and `test_dev` via symlinks.
    - Writes `output_dataset/dataset.yaml` for Ultralytics.
    """

    if not (0.0 <= overlap < 1.0):
        raise typer.BadParameter("overlap must be in [0, 1).")
    if not (0.0 <= keep_empty_prob <= 1.0):
        raise typer.BadParameter("keep_empty_prob must be in [0, 1].")

    rng = random.Random(seed)

    src_train_images = source_dataset / "train" / "images"
    src_train_labels = source_dataset / "train" / "labels"
    src_val = source_dataset / "val"
    src_test = source_dataset / "test_dev"

    if not src_train_images.exists():
        raise FileNotFoundError(f"Missing: {src_train_images}")
    if not src_train_labels.exists():
        raise FileNotFoundError(f"Missing: {src_train_labels}")
    if not src_val.exists():
        raise FileNotFoundError(f"Missing: {src_val}")
    if not src_test.exists():
        raise FileNotFoundError(f"Missing: {src_test}")

    out_train_images = output_dataset / "train" / "images"
    out_train_labels = output_dataset / "train" / "labels"
    _safe_mkdir(out_train_images)
    _safe_mkdir(out_train_labels)

    # Symlink full-frame val/test_dev.
    for split_name, src_split in [("val", src_val), ("test_dev", src_test)]:
        dst_split = output_dataset / split_name
        
        if dst_split.is_symlink():
            dst_split.unlink()
        elif dst_split.exists():
            continue
            
        abs_src = src_split.resolve()
        
        logger.info(f"Symlinking {dst_split} -> {abs_src}")
        os.symlink(abs_src, dst_split, target_is_directory=True)

    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    logger.info(
        "Tiling train split: tile=%s stride=%s min_intersection=%s keep_empty_prob=%s",
        tile_size,
        stride,
        min_intersection_ratio,
        keep_empty_prob,
    )

    # Fixed VisDrone class ordering used elsewhere in the project.
    class_names = [
        "pedestrian",
        "people",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "awning-tricycle",
        "bus",
        "motor",
    ]

    image_files = sorted([p for p in src_train_images.iterdir() if p.suffix.lower() == ".jpg"])
    if not image_files:
        raise FileNotFoundError(f"No .jpg files found in {src_train_images}")

    tile_count = 0
    kept_empty = 0

    for img_path in tqdm(image_files, desc="Tiling train"):
        label_path = src_train_labels / f"{img_path.stem}.txt"
        label_lines: list[str] = []
        if label_path.exists():
            label_lines = label_path.read_text(encoding="utf-8").splitlines()

        labels = []
        for raw in label_lines:
            parts = raw.strip().split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue
            labels.append(YoloLabel(class_id=class_id, cx=cx, cy=cy, w=w, h=h))

        with Image.open(img_path) as img:
            img_w, img_h = img.size
            img_rgb = img.convert("RGB")

            for x0, y0, x1, y1 in iter_tiles(img_w, img_h, tile=tile_size, stride=stride):
                tile_xyxy = BoxXYXY(float(x0), float(y0), float(x1), float(y1))
                tile_labels: list[YoloLabel] = []

                for lab in labels:
                    mapped = remap_label_to_tile(
                        lab,
                        image_w=img_w,
                        image_h=img_h,
                        tile_xyxy=tile_xyxy,
                        min_intersection_ratio=min_intersection_ratio,
                    )
                    if mapped is not None:
                        tile_labels.append(mapped)

                if not tile_labels:
                    if rng.random() > keep_empty_prob:
                        continue
                    kept_empty += 1

                tile_img = img_rgb.crop((x0, y0, x1, y1))
                tile_stem = f"{img_path.stem}__x{x0}_y{y0}_w{(x1-x0)}_h{(y1-y0)}"
                out_img_path = out_train_images / f"{tile_stem}.jpg"
                out_lbl_path = out_train_labels / f"{tile_stem}.txt"

                tile_img.save(out_img_path, quality=95)
                out_lbl_path.write_text(format_yolo_labels(tile_labels), encoding="utf-8")
                tile_count += 1

    # Write dataset.yaml at root.
    yaml_path = output_dataset / "dataset.yaml"
    _write_ultralytics_dataset_yaml(
        dataset_root=output_dataset,
        yaml_path=yaml_path,
        train_images="train/images",
        val_images="val/images",
        test_images="test_dev/images",
        class_names=class_names,
    )

    logger.success(
        "Tiled dataset created at %s (train tiles: %s, kept empty: %s). YAML: %s",
        output_dataset,
        tile_count,
        kept_empty,
        yaml_path,
    )


@app.command()
def main(
    input_dir: Path = typer.Option(  # noqa: B008
        ...,
        help="Path to raw VisDrone folder (contains 'images' and 'annotations').",
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        ...,
        help="Path where YOLO formatted data will be saved.",
    ),
):
    logger.info("Starting dataset processing...")

    input_str = str(input_dir)
    output_str = str(output_dir)

    convert_visdrone_to_yolo(input_str, output_str)

    logger.success(f"Processing complete! Data saved to {output_dir}")


if __name__ == "__main__":
    app()
