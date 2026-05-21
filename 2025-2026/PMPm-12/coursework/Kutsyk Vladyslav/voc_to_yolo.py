"""
voc_to_yolo.py

Convert Pascal VOC XML annotations into YOLO label files.

This script is intended for datasets organized like:

    data/DUT-Anti-UAV/detection/
      train/
        img/ or images/
        xml/ or annotations/
      val/
      test/

It writes YOLO labels to:

    <split>/labels/*.txt

The project expects YOLO-style labels in the format:

    class_id cx cy w h

with all coordinates normalized to [0, 1].
"""

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class ConvertResult:
    """Summary of one conversion run."""

    xml_files: int
    label_files: int


def _find_existing_dir(base_dir: Path, candidates: Iterable[str]) -> Path | None:
    for name in candidates:
        candidate = base_dir / name
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _normalize_bbox(xmin: float, ymin: float, xmax: float, ymax: float, width: float, height: float) -> tuple[float, float, float, float]:
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def _parse_xml(xml_path: Path, class_name_to_id: dict[str, int]) -> tuple[int, list[str]]:
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        print(f"[VOC->YOLO] WARNING: skipped malformed XML: {xml_path}")
        return 0, []
    root = tree.getroot()

    size_node = root.find("size")
    if size_node is None:
        raise ValueError(f"Missing <size> node in {xml_path}")

    width_node = size_node.find("width")
    height_node = size_node.find("height")
    if width_node is None or height_node is None:
        raise ValueError(f"Missing width/height in {xml_path}")

    width = float(width_node.text or 0)
    height = float(height_node.text or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}: {width}x{height}")

    lines: list[str] = []
    for object_node in root.findall("object"):
        name_node = object_node.find("name")
        bndbox_node = object_node.find("bndbox")
        if name_node is None or bndbox_node is None:
            continue

        class_name = (name_node.text or "").strip()
        if not class_name:
            continue

        if class_name not in class_name_to_id:
            class_name_to_id[class_name] = len(class_name_to_id)

        xmin_node = bndbox_node.find("xmin")
        ymin_node = bndbox_node.find("ymin")
        xmax_node = bndbox_node.find("xmax")
        ymax_node = bndbox_node.find("ymax")
        if None in (xmin_node, ymin_node, xmax_node, ymax_node):
            continue

        xmin = float(xmin_node.text or 0)
        ymin = float(ymin_node.text or 0)
        xmax = float(xmax_node.text or 0)
        ymax = float(ymax_node.text or 0)
        if xmax <= xmin or ymax <= ymin:
            continue

        cx, cy, bw, bh = _normalize_bbox(xmin, ymin, xmax, ymax, width, height)
        class_id = class_name_to_id[class_name]
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return len(lines), lines


def convert_split(split_dir: Path) -> ConvertResult:
    """Convert one detection split, for example train/val/test."""
    xml_dir = _find_existing_dir(split_dir, ("xml", "annotations"))

    if xml_dir is None:
        raise FileNotFoundError(f"No XML directory found in {split_dir}")

    labels_dir = split_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    class_name_to_id: dict[str, int] = {}
    xml_files = sorted(xml_dir.glob("*.xml"))
    written_labels = 0

    for xml_path in xml_files:
        _, lines = _parse_xml(xml_path, class_name_to_id)
        label_path = labels_dir / f"{xml_path.stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written_labels += 1

    print(f"[VOC->YOLO] {split_dir.name}: XML={len(xml_files)}, labels={written_labels}, classes={class_name_to_id or {'UAV': 0}}")
    return ConvertResult(xml_files=len(xml_files), label_files=written_labels)


def convert_dataset(dataset_root: Path) -> list[ConvertResult]:
    """Convert train/val/test splits under the detection directory."""
    results: list[ConvertResult] = []
    detection_root = dataset_root / "detection"

    for split_name in ("train", "val", "test"):
        split_dir = detection_root / split_name
        if split_dir.exists():
            results.append(convert_split(split_dir))

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML annotations to YOLO labels.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/DUT-Anti-UAV"),
        help="Dataset root containing the detection/ folder.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")

    convert_dataset(args.dataset_root)


if __name__ == "__main__":
    main()