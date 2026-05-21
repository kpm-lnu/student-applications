from __future__ import annotations

import csv
from pathlib import Path


# Configuration
INPUT_ROOT = Path("dataset/clean")
OUTPUT_DIR = Path("dataset/fragments_generated")
OUTPUT_CSV = Path("dataset/labels_generated.csv")

FRAGMENT_SIZE = 1024
STRIDE = FRAGMENT_SIZE
INCLUDE_PARTIAL_TAIL = False


def iter_class_files(root: Path):
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        label = class_dir.name
        for file_path in sorted(class_dir.iterdir()):
            if file_path.is_file():
                yield label, file_path


def fragment_offsets(file_size: int, fragment_size: int, stride: int, include_partial_tail: bool):
    if file_size <= 0:
        return

    if file_size < fragment_size:
        if include_partial_tail:
            yield 0
        return

    last_full_start = file_size - fragment_size
    offset = 0

    while offset <= last_full_start:
        yield offset
        offset += stride

    if include_partial_tail and last_full_start % stride != 0:
        yield last_full_start


def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input root does not exist: {INPUT_ROOT}")

    if FRAGMENT_SIZE <= 0:
        raise ValueError("FRAGMENT_SIZE must be positive.")

    if STRIDE <= 0:
        raise ValueError("STRIDE must be positive.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    fragment_index = 0

    for label, file_path in iter_class_files(INPUT_ROOT):
        data = file_path.read_bytes()

        for offset in fragment_offsets(
            file_size=len(data),
            fragment_size=FRAGMENT_SIZE,
            stride=STRIDE,
            include_partial_tail=INCLUDE_PARTIAL_TAIL,
        ):
            fragment = data[offset : offset + FRAGMENT_SIZE]

            if len(fragment) < FRAGMENT_SIZE:
                fragment = fragment.ljust(FRAGMENT_SIZE, b"\x00")

            fragment_name = f"frag_{fragment_index}.bin"
            (OUTPUT_DIR / fragment_name).write_bytes(fragment)

            rows.append(
                {
                    "file": fragment_name,
                    "label": label,
                    "source_file": file_path.name,
                }
            )
            fragment_index += 1

    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "source_file"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Classes scanned: {len([p for p in INPUT_ROOT.iterdir() if p.is_dir()])}")
    print(f"Fragments written: {len(rows)}")
    print(f"Fragment size: {FRAGMENT_SIZE} bytes")
    print(f"Stride: {STRIDE} bytes")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Output CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
