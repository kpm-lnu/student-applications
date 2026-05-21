from pathlib import Path
import zipfile
import csv
import random
import shutil
import tempfile


ARCHIVES_DIR = Path("napier_archives")
OUTPUT_DIR = Path("dataset/fragments")
OUTPUT_CSV = Path("dataset/labels.csv")

FRAGMENT_SIZE = 1024
FRAGMENTS_PER_CLASS = 2000
MAX_FRAGMENTS_PER_FILE = 20

SKIP_HEAD_RATIO = 0.10
SKIP_TAIL_RATIO = 0.10

RANDOM_SEED = 42


CLASS_ARCHIVES = {
    "text": [
        "TXT-small.zip",
    ],
    "pdf": [
        "PDF-small.zip",
    ],
    # "compressed": [
    #     "ZIP-DEFLATE-tiny.zip",
    #     "GZIP-tiny.zip",
    #     "7ZIP-LZMA2-tiny.zip",
    # ],
    "executable": [
        "exe-small.zip",
    ],
    "image": [
        "JPG-q100-small.zip",
    ],
}


def reset_output():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def valid_offsets(file_size: int):
    if file_size < FRAGMENT_SIZE:
        return []

    start = int(file_size * SKIP_HEAD_RATIO)
    end = int(file_size * (1.0 - SKIP_TAIL_RATIO)) - FRAGMENT_SIZE

    if end <= start:
        start = 0
        end = file_size - FRAGMENT_SIZE

    if end < 0:
        return []

    offsets = list(range(start, end + 1, FRAGMENT_SIZE))
    random.shuffle(offsets)

    return offsets[:MAX_FRAGMENTS_PER_FILE]


def safe_name(name: str):
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def generate_fragments_from_bytes(data: bytes, label: str, source_file: str, start_id: int):
    rows = []
    offsets = valid_offsets(len(data))

    for local_idx, offset in enumerate(offsets):
        fragment = data[offset: offset + FRAGMENT_SIZE]

        if len(fragment) != FRAGMENT_SIZE:
            continue

        frag_name = f"{label}_{start_id + len(rows):06d}.bin"
        frag_path = OUTPUT_DIR / label / frag_name
        frag_path.write_bytes(fragment)

        rows.append({
            "file": str(frag_path),
            "label": label,
            "source_file": source_file,
            "offset": offset,
            "fragment_size": FRAGMENT_SIZE,
        })

    return rows


def process_archive(label: str, archive_path: Path, current_count: int, global_id: int):
    rows = []

    if not archive_path.exists():
        print(f"[warning] missing archive: {archive_path}")
        return rows, current_count, global_id

    print(f"[archive] {label}: {archive_path.name}")

    with zipfile.ZipFile(archive_path, "r") as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        random.shuffle(members)

        for member in members:
            if current_count >= FRAGMENTS_PER_CLASS:
                break

            try:
                data = zf.read(member)
            except Exception as e:
                print(f"  [skip] {member.filename}: {e}")
                continue

            source_name = safe_name(member.filename)

            new_rows = generate_fragments_from_bytes(
                data=data,
                label=label,
                source_file=source_name,
                start_id=global_id,
            )

            remaining = FRAGMENTS_PER_CLASS - current_count
            new_rows = new_rows[:remaining]

            rows.extend(new_rows)
            current_count += len(new_rows)
            global_id += len(new_rows)

            if new_rows:
                print(f"  {source_name}: +{len(new_rows)} fragments, total={current_count}")

    return rows, current_count, global_id


def main():
    random.seed(RANDOM_SEED)
    reset_output()

    all_rows = []

    for label, archives in CLASS_ARCHIVES.items():
        print(f"\n==============================")
        print(f"CLASS: {label}")
        print(f"target: {FRAGMENTS_PER_CLASS}")
        print(f"==============================")

        (OUTPUT_DIR / label).mkdir(parents=True, exist_ok=True)

        class_count = 0
        global_id = 0

        archive_list = archives[:]
        random.shuffle(archive_list)

        for archive_name in archive_list:
            if class_count >= FRAGMENTS_PER_CLASS:
                break

            archive_path = ARCHIVES_DIR / archive_name

            rows, class_count, global_id = process_archive(
                label=label,
                archive_path=archive_path,
                current_count=class_count,
                global_id=global_id,
            )

            all_rows.extend(rows)

        if class_count < FRAGMENTS_PER_CLASS:
            print(f"[warning] class {label}: only {class_count}/{FRAGMENTS_PER_CLASS} fragments generated")
        else:
            print(f"[done] class {label}: {class_count} fragments")

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "label",
                "source_file",
                "offset",
                "fragment_size",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nDONE")
    print(f"Fragments dir: {OUTPUT_DIR}")
    print(f"Labels CSV: {OUTPUT_CSV}")
    print(f"Total fragments: {len(all_rows)}")


if __name__ == "__main__":
    main()