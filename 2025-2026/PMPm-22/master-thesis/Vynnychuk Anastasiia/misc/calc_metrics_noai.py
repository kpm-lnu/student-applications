from pathlib import Path

import torch
import json
from ultralytics import YOLO
import sys


def parse_label_file(file_stem: str, start_img_id: int):
    video_uid = file_stem[:5]

    with open(f'./data/CURE-TSD_cut/labels/{video_uid}.txt', 'r') as f:
        lines = f.readlines()[1:]

    annotations = []
    for line in lines:
        parts = line.strip().split('_')
        if len(parts) != 10:
            print(f"Warning: Skipping malformed line: {line.strip()}")
            continue

        frame_num = int(parts[0])
        sign_type = int(parts[1])
        llx, lly, lrx, lry, ulx, uly, urx, ury = map(int, parts[2:])

        x = llx  # left x coordinate
        y = lly  # top y coordinate (smaller y value)
        width = lrx - llx  # right x - left x
        height = uly - lly  # bottom y - top y (larger y - smaller y)

        annotations.append({
            "image_id": start_img_id + frame_num - 1,
            "category_id": dfg_to_gt_class_mapping[sign_type],
            "bbox": [x, y, width, height],
            "area": width * height,
            "iscrowd": 0
        })

    return annotations


allowed_dfg_classes = ["II-30-10", "II-30-30", "II-30-40", "II-30-50", "II-30-60", "II-30-70", "II-7", "II-28", "II-34", "II-35", "II-2",
                       "II-40", "I-10", "II-26", "II-26.1", "II-33", "II-4", "II-1", "III-35"]

gt_classes = [
    {'id': 1, 'name': "II-30"},
    {'id': 2, 'name': "II-7"},
    {'id': 3, 'name': "II-28"},
    {'id': 4, 'name': "II-34"},
    {'id': 5, 'name': "II-35"},
    {'id': 6, 'name': "II-2"},
    {'id': 7, 'name': "II-40"},
    {'id': 8, 'name': "I-10"},
    {'id': 9, 'name': "II-26"},
    {'id': 10, 'name': "II-26.1"},
    {'id': 11, 'name': "II-33"},
    {'id': 12, 'name': "II-4"},
    {'id': 13, 'name': "II-1"},
    {'id': 14, 'name': "III-35"},
]

dfg_to_gt_class_mapping = {
    51: 1,
    52: 1,
    53: 1,
    54: 1,
    55: 1,
    56: 1,
    78: 2,
    49: 3,
    59: 4,
    60: 5,
    43: 6,
    63: 7,
    2: 8,
    47: 9,
    48: 10,
    58: 11,
    62: 12,
    37: 13,
    123: 14
}

model = YOLO('./models/kursova.pt')

dfg_inv_names = {v: k for (k, v) in model.names.items()}
allowed_dfg_class_idxs = [dfg_inv_names[c] for c in allowed_dfg_classes]

frame_num = 1
prev_frame_path = None
video_annotations = None
file_stem = None
img_id = 1

gt_images = []
gt_annotations = []

dt_annotations = []

for frame_result in model.track('./data/CURE-TSD_cut/data2', tracker='bytetrack.yaml', stream=True):
# for frame_result in model.predict('./data/CURE-TSD_cut/data2', stream=True):
    if prev_frame_path is not None and frame_result.path != prev_frame_path:
        frame_num = 1
        prev_frame_path = frame_result.path
        file_stem = None
        video_annotations = None

    if prev_frame_path is None:
        prev_frame_path = frame_result.path

    if file_stem is None:
        file_stem = Path(frame_result.path).stem

    if video_annotations is None:
        video_annotations = parse_label_file(file_stem, img_id)
        gt_annotations.extend(video_annotations)

    img_file_name = f"{file_stem}_{frame_num:04d}"

    gt_images.append({
        "id": img_id,
        "file_name": img_file_name,
        "width": frame_result.orig_shape[1],
        "height": frame_result.orig_shape[0]
    })

    for (cls, conf, xywh) in zip(frame_result.boxes.cls, frame_result.boxes.conf, frame_result.boxes.xywh.to(torch.int32)):
        if cls.item() not in allowed_dfg_class_idxs:
            continue

        dt_annotations.append({
            "image_id": img_id,
            "category_id": dfg_to_gt_class_mapping[cls.item()],
            "bbox": xywh.tolist(),
            "area": xywh[2].item() * xywh[3].item(),
            "iscrowd": 0
        })

    frame_num += 1
    img_id += 1

with open('coco_gt.json', 'w') as f:
    json.dump({
        "images": gt_images,
        "annotations": gt_annotations,
        "categories": gt_classes
    }, f)

with open('coco_dt.json', 'w') as f:
    json.dump(dt_annotations, f)

print(sys.getsizeof(gt_images))
print(sys.getsizeof(gt_annotations))
print(sys.getsizeof(dt_annotations))

input()
