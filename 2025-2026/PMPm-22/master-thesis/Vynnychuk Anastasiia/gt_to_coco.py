import os
import json
from pathlib import Path
import cv2 as cv

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

video_root = './data/CURE-TSD_cut/data2'

gt_images = []
gt_annotations = []

img_id = 1

for vide_file_name in sorted(os.listdir(video_root)):
    video_path = os.path.join(video_root, vide_file_name)
    file_stem = Path(video_path).stem

    video_annotations = parse_label_file(file_stem, img_id)
    gt_annotations.extend(video_annotations)

    cap = cv.VideoCapture(video_path)
    frame_num = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_file_name = f"{file_stem}_{frame_num:04d}"

        gt_images.append({
            "id": img_id,
            "file_name": img_file_name,
            "width": frame.shape[1],
            "height": frame.shape[0]
        })

        frame_num += 1
        img_id += 1

with open('coco_gt.json', 'w') as f:
    json.dump({
        "images": gt_images,
        "annotations": gt_annotations,
        "categories": gt_classes
    }, f)
