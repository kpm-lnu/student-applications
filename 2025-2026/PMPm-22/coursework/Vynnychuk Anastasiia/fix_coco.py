import json
import os

base_path = "./data"

for dir in os.listdir(base_path):
    if "-" not in dir:
        continue

    coco_file = os.path.join(base_path, dir, "coco.json")
    with open(coco_file) as f:
        coco_obj = json.load(f)

    for ann in coco_obj["annotations"]:
        (x, y, w, h) = ann["bbox"]
        ann["iscrowd"] = 0
        ann["area"] = w * h

    with open(coco_file, "w") as f:
        json.dump(coco_obj, f)
