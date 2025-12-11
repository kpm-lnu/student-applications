import json
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

datasets = [
    {"name": "base", "path": "./data/dfg-ultralytics"},
    {"name": "color", "path": "./data/dfg-color"},
    {"name": "full", "path": "./data/dfg-fullaug"},
    {"name": "image compression", "path": "./data/dfg-image_compression"},
    {"name": "image quality", "path": "./data/dfg-image_quality"},
    {"name": "lightning", "path": "./data/dfg-lightning"},
    {"name": "noise", "path": "./data/dfg-noise"},
    {"name": "weather", "path": "./data/dfg-weather"},
]

for dataset in datasets:
    # base_path = "./data/dfg-ultralytics"
    coco_gt = COCO(os.path.join(dataset["path"], "coco.json"))
    coco_dt = coco_gt.loadRes(os.path.join(dataset["path"], "annotations.json"))

    E = COCOeval(coco_gt, coco_dt, "bbox")

    E.evaluate()
    E.accumulate()
    E.summarize()

    dataset["stats"] = E.stats.tolist()

with open("./data/stats_by_dataset.json", "w") as f:
    json.dump(datasets, f)
