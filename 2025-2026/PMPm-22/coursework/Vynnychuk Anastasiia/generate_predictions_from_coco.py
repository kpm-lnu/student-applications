import json
import os

from tqdm import tqdm
from ultralytics.models import YOLO

coco_file = "./data/dfg-weather/coco.json"
img_path = "./data/dfg-weather/images/test/"
model_weights = "./runs/train2/weights/best.pt"
results_file = os.path.join("/".join(coco_file.split("/")[:-1]), "annotations.json")

model = YOLO(model_weights)

with open(coco_file, "r") as f:
    coco_obj = json.load(f)

annotations = []

for image in tqdm(coco_obj["images"]):
    image_file_path = os.path.join(img_path, image["file_name"])

    outputs = model.predict(
        image_file_path, verbose=False, imgsz=640, augment=False, half=False
    )

    if outputs[0].boxes is None:
        continue

    for cls, conf, bbox in zip(
        outputs[0].boxes.cls, outputs[0].boxes.conf, outputs[0].boxes.xywh
    ):
        xcenter, ycenter, w, h = bbox.cpu().numpy().tolist()
        annotations.append(
            {
                "category_id": int(cls),
                "bbox": [xcenter - w / 2.0, ycenter - h / 2.0, w, h],
                "image_id": image["id"],
                "score": conf.item(),
                "iscrowd": 0,
            }
        )

with open(results_file, "w") as f:
    json.dump(annotations, f)
