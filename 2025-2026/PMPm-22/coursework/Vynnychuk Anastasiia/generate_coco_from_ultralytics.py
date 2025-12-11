import json
import os
from typing import Generator

import cv2 as cv
import yaml
from ultralytics.models import YOLO


def get_coco_img_data(idx: int, file_name: str, path: str) -> dict:
    img = cv.imread(os.path.join(path, file_name))
    if img is None:
        raise Exception(f"image {file_name} cannot be opened")
    (h, w, _) = img.shape
    return {"id": idx, "file_name": file_name, "width": w, "height": h}


def get_coco_ann_data(
    file_name: str, path: str, images_dict: dict[str, dict]
) -> Generator[dict, None, None]:
    image_wo_ext = file_name.split(".")[0]
    image_data = images_dict[image_wo_ext]
    image_id = image_data["id"]
    with open(os.path.join(path, file_name), "r") as f:
        lines = f.readlines()
    for line in lines:
        (cls, xcenterrel, ycenterrel, wrel, hrel) = [float(x) for x in line.split(" ")]
        xminrel = xcenterrel - (wrel / 2.0)
        xmin = int(xminrel * image_data["width"])
        yminrel = ycenterrel - (hrel / 2.0)
        ymin = int(yminrel * image_data["height"])
        w = int(wrel * image_data["width"])
        h = int(hrel * image_data["height"])

        yield {
            "image_id": image_id,
            "category_id": int(cls),
            "bbox": [xmin, ymin, w, h],
        }


dataset_path = "./data/dfg-noise/data.yaml"
coco_output_file = "./data/dfg-noise/coco.json"

with open(dataset_path, "r") as f:
    dataset_data = yaml.full_load(f)

base_path = dataset_data["path"]
images_path = os.path.join(base_path, dataset_data["test"])
labels_path = os.path.join(base_path, "./labels/test")

coco_images = [
    get_coco_img_data(idx, file_name, images_path)
    for (idx, file_name) in enumerate(os.listdir(images_path), 1)
]
images_dict = {img["file_name"].split(".")[0]: img for img in coco_images}
coco_categories = [
    {"id": id, "name": name, "supercategory": "traffic_sign"}
    for (id, name) in dataset_data["names"].items()
]
coco_annotations = []
idx = 1
for file_name in os.listdir(labels_path):
    for ann in get_coco_ann_data(file_name, labels_path, images_dict):
        coco_annotations.append({**ann, "id": idx})
        idx += 1

coco_obj = {
    "images": coco_images,
    "categories": coco_categories,
    "annotations": coco_annotations,
}

with open(coco_output_file, "w") as f:
    json.dump(coco_obj, f)

exit()

images_dict_reverse = {name: idx for (idx, name) in images_dict.items()}

model = YOLO("./runs/train2/weights/best.pt")

results = model(images_path, stream=True)

annotations = []
images = []

for r in results:
    image_name = r.path.split("/")[-1].split(".")[0]
    for cls, bbox in zip(r.boxes.cls, r.boxes.xywh):
        annotations.append(
            {
                "category_id": int(cls),
                "bbox": bbox.numpy().astype(int).tolist(),
                "image_id": 0,
            }
        )
    print(annotations)
    break
