import json
import os
import shutil

from sklearn.model_selection import train_test_split

os.makedirs("./data/dfg-ultralytics")
os.makedirs("./data/dfg-ultralytics/images/train")
os.makedirs("./data/dfg-ultralytics/images/val")
os.makedirs("./data/dfg-ultralytics/images/test")
os.makedirs("./data/dfg-ultralytics/labels/train")
os.makedirs("./data/dfg-ultralytics/labels/val")
os.makedirs("./data/dfg-ultralytics/labels/test")

with open("./data/dfg/train.json", "r") as f:
    train_annotations_json = json.load(f)
train_annotations_json.keys()

train_images = train_annotations_json["images"][:]
train_annotations = {}
for annotation in train_annotations_json["annotations"]:
    if annotation["image_id"] not in train_annotations:
        train_annotations[annotation["image_id"]] = []
    train_annotations[annotation["image_id"]].append(annotation)

train_images, val_images = train_test_split(
    train_images, test_size=0.2, random_state=42
)

for image in train_images:
    w = image["width"]
    h = image["height"]
    fname = image["file_name"].split(".")[0]

    if image["id"] not in train_annotations:
        with open(f"./data/dfg-ultralytics/labels/train/{fname}.txt", "w") as f:
            f.write("")
        continue

    annotations = [
        "{} {} {} {} {}".format(
            ann["category_id"],
            ann["bbox"][0] / w + ann["bbox"][2] / (2 * w),
            ann["bbox"][1] / h + ann["bbox"][3] / (2 * h),
            ann["bbox"][2] / w,
            ann["bbox"][3] / h,
        )
        for ann in train_annotations[image["id"]]
        if ann["bbox"][2] != -1
    ]
    with open(f"./data/dfg-ultralytics/labels/train/{fname}.txt", "w") as f:
        f.write("\n".join(annotations))

for image in val_images:
    w = image["width"]
    h = image["height"]
    fname = image["file_name"].split(".")[0]

    if image["id"] not in train_annotations:
        with open(f"./data/dfg-ultralytics/labels/val/{fname}.txt", "w") as f:
            f.write("")
        continue

    annotations = [
        "{} {} {} {} {}".format(
            ann["category_id"],
            ann["bbox"][0] / w + ann["bbox"][2] / (2 * w),
            ann["bbox"][1] / h + ann["bbox"][3] / (2 * h),
            ann["bbox"][2] / w,
            ann["bbox"][3] / h,
        )
        for ann in train_annotations[image["id"]]
        if ann["bbox"][2] != -1
    ]
    with open(f"./data/dfg-ultralytics/labels/val/{fname}.txt", "w") as f:
        f.write("\n".join(annotations))

for image in train_images:
    shutil.copy(
        f"./data/dfg/JPEGImages/{image['file_name']}",
        f"./data/dfg-ultralytics/images/train/{image['file_name']}",
    )

for image in val_images:
    shutil.copy(
        f"./data/dfg/JPEGImages/{image['file_name']}",
        f"./data/dfg-ultralytics/images/val/{image['file_name']}",
    )

with open("./data/dfg/test.json", "r") as f:
    test_annotations = json.load(f)

test_images = {x["id"]: x for x in test_annotations["images"]}
for annotation in test_annotations["annotations"]:
    if annotation["bbox"][2] == -1:
        continue
    image = test_images[annotation["image_id"]]
    fname = image["file_name"].split(".")[0]
    x = (annotation["bbox"][0] + annotation["bbox"][2] / 2) / image["width"]
    y = (annotation["bbox"][1] + annotation["bbox"][3] / 2) / image["height"]
    w = annotation["bbox"][2] / image["width"]
    h = annotation["bbox"][3] / image["height"]
    with open(f"./data/dfg-ultralytics/labels/test/{fname}.txt", "a") as f:
        f.write("{} {} {} {} {}\n".format(annotation["category_id"], x, y, w, h))

for image in test_annotations["images"]:
    shutil.copy(
        f"./data/dfg/JPEGImages/{image['file_name']}",
        f"./data/dfg-ultralytics/images/test/{image['file_name']}",
    )

with open("./data/dfg-ultralytics/data.yaml", "w") as f:
    f.write("path: ./data/dfg-ultralytics/\n")
    f.write("train: ./images/train/\n")
    f.write("val: ./images/val/\n")
    f.write("test: ./images/test/\n")
    f.write("\n")

    f.write("names:\n")
    for cat in train_annotations_json["categories"]:
        f.write(f"\t{cat['id']}: {cat['name']}\n")
