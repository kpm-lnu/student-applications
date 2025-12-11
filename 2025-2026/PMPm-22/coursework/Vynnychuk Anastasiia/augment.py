import json
import os
import random

import albumentations as A
import cv2 as cv
import numpy as np


def save_augmented_in_ultralytics(
    images, annotations_by_img, pipeline, dataset_path, show_only=False, limit=None
):
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "images/test"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "labels"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "labels/test"), exist_ok=True)

    idx = 0

    for img_id, annotations in annotations_by_img.items():
        if limit is not None and idx >= limit:
            break
        idx += 1

        image = cv.imread(
            os.path.join(base_jpeg_path, images[img_id]), cv.IMREAD_COLOR_RGB
        )
        if image is None:
            continue

        image_file_name = images[img_id].split(".")[0]
        (h, w, _) = image.shape

        augmentations_count = random.randint(0, 5)
        for i in range(augmentations_count):
            augmented = pipeline(image=image)

            if show_only:
                to_show = cv.cvtColor(augmented["image"], cv.COLOR_RGB2BGR)
                for bbox in augmented["bboxes"]:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (p1[0] + int(bbox[2]), p1[1] + int(bbox[3]))
                    to_show = cv.rectangle(to_show, p1, p2, [255, 0, 0, 255])
                cv.imshow("augmented", to_show)
                break

            augmented_file_name = f"{image_file_name}_{i}"
            cv.imwrite(
                os.path.join(dataset_path, f"images/test/{augmented_file_name}.jpg"),
                cv.cvtColor(augmented["image"], cv.COLOR_RGB2BGR),
            )

            txt_annotations = [
                "{} {} {} {} {}".format(
                    ann["category_id"],
                    ann["bbox"][0] / w + ann["bbox"][2] / (2 * w),
                    ann["bbox"][1] / h + ann["bbox"][3] / (2 * h),
                    ann["bbox"][2] / w,
                    ann["bbox"][3] / h,
                )
                for ann in annotations
                if ann["bbox"][2] != -1
            ]
            with open(
                os.path.join(dataset_path, f"labels/test/{augmented_file_name}.txt"),
                "w",
            ) as f:
                f.write("\n".join(txt_annotations))

        if show_only:
            break

    if show_only:
        cv.waitKey()

    with open(os.path.join(dataset_path, "data.yaml"), "w") as f:
        f.write(f"path: {dataset_path}\n")
        f.write("test: ./images/test/\n")
        f.write("\n")

        f.write("names:\n")
        for cat in test_json["categories"]:
            f.write(f"\t{cat['id']}: {cat['name']}\n")


base_jpeg_path = "./data/dfg/JPEGImages"
json_config_path = "./data/dfg/test.json"

testaug_path = "./data/dfg-testaug"

# Weather conditions (apply one at a time for cleaner analysis)
weather_augmentations = lambda p: A.OneOf(
    [
        A.RandomRain(
            slant_range=(-10, 10),
            drop_length=20,
            drop_width=1,
            drop_color=(200, 200, 200),
            blur_value=3,
            brightness_coefficient=0.9,
            rain_type="drizzle",  # or 'heavy' for more extreme
            p=1.0,
        ),
        A.RandomFog(fog_coef_range=(0.1, 0.2), alpha_coef=0.1, p=1.0),
        A.RandomSnow(
            snow_point_range=(0.1, 0.3),
            brightness_coeff=1.5,
            p=1.0,
        ),
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_range=(0, 1),
            num_flare_circles_range=(4, 8),
            src_radius=200,
            p=1.0,
        ),
    ],
    p=p,
)

# Lighting variations (very common in real scenarios)
lightning_augmentations = lambda p: A.OneOf(
    [
        A.RandomBrightnessContrast(
            brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=1.0
        ),
        A.RandomToneCurve(scale=0.3, p=1.0),
        A.RandomGamma(gamma_limit=(70, 130), p=1.0),
    ],
    p=p,
)

# Image quality degradation (camera issues, distance)
image_quality_augmentations = lambda p: A.OneOf(
    [
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.MotionBlur(blur_limit=7, p=1.0),
        A.MedianBlur(blur_limit=5, p=1.0),
    ],
    p=p,
)

# Compression artifacts (dashcam footage, transmitted images)
image_compression_augmentations = lambda p: A.ImageCompression(
    quality_range=(60, 90), p=p
)

# Noise (low light conditions, sensor noise)
noise_augmentations = lambda p: A.OneOf(
    [
        A.GaussNoise(std_range=(0.05, 0.15), p=1.0),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
    ],
    p=p,
)

# Color shifts (time of day, camera white balance)
color_augmentations = lambda p: A.HueSaturationValue(
    hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=p
)

full_pipeline = A.Compose(
    [
        weather_augmentations(0.4),
        lightning_augmentations(0.5),
        image_quality_augmentations(0.3),
        image_compression_augmentations(0.2),
        noise_augmentations(0.25),
        color_augmentations(0.3),
    ],
    # bbox_params=A.BboxParams(
    #     format="coco", label_fields=["class_labels"], min_visibility=0.5
    # ),
)

with open(json_config_path) as f:
    test_json = json.load(f)
images = {img["id"]: img["file_name"] for img in test_json["images"]}
annotations_by_img = dict()
for ann in test_json["annotations"]:
    if ann["image_id"] not in annotations_by_img:
        annotations_by_img[ann["image_id"]] = []
    annotations_by_img[ann["image_id"]].append(ann)

# save_augmented_in_ultralytics(
#     images, annotations_by_img, weather_augmentations(1), "./data/dfg-weather"
# )
# save_augmented_in_ultralytics(
#     images,
#     annotations_by_img,
#     lightning_augmentations(1),
#     "./data/dfg-lightning",
# )
# save_augmented_in_ultralytics(
#     images,
#     annotations_by_img,
#     image_quality_augmentations(1),
#     "./data/dfg-image_quality",
# )
# save_augmented_in_ultralytics(
#     images,
#     annotations_by_img,
#     image_compression_augmentations(1),
#     "./data/dfg-image_compression",
# )
# save_augmented_in_ultralytics(
#     images, annotations_by_img, noise_augmentations(1), "./data/dfg-noise"
# )
# save_augmented_in_ultralytics(
#     images, annotations_by_img, color_augmentations(1), "./data/dfg-color"
# )
save_augmented_in_ultralytics(
    images, annotations_by_img, full_pipeline, "./data/dfg-fullaug"
)
