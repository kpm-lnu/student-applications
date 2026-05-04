from pathlib import Path
import torch
import json
from ultralytics import YOLO
import argparse


def calculate_letterbox_shape(orig_shape, imgsz=640, stride=32):
    """
    Calculate the actual inference shape after letterbox resizing.

    Args:
        orig_shape: (height, width) of original image
        imgsz: target size (default 640)
        stride: model stride (default 32 for YOLO)

    Returns:
        (inf_height, inf_width) - actual tensor dimensions fed to model
    """
    h, w = orig_shape

    # Scale to make longest side = imgsz
    scale = imgsz / max(h, w)
    new_h = h * scale
    new_w = w * scale

    # Round up to nearest multiple of stride
    inf_h = int((new_h + stride - 1) // stride * stride)
    inf_w = int((new_w + stride - 1) // stride * stride)

    return inf_w, inf_h


imgsz = 640

allowed_dfg_classes = ["II-30-10", "II-30-30", "II-30-40", "II-30-50", "II-30-60", "II-30-70", "II-7", "II-28", "II-34", "II-35", "II-2",
                       "II-40", "I-10", "II-26", "II-26.1", "II-33", "II-4", "II-1", "III-35"]

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


def det_stream(model: YOLO, data_dir: str, batch_size: int): return model.predict(
    data_dir, stream=True, imgsz=imgsz, conf=0, batch=batch_size, save=True)


def track_stream(model: YOLO, data_dir: str, batch_size: int): return model.track(data_dir,
                                                                                  tracker='bytetrack.yaml',
                                                                                  stream=True,
                                                                                  imgsz=imgsz,
                                                                                  save=True,
                                                                                  batch=batch_size)


stream_funcs = {
    'det': det_stream,
    'track': track_stream
}

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
                    default='./models/kursova.pt', help='Path to the YOLO model')
parser.add_argument('--data_dir', type=str, default='./data/CURE-TSD_cut/data2',
                    help='Path to the input video frames')
parser.add_argument('--output_file', type=str, default='./data/out/coco/coco_dt.json',
                    help='Path to save the output COCO JSON files')
parser.add_argument('--gt_file', type=str, default='./data/out/coco/coco_gt.json',
                    help='Path to the ground truth COCO JSON file')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size for inference')
parser.add_argument('--det_type', type=str, choices=[
                    'det', 'track'], default='track', help='Whether to run detection or tracking')
args = parser.parse_args()

with open(args.gt_file, 'r') as f:
    gt_data = json.load(f)

filename_to_img_id = {img['file_name']: img['id'] for img in gt_data['images']}

model = YOLO(args.model)

dfg_inv_names = {v: k for (k, v) in model.names.items()}
allowed_dfg_class_idxs = [dfg_inv_names[c] for c in allowed_dfg_classes]

frame_num = 1
prev_frame_path = None
video_annotations = None
file_stem = None

dt_annotations = []

no_id = []

stream_func = stream_funcs[args.det_type]

for frame_result in stream_func(model, args.data_dir, args.batch_size):
    if prev_frame_path is not None and frame_result.path != prev_frame_path:
        frame_num = 1
        prev_frame_path = frame_result.path
        file_stem = None
        video_annotations = None

    if prev_frame_path is None:
        prev_frame_path = frame_result.path

    if file_stem is None:
        file_stem = Path(frame_result.path).stem

    img_file_name = f"{file_stem}_{frame_num:04d}"
    img_id = filename_to_img_id.get(img_file_name)

    if img_id is None:
        print(f"Warning: No img_id found for {img_file_name}, skipping frame")
        no_id.append(img_file_name)
        frame_num += 1
        continue

    for (cls, conf, xywh) in zip(frame_result.boxes.cls, frame_result.boxes.conf, frame_result.boxes.xywh.to(torch.int32)):
        if cls.item() not in allowed_dfg_class_idxs:
            continue

        inf_w, inf_h = calculate_letterbox_shape(
            frame_result.orig_shape, imgsz=imgsz)
        orig_h, orig_w = frame_result.orig_shape
        scale_w = inf_w / orig_w
        scale_h = inf_h / orig_h

        scale_w = inf_w / frame_result.orig_shape[1]
        scale_h = inf_h / frame_result.orig_shape[0]
        inf_area = (xywh[2] * scale_w) * (xywh[3] * scale_h)

        dt_annotations.append({
            "image_id": img_id,
            "category_id": dfg_to_gt_class_mapping[cls.item()],
            "bbox": xywh.tolist(),
            "area": inf_area.item(),
            "score": conf.item(),
            "iscrowd": 0
        })

    frame_num += 1

with open(args.output_file, 'w') as f:
    json.dump(dt_annotations, f)

print(f"Frames with no img_id: {no_id}")
