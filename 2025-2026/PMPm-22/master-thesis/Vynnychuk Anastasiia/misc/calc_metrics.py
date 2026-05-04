import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json
from pathlib import Path

allowed_classes = ["II-30-40", "II-7", "II-28", "II-34", "II-35", "II-2",
                   "II-40", "I-10", "II-26", "II-26.1", "II-33", "II-4", "II-1", "III-35"]

# Load tracking results
tracking_result = torch.load('tracking_result.pt', weights_only=False)

inv_names = {v: k for (k, v) in tracking_result[0].names.items()}
allowed_class_idxs = [inv_names[c] for c in allowed_classes]


def parse_label_file(label_path, class_names, allowed_classes):
    """Parse a label file and return ground truth annotations (filtered by allowed classes)."""
    annotations = []
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Skip header
    for line in lines[1:]:
        parts = line.strip().split('_')
        if len(parts) != 10:
            continue

        frame_num = int(parts[0])
        model_class_id = int(parts[1])  # This is the model's class index

        # Map to class name
        if model_class_id not in class_names:
            continue

        class_name = class_names[model_class_id]

        # Filter by allowed classes
        if class_name not in allowed_classes:
            continue

        llx, lly, lrx, lry, ulx, uly, urx, ury = map(int, parts[2:])

        # Convert to [x, y, width, height] format
        # Note: In the label format, "lower" refers to smaller y (top), "upper" to larger y (bottom)
        x = llx  # left x coordinate
        y = lly  # top y coordinate (smaller y value)
        width = lrx - llx  # right x - left x
        height = uly - lly  # bottom y - top y (larger y - smaller y)

        annotations.append({
            'frame': frame_num,
            'class_name': class_name,
            'bbox': [x, y, width, height],
            'area': width * height
        })

    return annotations


def extract_video_id(path):
    """Extract video ID (first two numbers) from filename like '01_04_01_02_01.mp4'."""
    filename = Path(path).stem  # Get filename without extension
    parts = filename.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return None


def create_coco_format(tracking_result, allowed_classes):
    """Convert tracking results and ground truth to COCO format."""

    # Initialize COCO structures
    coco_gt = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    coco_dt = []  # Detection results

    # Create category mapping
    # First, get the class names from the model
    if len(tracking_result) > 0:
        class_names = tracking_result[0].names
    else:
        print("No tracking results found!")
        return None, None

    # Create category ID mapping for allowed classes
    category_map = {}
    for idx, class_name in enumerate(allowed_classes):
        category_id = idx + 1
        coco_gt['categories'].append({
            'id': category_id,
            'name': class_name
        })
        category_map[class_name] = category_id

    # Reverse lookup: model class_id -> class_name -> category_id
    model_class_to_category = {}
    for class_id, class_name in class_names.items():
        if class_name in category_map:
            model_class_to_category[class_id] = category_map[class_name]

    annotation_id = 1
    image_id = 1

    # Group results by video
    video_results = {}
    for result in tracking_result:
        video_id = extract_video_id(result.path)
        if video_id not in video_results:
            video_results[video_id] = []
        video_results[video_id].append(result)

    print(f"Found {len(video_results)} videos in tracking results")

    # Process each video
    for video_id, results in video_results.items():
        print(f"\nProcessing video: {video_id}")

        # Load ground truth labels
        label_path = f"./data/CURE-TSD_cut/labels/{video_id}.txt"
        if not os.path.exists(label_path):
            print(f"  Warning: Label file not found: {label_path}")
            continue

        gt_annotations = parse_label_file(
            label_path, class_names, allowed_classes)
        gt_by_frame = {}
        for ann in gt_annotations:
            frame = ann['frame']
            if frame not in gt_by_frame:
                gt_by_frame[frame] = []
            gt_by_frame[frame].append(ann)

        print(
            f"  Loaded {len(gt_annotations)} ground truth annotations (filtered) across {len(gt_by_frame)} frames")

        # Process each frame result
        for frame_idx, result in enumerate(results, start=1):
            # Add image to COCO GT
            coco_gt['images'].append({
                'id': image_id,
                'file_name': f"{video_id}_frame_{frame_idx:03d}",
                'width': result.orig_shape[1] if hasattr(result, 'orig_shape') else 1920,
                'height': result.orig_shape[0] if hasattr(result, 'orig_shape') else 1080
            })

            # Add ground truth annotations for this frame
            if frame_idx in gt_by_frame:
                for gt_ann in gt_by_frame[frame_idx]:
                    coco_gt['annotations'].append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': category_map[gt_ann['class_name']],
                        'bbox': gt_ann['bbox'],
                        'area': gt_ann['area'],
                        'iscrowd': 0
                    })
                    annotation_id += 1

            # Add predictions (filtered by allowed classes)
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                classes = result.boxes.cls.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()

                for box, cls, score in zip(boxes, classes, scores):
                    cls_int = int(cls)

                    # Only include if class is in allowed classes
                    if cls_int in model_class_to_category:
                        x1, y1, x2, y2 = box
                        width = x2 - x1
                        height = y2 - y1

                        coco_dt.append({
                            'image_id': image_id,
                            'category_id': model_class_to_category[cls_int],
                            'bbox': [float(x1), float(y1), float(width), float(height)],
                            'score': float(score)
                        })

            image_id += 1

    print(f"\n{'='*60}")
    print(f"COCO Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total images: {len(coco_gt['images'])}")
    print(f"Total ground truth annotations: {len(coco_gt['annotations'])}")
    print(f"Total predictions: {len(coco_dt)}")
    print(f"Categories: {len(coco_gt['categories'])}")

    return coco_gt, coco_dt


# Create COCO format data
coco_gt_data, coco_dt_data = create_coco_format(
    tracking_result, allowed_classes)

if coco_gt_data is None:
    print("Failed to create COCO format data")
    exit()

# Save to temporary files for COCO API
with open('coco_gt_temp.json', 'w') as f:
    json.dump(coco_gt_data, f)

with open('coco_dt_temp.json', 'w') as f:
    json.dump(coco_dt_data, f)

# Load with COCO API
coco_gt = COCO('coco_gt_temp.json')
coco_dt = coco_gt.loadRes('coco_dt_temp.json')

# Run COCO evaluation
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Clean up temporary files
os.remove('coco_gt_temp.json')
os.remove('coco_dt_temp.json')

print("\nEvaluation complete!")
