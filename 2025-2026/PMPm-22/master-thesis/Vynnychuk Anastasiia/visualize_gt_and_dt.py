import cv2 as cv
from ultralytics import YOLO
import os

def read_bboxes(label_path) -> dict[int, list[tuple[int, int, int, int, int]]]:
    """Returns dict mapping frame_num to list of bboxes as (x, y, width, height, class_id)"""
    bboxes_by_frame = {}
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split('_')

        frame_num = int(parts[0])
        sign_type = int(parts[1])
        llx, lly, lrx, lry, ulx, uly, urx, ury = map(int, parts[2:])

        x = llx  # left x coordinate
        y = lly  # top y coordinate (smaller y value)
        width = lrx - llx  # right x - left x
        height = uly - lly  # bottom y - top y (larger y - smaller y)

        if frame_num not in bboxes_by_frame:
            bboxes_by_frame[frame_num] = []

        bboxes_by_frame[frame_num].append((x, y, width, height, sign_type))

    return bboxes_by_frame


video_root = './data/CURE-TSD_orig/data'
label_root = './data/CURE-TSD_orig/labels'

video_file_name = '01_23_00_00_00.mp4'


video_path = os.path.join(video_root, video_file_name)
label_path = os.path.join(label_root, video_file_name[:5] + '.txt')

model = YOLO('./models/kursova.pt')

cap = cv.VideoCapture(video_path)
canvas_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
canvas_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
cap.release()

gt_bboxes = read_bboxes(label_path)

output_path = './data/out/' + video_file_name[:5] + '_gt_and_dt.mp4'

# Setup video writer
cap = cv.VideoCapture(video_path)
fps = cap.get(cv.CAP_PROP_FPS)
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(output_path, fourcc, fps, (canvas_width, canvas_height))

frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    dt = model.predict(frame, conf=0.2)

    # print(frame.shape)

    dt_boxes = dt[0].boxes.xyxy.cpu().numpy() if len(dt) > 0 and dt[0].boxes is not None else []

    # Draw bounding boxes if they exist for this frame
    if frame_num in gt_bboxes and gt_bboxes[frame_num] is not None:
        for bbox in gt_bboxes[frame_num]:
            x, y, w, h, class_id = bbox
            if w * h < 197:
                continue
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    for bbox in dt_boxes:
        x1, y1, x2, y2 = map(int, bbox)
        frame = cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Write the frame to output video
    out.write(frame)

    frame_num += 1

cap.release()
out.release()
