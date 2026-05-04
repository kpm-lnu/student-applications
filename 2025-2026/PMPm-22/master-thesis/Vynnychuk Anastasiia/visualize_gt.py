import cv2 as cv
import os
from tqdm import tqdm


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

# Get all video files from video_root
video_files = sorted([f for f in os.listdir(video_root) if f.endswith('.mp4')])

# Create output directory if it doesn't exist
output_dir = './data/out/gt/'
os.makedirs(output_dir, exist_ok=True)

# Process each video file
for video_file_name in tqdm(video_files, desc="Processing videos"):
    video_path = os.path.join(video_root, video_file_name)
    label_path = os.path.join(label_root, video_file_name[:5] + '.txt')
    
    # Skip if label file doesn't exist
    if not os.path.exists(label_path):
        continue
    
    # Get canvas size from video
    cap = cv.VideoCapture(video_path)
    canvas_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    canvas_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    bboxes = read_bboxes(label_path)
    
    output_path = os.path.join(output_dir, video_file_name)
    
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
    
        frame = cv.putText(frame, str(frame_num), (100, 100), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)
    
        # Draw bounding boxes if they exist for this frame
        if frame_num in bboxes and bboxes[frame_num] is not None:
            for bbox in bboxes[frame_num]:
                x, y, w, h, class_id = bbox
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
        # Write the frame to output video
        out.write(frame)
    
        frame_num += 1
    
    cap.release()
    out.release()
