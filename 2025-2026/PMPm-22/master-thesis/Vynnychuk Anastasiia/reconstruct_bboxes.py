import os
import numpy as np
import cv2 as cv
from argparse import ArgumentParser
import json

from belgiumts_lib import CameraData, ConfigFilePaths, Sign, VehiclePose, parse_poles

def load_config(seq_id: int, camera_id: int) -> ConfigFilePaths:
    sequences_root = './data/belgiumts/sequences/'
    seq_root = os.path.join(sequences_root, f'Seq{seq_id:02d}')
    video_files_path = os.path.join(seq_root, f'{camera_id:02d}')
    gt_file_path = os.path.join(sequences_root, f'sequence{seq_id}_GT.txt')
    camera_set_file_path = os.path.join(seq_root, 'camera_set.txt')
    poses_file_path = os.path.join(
        seq_root, [x for x in os.listdir(seq_root) if x.endswith('.poses')][0])

    return ConfigFilePaths(
        video_files_path=video_files_path,
        gt_file_path=gt_file_path,
        camera_set_file_path=camera_set_file_path,
        poses_file_path=poses_file_path
    )

def parse_poses(poses_file_path: str) -> dict[int, VehiclePose]:
    with open(poses_file_path, 'r') as f:
        lines = f.readlines()

    poses_lines = [l.strip() for l in lines[4:] if l.strip()]
    poses = {int((parts := line.split())[0]): VehiclePose(
        R_vehicle=np.array(parts[1:10], dtype=float).reshape(3, 3),
        t_vehicle=np.array(parts[10:13], dtype=float)
    ) for line in poses_lines}

    return poses

def parse_camera_data(camera_set_file_path: str, camera_id: int) -> CameraData:
    # TODO: parse the file, for now hardcoded for camera 01
    R_cam_str = """
    0.26074615509705534322 -0.23937308575402516109 0.93526037466509825968
    -0.96532572405317029762 -0.077247832271888527966 0.24935721142673666906
    0.012557431398436305625 -0.96784983247706957155 -0.25121507257882286224
    """
    R_cam = np.array(R_cam_str.split(), dtype=float).reshape(3, 3)

    K_str = """
    1401.4490697891651507 -0.13022024734980200411 813.55302388514940048
    0 1403.2649118299896145 598.60305944790320609
    0 0 1
    """
    K = np.array(K_str.split(), dtype=float).reshape(3, 3)

    t_cam_str = """
    1.506889599053904405 -0.3303040418872391637 -0.082612542681732570315
    """
    t_cam = np.array(t_cam_str.split(), dtype=float)

    distortion_coeffs_str = """
    -0.19926509974069675502 0.13345850034658560124 0.061006889380811869794
    """
    distorsion_coeffs_components = distortion_coeffs_str.split()
    distortion_coeffs = np.array([distorsion_coeffs_components[0], distorsion_coeffs_components[1], 0, 0, distorsion_coeffs_components[2]], dtype=float)

    return CameraData(K=K, R=R_cam, t=t_cam, distorsion_matrix=distortion_coeffs, dimensions=(1628, 1236))

parser = ArgumentParser()
parser.add_argument('--seq_id', type=int, required=True,
                    help='Sequence ID to visualize')
parser.add_argument('--camera_id', type=int, required=True,
                    help='Camera ID to visualize')
parser.add_argument('--visualize', action='store_true', help='Whether to visualize the results')
parser.add_argument('--video_output_path', type=str, default=None, help='Path to save the output video (if not provided, will not save)')
parser.add_argument('--output_path', type=str, default=None, help='Path to save the bboxes as json')
args = parser.parse_args()

seq_id = args.seq_id
camera_id = args.camera_id
visualize = args.visualize
video_output_path = args.video_output_path
output_path = args.output_path
detector_dimensions = 640

config = load_config(seq_id, camera_id)

poses = parse_poses(config.poses_file_path)
camera_data = parse_camera_data(config.camera_set_file_path, camera_id)
poles = parse_poles(config.gt_file_path)

scale_factor = detector_dimensions / max(camera_data.dimensions)

signs = [Sign(pole.idx, sign.type, [bbox for bbox in pole.bboxes if bbox.sign_idx == sign.idx and bbox.camera == camera_id]) for pole in poles for sign in pole.signs]
for sign in signs:
    sign.calculate_geometry(poses, camera_data)

if video_output_path is not None:
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(video_output_path, fourcc, 30, (camera_data.dimensions[0], camera_data.dimensions[1]))

annotations = []

for imfile in sorted(os.listdir(config.video_files_path)):
    frame_num = int(imfile.split('.')[1])
    img_path = os.path.join(config.video_files_path, imfile)
    if visualize or video_output_path is not None:
        img = cv.imread(img_path)

    for sign in signs:
        gen_bbox = sign.project_bbox(camera_data, poses[frame_num], camera_data.dimensions[0], camera_data.dimensions[1])
        if gen_bbox is not None:
            x1, y1, x2, y2 = map(int, gen_bbox)
            width = x2 - x1
            height = y2 - y1
            area = width * height
            if area * scale_factor > 15**2:
                if visualize or video_output_path is not None:
                    img = cv.putText(img, f"{sign.type} (pole {sign.pole_idx})", (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                    img = cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                annotations.append({
                    'image': imfile,
                    'class': sign.type,
                    'bbox': [x1 + width/2, y1 + height/2, width, height]
                })

        for bbox in sign.bboxes:
            if bbox.frame != frame_num:
                continue
            x1, y1, x2, y2 = map(int, bbox.bbox)
            if visualize or video_output_path is not None:
                img = cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if visualize:
        cv.imshow('Blue=generated, Green=GT', img)
        if cv.waitKey(33) == ord('q'):
            break

    if video_output_path is not None:
        out.write(img)

if visualize:
    cv.destroyAllWindows()

if video_output_path is not None:
    out.release()

if output_path is not None:
    with open(output_path, 'w') as f:
        json.dump(annotations, f)
