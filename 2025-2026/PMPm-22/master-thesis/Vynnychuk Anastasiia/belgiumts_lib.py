from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.optimize import least_squares
import cv2 as cv

@dataclass
class VehiclePose:
    R_vehicle: np.ndarray  # 3x3 rotation matrix
    t_vehicle: np.ndarray  # 3D translation vector

@dataclass
class ConfigFilePaths:
    video_files_path: str
    gt_file_path: str
    camera_set_file_path: str
    poses_file_path: str

@dataclass
class CameraData:
    K: np.ndarray  # Intrinsic matrix
    R: np.ndarray  # Rotation matrix
    t: np.ndarray  # Translation vector
    distorsion_matrix: np.ndarray  # Distortion coefficients
    dimensions: tuple[int, int]

@dataclass
class PoleSignData:
    type: str
    idx: int

@dataclass
class PoleBboxData:
    camera: int
    frame: int
    sign_idx: int
    bbox: tuple[float, float, float, float]
    type: str
    coords: tuple[float, float, float]

@dataclass
class PoleData:
    idx: int
    id: int
    coords: tuple[float, float, float]
    signs: list[PoleSignData]
    bboxes: list[PoleBboxData]

class Sign:
    def __init__(self, pole_idx: int, type: str, bboxes: list[PoleBboxData]):
        self.pole_idx = pole_idx
        self.type = type
        self.bboxes = bboxes

    def calculate_geometry(self, poses: dict[int, VehiclePose], camera_data: CameraData) -> bool:
        gt_obs = []
        for bbox in self.bboxes:
            if bbox.frame not in poses:
                continue
            x1, y1, x2, y2 = bbox.bbox
            gt_obs.append((bbox.frame, (x1 + x2) / 2, (y1 + y2) / 2))

        if len(gt_obs) < 1:
            return False

        c0 = np.mean([bbox.coords for bbox in self.bboxes], axis=0)

        # With 1 bbox (2 residuals), only optimize X,Y; with 2+ bboxes optimize all 3
        optimize_z = len(gt_obs) >= 2

        def residuals(params):
            if optimize_z:
                center_3d = params
            else:
                center_3d = np.array([params[0], params[1], c0[2]])
            res = []
            for frame, u_gt, v_gt in gt_obs:
                pose = poses[frame]
                P_v = pose.R_vehicle.T @ (center_3d - pose.t_vehicle)
                P_c = camera_data.R.T @ (P_v - camera_data.t)
                if P_c[2] <= 0:
                    res.extend([1e3, 1e3])
                    continue
                pts_2d, _ = cv.projectPoints(
                    P_c.reshape(1, 1, 3), np.zeros(3), np.zeros(3), camera_data.K, camera_data.distorsion_matrix)
                u, v = pts_2d.ravel()
                res.extend([u - u_gt, v - v_gt])
            return res

        x0 = c0.copy() if optimize_z else c0[:2].copy()
        result = least_squares(residuals, x0, method='lm')
        if optimize_z:
            self.optimized_3d_center = result.x
        else:
            self.optimized_3d_center = np.array([result.x[0], result.x[1], c0[2]])

        self.shift_mm = np.linalg.norm(self.optimized_3d_center - c0) * 1000

        half_ws, half_hs = [], []
        for bbox in self.bboxes:
            if bbox.frame not in poses:
                continue
            pose = poses[bbox.frame]
            P_v = pose.R_vehicle.T @ (self.optimized_3d_center - pose.t_vehicle)
            P_c = camera_data.R.T @ (P_v - camera_data.t)
            if P_c[2] <= 0:
                continue
            x1, y1, x2, y2 = bbox.bbox
            Z = P_c[2]
            half_ws.append((x2 - x1) / 2 * Z / camera_data.K[0, 0])
            half_hs.append((y2 - y1) / 2 * Z / camera_data.K[1, 1])

        if not half_ws or not half_hs:
            return False
        
        self.half_w = np.median(half_ws)
        self.half_h = np.median(half_hs)
        
        return True

    @property
    def is_valid(self) -> bool:
        return hasattr(self, 'optimized_3d_center') and hasattr(self, 'half_w') and hasattr(self, 'half_h')

    def project_bbox(self, camera_data: CameraData, pose: VehiclePose, img_w: int, img_h: int) -> Optional[tuple[float, float, float, float]]:
        if not self.is_valid:
            return None
        
        P_v = pose.R_vehicle.T @ (self.optimized_3d_center - pose.t_vehicle)
        P_c = camera_data.R.T @ (P_v - camera_data.t)
        if P_c[2] <= 0:
            return None

        pts_2d, _ = cv.projectPoints(
            P_c.reshape(1, 1, 3), np.zeros(3), np.zeros(3), camera_data.K, camera_data.distorsion_matrix)
        u, v = pts_2d.ravel()

        Z = P_c[2]
        half_w = self.half_w * camera_data.K[0, 0] / Z
        half_h = self.half_h * camera_data.K[1, 1] / Z

        x1 = max(0, u - half_w)
        y1 = max(0, v - half_h)
        x2 = min(img_w, u + half_w)
        y2 = min(img_h, v + half_h)

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)


def parse_poles(gt_file_path: str) -> list[PoleData]:
    with open(gt_file_path, 'r') as f:
        lines = f.readlines()

    result = []

    i = 0
    num_poles = int(lines[i].strip())
    i += 1

    for _ in range(num_poles):
        # Skip empty lines
        while i < len(lines) and lines[i].strip() == '':
            i += 1

        # Pole number and ID
        pole_info = lines[i].strip().split()
        pole_idx = int(pole_info[0])
        pole_id = int(pole_info[1])
        i += 1
        
        # 3D coordinates
        pole_coords = list(map(float, lines[i].strip().split()))
        i += 1
        
        # Number of traffic signs and their types
        signs_data = lines[i].strip()[:-1].split(';')
        num_signs = int(signs_data[0])
        sign_types = signs_data[1:]
        signs = [PoleSignData(type=t, idx=idx) for idx, t in enumerate(sign_types, 1)]
        i += 1
        
        # Number of centers
        num_centers = int(lines[i].strip())
        i += 1
        
        # Skip center coordinates
        pole_centers = []
        for _ in range(num_centers):
            pole_centers.append(list(map(float, lines[i].strip().split())))
            i += 1

        # Number of bounding boxes
        num_bboxes = int(lines[i].strip())
        i += 1
        
        bboxes = []
        # Parse each bounding box
        for _ in range(num_bboxes):
            # Bounding box coordinates
            bbox_coords = list(map(float, lines[i].strip().split()))
            # x1, y1, x2, y2 = bbox_coords
            y1, x1, y2, x2= bbox_coords
            i += 1
            
            # Camera, frame, order number
            metadata = lines[i].strip().split()
            camera = int(metadata[0])
            frame = int(metadata[1])
            sign_idx = int(metadata[2])
            i += 1
            
            # Class label
            class_label = lines[i].strip().rstrip(';')
            i += 1

            # 3D center coordinates
            bboxes.append(PoleBboxData(
                camera=camera,
                frame=frame,
                sign_idx=sign_idx,
                bbox=(x1, y1, x2, y2),
                type=class_label,
                coords=tuple(map(float, lines[i].strip().split()))
            ))
            i += 1

        pole_data = PoleData(
            idx=pole_idx,
            id=pole_id,
            coords=tuple(pole_coords),
            signs=signs,
            bboxes=bboxes
        )

        result.append(pole_data)

    return result
