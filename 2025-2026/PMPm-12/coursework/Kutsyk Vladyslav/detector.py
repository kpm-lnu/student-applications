"""
detector.py
Обгортка над моделлю YOLOv8s-P2-SimAM.
Приймає BGR-кадр, повертає numpy-масив детекцій.
"""

import numpy as np
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO

from config import (CONF_THRESH, NMS_IOU_THRESH,
                    MAX_DETS, INFER_IMG_SIZE, DEVICE)


class Detector:
    """
    Завантажує навчену модель і виконує інференс.

    Використання:
        det = Detector("checkpoints/best.pt")
        bboxes = det.detect(frame)   # np.ndarray (N, 5): u,v,w,h,score
    """

    def __init__(self,
                 weights: str,
                 conf:    float = CONF_THRESH,
                 iou:     float = NMS_IOU_THRESH,
                 max_det: int   = MAX_DETS,
                 device:  str   = DEVICE):

        self.model   = YOLO(weights)
        self.conf    = conf
        self.iou     = iou
        self.max_det = max_det
        self.device  = device
        print(f"[Detector] Завантажено: {weights}")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            frame: BGR зображення (H, W, 3) у форматі NumPy

        Returns:
            np.ndarray shape (N, 5): [u_center, v_center, w, h, score]
            Координати у пікселях вхідного кадру.
            Якщо детекцій немає — повертає пустий масив (0, 5).
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
            imgsz=max(INFER_IMG_SIZE),
        )

        if not results or results[0].boxes is None:
            return np.empty((0, 5), dtype=np.float32)

        boxes  = results[0].boxes
        xyxy   = boxes.xyxy.cpu().numpy()    # x1,y1,x2,y2
        scores = boxes.conf.cpu().numpy()

        # Перетворення xyxy → xywh (center format)
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
        w  =  xyxy[:, 2] - xyxy[:, 0]
        h  =  xyxy[:, 3] - xyxy[:, 1]

        return np.stack([cx, cy, w, h, scores], axis=1).astype(np.float32)

    def warmup(self) -> None:
        """Прогрів моделі на dummy-вхід для стабільних FPS."""
        dummy = np.zeros((*INFER_IMG_SIZE[::-1], 3), dtype=np.uint8)
        for _ in range(3):
            self.detect(dummy)
        print("[Detector] Warmup завершено.")
