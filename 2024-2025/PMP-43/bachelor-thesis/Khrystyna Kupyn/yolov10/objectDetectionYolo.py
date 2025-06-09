import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import supervision as sv
import argparse


class ObjectDetection:
    def __init__(self, capture_index=0):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device:", self.device)

        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names

        self.box_annotator = sv.BoxAnnotator(thickness=3)

    def load_model(self):
        model = YOLO("yolo10m.pt")
        model.fuse()  
        return model

    def predict(self, frame: np.ndarray):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame: np.ndarray) -> np.ndarray:
        result = results[0]
        boxes = result.boxes

        if boxes is None or boxes.cls is None or len(boxes) == 0:
            return frame

        detections = sv.Detections(
            xyxy=boxes.xyxy.cpu().numpy(),
            confidence=boxes.conf.cpu().numpy(),
            class_id=boxes.cls.cpu().numpy().astype(int),
        )

        labels = [
            f"{self.CLASS_NAMES_DICT[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections,
            labels=labels
        )
        return annotated_frame

    def __call__(self):
        for i in range(5):
            cap_test = cv2.VideoCapture(i)
            if cap_test.isOpened():
                print(f"Камера з індексом {i} доступна")
                cap_test.release()

        cap = cv2.VideoCapture(self.capture_index)
        if not cap.isOpened():
            print(f"Не вдалося відкрити камеру з індексом {self.capture_index}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not hasattr(cv2, 'imshow'):
            print("Error: OpenCV was built without GUI support. Метод imshow відсутній.")
            cap.release()
            return

        while True:
            start_time = time()

            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # передбачення та бокси
            results = self.predict(frame)
            annotated_frame = self.plot_bboxes(results, frame)

            # FPS
            end_time = time()
            fps = 1.0 / max(end_time - start_time, 1e-5)
            cv2.putText(
                annotated_frame,
                f"FPS: {int(fps)}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                2
            )

            cv2.imshow('YOLO Detection', annotated_frame)

            # Вийти по ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    args = parser.parse_args()

    detector = ObjectDetection(capture_index=args.camera)
    detector()


