from torchvision.models.detection import ssd300_vgg16
import torch
import cv2
import numpy as np
from torchvision import transforms
from time import time

class ObjectDetection:
    def __init__(self, capture_index=0):
        self.capture_index = capture_index
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device:", self.device)

        self.model = ssd300_vgg16(weights=None, weights_backbone=None, num_classes=10).to(self.device)
        self.model.load_state_dict(torch.load("ssd300_custom.pth", map_location=self.device))
        self.model.eval()

        self.CLASS_NAMES = ["__background__", "car", "chair", "cup", "door", "potted plant",
                            "orange", "person", "phone", "tree"]

    def predict(self, frame: np.ndarray):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (300, 300))
        image_tensor = transforms.ToTensor()(image_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)[0]

        return outputs

    def plot_bboxes(self, outputs, frame: np.ndarray, threshold=0.4) -> np.ndarray:
        h_orig, w_orig = frame.shape[:2]

        boxes = outputs['boxes'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            if score < threshold:
                continue
            x1, y1, x2, y2 = box

            x1 = int(x1 / 300 * w_orig)
            x2 = int(x2 / 300 * w_orig)
            y1 = int(y1 / 300 * h_orig)
            y2 = int(y2 / 300 * h_orig)

            class_name = self.CLASS_NAMES[label] if label < len(self.CLASS_NAMES) else f"id_{label}"
            label_text = f"{class_name} {score:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not hasattr(cv2, 'imshow'):
            print("Error: OpenCV was built without GUI support.")
            cap.release()
            return

        while True:
            start_time = time()
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            outputs = self.predict(frame)
            annotated_frame = self.plot_bboxes(outputs, frame)

            fps = 1.0 / max(time() - start_time, 1e-5)
            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

            cv2.imshow('SSD Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

