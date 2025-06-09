import os
from ultralytics import YOLO


def load_latest_model():
    latest_index = max(int(x[len('train'):] or '0')
               for x in os.listdir('../runs/detect')
               if x.startswith('train'))
    weights_path = f'../runs/detect/train{latest_index}/weights/best.pt'
    print(f'Loading the weights from {weights_path}')
    model = YOLO(weights_path)
    return model
