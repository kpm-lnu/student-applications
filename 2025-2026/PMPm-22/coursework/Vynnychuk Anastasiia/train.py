from ultralytics.models import YOLO

model = YOLO("yolov8m.pt")

model.train(
    data="./data/dfg-ultralytics/data.yaml",
    epochs=40,
    fliplr=0.0,
)
