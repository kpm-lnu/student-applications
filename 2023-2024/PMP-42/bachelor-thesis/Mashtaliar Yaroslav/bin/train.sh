yolo task=detect \
  mode=train \
  model=runs/detect/train5/weights/best.pt \
  data=datasets/box-v5/data.yaml \
  epochs=5 \
  imgsz=720
