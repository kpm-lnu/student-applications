from ultralytics import YOLO
import cv2 as cv
import os
import time

video_root = './data/CURE-TSD_cut/data'
label_root = './data/CURE-TSD_cut/labels'

model = YOLO('./models/kursova.pt')

video_file_name = '02_02_00_00_00.mp4'
# video_file_name = '01_04_00_00_00.mp4'
video_path = os.path.join(video_root, video_file_name)

# Visualize video with model predictions
cap = cv.VideoCapture(video_path)
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Measure time for prediction
    start_time = time.time()

    # Run model prediction on the frame
    results = model.predict(frame, verbose=False, conf=0.25)

    # Draw bounding boxes from predictions
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())

            # Draw rectangle
            cv.rectangle(frame, (int(x1), int(y1)),
                         (int(x2), int(y2)), (0, 255, 0), 2)

            # Add label with class and confidence
            # label = f"Class {cls}: {conf:.2f}"
            # cv.putText(frame, label, (int(x1), int(y1) - 10),
            #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv.imshow('Model Predictions', frame)

    # Calculate remaining wait time for 10 fps (100ms per frame)
    elapsed_ms = (time.time() - start_time) * 1000
    # Minimum 1ms for event processing
    wait_time = max(1, int(100 - elapsed_ms))

    # Press 'q' to quit
    key = cv.waitKey(wait_time)
    if key == ord('q'):
        break

    frame_num += 1

cap.release()
cv.destroyAllWindows()
