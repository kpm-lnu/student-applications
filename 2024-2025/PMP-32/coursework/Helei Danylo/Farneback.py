import cv2
import numpy as np

# Введення шляху до відеофайлу
video_path = "test_video5.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Не вдалося відкрити відео.")
    exit()

# Зчитування першого кадру
ret, prev_frame = cap.read()
if not ret:
    print("Не вдалося зчитати перший кадр.")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Цикл по кадрах
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обчислення оптичного потоку Farneback
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    # Отримання горизонтальних і вертикальних компонент потоку
    fx, fy = flow[..., 0], flow[..., 1]

    # Малювання стрілок
    step = 15
    h, w = gray.shape
    for y in range(0, h, step):
        for x in range(0, w, step):
            pt1 = (x, y)
            pt2 = (int(x + fx[y, x]), int(y + fy[y, x]))
            cv2.arrowedLine(frame, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)

    cv2.imshow("Оптичний потік Farneback", frame)

    key = cv2.waitKey(30)
    if key == 27:  # ESC
        break

    prev_gray = gray.copy()

cap.release()
cv2.destroyAllWindows()
