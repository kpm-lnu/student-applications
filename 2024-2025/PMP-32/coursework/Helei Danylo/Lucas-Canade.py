import cv2
import numpy as np


# Введення шляху до відео
video_path = "test_video5.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Не вдалося відкрити відео.")
    exit()

# Параметри детектора кутів Shi-Tomasi
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Параметри Лукаса-Канаде
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Синій колір для всіх точок
color = (255, 0, 0)

# Зчитування першого кадру
ret, old_frame = cap.read()
if not ret:
    print("Не вдалося зчитати перший кадр.")
    exit()

# Вибір області ROI (Region of Interest)
roi = cv2.selectROI("Виберіть об'єкт для відстеження", old_frame, fromCenter=False, showCrosshair=True)
x, y, w, h = roi
roi_frame = old_frame[y:y+h, x:x+w]

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
roi_gray = old_gray[y:y+h, x:x+w]

# Знаходження початкових точок для відстеження
p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)

# Компенсація координат ROI до глобального зображення
if p0 is not None:
    p0[:, 0, 0] += x
    p0[:, 0, 1] += y

# Маска для відображення траєкторій
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обчислення оптичного потоку
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is None or st is None:
        break

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Малювання траєкторій
    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color, 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color, -1)

    img = cv2.add(frame, mask)
    cv2.imshow('Оптичний потік Лукаса-Канаде', img)

    key = cv2.waitKey(30)
    if key == 27:  # ESC для виходу
        break

    # Підготовка до наступного кроку
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()
