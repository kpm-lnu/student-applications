import cv2
from ultralytics import YOLO
import numpy as np
import re
import os
import time
from datetime import datetime
from paddleocr import PaddleOCR
import psycopg2
from collections import Counter
import Levenshtein

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DB_PARAMS = {
    'dbname': 'licence_plates',
    'user': 'shum430',
    'password': 'admin111',
    'host': 'localhost',
    'port': 5432
}

def create_table():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS license_plates (
                id SERIAL PRIMARY KEY,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                license_plate VARCHAR(20)
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("Error creating table:", e)

def save_to_postgres(license_plates, start_time, end_time):
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        for plate in license_plates:
            cursor.execute('''
                SELECT COUNT(*) FROM license_plates
                WHERE license_plate = %s
                AND start_time >= %s - interval '30 seconds'
                AND end_time <= %s + interval '30 seconds'
            ''', (plate, start_time, end_time))
            count = cursor.fetchone()[0]
            if count == 0:
                cursor.execute('''
                    INSERT INTO license_plates(start_time, end_time, license_plate)
                    VALUES (%s, %s, %s)
                ''', (start_time, end_time, plate))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("Error saving to DB:", e)

def group_similar_plates(plates, threshold=1, distance_limit=3):
    counts = Counter(plates)
    filtered = [plate for plate, freq in counts.items() if freq >= threshold]
    grouped = []
    used = set()
    for plate in filtered:
        if plate in used:
            continue
        group = [plate]
        used.add(plate)
        for other in filtered:
            if other in used:
                continue
            if Levenshtein.distance(plate, other) <= distance_limit:
                group.append(other)
                used.add(other)
        best = max(group, key=lambda x: counts[x])
        grouped.append(best)
    return grouped

def paddle_ocr(frame, x1, y1, x2, y2):
    frame = frame[y1:y2, x1:x2]
    result = ocr.ocr(frame, det=False, rec=True, cls=False)
    text = ""
    for r in result:
        score = r[0][1]
        raw_text = r[0][0]
        if np.isnan(score):
            score = 0
        else:
            score = int(score * 100)
        if score > 70:
            text = raw_text
            confidence_scores.append(score)
        else:
            rejected_ocr_set.add(raw_text)
    pattern = re.compile('[\W]')
    text = pattern.sub('', text)
    text = text.replace("???", "").replace("O", "0").replace("粤", "")
    return str(text)

# --- INIT ---
create_table()
cap = cv2.VideoCapture("data/carLicence1.mp4")
model = YOLO("license_plate_detector.pt")
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

startTime = datetime.now()
video_start = time.time()

all_plates = []
unique_plates_overall = set()
confidence_scores = []
rejected_ocr_set = set()
total_frames = 0
detected_frames = 0

# --- MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1
    currentTime = datetime.now()
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    results = model.predict(frame, conf=0.7)

    for result in results:
        boxes = result.boxes
        if boxes:
            detected_frames += 1
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = paddle_ocr(frame, x1, y1, x2, y2)
            if label:
                print(label)
                all_plates.append(label)
            textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
            c2 = x1 + textSize[0], y1 - textSize[1] - 3
            cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    if (currentTime - startTime).seconds >= 5:
        endTime = currentTime
        cleaned_plates = group_similar_plates(all_plates)
        plate_pattern = re.compile(r'^[A-Z0-9]{6,9}$')
        cleaned_plates = [p for p in cleaned_plates if plate_pattern.match(p)]

        save_to_postgres(cleaned_plates, startTime, endTime)
        unique_plates_overall.update(cleaned_plates)
        startTime = currentTime
        all_plates.clear()

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- FINAL REPORT ---
duration = time.time() - video_start
print("\n========= 🧪 ЗВІТ ПРО ОБРОБКУ =========")
print(f"⏱ Загальний час обробки: {duration:.2f} сек")
print(f"🎞 Загальна кількість кадрів: {total_frames}")
print(f"✅ Кадрів з виявленням номерів: {detected_frames}")
with open("screenshots/confidence_log.txt", "a") as f:
    for score in confidence_scores:
        f.write(f"{score}\n")

if confidence_scores:
    avg_conf = sum(confidence_scores) / len(confidence_scores)
    print(f"📊 Середній confidence OCR: {avg_conf:.2f}%")
else:
    print("⚠️ Жоден результат не мав довіри > 70%")

print("📌 Унікальні номери за весь час:", unique_plates_overall)

print(f"\n❌ Унікальних відкинутих OCR-рядків: {len(rejected_ocr_set)}")
if rejected_ocr_set:
    print("🔍 Відкинуті приклади:")
    for r in sorted(rejected_ocr_set):
        print(f" - {r}")
