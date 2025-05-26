import cv2
import mediapipe as mp
import numpy as np
import os

# === Лістинг модуля попередньої обробки відео (рис. 2.4) ===
def preprocess_video(video_path, output_folder="frames"):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (640, 480))
        normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f"{output_folder}/frame_{frame_idx:04d}.jpg", normalized)
        frame_idx += 1
    cap.release()
    print(f"✅ Збережено {frame_idx} кадрів у {output_folder}/")

# === Лістинг модуля для детекції ключових точок (рис. 2.6) ===
def detect_keypoints(image, pose_model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb)
    return results.pose_landmarks

# === Лістинг модуля скелетонізації (рис. 2.5) ===
def build_skeleton(image, landmarks, draw_utils, connections):
    h, w, _ = image.shape
    if landmarks:
        draw_utils.draw_landmarks(
            image,
            landmarks,
            connections,
            landmark_drawing_spec=draw_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            connection_drawing_spec=draw_utils.DrawingSpec(color=(255,0,0), thickness=2)
        )
        for idx, lm in enumerate(landmarks.landmark):
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (x, y), 3, (0, 255, 255), -1)
            cv2.putText(image, str(idx), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return image

# === Основна функція запуску всіх модулів ===
def main(source=0):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(source)
    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      enable_segmentation=False, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            landmarks = detect_keypoints(frame, pose)
            frame_with_skeleton = build_skeleton(frame, landmarks, mp_drawing, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('Pose Estimation', frame_with_skeleton)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

# === Точка входу ===
if __name__ == "__main__":
    main(0) 
