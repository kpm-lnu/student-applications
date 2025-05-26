import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy.ndimage
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
MOUTH_IDX = [13, 14, 78, 308]

cap = cv2.VideoCapture('video.mp4')

ear_open = None
ear_closed = None
ear_threshold = None
calibration_frames = 50
calibration_counter = 0
ears_open_list = []
ears_closed_list = []
calibrated = False

blink_counter = 0
consecutive_closed_frames = 0
BLINK_CONSEC_FRAMES = 3
yawn_counter = 0
YAWN_MAR_THRESHOLD = 0.7
yawn_active = False
yawn_cooldown = 0
YAWN_COOLDOWN_FRAMES = 150
sleeping_counter = 0
SLEEPING_FRAMES = 90

tilt_over_threshold_counter = 0
TILT_ANGLE_THRESHOLD = 30
TILT_DURATION_FRAMES = 60

was_sleeping = False
woke_up_after_sleep = False
stable_head_counter = 0
STABLE_FRAMES_AFTER_SLEEP = 60

blink_list = []
yawn_list = []
tilt_list = []
timestamps = []
start_time = time.time()

def calculate_EAR(landmarks, eye_indices, iw, ih):
    eps = 1e-6
    p = [np.array([landmarks[idx].x * iw, landmarks[idx].y * ih]) for idx in eye_indices]
    horizontal = np.linalg.norm(p[0] - p[3])
    vertical1 = np.linalg.norm(p[1] - p[5])
    vertical2 = np.linalg.norm(p[2] - p[4])
    vertical_min = min(vertical1, vertical2)
    vertical_max = max(vertical1, vertical2)
    ear_closed = (vertical_min + vertical_min) / (2.0 * horizontal + eps)
    ear_open = (vertical_max + vertical_max) / (2.0 * horizontal + eps)
    ear_threshold = (ear_closed + ear_open) / 2.0
    return (ear_closed, ear_open, ear_threshold)

def calculate_MAR(landmarks, mouth_indices, iw, ih):
    upper = np.array([landmarks[mouth_indices[0]].x * iw, landmarks[mouth_indices[0]].y * ih])
    lower = np.array([landmarks[mouth_indices[1]].x * iw, landmarks[mouth_indices[1]].y * ih])
    left = np.array([landmarks[mouth_indices[2]].x * iw, landmarks[mouth_indices[2]].y * ih])
    right = np.array([landmarks[mouth_indices[3]].x * iw, landmarks[mouth_indices[3]].y * ih])
    vertical = np.linalg.norm(upper - lower)
    horizontal = np.linalg.norm(left - right)
    return vertical / horizontal

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    ih, iw, _ = frame.shape
    current_time = time.time() - start_time

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear_closed, left_ear_open, left_ear_threshold = calculate_EAR(face_landmarks.landmark, LEFT_EYE_IDX, iw, ih)
            right_ear_closed, right_ear_open, right_ear_threshold = calculate_EAR(face_landmarks.landmark, RIGHT_EYE_IDX, iw, ih)
            avg_ear_closed = (left_ear_closed + right_ear_closed) / 2.0
            avg_ear_open = (left_ear_open + right_ear_open) / 2.0
            avg_ear_threshold = (left_ear_threshold + right_ear_threshold) / 2.0
            mar = calculate_MAR(face_landmarks.landmark, MOUTH_IDX, iw, ih)

            delta_x = int((face_landmarks.landmark[263].x - face_landmarks.landmark[33].x) * iw)
            delta_y = int((face_landmarks.landmark[263].y - face_landmarks.landmark[33].y) * ih)
            angle = np.arctan2(delta_y, delta_x) * 180.0 / np.pi

            if not calibrated:
                calibration_counter += 1
                if calibration_counter <= calibration_frames // 2:
                    ears_open_list.append(avg_ear_open)
                elif calibration_counter <= calibration_frames:
                    ears_closed_list.append(avg_ear_closed)
                else:
                    ear_open = max(ears_open_list)
                    ear_closed = min(ears_closed_list)
                    ear_threshold = (ear_open + ear_closed) / 2.0 * 0.90
                    EAR_THRESH_LOW = ear_threshold * 0.95
                    EAR_THRESH_HIGH = ear_threshold * 1.05
                    calibrated = True

                blink_list.append(0)
                yawn_list.append(0)
                tilt_list.append(angle)
                timestamps.append(current_time)
            else:
                avg_ear = (avg_ear_open + avg_ear_closed) / 2.0

                if avg_ear <= EAR_THRESH_LOW:
                    consecutive_closed_frames += 1
                    cv2.putText(frame, "Eyes Closed", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    sleeping_counter += 1
                    blink = 1
                else:
                    if consecutive_closed_frames >= BLINK_CONSEC_FRAMES:
                        blink_counter += 1
                    consecutive_closed_frames = 0
                    cv2.putText(frame, "Eyes Open", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
                    sleeping_counter = 0
                    blink = 0

                if mar > YAWN_MAR_THRESHOLD:
                    if not yawn_active and yawn_cooldown == 0:
                        yawn_counter += 1
                        yawn_active = True
                        yawn = 1
                    else:
                        yawn = 0
                else:
                    if yawn_active:
                        yawn_active = False
                        yawn_cooldown = YAWN_COOLDOWN_FRAMES
                    yawn = 0

                if yawn_cooldown > 0:
                    yawn_cooldown -= 1

                if abs(angle) > TILT_ANGLE_THRESHOLD:
                    tilt_over_threshold_counter += 1
                else:
                    tilt_over_threshold_counter = 0

                cv2.putText(frame, f"EAR: {avg_ear:.3f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"MAR: {mar:.3f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
                cv2.putText(frame, f"Blinks: {blink_counter}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.putText(frame, f"Yawns: {yawn_counter}", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 50, 255), 2)

                if sleeping_counter > SLEEPING_FRAMES:
                    cv2.putText(frame, "SLEEPING!!!", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    was_sleeping = True
                    woke_up_after_sleep = False
                    stable_head_counter = 0
                else:
                    if was_sleeping and not woke_up_after_sleep:
                        woke_up_after_sleep = True
                    if woke_up_after_sleep:
                        if abs(angle) < TILT_ANGLE_THRESHOLD:
                            stable_head_counter += 1
                        else:
                            stable_head_counter = 0
                        if stable_head_counter >= STABLE_FRAMES_AFTER_SLEEP:
                            cv2.putText(frame, "DROWSY!", (30, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 140, 255), 3)

                blink_list.append(blink)
                yawn_list.append(yawn)
                tilt_list.append(angle)
                timestamps.append(current_time)

    cv2.imshow("Driver Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

if blink_list and yawn_list and tilt_list:
    coeffs_blink = pywt.wavedec(blink_list, 'db4', level=2)
    coeffs_yawn = pywt.wavedec(yawn_list, 'db4', level=2)
    coeffs_tilt = pywt.wavedec(tilt_list, 'db4', level=2)

    reconstructed_blink = pywt.waverec(coeffs_blink, 'db4')
    reconstructed_yawn = pywt.waverec(coeffs_yawn, 'db4')
    reconstructed_tilt = pywt.waverec(coeffs_tilt, 'db4')

    smooth_blink = scipy.ndimage.gaussian_filter1d(reconstructed_blink, sigma=2)
    smooth_yawn = scipy.ndimage.gaussian_filter1d(reconstructed_yawn, sigma=2)
    smooth_tilt = scipy.ndimage.gaussian_filter1d(reconstructed_tilt, sigma=3)

    fig, axs = plt.subplots(3, 1, figsize=(20, 15))

    axs[0].plot(timestamps[:len(smooth_blink)], smooth_blink, label='Blinking', color='blue')
    axs[0].fill_between(timestamps[:len(smooth_blink)], smooth_blink, alpha=0.3, color='blue')
    axs[0].set_title('Blinking Detection', fontsize=18)
    axs[0].set_ylabel('Blink Signal')
    axs[0].grid(True)

    axs[1].plot(timestamps[:len(smooth_yawn)], smooth_yawn, label='Yawning', color='red')
    axs[1].fill_between(timestamps[:len(smooth_yawn)], smooth_yawn, alpha=0.3, color='red')
    axs[1].set_title('Yawning Detection', fontsize=18)
    axs[1].set_ylabel('Yawn Signal')
    axs[1].grid(True)

    axs[2].plot(timestamps[:len(smooth_tilt)], smooth_tilt, label='Head Tilt', color='green')
    axs[2].fill_between(timestamps[:len(smooth_tilt)], smooth_tilt, alpha=0.3, color='green')
    axs[2].set_title('Head Tilt Detection', fontsize=18)
    axs[2].set_ylabel('Tilt Signal')
    axs[2].set_xlabel('Time (s)')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
else:
    print("Error")
