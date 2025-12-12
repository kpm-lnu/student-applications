import math
import time
import sys
import csv
from pathlib import Path
from collections import deque

import numpy as np
import cv2 as cv


FEATURE_PARAMS = dict(maxCorners=600, qualityLevel=0.01, minDistance=7, blockSize=7)
WIN_SIZE = (21, 21)
LK_CRITERIA = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)

MIN_VEC_LEN = 2.5
DIFF_BLUR_K = 5
DIFF_THRESH = 18
MORPH_OP_K = 3

ARROW_TTL = 0.5
CLEANUP_EVERY = 0.5
MIN_TRACKS = 150

MAX_CORNERS = 500
CORNER_QUALITY = 0.01
MIN_DISTANCE_OWN = 7

LK_WINDOW_RADIUS = 5
LK_EPS = 1e-4

MOTION_DIFF_THRESH = 0.16

MIN_FLOW_MAG = 0.5

COLOR_RED   = (0, 0, 255)
COLOR_BLUE  = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)

HIGHLIGHT_ALPHA = 0.5
CONTOUR_THICKNESS = 1

TRACK_BOX_SIZE = 10

INIT_MIN_DISP = 1.0


def fmt_hms(sec: float) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def print_progress(processed, total, start_time, prefix=""):
    elapsed = time.time() - start_time
    if total and total > 0:
        msg = f"\r{prefix}Кадр {processed}/{total}    Elapsed {fmt_hms(elapsed)}"
    else:
        msg = f"\r{prefix}Frames: {processed}  Elapsed {fmt_hms(elapsed)}"
    sys.stdout.write(msg)
    sys.stdout.flush()


def list_videos():
    videos_dir = Path("videos")
    if not videos_dir.exists():
        return []
    exts = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")
    return sorted([p for p in videos_dir.iterdir() if p.suffix in exts])


def choose_video():
    videos = list_videos()
    if not videos:
        print("У папці 'videos' немає відеофайлів.")
        return None

    print("Доступні відео:")
    for i, vf in enumerate(videos):
        print(f"[{i}] {vf.name}")

    while True:
        try:
            idx = int(input("Виберіть номер відео для обробки: "))
            if 0 <= idx < len(videos):
                return videos[idx]
            print("Невірний номер, спробуй ще раз.")
        except ValueError:
            print("Введи ціле число.")


def make_motion_mask_opencv(prev_gray_u8_blur, gray_u8_blur):
    diff = cv.absdiff(gray_u8_blur, prev_gray_u8_blur)
    diff = cv.GaussianBlur(diff, (DIFF_BLUR_K, DIFF_BLUR_K), 0)
    _, mask = cv.threshold(diff, DIFF_THRESH, 255, cv.THRESH_BINARY)
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (MORPH_OP_K, MORPH_OP_K))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_DILATE, k, iterations=1)
    return mask


def highlight_motion_with_contour(frame_bgr, motion_mask_u8, overlay_color, contour_color, alpha=HIGHLIGHT_ALPHA):
    if motion_mask_u8 is None:
        return frame_bgr

    out = frame_bgr.copy()

    color_mask = np.full_like(frame_bgr, overlay_color, dtype=np.uint8)
    blended = cv.addWeighted(frame_bgr, 1 - alpha, color_mask, alpha, 0)

    m = motion_mask_u8 > 0
    out[m] = blended[m]

    contours, _ = cv.findContours(motion_mask_u8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        cv.drawContours(out, contours, -1, contour_color, CONTOUR_THICKNESS)

    return out


def draw_arrow(img, p0, p1, color):
    p0 = tuple(np.round(p0).astype(int))
    p1 = tuple(np.round(p1).astype(int))
    cv.arrowedLine(img, p0, p1, color, 1, tipLength=0.15)


def draw_track_box(img, pt_xy, size=TRACK_BOX_SIZE, color=COLOR_GREEN, thickness=1):
    if pt_xy is None:
        return
    x, y = pt_xy
    if not np.isfinite(x) or not np.isfinite(y):
        return
    x = int(round(float(x)))
    y = int(round(float(y)))
    half = size // 2
    cv.rectangle(img, (x - half, y - half), (x + half, y + half), color, thickness)


def pick_corner_on_second_frame_that_moved(video_path: Path):
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    ret0, f0 = cap.read()
    ret1, f1 = cap.read()
    cap.release()
    if not ret0 or not ret1:
        return None

    h, w = f0.shape[:2]

    g0u = cv.cvtColor(f0, cv.COLOR_BGR2GRAY)
    g1u = cv.cvtColor(f1, cv.COLOR_BGR2GRAY)
    g0b = cv.GaussianBlur(g0u, (5, 5), 0)
    g1b = cv.GaussianBlur(g1u, (5, 5), 0)

    motion_mask = make_motion_mask_opencv(g0b, g1b)

    corners = cv.goodFeaturesToTrack(
        g1b,
        mask=None,
        maxCorners=300,
        qualityLevel=FEATURE_PARAMS["qualityLevel"],
        minDistance=FEATURE_PARAMS["minDistance"],
        blockSize=FEATURE_PARAMS["blockSize"],
    )
    if corners is None:
        return (w / 2.0, h / 2.0)

    pts1 = corners.reshape(-1, 1, 2).astype(np.float32)

    pts0, st, _ = cv.calcOpticalFlowPyrLK(
        g1b, g0b, pts1, None,
        winSize=WIN_SIZE, maxLevel=0, criteria=LK_CRITERIA
    )
    if pts0 is None or st is None:
        x, y = pts1.reshape(-1, 2)[0]
        return (float(x), float(y))

    pts1_flat = pts1.reshape(-1, 2)
    pts0_flat = pts0.reshape(-1, 2)
    st_flat = st.reshape(-1)

    best = None
    best_disp = -1.0
    best_pref = -1

    for (x1, y1), (x0, y0), ok in zip(pts1_flat, pts0_flat, st_flat):
        if int(ok) != 1:
            continue

        if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x0 < w and 0 <= y0 < h):
            continue

        disp = float(math.hypot(x1 - x0, y1 - y0))
        if disp < INIT_MIN_DISP:
            continue

        ix, iy = int(round(x1)), int(round(y1))
        pref = 1 if motion_mask[iy, ix] > 0 else 0

        if (pref > best_pref) or (pref == best_pref and disp > best_disp):
            best_pref = pref
            best_disp = disp
            best = (float(x1), float(y1))

    if best is not None:
        return best


    x, y = pts1_flat[0]
    return (float(x), float(y))


def reinit_point_on_frame(gray_u8, motion_mask_u8, last_xy=None, search_radius=40):
    corners = cv.goodFeaturesToTrack(
        gray_u8,
        mask=motion_mask_u8,
        maxCorners=80,
        qualityLevel=FEATURE_PARAMS["qualityLevel"],
        minDistance=FEATURE_PARAMS["minDistance"],
        blockSize=FEATURE_PARAMS["blockSize"],
    )
    if corners is None:
        corners = cv.goodFeaturesToTrack(
            gray_u8,
            mask=None,
            maxCorners=80,
            qualityLevel=FEATURE_PARAMS["qualityLevel"],
            minDistance=FEATURE_PARAMS["minDistance"],
            blockSize=FEATURE_PARAMS["blockSize"],
        )
    if corners is None:
        return None

    pts = corners.reshape(-1, 2).astype(np.float32)

    if last_xy is None or not (np.isfinite(last_xy[0]) and np.isfinite(last_xy[1])):
        x, y = pts[0]
        return (float(x), float(y))

    lx, ly = float(last_xy[0]), float(last_xy[1])
    d2 = (pts[:, 0] - lx) ** 2 + (pts[:, 1] - ly) ** 2
    best = int(np.argmin(d2))
    if d2[best] <= (search_radius ** 2):
        x, y = pts[best]
        return (float(x), float(y))

    x, y = pts[0]
    return (float(x), float(y))



def compute_image_gradients(gray):
    H, W = gray.shape
    Ix = np.zeros_like(gray, dtype=np.float32)
    Iy = np.zeros_like(gray, dtype=np.float32)
    Ix[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    Iy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
    return Ix, Iy


def shi_tomasi_corners(gray, max_corners=MAX_CORNERS, quality=CORNER_QUALITY, min_distance=MIN_DISTANCE_OWN):
    Ix, Iy = compute_image_gradients(gray)

    win = 3
    pad = win // 2
    H, W = gray.shape

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    def integral(img):
        ii = img.cumsum(axis=0).cumsum(axis=1)
        ii = np.pad(ii, ((1, 0), (1, 0)), mode="constant")
        return ii

    Ixx_i = integral(Ixx)
    Iyy_i = integral(Iyy)
    Ixy_i = integral(Ixy)

    out_h = H - win + 1
    out_w = W - win + 1
    if out_h <= 0 or out_w <= 0:
        return np.empty((0, 2), dtype=np.float32)

    ys = np.arange(out_h)
    xs = np.arange(out_w)
    Y, X = np.meshgrid(ys, xs, indexing="ij")

    y1 = Y
    x1 = X
    y2 = Y + win
    x2 = X + win

    def window_sum(ii):
        A = ii[y2, x2]
        B = ii[y1, x2]
        C = ii[y2, x1]
        D = ii[y1, x1]
        return A - B - C + D

    Sxx = window_sum(Ixx_i)
    Syy = window_sum(Iyy_i)
    Sxy = window_sum(Ixy_i)

    trace = Sxx + Syy
    det = Sxx * Syy - Sxy * Sxy
    tmp = trace * trace - 4.0 * det
    tmp[tmp < 0] = 0.0

    sqrt_tmp = np.sqrt(tmp)
    lambda1 = (trace + sqrt_tmp) * 0.5
    lambda2 = (trace - sqrt_tmp) * 0.5
    lambda_min = np.minimum(lambda1, lambda2)

    score = np.zeros_like(gray, dtype=np.float32)
    score[pad:H - pad, pad:W - pad] = lambda_min

    max_score = score.max()
    if max_score <= 0:
        return np.empty((0, 2), dtype=np.float32)

    thresh = max_score * quality
    mask = score > thresh

    ys2, xs2 = np.where(mask)
    cand_scores = score[ys2, xs2]
    order = np.argsort(-cand_scores)

    corners = []
    for idx in order:
        y = int(ys2[idx])
        x = int(xs2[idx])
        good = True
        for (cx, cy) in corners:
            if (cx - x) ** 2 + (cy - y) ** 2 < min_distance ** 2:
                good = False
                break
        if good:
            corners.append((x, y))
        if len(corners) >= max_corners:
            break

    if len(corners) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(corners, dtype=np.float32)


def compute_lk_single(prev, curr, points, win_r=LK_WINDOW_RADIUS):
    if len(points) == 0:
        return points.copy(), np.zeros((0,), dtype=np.uint8)

    H, W = prev.shape
    Ix, Iy = compute_image_gradients(prev)
    It = curr - prev

    new_points = np.zeros_like(points, dtype=np.float32)
    status = np.zeros((points.shape[0],), dtype=np.uint8)

    for i, (x, y) in enumerate(points):
        x = float(x)
        y = float(y)
        xi = int(round(x))
        yi = int(round(y))
        if xi < win_r or xi >= W - win_r or yi < win_r or yi >= H - win_r:
            status[i] = 0
            continue

        x1 = xi - win_r
        x2 = xi + win_r
        y1 = yi - win_r
        y2 = yi + win_r

        Ix_win = Ix[y1:y2 + 1, x1:x2 + 1].reshape(-1)
        Iy_win = Iy[y1:y2 + 1, x1:x2 + 1].reshape(-1)
        It_win = It[y1:y2 + 1, x1:x2 + 1].reshape(-1)

        Gxx = np.sum(Ix_win * Ix_win)
        Gxy = np.sum(Ix_win * Iy_win)
        Gyy = np.sum(Iy_win * Iy_win)
        Bx = -np.sum(Ix_win * It_win)
        By = -np.sum(Iy_win * It_win)

        det = Gxx * Gyy - Gxy * Gxy
        if abs(det) < LK_EPS:
            status[i] = 0
            continue

        vx = (Gyy * Bx - Gxy * By) / det
        vy = (-Gxy * Bx + Gxx * By) / det

        new_points[i, 0] = x + vx
        new_points[i, 1] = y + vy
        status[i] = 1

    return new_points, status



def process_video_lk_opencv(video_path: Path, track_init_xy_on_frame1):
    print(f"[OpenCV] Обробка відео: {video_path}")

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("[OpenCV] Не вдалось відкрити відео.")
        return [], 0.0

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None

    fps = cap.get(cv.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 30.0

    ret, frame0 = cap.read()
    if not ret:
        print("[OpenCV] Не вдалось зчитати перший кадр.")
        cap.release()
        return [], fps

    h, w = frame0.shape[:2]

    out_dir = Path("processed")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{video_path.stem}_lucas-kanade_opencv.mp4"
    writer = cv.VideoWriter(str(out_path), cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    prev_gray_u8 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
    prev_gray = cv.GaussianBlur(prev_gray_u8, (5, 5), 0)

    # multi-points init
    p0 = cv.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)

    arrows = deque()
    last_cleanup = time.time()
    frame_idx = 0
    reinit_every = int(fps * 2)


    track_xy = None
    track_initialized = False
    track_coords = []

    print(f"[OpenCV] Запис у {out_path}")
    start_time = time.time()


    writer.write(frame0)
    track_coords.append((float("nan"), float("nan"), 3))
    processed = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_u8 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray_u8, (5, 5), 0)

        motion_mask = make_motion_mask_opencv(prev_gray, gray)


        if p0 is None or len(p0) < MIN_TRACKS or (frame_idx % reinit_every == 0):
            p0 = cv.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)


        if p0 is not None:
            p1, st, _ = cv.calcOpticalFlowPyrLK(
                prev_gray, gray, p0, None,
                winSize=WIN_SIZE, maxLevel=0, criteria=LK_CRITERIA
            )
            if p1 is not None and st is not None:
                good_old = p0[st.flatten() == 1].reshape(-1, 2)
                good_new = p1[st.flatten() == 1].reshape(-1, 2)

                for (x0, y0), (x1, y1) in zip(good_old, good_new):
                    dx = x1 - x0
                    dy = y1 - y0
                    vec_len = math.hypot(dx, dy)
                    if vec_len < MIN_VEC_LEN:
                        continue
                    ix, iy = int(round(x1)), int(round(y1))
                    if 0 <= ix < w and 0 <= iy < h and motion_mask[iy, ix] == 0:
                        continue
                    arrows.append((time.time(), (x0, y0), (x1, y1), COLOR_BLUE))

                p0 = good_new.reshape(-1, 1, 2)


        now = time.time()
        if now - last_cleanup >= CLEANUP_EVERY:
            while arrows and (now - arrows[0][0]) > ARROW_TTL:
                arrows.popleft()
            last_cleanup = now


        vis = highlight_motion_with_contour(frame, motion_mask, overlay_color=COLOR_RED, contour_color=COLOR_RED)
        for t0, p_from, p_to, color in arrows:
            if (now - t0) <= ARROW_TTL:
                draw_arrow(vis, p_from, p_to, color)


        if not track_initialized:
            track_xy = track_init_xy_on_frame1
            track_initialized = True
            draw_track_box(vis, track_xy)
            writer.write(vis)
            track_coords.append((track_xy[0], track_xy[1], 3))

            prev_gray = gray
            prev_gray_u8 = gray_u8
            frame_idx += 1
            processed += 1
            print_progress(processed, total_frames, start_time, prefix="[OpenCV] ")
            continue


        st_val = 0
        pt_prev = np.array([[track_xy]], dtype=np.float32).reshape(1, 1, 2)
        pt_next, st_pt, _ = cv.calcOpticalFlowPyrLK(
            prev_gray, gray, pt_prev, None,
            winSize=WIN_SIZE, maxLevel=0, criteria=LK_CRITERIA
        )
        if pt_next is not None and st_pt is not None and int(st_pt[0, 0]) == 1:
            nx, ny = pt_next.reshape(-1, 2)[0]
            if 0 <= nx < w and 0 <= ny < h:
                track_xy = (float(nx), float(ny))
                st_val = 1

        if st_val == 0:
            new_xy = reinit_point_on_frame(gray_u8, motion_mask, last_xy=track_xy)
            if new_xy is not None:
                track_xy = new_xy
                st_val = 2
            else:
                track_xy = (float("nan"), float("nan"))
                st_val = 0

        draw_track_box(vis, track_xy)
        writer.write(vis)
        track_coords.append((track_xy[0], track_xy[1], st_val))

        prev_gray = gray
        prev_gray_u8 = gray_u8
        frame_idx += 1
        processed += 1
        print_progress(processed, total_frames, start_time, prefix="[OpenCV] ")

    cap.release()
    writer.release()
    sys.stdout.write("\n")
    print(f"[OpenCV] Обробка завершена. Результат збережено у: {out_path}")
    return track_coords, fps



def process_video_lk_mine(video_path: Path, track_init_xy_on_frame1):
    print(f"[Mine] Обробка відео: {video_path}")

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("[Mine] Не вдалося відкрити відео.")
        return [], 0.0

    fps = cap.get(cv.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None

    processed_dir = Path("processed")
    processed_dir.mkdir(exist_ok=True)
    out_path = processed_dir / f"{video_path.stem}_lucas-kanade_mine.mp4"

    writer = cv.VideoWriter(str(out_path), cv.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    prev_gray = None
    prev_gray_u8 = None

    track_xy = None
    track_initialized = False
    track_coords = []

    start_time = time.time()
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_bgr = frame.copy()
        gray_u8 = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        gray = gray_u8.astype(np.float32) / 255.0

        if prev_gray is None:
            prev_gray = gray
            prev_gray_u8 = gray_u8


            writer.write(frame_bgr)
            track_coords.append((float("nan"), float("nan"), 3))
            processed += 1
            continue


        diff = np.abs(gray - prev_gray)
        diff = cv.GaussianBlur(diff, (5, 5), 0)
        motion_mask = (diff > MOTION_DIFF_THRESH).astype(np.uint8) * 255

        kernel = np.ones((3, 3), np.uint8)
        motion_mask = cv.morphologyEx(motion_mask, cv.MORPH_OPEN, kernel, iterations=1)
        motion_mask = cv.dilate(motion_mask, kernel, iterations=1)


        vis = highlight_motion_with_contour(frame_bgr, motion_mask, overlay_color=COLOR_BLUE, contour_color=COLOR_BLUE)

        all_corners_prev = shi_tomasi_corners(prev_gray)
        if all_corners_prev.shape[0] > 0:
            all_corners_curr, all_status = compute_lk_single(prev_gray, gray, all_corners_prev)
            valid = all_status == 1
            p0_all = all_corners_prev[valid]
            p1_all = all_corners_curr[valid]
        else:
            p0_all = np.empty((0, 2), dtype=np.float32)
            p1_all = np.empty((0, 2), dtype=np.float32)

        if p0_all.shape[0] > 0:
            for (x0, y0), (x1, y1) in zip(p0_all, p1_all):
                mag = math.hypot(x1 - x0, y1 - y0)
                if mag < MIN_FLOW_MAG:
                    continue
                ix, iy = int(round(x1)), int(round(y1))
                if not (0 <= ix < width and 0 <= iy < height):
                    continue
                if motion_mask[iy, ix] == 0:
                    continue
                cv.arrowedLine(
                    vis,
                    (int(round(x0)), int(round(y0))),
                    (int(round(x1)), int(round(y1))),
                    COLOR_RED,
                    2,
                    tipLength=0.3
                )


        if not track_initialized:
            track_xy = track_init_xy_on_frame1
            track_initialized = True
            draw_track_box(vis, track_xy)
            writer.write(vis)
            track_coords.append((track_xy[0], track_xy[1], 3))

            prev_gray = gray
            prev_gray_u8 = gray_u8
            processed += 1
            if processed % 10 == 0:
                print_progress(processed, total_frames, start_time, prefix="[Mine] ")
            continue


        st_val = 0
        pt_prev = np.array([[track_xy[0], track_xy[1]]], dtype=np.float32)
        pt_next, st_pt = compute_lk_single(prev_gray, gray, pt_prev)
        if pt_next is not None and st_pt is not None and int(st_pt[0]) == 1:
            nx, ny = pt_next[0]
            if 0 <= nx < width and 0 <= ny < height:
                track_xy = (float(nx), float(ny))
                st_val = 1

        if st_val == 0:
            new_xy = reinit_point_on_frame(gray_u8, motion_mask, last_xy=track_xy)
            if new_xy is not None:
                track_xy = new_xy
                st_val = 2
            else:
                track_xy = (float("nan"), float("nan"))
                st_val = 0

        draw_track_box(vis, track_xy)
        writer.write(vis)
        track_coords.append((track_xy[0], track_xy[1], st_val))

        prev_gray = gray
        prev_gray_u8 = gray_u8

        processed += 1
        if processed % 10 == 0:
            print_progress(processed, total_frames, start_time, prefix="[Mine] ")

    cap.release()
    writer.release()
    sys.stdout.write("\n")
    print(f"[Mine] Обробка завершена. Результат збережено у: {out_path}")
    return track_coords, fps


# -------------------- Compare output --------------------
def print_and_save_compare(video_stem, coords_cv, coords_mn, fps):
    out_dir = Path("processed")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"track_compare_{video_stem}.csv"

    n = min(len(coords_cv), len(coords_mn))
    rows = []
    for i in range(n):
        xcv, ycv, scv = coords_cv[i]
        xmn, ymn, smn = coords_mn[i]
        t = i / fps if fps and fps > 0 else float(i)

        dist = float("nan")
        if np.isfinite(xcv) and np.isfinite(ycv) and np.isfinite(xmn) and np.isfinite(ymn):
            dist = math.hypot(xcv - xmn, ycv - ymn)

        rows.append((i, t, xcv, ycv, scv, xmn, ymn, smn, dist))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_sec", "x_opencv", "y_opencv", "st_opencv", "x_mine", "y_mine", "st_mine", "dist_px"])
        w.writerows(rows)

    def fmtv(v):
        if v is None or not np.isfinite(v):
            return "--"
        return f"{v:8.2f}"

    print("\nПорівняння координат контрольної точки")
    header = f"{'fr':>6} | {'x_cv':>8} {'y_cv':>8} {'st':>3} || {'x_mn':>8} {'y_mn':>8} {'st':>3} || {'dist':>8}"
    print(header)
    print("-" * len(header))

    if n <= 200:
        show_idx = list(range(n))
    else:
        show_idx = list(range(60)) + list(range(max(60, n - 20), n))

    for i in show_idx:
        fr, t, xcv, ycv, scv, xmn, ymn, smn, dist = rows[i]
        line = f"{fr:6d}  {fmtv(xcv)} {fmtv(ycv)} {scv:3d} || {fmtv(xmn)} {fmtv(ymn)} {smn:3d} || {fmtv(dist)}"
        print(line)

    if n > 200:
        print(f"\n(Показано перші 60 та останні 20 рядків; повна таблиця у CSV: {csv_path})")
    else:
        print(f"\nCSV збережено: {csv_path}")



def main():
    video_path = choose_video()
    if video_path is None:
        return

    track_init = pick_corner_on_second_frame_that_moved(video_path)
    print(f"\nКонтрольна точка (на 2-му кадрі): {track_init}")

    print("\n--- Лукас–Канаде в OpenCV реалізації ---")
    t0 = time.time()
    coords_cv, fps_cv = process_video_lk_opencv(video_path, track_init)
    t1 = time.time()

    print("\n--- Лукас–Канаде у власній реалізації ---")
    coords_mn, fps_mn = process_video_lk_mine(video_path, track_init)
    t2 = time.time()

    fps = fps_cv if fps_cv and fps_cv > 0 else fps_mn

    dt_opencv = t1 - t0
    dt_mine = t2 - t1

    print("\nЧас роботи програми")
    print(f"Lucas–Kanade OpenCV : {dt_opencv:.2f} с ({fmt_hms(dt_opencv)})")
    print(f"Lucas–Kanade ручний : {dt_mine:.2f} с ({fmt_hms(dt_mine)})")

    print_and_save_compare(video_path.stem, coords_cv, coords_mn, fps)


if __name__ == "__main__":
    main()
