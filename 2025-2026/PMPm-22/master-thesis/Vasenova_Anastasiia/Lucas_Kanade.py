import math
import time
import sys
import os
from pathlib import Path
from collections import deque
import cv2
import numpy as np
import cv2 as cv
import cupy as cp
from cupyx.scipy.ndimage import uniform_filter
from numba import njit, prange

cv.setUseOptimized(True)
try:
    cv.setNumThreads(cv.getNumberOfCPUs())
except Exception:
    pass


r = 3
max_num_corners = 350
alpha = 0.01 # надійність того, що точка є кутовою
min_dist = 9 # мінімальна відстань між кутовими точками
eps = 1e-4
delta = 0.4

params_shi_tomasi_opencv = dict(maxCorners=max_num_corners, qualityLevel=alpha, minDistance=min_dist, blockSize=7)
win_opencv = (2 * r + 1, 2 * r + 1)
lk_opencv_criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 1, eps)

size_gauss_core = 5
diff_threshold = 18

arrow_ttl_sec = 0.5 # скільки секунд живе стрілка
min_vec_len = 3.0
min_num_corners = 150
reinit_sec = 0.5 # час, через який обираються нові кутові точки
mask_hit_r = 2 # радіус, в якому перевіряємо наявність руху
arrow_thickness = 1
alpha_fill = 0.40 # прозорість заливки
arrow_scale = 0.8
color_opencv = (255, 0, 0)
color_mine = (0, 0, 255)

square_size = 25
track_color = (0, 255, 0)
track_thickness = 2
track_tail_len = 40 # скільки кадрів живе траєкторія точки

output_dir = "processed"



def time_format(sec: float) -> str:

    sec = int(max(0, sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60

    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"



def print_progress(processed, total, start_time, prefix=""):

    elapsed = time.time() - start_time
    if total and total > 0:
        msg = f"\r{prefix}Кадр {processed}/{total}    Elapsed {time_format(elapsed)}"
    else:
        msg = f"\r{prefix}Frames: {processed}  Elapsed {time_format(elapsed)}"
    sys.stdout.write(msg)
    sys.stdout.flush()



def choose_video():

    videos_dir = Path("videos")
    extentions = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")
    videos = sorted([p for p in videos_dir.iterdir() if p.suffix in extentions])

    print("Доступні відео:")
    for i, vf in enumerate(videos):
        print(f"[{i}] {vf.name}")

    while True:
        try:
            idx = int(input("Виберіть номер відео для обробки: "))
            if 0 <= idx < len(videos):
                return videos[idx]
            print("Невірний номер!")
        except ValueError:
            print("Введіть ціле число!")



def choose_track_point(video_path) -> tuple:

    cap = cv.VideoCapture(str(video_path))
    ret, frame0 = cap.read()
    cap.release()
    if not ret:
        print("Не вдалося прочитати перший кадр.")
        sys.exit(1)

    H, W = frame0.shape[:2]
    point = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            point[0] = (x, y)

    win = "Натисніть на точку для відстеження, потім – Enter"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.setMouseCallback(win, on_mouse)

    while True:
        show = frame0.copy()
        if point[0] is not None:
            px, py = point[0]
            sr = square_size // 2
            cv.rectangle(show, (px - sr, py - sr), (px + sr, py + sr),
                         track_color, track_thickness)
        cv.imshow(win, show)
        key = cv.waitKey(20) & 0xFF
        if key == 13 and point[0] is not None:   # Enter
            break

    cv.destroyWindow(win)
    px, py = point[0]
    px = int(np.clip(px, 0, W - 1))
    py = int(np.clip(py, 0, H - 1))
    print(f"Обрана точка для відстеження: ({px}, {py})")

    return px, py



def if_point_in_area(mask, x, y, r):

    h, w = mask.shape
    x1 = max(0, x - r);  x2 = min(w - 1, x + r)
    y1 = max(0, y - r);  y2 = min(h - 1, y + r)

    return np.any(mask[y1:y2 + 1, x1:x2 + 1] > 0)



def cleanup_arrows(arrows_deque, frame_idx, max_num_frames):

    while arrows_deque and (frame_idx - arrows_deque[0][0]) > max_num_frames:
        arrows_deque.popleft()



def make_motion_mask(prev_frame, cur_frame):

    diff = cv.absdiff(cur_frame, prev_frame)
    diff = cv.GaussianBlur(diff, (size_gauss_core, size_gauss_core), 0)
    _, mask = cv.threshold(diff, diff_threshold, 255, cv.THRESH_BINARY)
    k_small = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k_small, iterations=1)
    k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k_close, iterations=2)

    return mask



def draw_detection(frame, motion_mask, arrows_deque, frame_idx, arrow_ttl_frames, color):

    res = frame.copy()
    overlay = frame.copy()
    overlay[motion_mask > 0] = color
    res = cv.addWeighted(overlay, alpha_fill, res, 1.0 - alpha_fill, 0)
    contours, _ = cv.findContours(motion_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv.contourArea(c) >= 40:
            cv.drawContours(res, [c], -1, color, arrow_thickness)
    for f0, p_from, p_to in arrows_deque:
        if (frame_idx - f0) <= arrow_ttl_frames:
            p0 = np.array(p_from, dtype=np.float32)
            p1 = np.array(p_to,   dtype=np.float32)
            p2 = p0 + arrow_scale * (p1 - p0)
            a = tuple(np.round(p_from).astype(int))
            b = tuple(np.round(p2).astype(int))
            cv.arrowedLine(res, a, b, color, arrow_thickness, tipLength=0.25)

    return res



def draw_track(frame_bgr, px, py, history):

    out = frame_bgr.copy()
    H, W = out.shape[:2]
    tail = history[-track_tail_len:] + [(px, py)]
    for i in range(1, len(tail)):
        x1t = int(round(tail[i - 1][0]))
        y1t = int(round(tail[i - 1][1]))
        x2t = int(round(tail[i][0]))
        y2t = int(round(tail[i][1]))
        a = i / max(len(tail), 1)
        clr = (0, int(80 + 175 * a), 0)
        cv.line(out, (x1t, y1t), (x2t, y2t), clr, 1, cv.LINE_AA)
    ix, iy = int(round(px)), int(round(py))
    hs = square_size // 2
    pt1 = (max(0, ix - hs), max(0, iy - hs))
    pt2 = (min(W - 1, ix + hs), min(H - 1, iy + hs))
    cv.rectangle(out, pt1, pt2, track_color, track_thickness)
    lx = min(ix + hs + 3, W - 90)
    ly = max(iy, 14)
    cv.putText(out, f"({ix},{iy})", (lx, ly), cv.FONT_HERSHEY_SIMPLEX, 0.45, track_color, 1, cv.LINE_AA)

    return out



@njit(parallel=True, fastmath=True)
def compute_image_gradients(I):

    Ix = np.zeros_like(I, dtype=np.float32)
    Iy = np.zeros_like(I, dtype=np.float32)
    Ix[:, 1:-1] = (I[:, 2:] - I[:, :-2]) * 0.5
    Iy[1:-1, :] = (I[2:, :] - I[:-2, :]) * 0.5
    return Ix, Iy



def shi_tomasi_corners_mine(Is, max_num_corners, alpha, min_dist):

    h, w = Is.shape
    I = Is.astype(np.float32)
    win = 2 * r + 1

    if h - 2 * r <= 0 or w - 2 * r <= 0:
        return np.empty((0, 2), dtype=np.float32)

    Ix, Iy = compute_image_gradients(I)
    Ixx = Ix * Ix;  Iyy = Iy * Iy;  Ixy = Ix * Iy

    Ixx_gpu = cp.asarray(Ixx)
    Iyy_gpu = cp.asarray(Iyy)
    Ixy_gpu = cp.asarray(Ixy)

    Mxx = uniform_filter(Ixx_gpu, size=win)[r:h - r, r:w - r]
    Myy = uniform_filter(Iyy_gpu, size=win)[r:h - r, r:w - r]
    Mxy = uniform_filter(Ixy_gpu, size=win)[r:h - r, r:w - r]
    Mxx = cp.asnumpy(Mxx)
    Myy = cp.asnumpy(Myy)
    Mxy = cp.asnumpy(Mxy)

    trace = Mxx + Myy
    det = Mxx * Myy - Mxy * Mxy
    tmp = np.sqrt(np.maximum(trace * trace - 4 * det, 0))
    lambda_min = np.minimum((trace + tmp) * 0.5, (trace - tmp) * 0.5)

    pad = win // 2
    R = np.zeros((h, w), dtype=np.float32)
    R[pad:h - pad, pad:w - pad] = lambda_min

    R_max = float(R.max())
    if R_max <= 0:
        return np.empty((0, 2), dtype=np.float32)
    y_z, x_z = np.where(R > R_max * alpha)
    if y_z.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    R_z = R[y_z, x_z]
    M = min(R_z.size, max_num_corners * 20)
    if R_z.size > M:
        idx = np.argpartition(R_z, -M)[-M:]
        y_z, x_z, R_z = y_z[idx], x_z[idx], R_z[idx]

    order = np.argsort(-R_z)
    x_z, y_z = x_z[order], y_z[order]

    cell = max(1, int(min_dist))
    gw = (w + cell - 1) // cell
    gh = (h + cell - 1) // cell
    grid = -np.ones((gh, gw), dtype=np.int32)
    md2 = float(min_dist) ** 2
    corners = []

    for x, y in zip(x_z, y_z):
        gx, gy, ok = int(x) // cell, int(y) // cell, True
        for ny in range(max(0, gy - 1), min(gh, gy + 2)):
            for nx in range(max(0, gx - 1), min(gw, gx + 2)):
                j = grid[ny, nx]

                if j != -1:
                    cx, cy = corners[j]
                    if (cx - x) ** 2 + (cy - y) ** 2 < md2:
                        ok = False;  break

            if not ok:
                break

        if ok:
            corners.append((int(x), int(y)))
            grid[gy, gx] = len(corners) - 1

            if len(corners) >= max_num_corners:
                break

    if not corners:
        return np.empty((0, 2), dtype=np.float32)

    return np.array(corners, dtype=np.float32)



@njit(parallel=True, fastmath=True)
def lk_mine(prev, curr, points, r, eps):

    h, w = prev.shape
    n = points.shape[0]
    new_points = np.zeros((n, 2), np.float32)
    status     = np.zeros((n,),   np.uint8)

    Ix, Iy = compute_image_gradients(prev)
    Ik = curr - prev

    for i in prange(n):
        x = float(points[i, 0])
        y = float(points[i, 1])

        if x < r or x >= w - r or y < r or y >= h - r:
            continue

        Mxx = Myy = Mxy = Dx = Dy = 0.0

        for yy in range(int(y) - r, int(y) + r + 1):
            for xx in range(int(x) - r, int(x) + r + 1):
                Mxx += Ix[yy, xx] * Ix[yy, xx]
                Mxy += Ix[yy, xx] * Iy[yy, xx]
                Myy += Iy[yy, xx] * Iy[yy, xx]
                Dx  += -Ix[yy, xx] * Ik[yy, xx]
                Dy  += -Iy[yy, xx] * Ik[yy, xx]

        det = Mxx * Myy - Mxy * Mxy

        if det < delta:
            continue

        u = (Myy * Dx - Mxy * Dy) / det
        v = (-Mxy * Dx + Mxx * Dy) / det
        new_points[i, 0] = x + u
        new_points[i, 1] = y + v
        status[i] = 1

    return new_points, status



def process_video_lk_opencv(video_path: Path, track_x0: float, track_y0: float) -> list:

    print(f"[OpenCV] Обробка відео: {video_path}")
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("[OpenCV] Не вдалось відкрити відео.")
        return []

    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or None

    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"{video_path.stem}_lucas-kanade_opencv.mp4"
    writer = cv.VideoWriter(str(out_path), cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    ttl_frames = max(1, int(round(arrow_ttl_sec * fps)))
    arrows = deque()
    reinit_every = max(1, int(fps * reinit_sec))

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return []

    prev_frame = cv.GaussianBlur(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), (5, 5), 0)
    p0 = cv.goodFeaturesToTrack(prev_frame, mask=None, **params_shi_tomasi_opencv)

    # відстеження обраної точки
    tracked_pt = np.array([[track_x0, track_y0]], dtype=np.float32).reshape(1, 1, 2)
    track_history = []
    track_log     = []

    frame_idx  = 0
    processed  = 0
    start_time = time.time()
    print(f"[OpenCV] Запис у {out_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        cur_frame   = cv.GaussianBlur(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), (5, 5), 0)
        motion_mask = make_motion_mask(prev_frame, cur_frame)

        if p0 is None or len(p0) < min_num_corners or (frame_idx % reinit_every == 0):
            p0 = cv.goodFeaturesToTrack(prev_frame, mask=None, **params_shi_tomasi_opencv)

        if p0 is not None and len(p0) > 0:
            p1, status, _ = cv.calcOpticalFlowPyrLK(prev_frame, cur_frame, p0, None,
                                                    win_opencv, maxLevel=0, criteria=lk_opencv_criteria)

            valid = status.reshape(-1) == 1

            prev_valid = p0.reshape(-1, 2)[valid]
            new_valid = p1.reshape(-1, 2)[valid]

            h_m, w_m = motion_mask.shape

            for (x0c, y0c), (x1c, y1c) in zip(prev_valid, new_valid):
                if math.hypot(x1c - x0c, y1c - y0c) < min_vec_len:
                    continue
                ix, iy = int(round(x1c)), int(round(y1c))

                if 0 <= ix < w_m and 0 <= iy < h_m:
                    if if_point_in_area(motion_mask, ix, iy, r=mask_hit_r):
                        arrows.append((frame_idx, (x0c, y0c), (x1c, y1c)))

            p0 = new_valid.reshape(-1, 1, 2).astype(np.float32)

        #відстеження обраної точки
        new_tracked, st, _ = cv.calcOpticalFlowPyrLK(prev_frame, cur_frame, tracked_pt, None,
                                                     win_opencv, maxLevel=0, criteria=lk_opencv_criteria)

        if st[0][0] == 1:
            tracked_pt = new_tracked

        px = float(tracked_pt[0, 0, 0])
        py = float(tracked_pt[0, 0, 1])

        track_log.append((frame_idx, px, py))
        track_history.append((px, py))

        cleanup_arrows(arrows, frame_idx, ttl_frames)
        vis = draw_detection(frame, motion_mask, arrows, frame_idx, ttl_frames, color_opencv)
        vis = draw_track(vis, px, py, track_history)

        writer.write(vis)
        prev_frame = cur_frame
        processed += 1
        print_progress(processed, total_frames, start_time, prefix="[OpenCV] ")

    cap.release()
    writer.release()
    sys.stdout.write("\n")
    print(f"[OpenCV] Відео збережено в {out_path}")

    return track_log



def process_video_lk_mine(video_path: Path, track_x0: float, track_y0: float) -> list:

    print(f"[Mine] Обробка відео: {video_path}")
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("[Mine] Не вдалося відкрити відео.")
        return []

    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or None

    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"{video_path.stem}_lucas-kanade_mine.mp4"
    writer = cv.VideoWriter(str(out_path), cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    ttl_frames = max(1, int(round(arrow_ttl_sec * fps)))
    arrows = deque()
    reinit_every = max(1, int(fps * reinit_sec))

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return []

    prev_u8 = cv.GaussianBlur(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), (5, 5), 0)
    prev_f32 = prev_u8.astype(np.float32)
    p0 = shi_tomasi_corners_mine(prev_u8, max_num_corners, alpha, min_dist)

    tracked_pt = np.array([[track_x0, track_y0]], dtype=np.float32)
    track_history = []
    track_log = []

    frame_idx = 0
    processed = 0
    start_time = time.time()
    print(f"[Mine] Запис у {out_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        cur_u8 = cv.GaussianBlur(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), (5, 5), 0)
        cur_f32 = cur_u8.astype(np.float32)
        motion_mask = make_motion_mask(prev_u8, cur_u8)

        if p0 is None or p0.shape[0] < min_num_corners or (frame_idx % reinit_every == 0):
            p0 = shi_tomasi_corners_mine(prev_u8, max_num_corners, alpha, min_dist)

        if p0 is not None and p0.shape[0] > 0:
            p1, if_lost = lk_mine(prev_f32, cur_f32, p0, r, eps)
            valid = if_lost == 1
            prev_valid = p0[valid]
            new_valid = p1[valid]
            h_m, w_m = motion_mask.shape

            for (x0c, y0c), (x1c, y1c) in zip(prev_valid, new_valid):
                if math.hypot(x1c - x0c, y1c - y0c) < min_vec_len:
                    continue

                ix, iy = int(round(x1c)), int(round(y1c))
                if 0 <= ix < w_m and 0 <= iy < h_m:
                    if if_point_in_area(motion_mask, ix, iy, r=mask_hit_r):
                        arrows.append((frame_idx, (x0c, y0c), (x1c, y1c)))

            p0 = new_valid.copy() if new_valid.shape[0] > 0 else None

        #відстеження обраної точки
        new_pts, st = lk_mine(prev_f32, cur_f32, tracked_pt, r, eps)

        if st[0] == 1:
            tracked_pt = new_pts

        px = float(tracked_pt[0, 0])
        py = float(tracked_pt[0, 1])
        track_log.append((frame_idx, px, py))
        track_history.append((px, py))

        cleanup_arrows(arrows, frame_idx, ttl_frames)
        vis = draw_detection(frame, motion_mask, arrows, frame_idx, ttl_frames, color_mine)
        vis = draw_track(vis, px, py, track_history)

        writer.write(vis)
        prev_u8  = cur_u8
        prev_f32 = cur_f32
        processed += 1
        print_progress(processed, total_frames, start_time, prefix="[Mine] ")

    cap.release()
    writer.release()
    sys.stdout.write("\n")
    print(f"[Mine] Відео збережено в {out_path}")
    return track_log



def print_comparison_table(track_opencv: list, track_mine: list, video_name: str, x0: float, y0: float, H: str, W: str):

    n = min(len(track_opencv), len(track_mine))
    if n == 0:
        print("Немає даних для порівняння траєкторій!")
        return

    sep = ("+" + "-" * 7 + "+" + "-" * 13 + "+" + "-" * 13 +
           "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 11 + "+" + "-" * 11 + "+")
    hdr = (f"| {'Кадр':^5} | {'x_o':^11} | {'y_o':^11} "
           f"| {'x_m':^11} | {'y_m':^11} "
           f"| {'Δx^k':^9} | {'Δy^k':^9} |")

    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append(f" Початкова точка: ({x0}, {y0})")
    lines.append("=" * 90)
    lines.append(sep)
    lines.append(hdr)
    lines.append(sep)

    dxs, dys = [], []
    for i in range(n):
        fi, ox, oy = track_opencv[i]
        _,  mx, my = track_mine[i]
        dx = ox - mx
        dy = oy - my
        dxs.append(dx)
        dys.append(dy)
        lines.append(
            f"| {fi:^5d} | {ox:^11.2f} | {oy:^11.2f} "
            f"| {mx:^11.2f} | {my:^11.2f} "
            f"| {dx:^9.2f} | {dy:^9.2f} |"
        )

    lines.append(sep)
    lines.append(f"  Δx^a = {np.mean(np.abs(dxs)):.3f} px   "
                 f"Δy^a = {np.mean(np.abs(dys)):.3f} px")
    lines.append(f"  Δx^m = {np.max(np.abs(dxs)):.3f} px   "
                 f"Δy^m = {np.max(np.abs(dys)):.3f} px")

    rms_x = math.sqrt(sum(d ** 2 for d in dxs) / n)
    rms_y = math.sqrt(sum(d ** 2 for d in dys) / n)
    lines.append(f"  x^RMS = {rms_x:.3f} px   "
                 f"y^RMS = {rms_y:.3f} px")

    lines.append("=" * 90)

    lines.append(H)
    lines.append(W)

    output = "\n".join(lines)
    print(output)

    os.makedirs(output_dir, exist_ok=True)
    table_path = os.path.join(output_dir, f"{video_name}_table_LK.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"\nТаблицю збережено: {table_path}")



def main():

    video_path = choose_video()
    if video_path is None:
        return

    print("\nОберіть точку для відстеження на першому кадрі")
    print("Клікніть мишею -> натисніть Enter")
    track_x0, track_y0 = choose_track_point(video_path)

    print("\n---- OpenCV реалізація [СИНІЙ] ----")
    t0 = time.time()
    track_opencv = process_video_lk_opencv(video_path, track_x0, track_y0)
    t1 = time.time()

    print("\n---- Власна реалізація [ЧЕРВОНИЙ] ----")
    track_mine = process_video_lk_mine(video_path, track_x0, track_y0)
    t2 = time.time()

    dt_cv   = t1 - t0
    dt_mine = t2 - t1
    print(f"\nOpenCV ЛК :  {dt_cv:.2f} с  ({time_format(dt_cv)})")
    print(f"Власний ЛК : {dt_mine:.2f} с  ({time_format(dt_mine)})")

    cap = cv2.VideoCapture(video_path)
    H = str(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    W = str(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    video_name = video_path.stem
    print_comparison_table(track_opencv, track_mine, video_name, track_x0, track_y0, H, W)

    print(f"\nВихід у папці : {output_dir}/")


if __name__ == "__main__":
    main()