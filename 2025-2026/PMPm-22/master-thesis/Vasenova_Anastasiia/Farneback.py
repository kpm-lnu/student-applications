import os
import sys
import math
import time
import cv2 as cv
import numpy as np
from pathlib import Path
from numba import njit, prange


output_dir = "processed"

perc_flow = 70.0
T_d = 0.2 #порогове значення для оптичного потоку
T_f = 12 #порогове значення для різниці кадрів

morph_kernel_size = 3
dil_ker_size = 3

min_obj_area = 50
max_area_coeff = 0.60
cutting_bound_coeff = 0.04

min_num_mov_points = 250

alpha_fill = 0.25
arrow_step = 18
arrow_min_flow_len = 0.6
arrow_coef = 3.0

color_opencv = (255, 0, 0)
arrow_color_opencv = (0, 0, 255)
color_mine = (0, 0, 255)
arrow_color_mine   = (255, 0, 0)

square_size = 25
track_color = (0, 255, 0)
track_thickness = 2
track_tail_len = 40 #довжина вектора, що показує траєкторію відстежуваної точки

win_r = 7  #радіус локального вікна для апроксимації
sigma_gauss = 3.5  #σ для гаусового зважування
avg_r = 5  #радіус вікна усереднення
sigma_avg = 2.5  #σ для гаусового вікна усереднення
lambda_flow = 1e-4  #регуляризація для det
flow_iters = 3
s = 2  #зменшення кадру у s разів
flow_step = 1
pyr_scale  = 0.5

_global_pinv_kernels = None
_global_avg_kernel   = None


def time_format(sec: float) -> str:

    sec = max(0, int(sec))
    h   = sec // 3600
    m   = (sec % 3600) // 60
    s   = sec % 60

    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"



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

    win = "Натисніть на точку, яку бажаєте відстежити, і натисніть Enter!"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.setMouseCallback(win, on_mouse)

    while True:
        show = frame0.copy()
        if point[0] is not None:
            px, py = point[0]
            sr = square_size // 2
            cv.rectangle(show, (px - sr, py - sr), (px + sr, py + sr), track_color, track_thickness)

        cv.imshow(win, show)
        key = cv.waitKey(20) & 0xFF

        if key == 13 and point[0] is not None:
            break

    cv.destroyWindow(win)

    px, py = point[0]
    px = int(np.clip(px, 0, W - 1))
    py = int(np.clip(py, 0, H - 1))
    print(f"Обрана точка для відстеження: ({px}, {py})")

    return px, py



def downscale(frame: np.ndarray, factor: int) -> np.ndarray:

    H, W = frame.shape[:2]
    new_w = max(1, W // factor)
    new_h = max(1, H // factor)

    return cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)



def upscale(u_small: np.ndarray, v_small: np.ndarray,
            factor: int, out_shape: tuple):

    H, W = out_shape
    u_big = cv.resize(u_small, (W, H), interpolation=cv.INTER_LINEAR)
    v_big = cv.resize(v_small, (W, H), interpolation=cv.INTER_LINEAR)

    return u_big * factor, v_big * factor # повертаємо до початкового масштабу вектор переміщення



@njit(parallel=True, fastmath=True, cache=True)
def conv(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    r_y = kh // 2
    r_x = kw // 2
    H, W = image.shape
    out = np.empty((H, W), np.float32)

    for y in prange(H):
        for x in range(W):
            acc = np.float32(0.0)
            for ky in range(kh):
                for kx in range(kw):
                    iy = y + ky - r_y
                    ix = x + kx - r_x

                    if iy < 0:
                        iy = -iy

                    elif iy >= H:
                        iy = 2 * (H - 1) - iy

                    if ix < 0:
                        ix = -ix

                    elif ix >= W:
                        ix = 2 * (W - 1) - ix

                    acc += image[iy, ix] * kernel[ky, kx]

            out[y, x] = acc

    return out



def binary_dilate(mask_bin, ksize, iterations=1):

    kernel = np.ones((ksize, ksize), dtype=np.float32)
    cur    = mask_bin.astype(np.float32)

    for _ in range(iterations):
        cur = (conv(cur, kernel) > 0).astype(np.float32)

    return cur.astype(np.uint8)



def binary_erode(mask_bin, ksize, iterations=1):

    kernel = np.ones((ksize, ksize), dtype=np.float32)
    need   = ksize * ksize
    cur    = mask_bin.astype(np.float32)

    for _ in range(iterations):
        cur = (conv(cur, kernel) >= need).astype(np.float32)

    return cur.astype(np.uint8)



def binary_open(mask_bin, ksize, iterations=1):

    return binary_dilate(binary_erode(mask_bin, ksize, iterations), ksize, iterations)



def cut_boundaries(mask):

    h, w   = mask.shape
    margin = int(cutting_bound_coeff * min(h, w))

    if margin <= 0:
        return mask

    out = mask.copy()

    out[:margin, :]  = 0
    out[-margin:, :] = 0
    out[:, :margin]  = 0
    out[:, -margin:] = 0

    return out



def filter_components(mask):

    h, w = mask.shape
    max_area = max_area_coeff * h * w
    num, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)

    if num <= 1:
        return mask

    keep = np.zeros_like(mask)
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]

        if area < min_obj_area or area > max_area:
            continue

        if (x == 0) or (y == 0) or ((x + ww) >= w) or ((y + hh) >= h):
            continue

        keep[labels == i] = 255

    return keep



def build_motion_mask_flow(d):

    d_valid = d[d > 1e-12]

    if d_valid.size == 0:
        return np.zeros_like(d, dtype=np.uint8)

    thr = max(T_d, float(np.percentile(d_valid, perc_flow)))
    mask = (d > thr).astype(np.uint8)

    mask = binary_open(mask, morph_kernel_size, 1)
    mask = binary_dilate(mask, morph_kernel_size, 1)

    return (mask * 255).astype(np.uint8)



def build_motion_mask_diff(diff_full):

    diff_255 = diff_full.astype(np.uint8)
    mask = (diff_255 > T_f).astype(np.uint8)

    mask = binary_open(mask, morph_kernel_size, 1)
    mask = binary_dilate(mask, morph_kernel_size, 1)

    return (mask * 255).astype(np.uint8)



def combine_masks(mask_flow, mask_diff):

    motion_and = (mask_flow & mask_diff)
    motion_and = cut_boundaries(motion_and)
    motion_and = filter_components(motion_and)

    if int(np.count_nonzero(motion_and)) < min_num_mov_points:
        motion_or = (mask_flow | mask_diff)
        motion_or = cut_boundaries(motion_or)
        motion_or = filter_components(motion_or)
        return motion_or

    return motion_and



def build_motion_mask_common(mag, diff_full):

    mask_flow = build_motion_mask_flow(mag)
    mask_diff = build_motion_mask_diff(diff_full)
    motion = combine_masks(mask_flow, mask_diff)

    if np.any(motion):
        mb = (motion > 0).astype(np.uint8)
        mb = binary_dilate(mb, dil_ker_size, 1)
        motion = (mb * 255).astype(np.uint8)

    return motion



def find_contours(mask_bin):

    padded = np.pad(mask_bin, 1, mode='constant', constant_values=0)
    kernel = np.ones((3, 3), dtype=np.float32)
    local_sum = conv(padded.astype(np.float32), kernel)[1:-1, 1:-1]

    return ((mask_bin == 1) & (local_sum < 9)).astype(np.uint8)



def update_track_point(px: float, py: float, u: np.ndarray, v: np.ndarray, frame_shape) -> tuple:

    H, W = frame_shape[:2]
    cx = float(np.clip(px, 0, W - 1))
    cy = float(np.clip(py, 0, H - 1))

    x0, y0 = int(cx), int(cy)
    x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
    fx, fy = cx - x0, cy - y0

    du = ((1 - fx) * (1 - fy) * u[y0, x0] +
             fx    * (1 - fy) * u[y0, x1] +
          (1 - fx) *    fy    * u[y1, x0] +
             fx    *    fy    * u[y1, x1])

    dv = ((1 - fx) * (1 - fy) * v[y0, x0] +
             fx    * (1 - fy) * v[y0, x1] +
          (1 - fx) *    fy    * v[y1, x0] +
             fx    *    fy    * v[y1, x1])

    nx = float(np.clip(cx + du, 0, W - 1))
    ny = float(np.clip(cy + dv, 0, H - 1))

    return nx, ny



def draw_track(frame_bgr: np.ndarray, px: float, py: float, history: list) -> np.ndarray:

    out  = frame_bgr.copy()
    H, W = out.shape[:2]

    tail = history[-track_tail_len:] + [(px, py)]
    for i in range(1, len(tail)):
        x1t = int(round(tail[i - 1][0]))
        y1t = int(round(tail[i - 1][1]))
        x2t = int(round(tail[i][0]))
        y2t = int(round(tail[i][1]))
        alpha = i / max(len(tail), 1)
        color = (0, int(80 + 175 * alpha), 0)
        cv.line(out, (x1t, y1t), (x2t, y2t), color, 1, cv.LINE_AA)

    ix, iy = int(round(px)), int(round(py))
    hs = square_size // 2
    pt1 = (max(0, ix - hs), max(0, iy - hs))
    pt2 = (min(W - 1, ix + hs), min(H - 1, iy + hs))
    cv.rectangle(out, pt1, pt2, track_color, track_thickness)

    lx = min(ix + hs + 3, W - 90)
    ly = max(iy, 14)
    cv.putText(out, f"({ix},{iy})", (lx, ly), cv.FONT_HERSHEY_SIMPLEX,
               0.45, track_color, 1, cv.LINE_AA)

    return out



def draw_flow(frame_bgr, motion_mask, contours_mask, u, v, fill_color_bgr, arrow_color_bgr):

    out = frame_bgr.copy()

    if alpha_fill > 0 and np.any(motion_mask):
        overlay = out.copy()
        overlay[motion_mask > 0] = fill_color_bgr
        out = cv.addWeighted(overlay, alpha_fill, out, 1.0 - alpha_fill, 0)

    out[contours_mask == 1] = fill_color_bgr

    if np.any(motion_mask):
        h, w = motion_mask.shape
        for yy in range(0, h, arrow_step):
            for xx in range(0, w, arrow_step):
                if motion_mask[yy, xx] == 0:
                    continue

                du  = float(u[yy, xx])
                dv  = float(v[yy, xx])
                mag = math.hypot(du, dv)

                if mag < arrow_min_flow_len:
                    continue

                x2 = int(round(xx + du * arrow_coef))
                y2 = int(round(yy + dv * arrow_coef))

                if 0 <= x2 < w and 0 <= y2 < h:
                    cv.arrowedLine(out, (xx, yy), (x2, y2),
                                   arrow_color_bgr, 1, tipLength=0.35)

    return out



def build_gauss_weights(radius: int, sigma: float) -> np.ndarray:

    r = radius

    xs = np.arange(-r, r + 1, dtype=np.float64)
    ys = np.arange(-r, r + 1, dtype=np.float64)

    XX, YY = np.meshgrid(xs, ys)

    W = np.exp(-(XX ** 2 + YY ** 2) / (2.0 * sigma ** 2))
    W /= W.sum()

    return W.astype(np.float32)



def build_poly_kernels(radius: int, sigma: float):

    r = radius
    size = 2 * r + 1
    W = build_gauss_weights(r, sigma)
    N = size * size
    M = np.zeros((N, 6), dtype=np.float64)
    W_vec = np.zeros(N, dtype=np.float64)

    for k_idx, (row, col) in enumerate([(ry, rx) for ry in range(size) for rx in range(size)]):
        dy = row - r
        dx = col - r

        M[k_idx] = [dx*dx, dy*dy, dx*dy, dx, dy, 1.0] #A11, A22, A12, b1, b2, c
        W_vec[k_idx] = W[row, col]

    W_diag = np.diag(W_vec)
    MtWM = M.T @ W_diag @ M
    MtWM_inv = np.linalg.pinv(MtWM)
    filter_mat = MtWM_inv @ (M.T * W_vec[np.newaxis, :])

    return [filter_mat[i].reshape(size, size).astype(np.float32) for i in range(6)]



def compute_poly_coeffs(gray, kernels):

    A11 = conv(gray, kernels[0])
    A22 = conv(gray, kernels[1])
    A12 = conv(gray, kernels[2])
    bx = conv(gray, kernels[3])
    by = conv(gray, kernels[4])
    c = conv(gray, kernels[5])

    return A11, A22, A12, bx, by, c



def warp_image(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:

    H, W = img.shape[:2]

    xs   = np.arange(W, dtype=np.float32)
    ys   = np.arange(H, dtype=np.float32)

    XX, YY = np.meshgrid(xs, ys)

    map_x = (XX + u).astype(np.float32)
    map_y = (YY + v).astype(np.float32)

    return cv.remap(img, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)



def compute_flow_single_iter(prev, cur, kernels, avg_kernel):

    A11_1, A22_1, A12_1, bx1, by1, c1 = compute_poly_coeffs(prev, kernels)
    A11_2, A22_2, A12_2, bx2, by2, c2 = compute_poly_coeffs(cur, kernels)

    dA11 = A11_1 - A11_2
    dA22 = A22_1 - A22_2
    dbx = bx1 - bx2
    dby = by1 - by2
    dc = c1 - c2

    r = win_r
    size = 2 * r + 1
    xs = np.arange(-r, r + 1, dtype=np.float64)
    ys = np.arange(-r, r + 1, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys)
    N_w = float(size ** 2)
    m_xx = float((XX ** 2).sum())
    m_yy = float((YY ** 2).sum())

    S_RxRx = 4.0 * A11_2 ** 2 * m_xx + 4.0 * A12_2 ** 2 * m_yy + bx2 ** 2 * N_w
    S_RyRy = 4.0 * A12_2 ** 2 * m_xx + 4.0 * A22_2 ** 2 * m_yy + by2 ** 2 * N_w
    S_RxRy = (4.0 * A11_2 * A12_2 * m_xx + 4.0 * A12_2 * A22_2 * m_yy + bx2 * by2 * N_w)
    S_RxH = (2.0 * A11_2 * dbx * m_xx + 2.0 * A12_2 * dby * m_yy + bx2 * (dA11 * m_xx + dA22 * m_yy + dc * N_w))
    S_RyH = (2.0 * A12_2 * dbx * m_xx + 2.0 * A22_2 * dby * m_yy + by2 * (dA11 * m_xx + dA22 * m_yy + dc * N_w))

    S_RxRx = conv(S_RxRx, avg_kernel)
    S_RyRy = conv(S_RyRy, avg_kernel)
    S_RxRy = conv(S_RxRy, avg_kernel)
    S_RxH = conv(S_RxH, avg_kernel)
    S_RyH = conv(S_RyH, avg_kernel)

    det = S_RxRx * S_RyRy - S_RxRy ** 2
    det_reg = det + lambda_flow

    u = (S_RyRy * S_RxH - S_RxRy * S_RyH) / det_reg
    v = (-S_RxRy * S_RxH + S_RxRx * S_RyH) / det_reg

    bad = np.abs(det) < 1e-6
    u[bad] = 0.0
    v[bad] = 0.0

    return u.astype(np.float32), v.astype(np.float32)



def compute_flow_iter(prev, cur, kernels, avg_kernel):

    u_accum = np.zeros_like(prev, dtype=np.float32)
    v_accum = np.zeros_like(prev, dtype=np.float32)

    for _ in range(flow_iters):
        curr_warped = warp_image(cur, u_accum, v_accum)
        du, dv = compute_flow_single_iter(prev, curr_warped, kernels, avg_kernel)
        u_accum = u_accum + du
        v_accum = v_accum + dv

    return u_accum, v_accum



def process_video_mine(video_path, track_x0: float, track_y0: float) -> list:

    global _global_pinv_kernels, _global_avg_kernel

    print(f"[Mine] Обробка відео: {video_path}")
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("[Mine] Не вдалося відкрити відео.")
        return []

    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        return []

    H, W = frame0.shape[:2]
    fps = cap.get(cv.CAP_PROP_FPS)

    if not fps or fps != fps or fps < 1:
        fps = 30.0

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None

    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(str(video_path)))[0]
    out_path = os.path.join(output_dir, f"{name}_farneback_mine.mp4")
    writer = cv.VideoWriter(out_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    u_full = np.zeros((H, W), dtype=np.float32)
    v_full = np.zeros((H, W), dtype=np.float32)
    last_diff = np.zeros((H, W), dtype=np.float32)

    if _global_pinv_kernels is None:
        _global_pinv_kernels = build_poly_kernels(win_r, sigma_gauss)

    if _global_avg_kernel is None:
        _global_avg_kernel = build_gauss_weights(avg_r, sigma_avg)

    prev_frame = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY).astype(np.float32)

    px, py = float(track_x0), float(track_y0)
    track_log = []
    history = []

    window_name = "Farneback Mine"
    start_time = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        cur_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float32)

        recompute = (frame_idx == 1) or (frame_idx % flow_step == 0)
        if recompute:
            last_diff = np.abs(cur_frame - prev_frame)

            prev_small = downscale(prev_frame, s)
            cur_small = downscale(cur_frame, s)

            u_s, v_s = compute_flow_iter(prev_small, cur_small,
                                               _global_pinv_kernels,
                                               _global_avg_kernel)

            u_full, v_full = upscale(u_s, v_s, s, (H, W))

            prev_frame = cur_frame

        history.append((px, py))
        px, py = update_track_point(px, py, u_full, v_full, frame.shape)
        track_log.append((frame_idx, px, py))

        mag = np.sqrt(u_full * u_full + v_full * v_full)

        motion_255 = build_motion_mask_common(mag, last_diff)
        motion_bin = (motion_255 > 0).astype(np.uint8)
        contours_mask = find_contours(motion_bin)

        vis = draw_flow(frame, motion_255, contours_mask, u_full, v_full, fill_color_bgr=color_mine, arrow_color_bgr=arrow_color_mine)
        vis = draw_track(vis, px, py, history)

        writer.write(vis)
        try:
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            prog_str = f"{frame_idx}/{total_frames}" if total_frames else str(frame_idx)
            cv.setWindowTitle(window_name, f"{window_name} [{prog_str}]")
        except Exception:
            pass
        cv.imshow(window_name, vis)

        prog = f"{frame_idx}/{total_frames}" if total_frames else str(frame_idx)
        sys.stdout.write(f"\r[Mine] {prog}  pt=({px:.1f},{py:.1f})  "
            f"Elapsed {time_format(time.time()-start_time)}")
        sys.stdout.flush()

        if (cv.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    writer.release()
    cv.destroyWindow(window_name)
    sys.stdout.write("\n")
    print(f"[Mine] Готово. Результат: {out_path}")

    return track_log



def process_video_opencv(video_path, track_x0: float, track_y0: float) -> list:

    print(f"[OpenCV] Обробка відео: {video_path}")
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("[OpenCV] Не вдалося відкрити відео.")
        return []

    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        return []

    H, W = frame0.shape[:2]
    fps = cap.get(cv.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 25.0
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None

    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(str(video_path)))[0]
    out_path = os.path.join(output_dir, f"{name}_farneback_opencv.mp4")
    writer = cv.VideoWriter(out_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    prev_frame = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)

    px, py = float(track_x0), float(track_y0)
    track_log = []
    history = []

    window_name = "Farneback OpenCV"
    start_time = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        cur_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        diff_full = cv.absdiff(cur_frame, prev_frame).astype(np.float32)

        flow = cv.calcOpticalFlowFarneback(prev_frame, cur_frame, None, pyr_scale, 1,
                                           2 * avg_r + 1 , flow_iters, 2 * win_r + 1, sigma_gauss, 0)
        u   = flow[..., 0]
        v   = flow[..., 1]
        mag = np.sqrt(u * u + v * v)

        history.append((px, py))
        px, py = update_track_point(px, py, u, v, frame.shape)
        track_log.append((frame_idx, px, py))

        motion_255 = build_motion_mask_common(mag, diff_full)
        motion_bin = (motion_255 > 0).astype(np.uint8)
        contours_mask = find_contours(motion_bin)

        vis = draw_flow( frame, motion_255, contours_mask, u, v,
                         fill_color_bgr=color_opencv, arrow_color_bgr=arrow_color_opencv)
        vis = draw_track(vis, px, py, history)

        writer.write(vis)
        try:
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            cv.setWindowTitle(window_name, f"{window_name} [{frame_idx}]")
        except Exception:
            pass
        cv.imshow(window_name, vis)

        prog = f"{frame_idx}/{total_frames}" if total_frames else str(frame_idx)
        sys.stdout.write(f"\r[OpenCV] {prog}  pt=({px:.1f},{py:.1f})  "
            f"Elapsed {time_format(time.time()-start_time)}")
        sys.stdout.flush()

        if (cv.waitKey(1) & 0xFF) == 27:
            break

        prev_frame = cur_frame

    cap.release()
    writer.release()
    cv.destroyWindow(window_name)
    sys.stdout.write("\n")
    print(f"[OpenCV] Готово. Результат: {out_path}")
    return track_log



def print_comparison_table(track_opencv: list, track_mine: list, video_name: str, out_dir: str, x0: float, y0: float, H: str, W: str):

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
    lines.append(f"  Початкова точка: ({x0}, {y0})")
    lines.append("=" * 90)
    lines.append(sep)
    lines.append(hdr)
    lines.append(sep)

    dxs, dys = [], []
    for i in range(n):
        fi,  ox, oy = track_opencv[i]
        _,   mx, my = track_mine[i]
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

    os.makedirs(out_dir, exist_ok=True)
    table_path = os.path.join(out_dir, f"{video_name}_table_Farn.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"\nТаблицю збережено: {table_path}")



def main():
    video_path = choose_video()

    print("\nОберіть точку для відстеження на першому кадрі")
    print("Клікніть мишею -> натисніть Enter")
    track_x0, track_y0 = choose_track_point(video_path)

    print("\n---- OpenCV реалізація [СИНІЙ] ----")
    t0 = time.time()
    track_opencv = process_video_opencv(video_path, track_x0, track_y0)
    t1 = time.time()

    print("\n---- Власна реалізація [ЧЕРВОНИЙ] ----")
    track_mine = process_video_mine(video_path, track_x0, track_y0)
    t2 = time.time()

    print(f"\nOpenCV Фарнебек : {t1 - t0:.2f} с  ({time_format(t1 - t0)})")
    print(f"Власний Фарнебек : {t2 - t1:.2f} с  ({time_format(t2 - t1)})")

    cap = cv.VideoCapture(video_path)
    H = str(int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    W = str(int(cap.get(cv.CAP_PROP_FRAME_WIDTH)))

    video_name = os.path.splitext(os.path.basename(str(video_path)))[0]
    print_comparison_table(track_opencv, track_mine, video_name, output_dir, track_x0, track_y0, H, W)

    print(f"\nВихід у папці : {output_dir}")


if __name__ == "__main__":
    main()