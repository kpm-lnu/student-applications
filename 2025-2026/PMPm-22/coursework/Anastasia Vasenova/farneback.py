import os
import sys
import math
import time
import cv2 as cv
import numpy as np

VIDEOS_DIR = "videos"

WINDOW_RADIUS = 2
FLOW_REG_LAMBDA = 1e-3

FLOW_PERC = 85.0
MIN_FLOW_THR = 1e-6

DIFF_THRESH = 12
MORPH_K = 3
MIN_COMPONENT_AREA = 60
MAX_COMPONENT_FRAC = 0.35
EDGE_CROP_FRAC = 0.05
DILATE_K = 3

TRAIL_DECAY = 0.92
ARROW_STEP = 16

DOWNSCALE = 2
FLOW_FRAME_STEP = 3

ECC_WARP_MODE = cv.MOTION_AFFINE
ECC_ITER = 80
ECC_TERM_EPS = 1e-6
ECC_MIN_CC = 0.92

ZERO_MASK_ON_BIG_MOTION = True
MAX_ROT_DEG = 3.5
MAX_TRANS_FRAC = 0.06

FB_PYR_SCALE = 0.5
FB_LEVELS = 3
FB_WINSIZE = 21
FB_ITERS = 3
FB_POLY_N = 5
FB_POLY_SIGMA = 1.2
FB_FLAGS = 0

DEBUG_EVERY_N = 30

ALPHA_FILL = 0.05
ALPHA_TRAIL = 0.5


COLOR_OPENCV = (0, 0, 255)  # red
COLOR_MINE   = (255, 0, 0)  # blue

_global_pinv_kernels = None


def fmt_hms(sec: float) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def list_videos(videos_dir: str):
    if not os.path.isdir(videos_dir):
        print(f"Папка '{videos_dir}' не знайдена.")
        return []
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    files = [f for f in os.listdir(videos_dir) if os.path.splitext(f)[1].lower() in exts]
    files.sort()
    return files

def choose_video(videos_dir: str) -> str:
    files = list_videos(videos_dir)
    if not files:
        print("У папці 'videos' немає відеофайлів.")
        sys.exit(1)
    print("Доступні відео:")
    for i, name in enumerate(files):
        print(f"[{i}] {name}")
    while True:
        choice = input("Вибери номер відео для обробки: ")
        try:
            idx = int(choice)
            if 0 <= idx < len(files):
                return os.path.join(videos_dir, files[idx])
        except ValueError:
            pass
        print("Невірний вибір. Спробуй ще раз.")

def frame_to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    b = frame_bgr[..., 0].astype(np.float32)
    g = frame_bgr[..., 1].astype(np.float32)
    r = frame_bgr[..., 2].astype(np.float32)
    gray = (0.114 * b + 0.587 * g + 0.299 * r) / 255.0
    return gray

def downscale_gray(gray: np.ndarray, factor: int = DOWNSCALE) -> np.ndarray:
    return gray[::factor, ::factor]

def upscale_flow(u_small: np.ndarray, v_small: np.ndarray, factor: int, out_shape):
    u_big = np.repeat(np.repeat(u_small, factor, axis=0), factor, axis=1)
    v_big = np.repeat(np.repeat(v_small, factor, axis=0), factor, axis=1)
    H, W = out_shape
    return u_big[:H, :W], v_big[:H, :W]


def build_poly_kernels(radius: int):
    r = radius
    coords = [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    N = len(coords)
    M = np.zeros((N, 6), dtype=np.float64)
    for k, (dx, dy) in enumerate(coords):
        M[k, 0] = dx * dx
        M[k, 1] = dy * dy
        M[k, 2] = dx * dy
        M[k, 3] = dx
        M[k, 4] = dy
        M[k, 5] = 1.0
    pinv = np.linalg.pinv(M)  # 6 x N
    return [row.reshape((2 * r + 1, 2 * r + 1)).astype(np.float32) for row in pinv]

def conv2d_valid_fast(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    r_y = kh // 2
    r_x = kw // 2
    padded = np.pad(image, ((r_y, r_y), (r_x, r_x)), mode='reflect').astype(np.float32)
    H, W = image.shape
    shape = (H, W, kh, kw)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
    patches = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    out = np.tensordot(patches, kernel, axes=([2, 3], [0, 1]))
    return out.astype(np.float32)

def compute_poly_coeffs(gray: np.ndarray, kernels):
    coeff_maps = [conv2d_valid_fast(gray, k) for k in kernels]
    a_xx, a_yy, a_xy, b_x, b_y, c = coeff_maps
    return a_xx, a_yy, a_xy, b_x, b_y, c

def compute_flow_farneback(prev_gray: np.ndarray, curr_gray: np.ndarray, kernels):
    a_xx1, a_yy1, a_xy1, b_x1, b_y1, _ = compute_poly_coeffs(prev_gray, kernels)
    a_xx2, a_yy2, a_xy2, b_x2, b_y2, _ = compute_poly_coeffs(curr_gray, kernels)

    a_xx = 0.5 * (a_xx1 + a_xx2)
    a_yy = 0.5 * (a_yy1 + a_yy2)
    a_xy = 0.5 * (a_xy1 + a_xy2)

    db_x = b_x2 - b_x1
    db_y = b_y2 - b_y1

    A12 = 0.5 * a_xy
    det = a_xx * a_yy - A12 * A12
    det_reg = det + FLOW_REG_LAMBDA

    invA11 = a_yy / det_reg
    invA22 = a_xx / det_reg
    invA12 = -A12 / det_reg

    u = -0.5 * (invA11 * db_x + invA12 * db_y)
    v = -0.5 * (invA12 * db_x + invA22 * db_y)

    mask_bad = np.abs(det) < 1e-6
    u[mask_bad] = 0.0
    v[mask_bad] = 0.0
    return u, v


def estimate_affine_ecc(prev_gray_full: np.ndarray, gray_full: np.ndarray):
    warp = np.array([[1, 0, 0],
                     [0, 1, 0]], dtype=np.float32)
    cc = 0.0
    try:
        prev_f = cv.GaussianBlur(prev_gray_full, (5, 5), 0)
        gray_f = cv.GaussianBlur(gray_full, (5, 5), 0)
        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, ECC_ITER, ECC_TERM_EPS)
        cc, warp = cv.findTransformECC(prev_f, gray_f, warp, ECC_WARP_MODE, criteria, None, 5)
        warp = warp.astype(np.float32)
        cc = float(cc)
    except cv.error:
        warp = np.array([[1, 0, 0],
                         [0, 1, 0]], dtype=np.float32)
        cc = 0.0
    return warp, cc

def warp_stats(warp: np.ndarray):
    a11, a12 = warp[0, 0], warp[0, 1]
    a21, a22 = warp[1, 0], warp[1, 1]
    tx, ty = warp[0, 2], warp[1, 2]
    rot_rad = math.atan2(a21, a11)
    rot_deg = abs(rot_rad * 180.0 / math.pi)
    trans = math.hypot(tx, ty)
    return rot_deg, trans


def binary_dilate(mask_bin: np.ndarray, ksize: int, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((ksize, ksize), dtype=np.float32)
    cur = mask_bin.astype(np.float32)
    for _ in range(iterations):
        conv = conv2d_valid_fast(cur, kernel)
        cur = (conv > 0).astype(np.float32)
    return cur.astype(np.uint8)

def binary_erode(mask_bin: np.ndarray, ksize: int, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((ksize, ksize), dtype=np.float32)
    cur = mask_bin.astype(np.float32)
    need = ksize * ksize
    for _ in range(iterations):
        conv = conv2d_valid_fast(cur, kernel)
        cur = (conv >= need).astype(np.float32)
    return cur.astype(np.uint8)

def binary_open(mask_bin: np.ndarray, ksize: int, iterations: int = 1) -> np.ndarray:
    return binary_dilate(binary_erode(mask_bin, ksize, iterations), ksize, iterations)

def connected_components_stats(mask_255: np.ndarray):
    mask = (mask_255 > 0)
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    stats_list = []
    cur_label = 1

    for y in range(H):
        for x in range(W):
            if mask[y, x] and labels[y, x] == 0:
                stack = [(y, x)]
                labels[y, x] = cur_label
                minx = maxx = x
                miny = maxy = y
                area = 0
                while stack:
                    cy, cx = stack.pop()
                    area += 1
                    minx = min(minx, cx); maxx = max(maxx, cx)
                    miny = min(miny, cy); maxy = max(maxy, cy)
                    for ny in range(cy - 1, cy + 2):
                        for nx in range(cx - 1, cx + 2):
                            if 0 <= ny < H and 0 <= nx < W:
                                if mask[ny, nx] and labels[ny, nx] == 0:
                                    labels[ny, nx] = cur_label
                                    stack.append((ny, nx))
                w = maxx - minx + 1
                h = maxy - miny + 1
                stats_list.append((minx, miny, w, h, area))
                cur_label += 1

    num_labels = cur_label
    stats = np.zeros((num_labels, 5), dtype=np.int32)
    for i, (x, y, w, h, a) in enumerate(stats_list, start=1):
        stats[i] = [x, y, w, h, a]
    return num_labels, labels, stats


def build_motion_mask_flow(mag_res: np.ndarray) -> np.ndarray:
    mag_valid = mag_res[mag_res > 1e-12]
    if mag_valid.size == 0:
        return np.zeros_like(mag_res, dtype=np.uint8)

    perc = float(np.percentile(mag_valid, FLOW_PERC))
    thr = max(MIN_FLOW_THR, perc)

    mask = (mag_res > thr).astype(np.uint8)
    mask = binary_open(mask, MORPH_K, iterations=1)
    mask = binary_dilate(mask, MORPH_K, iterations=1)
    return (mask * 255).astype(np.uint8)

def build_motion_mask_diff(diff_full: np.ndarray) -> np.ndarray:
    diff_255 = (diff_full * 255.0).astype(np.uint8)
    mask = (diff_255 > DIFF_THRESH).astype(np.uint8)
    mask = binary_open(mask, MORPH_K, iterations=1)
    mask = binary_dilate(mask, MORPH_K, iterations=1)
    return (mask * 255).astype(np.uint8)

def trim_edges(mask_255: np.ndarray) -> np.ndarray:
    h, w = mask_255.shape
    margin = int(EDGE_CROP_FRAC * min(h, w))
    if margin <= 0:
        return mask_255
    mask_255[:margin, :] = 0
    mask_255[-margin:, :] = 0
    mask_255[:, :margin] = 0
    mask_255[:, -margin:] = 0
    return mask_255

def filter_components_final(mask_255: np.ndarray) -> np.ndarray:
    h, w = mask_255.shape
    total_area = h * w
    max_area = MAX_COMPONENT_FRAC * total_area

    num, labels, stats = connected_components_stats(mask_255)
    if num <= 1:
        return mask_255

    keep = np.zeros_like(mask_255)
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < MIN_COMPONENT_AREA:
            continue
        touches_border = (x == 0) or (y == 0) or ((x + ww) >= w) or ((y + hh) >= h)
        if touches_border:
            continue
        if area > max_area:
            continue
        keep[labels == i] = 255
    return keep

def combine_motion_masks(mask_flow_255: np.ndarray, mask_diff_255: np.ndarray) -> np.ndarray:
    motion = (mask_flow_255 & mask_diff_255)
    motion = trim_edges(motion)
    motion = filter_components_final(motion)

    if not np.any(motion) and np.any(mask_flow_255):
        motion = mask_flow_255.copy()
        motion = trim_edges(motion)
        motion = filter_components_final(motion)

    return motion

def build_motion_mask_common(mag_res: np.ndarray,
                            diff_full: np.ndarray,
                            big_camera_motion: bool,
                            ecc_cc: float) -> np.ndarray:
    mask_flow = build_motion_mask_flow(mag_res)
    mask_diff = build_motion_mask_diff(diff_full)
    motion = combine_motion_masks(mask_flow, mask_diff)

    if ZERO_MASK_ON_BIG_MOTION and big_camera_motion and (ecc_cc < ECC_MIN_CC):
        motion[:] = 0

    if np.any(motion):
        mb = (motion > 0).astype(np.uint8)
        mb = binary_dilate(mb, DILATE_K, iterations=1)
        motion = (mb * 255).astype(np.uint8)

    return motion

def extract_contours(mask_bin: np.ndarray) -> np.ndarray:
    padded = np.pad(mask_bin, 1, mode='constant', constant_values=0)
    kernel = np.ones((3, 3), dtype=np.float32)
    local_sum_big = conv2d_valid_fast(padded.astype(np.float32), kernel)
    local_sum = local_sum_big[1:-1, 1:-1]
    return ((mask_bin == 1) & (local_sum < 9)).astype(np.uint8)


def draw_motion_overlay(frame_bgr: np.ndarray,
                        motion_mask_bin: np.ndarray,
                        contours_mask: np.ndarray,
                        u_res: np.ndarray,
                        v_res: np.ndarray,
                        trail: np.ndarray,
                        fill_color_bgr=(0, 0, 255),
                        arrow_color_bgr=None,
                        alpha_fill: float = ALPHA_FILL,
                        alpha_trail: float = ALPHA_TRAIL):

    if arrow_color_bgr is None:
        arrow_color_bgr = fill_color_bgr

    out = frame_bgr.astype(np.float32)

    # fill
    mask3 = motion_mask_bin[..., None].astype(np.float32)
    col = np.array(fill_color_bgr, dtype=np.float32).reshape(1, 1, 3)
    out = out * (1.0 - alpha_fill * mask3) + col * (alpha_fill * mask3)

    # contour
    out[contours_mask == 1] = np.array(fill_color_bgr, dtype=np.float32)

    # arrows
    h, w = motion_mask_bin.shape
    arrow_col = np.array(arrow_color_bgr, dtype=np.float32)
    for y in range(0, h, ARROW_STEP):
        for x in range(0, w, ARROW_STEP):
            du = float(u_res[y, x])
            dv = float(v_res[y, x])
            mag = math.hypot(du, dv)
            if mag < 1.0:
                continue
            x2 = int(x + du * 3.0)
            y2 = int(y + dv * 3.0)
            if 0 <= x2 < w and 0 <= y2 < h:
                steps = int(max(abs(x2 - x), abs(y2 - y))) + 1
                for i in range(steps):
                    t = i / steps
                    xi = int(x + t * (x2 - x))
                    yi = int(y + t * (y2 - y))
                    if 0 <= xi < w and 0 <= yi < h:
                        out[yi, xi] = arrow_col

    # trail (green)
    trail *= TRAIL_DECAY
    trail[motion_mask_bin == 1] = 1.0
    trail_3 = np.stack([np.zeros_like(trail), trail, np.zeros_like(trail)], axis=-1) * 255.0

    out = (1.0 - alpha_trail) * out + alpha_trail * trail_3
    return np.clip(out, 0, 255).astype(np.uint8), trail


def process_video_opencv(video_path: str):
    print(f"[OpenCV] Обробка відео: {video_path}")

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("[OpenCV] Не вдалося відкрити відео.")
        return

    ret, frame = cap.read()
    if not ret:
        print("[OpenCV] Порожнє відео.")
        cap.release()
        return

    H, W = frame.shape[:2]
    diag = math.hypot(W, H)

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None

    fps = cap.get(cv.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 30.0

    name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = "processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}_farneback_opencv.mp4")

    writer = cv.VideoWriter(out_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    prev_gray_full = frame_to_gray(frame)
    trail = np.zeros((H, W), dtype=np.float32)

    frame_idx = 0
    start_time = time.time()
    window_name = "Farneback OpenCV (RED)"

    print(f"[OpenCV] Запис у: {out_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_full = frame_to_gray(frame)
        frame_idx += 1

        warp, ecc_cc = estimate_affine_ecc(prev_gray_full, gray_full)
        rot_deg, trans = warp_stats(warp)
        big_camera_motion = (rot_deg > MAX_ROT_DEG) or (trans > MAX_TRANS_FRAC * diag)

        prev_aligned_full = cv.warpAffine(prev_gray_full, warp, (W, H), flags=cv.INTER_LINEAR)
        diff_full = np.abs(gray_full - prev_aligned_full)

        # IMPORTANT: scale to 0..255 for Farneback
        prev_fb = (prev_aligned_full * 255.0).astype(np.float32)
        curr_fb = (gray_full * 255.0).astype(np.float32)
        prev_blur = cv.GaussianBlur(prev_fb, (5, 5), 0)
        curr_blur = cv.GaussianBlur(curr_fb, (5, 5), 0)

        flow = cv.calcOpticalFlowFarneback(
            prev_blur, curr_blur, None,
            FB_PYR_SCALE, FB_LEVELS, FB_WINSIZE,
            FB_ITERS, FB_POLY_N, FB_POLY_SIGMA, FB_FLAGS
        )

        u_full = flow[..., 0]
        v_full = flow[..., 1]
        mag_res = np.sqrt(u_full * u_full + v_full * v_full)

        motion_mask_255 = build_motion_mask_common(mag_res, diff_full, big_camera_motion, ecc_cc)
        motion_mask_bin = (motion_mask_255 > 0).astype(np.uint8)
        contours_mask = extract_contours(motion_mask_bin)

        overlay, trail = draw_motion_overlay(
            frame, motion_mask_bin, contours_mask,
            u_full, v_full, trail,
            fill_color_bgr=COLOR_OPENCV,      # RED
            arrow_color_bgr=COLOR_OPENCV,     # RED arrows too
            alpha_fill=ALPHA_FILL,
            alpha_trail=ALPHA_TRAIL
        )

        writer.write(overlay)

        try:
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            cv.setWindowTitle(window_name, f"{window_name} [{frame_idx}]")
        except Exception:
            pass
        cv.imshow(window_name, overlay)

        if total_frames is not None:
            frac = min(1.0, frame_idx / total_frames)
            sys.stdout.write(
                f"\r[OpenCV] {frame_idx}/{total_frames} ({frac*100:5.1f}%)  Elapsed {fmt_hms(time.time()-start_time)}"
            )
        else:
            sys.stdout.write(f"\r[OpenCV] {frame_idx}  Elapsed {fmt_hms(time.time()-start_time)}")
        sys.stdout.flush()

        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break

        prev_gray_full = gray_full

    cap.release()
    writer.release()
    cv.destroyWindow(window_name)
    sys.stdout.write("\n")
    print(f"[OpenCV] Готово. Результат: {out_path}")


def process_video_mine(video_path: str):
    global _global_pinv_kernels
    print(f"[Mine] Обробка відео: {video_path}")

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("[Mine] Не вдалося відкрити відео.")
        return

    ret, frame = cap.read()
    if not ret:
        print("[Mine] Порожнє відео.")
        cap.release()
        return

    H, W = frame.shape[:2]
    diag = math.hypot(W, H)

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None

    fps = cap.get(cv.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 25.0

    name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = "processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}_farneback_mine.mp4")

    writer = cv.VideoWriter(out_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    if _global_pinv_kernels is None:
        print("[Mine] Обчислення поліноміальних ядер...")
        _global_pinv_kernels = build_poly_kernels(WINDOW_RADIUS)

    prev_gray_full = frame_to_gray(frame)
    trail = np.zeros((H, W), dtype=np.float32)

    u_full = np.zeros((H, W), dtype=np.float32)
    v_full = np.zeros_like(u_full)

    last_diff_full = np.zeros((H, W), dtype=np.float32)
    big_camera_motion = False
    last_ecc_cc = 0.0

    frame_idx = 0
    start_time = time.time()
    window_name = "Farneback Mine (BLUE)"

    print(f"[Mine] Запис у: {out_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_full = frame_to_gray(frame)
        frame_idx += 1

        recompute_flow = (frame_idx == 1) or (frame_idx % FLOW_FRAME_STEP == 0)

        if recompute_flow:
            warp, ecc_cc = estimate_affine_ecc(prev_gray_full, gray_full)
            last_ecc_cc = ecc_cc

            rot_deg, trans = warp_stats(warp)
            big_camera_motion = (rot_deg > MAX_ROT_DEG) or (trans > MAX_TRANS_FRAC * diag)

            prev_aligned_full = cv.warpAffine(prev_gray_full, warp, (W, H), flags=cv.INTER_LINEAR)
            last_diff_full = np.abs(gray_full - prev_aligned_full)

            prev_small = downscale_gray(prev_aligned_full, DOWNSCALE)
            curr_small = downscale_gray(gray_full, DOWNSCALE)

            u_small, v_small = compute_flow_farneback(prev_small, curr_small, _global_pinv_kernels)
            u_full, v_full = upscale_flow(u_small, v_small, DOWNSCALE, (H, W))

            prev_gray_full = gray_full

        mag_res = np.sqrt(u_full * u_full + v_full * v_full)

        motion_mask_255 = build_motion_mask_common(mag_res, last_diff_full, big_camera_motion, last_ecc_cc)
        motion_mask_bin = (motion_mask_255 > 0).astype(np.uint8)
        contours_mask = extract_contours(motion_mask_bin)

        overlay, trail = draw_motion_overlay(
            frame, motion_mask_bin, contours_mask,
            u_full, v_full, trail,
            fill_color_bgr=COLOR_MINE,       # BLUE
            arrow_color_bgr=COLOR_MINE,      # BLUE arrows too
            alpha_fill=ALPHA_FILL,
            alpha_trail=ALPHA_TRAIL
        )

        writer.write(overlay)

        try:
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            progress_str = f"{frame_idx}/{total_frames}" if total_frames else f"{frame_idx}"
            cv.setWindowTitle(window_name, f"{window_name} [{progress_str}]")
        except Exception:
            pass

        cv.imshow(window_name, overlay)

        if total_frames is not None:
            sys.stdout.write(f"\r[Mine] {frame_idx}/{total_frames}  Elapsed {fmt_hms(time.time()-start_time)}")
        else:
            sys.stdout.write(f"\r[Mine] {frame_idx}  Elapsed {fmt_hms(time.time()-start_time)}")
        sys.stdout.flush()

        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    writer.release()
    cv.destroyWindow(window_name)
    sys.stdout.write("\n")
    print(f"[Mine] Готово. Результат: {out_path}")


def main():
    video_path = choose_video(VIDEOS_DIR)

    print("\n=== КРОК 1: Farneback (OpenCV) ===")
    t0 = time.time()
    process_video_opencv(video_path)
    t1 = time.time()

    print("\n=== КРОК 2: Farneback (власний) ===")
    process_video_mine(video_path)
    t2 = time.time()

    print("\n=== ПІДСУМКИ ЧАСУ ===")
    print(f"Farneback OpenCV : {t1 - t0:.2f} с ({fmt_hms(t1 - t0)})")
    print(f"Farneback mine   : {t2 - t1:.2f} с ({fmt_hms(t2 - t1)})")

if __name__ == "__main__":
    main()
