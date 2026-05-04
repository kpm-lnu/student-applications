import os, sys, time, math, queue, threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2 as cv
import numpy as np

output_dir = "processed"

CFG = dict(
    gabor_xy_size=11, gabor_t_size=5, gabor_sigma=2.5,
    gabor_orient=6,   gabor_omega_xy=0.25, gabor_omega_t=(0.10, 0.25),

    fund_orb_features=2000, fund_match_ratio=0.75,
    fund_ransac_thresh=1.0, fund_confidence=0.99, fund_min_matches=30,
    flow_pyr_scale=0.5, flow_levels=3, flow_winsize=21,
    flow_iters=3, flow_poly_n=5, flow_poly_sigma=1.2,
    lambda_kappa=8.0, lambda_min=1.0, min_flow_median=0.3,
    e_norm_percentile=99.0,

    threshold_k=4.5, threshold_floor=0.03, ema_alpha=0.08,
    threshold_k_mine=5.5,  
    k_blur_sigma=2.0, fill_holes=True,
    morph_ksize=7, min_blob_area=400, max_blob_area=60000,
    aspect_ratio_max=4.5, solidity_min=0.30,

    overlay_alpha=0.25,
    color_opencv=(255, 0,   0), arrow_opencv=(0,   0, 255),
    color_mine  =(0,   0, 255), arrow_mine  =(255, 0,   0),
    square_size=25, track_color=(0, 255, 0), track_thick=2,
    tail_len=40, arrow_step=18, arrow_min=0.6, arrow_coef=3.0,

    processing_scale=0.5,
    cpu_threads=None, read_buf=2, write_buf=4,

    mine_win_r=7,
    mine_sigma=3.5,
    mine_avg_r=5,
    mine_avg_sigma=2.5,
    mine_lambda=1e-4,
    mine_iters=3,
    mine_scale=2,
)

TRACK_FARN = dict(pyr_scale=0.5, levels=0, winsize=2*5+1, iterations=3, poly_n=2*7+1, poly_sigma=3.5, flags=0)

_STOP = object()

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

    H, W  = frame0.shape[:2]
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
            sr = CFG["square_size"] // 2
            cv.rectangle(show, (px - sr, py - sr), (px + sr, py + sr), CFG["track_color"], CFG["track_thick"])
        cv.imshow(win, show)
        key = cv.waitKey(20) & 0xFF
        if key == 13 and point[0] is not None:
            break

    cv.destroyWindow(win)
    px, py = point[0]
    px = int(np.clip(px, 0, W - 1))
    py = int(np.clip(py, 0, H - 1))
    print(f"Обрана точка для відстеження: ({px}, {py})")
    return float(px), float(py)



def build_temporal(cfg):

    T = cfg["gabor_t_size"]
    sigma = cfg["gabor_sigma"]
    omega_k_list = cfg["gabor_omega_t"]

    k_ax = np.arange(T, dtype=np.float32) - (T - 1)

    G_t = np.exp(-k_ax ** 2 / (2 * sigma ** 2))

    rows_c, rows_s = [], []
    for omega_k in omega_k_list:
        phase = 2 * np.pi * omega_k * k_ax

        g_c = (G_t * np.cos(phase)).astype(np.float32)
        g_s = (G_t * np.sin(phase)).astype(np.float32)

        norm = np.sqrt((g_c ** 2 + g_s ** 2).sum()) + 1e-8

        rows_c.append(g_c / norm)
        rows_s.append(g_s / norm)

    return np.stack(rows_c).astype(np.float32), np.stack(rows_s).astype(np.float32)



def energy_quadrature(bufs, T, gtc_s, gts_s, E_prev_ref):

    n  = len(bufs)

    Re = np.stack([np.stack([b[0] for b in bufs[i]]) for i in range(n)])
    Ro = np.stack([np.stack([b[1] for b in bufs[i]]) for i in range(n)])

    Re_tc=np.einsum('otij,nt->onij',Re,gtc_s)
    Re_ts=np.einsum('otij,nt->onij',Re,gts_s)

    Ro_tc=np.einsum('otij,nt->onij',Ro,gtc_s)
    Ro_ts=np.einsum('otij,nt->onij',Ro,gts_s)

    E = ((Re_tc-Ro_ts)**2+(Re_ts+Ro_tc)**2).sum(axis=(0,1)).astype(np.float32)

    if E_prev_ref[0] is None:
        E_prev_ref[0]=E
        return None

    ep=np.abs(E-E_prev_ref[0])
    E_prev_ref[0]=E

    return ep



def _n_workers(cfg): return max(1,(os.cpu_count() or 4)//2)



def gabor_opencv_init(cfg):

    xy, s, n_or, wxy = cfg["gabor_xy_size"],cfg["gabor_sigma"],cfg["gabor_orient"],cfg["gabor_omega_xy"]
    lam = 1.0/(wxy+1e-12)
    spatial=[(cv.getGaborKernel((xy,xy),s,i*math.pi/n_or,lam,1.0,0, ktype=cv.CV_32F),
              cv.getGaborKernel((xy,xy),s,i*math.pi/n_or,lam,1.0,math.pi/2, ktype=cv.CV_32F))
             for i in range(n_or)]

    gtc_s, gts_s = build_temporal(cfg)
    T = cfg["gabor_t_size"]
    bufs = [deque(maxlen=T) for _ in spatial]
    pool = ThreadPoolExecutor(max_workers=cfg.get("cpu_threads") or _n_workers(cfg))

    return dict(spatial=spatial, bufs=bufs, T=T, E_prev=[None], gtc_s=gtc_s, gts_s=gts_s, pool=pool)



def gabor_opencv_process(g, gray):

    img = gray.astype(np.float32)

    def _one(i):
        ke, ko = g["spatial"][i]
        return i, cv.filter2D(img, cv.CV_32F, ke), cv.filter2D(img, cv.CV_32F, ko)

    for i, Re, Ro in g["pool"].map(_one, range(len(g["spatial"]))): g["bufs"][i].append((Re, Ro))

    if len(g["bufs"][0]) < g["T"]:
        return None

    return energy_quadrature(g["bufs"], g["T"], g["gtc_s"], g["gts_s"], g["E_prev"])



def build_gabor_kernels_3d(cfg):

    xy_sz = cfg["gabor_xy_size"]
    T = cfg["gabor_t_size"]
    sigma = cfg["gabor_sigma"]
    n_or = cfg["gabor_orient"]
    omega = cfg["gabor_omega_xy"]
    wts = cfg["gabor_omega_t"]

    r_xy = xy_sz // 2
    ax = np.arange(-r_xy, r_xy + 1, dtype=np.float64)
    X, Y = np.meshgrid(ax, ax)

    k_ax = np.arange(T, dtype=np.float64) - (T - 1)

    kernels = []
    for i in range(n_or):
        theta = i * math.pi / n_or   # 0, 30, 60, 90, 120, 150
        wx = omega * math.cos(theta)
        wy = omega * math.sin(theta)

        for wk in wts:
            ke_slices, ko_slices = [], []
            for k in k_ax:
                gauss = np.exp(-(X**2 + Y**2 + k**2) / (2.0 * sigma**2))
                phase = wx * X + wy * Y + wk * k
                ke_slices.append((gauss * np.cos(phase)).astype(np.float32))
                ko_slices.append((gauss * np.sin(phase)).astype(np.float32))

            norm = math.sqrt(sum(float((sl**2).sum()) for sl in ke_slices + ko_slices)) + 1e-8
            kernels.append(([sl / norm for sl in ke_slices],
                 [sl / norm for sl in ko_slices]))

    return kernels



def gabor_mine_init(cfg):

    T = cfg["gabor_t_size"]
    kernels = build_gabor_kernels_3d(cfg)
    pool = ThreadPoolExecutor(
                  max_workers=cfg.get("cpu_threads") or _n_workers(cfg))
    return dict(kernels=kernels, buf=deque(maxlen=T), T=T, E_prev=[None], pool=pool)



def gabor_mine_process(g, gray):

    g["buf"].append(gray.astype(np.float32))
    if len(g["buf"]) < g["T"]:
        return None

    frames = list(g["buf"])

    def _one_kernel(args):
        ke_slices, ko_slices = args

        Re = np.zeros_like(frames[0])
        Ro = np.zeros_like(frames[0])

        for t, (ke_t, ko_t) in enumerate(zip(ke_slices, ko_slices)):
            Re += cv.filter2D(frames[t], cv.CV_32F, ke_t)
            Ro += cv.filter2D(frames[t], cv.CV_32F, ko_t)

        return Re, Ro

    E = np.zeros_like(frames[0])
    for Re, Ro in g["pool"].map(_one_kernel, g["kernels"]):
        E += Re**2 + Ro**2

    E_prev = g["E_prev"][0]
    if E_prev is None:
        g["E_prev"][0] = E
        return None

    E_prime = np.abs(E - E_prev)
    g["E_prev"][0] = E

    return E_prime



def gabor_shutdown(g):

    if g["pool"]: g["pool"].shutdown(wait=False)



def w_map(F, flow, h, w):

    ys,xs = np.mgrid[0:h,0:w].astype(np.float32)

    mL = np.stack([xs,ys,np.ones_like(xs)],axis=-1)
    mR = np.stack([xs+flow[...,0], ys+flow[...,1], np.ones_like(xs)],axis=-1)

    FmL = mL @F.T
    FtmR = mR @ F

    num =  np.einsum('hwi,hwi->hw',mR,FmL)
    den = np.maximum(FmL[..., 0]**2 + FmL[..., 1]**2 + FtmR[..., 0]**2 + FtmR[..., 1]**2,1e-9)
    D = (num**2/den).astype(np.float32)
    lam = max(CFG["lambda_min"], CFG["lambda_kappa"] * float(np.median(D)))

    return D / (D + lam)



def epipolar_opencv_init(cfg):

    fp=dict(pyr_scale=cfg["flow_pyr_scale"],levels=cfg["flow_levels"], winsize=cfg["flow_winsize"],
            iterations=cfg["flow_iters"], poly_n=cfg["flow_poly_n"],poly_sigma=cfg["flow_poly_sigma"],flags=0)

    return dict(fp = fp, n_samples = cfg["fund_orb_features"], rt = cfg["fund_ransac_thresh"], conf = cfg["fund_confidence"])



def epipolar_opencv_process(e, prev, curr, shape):

    h, w = shape
    fl = cv.calcOpticalFlowFarneback(prev, curr, None, **e["fp"])
    u_fl, v_fl = fl[...,0], fl[...,1]

    if np.median(np.hypot(u_fl,v_fl)) < CFG["min_flow_median"]:
        return np.ones((h, w), np.float32), fl

    rng = np.random.default_rng(0)
    ys_all, xs_all = np.mgrid[0:h,0:w]
    idxs = rng.choice(h*w,size=min(e["n_samples"], h * w), replace=False)

    xs_s = xs_all.ravel()[idxs].astype(np.float32)
    ys_s = ys_all.ravel()[idxs].astype(np.float32)

    pL = np.column_stack([xs_s, ys_s]).reshape(-1, 1, 2)
    pR = np.column_stack([xs_s + u_fl.ravel()[idxs], ys_s + v_fl.ravel()[idxs]]).reshape(-1, 1, 2)

    F = None
    for method in (cv.FM_RANSAC, cv.USAC_MAGSAC):
        try:
            Ft, _ = cv.findFundamentalMat(pL,pR,method, ransacReprojThreshold = e["rt"], confidence = e["conf"])
            if Ft is not None and Ft.shape==(3,3):
                F=Ft.astype(np.float32); break
        except cv.error: pass

    if F is None:
        return np.ones((h,w),np.float32),fl

    return w_map(F, fl, h, w), fl



def farn_build_kernels(win_r, sigma):

    r    = win_r
    size = 2*r + 1
    xs   = np.arange(-r, r+1, dtype=np.float64)
    XX, YY = np.meshgrid(xs, xs)
    W    = np.exp(-(XX**2 + YY**2) / (2*sigma**2))
    W   /= W.sum()
    N    = size*size
    M    = np.zeros((N, 6), dtype=np.float64)
    Wv   = np.zeros(N, dtype=np.float64)

    for k, (row, col) in enumerate((ry, rx) for ry in range(size) for rx in range(size)):
        dy = row-r; dx = col-r
        M[k]  = [dx*dx, dy*dy, dx*dy, dx, dy, 1.0]
        Wv[k] = W[row, col]

    MtWM = M.T @ np.diag(Wv) @ M
    fm   = np.linalg.pinv(MtWM) @ (M.T * Wv)

    return [fm[i].reshape(size, size).astype(np.float32) for i in range(6)]



def farn_build_avg_kernel(avg_r, avg_sigma):

    r  = avg_r
    xs = np.arange(-r, r+1, dtype=np.float64)
    XX, YY = np.meshgrid(xs, xs)
    W  = np.exp(-(XX**2+YY**2) / (2*avg_sigma**2))
    W /= W.sum()

    return W.astype(np.float32)



def farneback_poly_coeffs(gray, kernels):

    return [cv.filter2D(gray, cv.CV_32F, kernels[i]) for i in range(6)]



def farneback_warp(img, u, v):

    H, W = img.shape[:2]
    xs   = np.arange(W, dtype=np.float32)
    ys   = np.arange(H, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)

    return cv.remap(img, (XX+u).astype(np.float32), (YY+v).astype(np.float32),
                    cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)



def farneback_mine(prev, cur, kernels, avg_k, lam):

    c1 = farneback_poly_coeffs(prev, kernels)
    c2 = farneback_poly_coeffs(cur,  kernels)
    A11_2, A22_2, A12_2, bx2, by2, _ = c2
    dA11 = c1[0] - c2[0]
    dA22 = c1[1] - c2[1]
    dbx  = c1[3] - c2[3]
    dby  = c1[4] - c2[4]
    dc = c1[5] - c2[5]

    r = kernels[0].shape[0]//2
    size = 2 * r+1
    xs = np.arange(-r, r+1, dtype=np.float64)
    XX, YY = np.meshgrid(xs, xs)
    Nw = float(size**2)
    mxx = float((XX**2).sum())
    myy = float((YY**2).sum())

    SRxRx = 4*A11_2**2*mxx + 4*A12_2**2*myy + bx2**2*Nw
    SRyRy = 4*A12_2**2*mxx + 4*A22_2**2*myy + by2**2*Nw
    SRxRy = 4*A11_2*A12_2*mxx + 4*A12_2*A22_2*myy + bx2*by2*Nw
    SRxH = 2*A11_2*dbx*mxx + 2*A12_2*dby*myy + bx2*(dA11*mxx+dA22*myy+dc*Nw)
    SRyH = 2*A12_2*dbx*mxx + 2*A22_2*dby*myy + by2*(dA11*mxx+dA22*myy+dc*Nw)

    SRxRx = cv.filter2D(SRxRx, cv.CV_32F, avg_k)
    SRyRy = cv.filter2D(SRyRy, cv.CV_32F, avg_k)
    SRxRy = cv.filter2D(SRxRy, cv.CV_32F, avg_k)
    SRxH  = cv.filter2D(SRxH,  cv.CV_32F, avg_k)
    SRyH  = cv.filter2D(SRyH,  cv.CV_32F, avg_k)

    det = SRxRx * SRyRy - SRxRy ** 2
    det_r = det + lam
    u = (SRyRy*SRxH - SRxRy*SRyH) / det_r
    v = (-SRxRy*SRxH + SRxRx*SRyH) / det_r
    bad = np.abs(det) < 1e-6
    u[bad]=0.0
    v[bad]=0.0

    return u.astype(np.float32), v.astype(np.float32)



def compute_farneback_mine(prev, cur, kernels, avg_k, lam, iters, scale):

    H, W = prev.shape[:2]
    sw, sh = max(1,W//scale), max(1,H//scale)
    ps = cv.resize(prev,(sw,sh),interpolation=cv.INTER_AREA)
    cs = cv.resize(cur, (sw,sh),interpolation=cv.INTER_AREA)

    ua = np.zeros_like(ps, dtype=np.float32)
    va = np.zeros_like(ps, dtype=np.float32)
    for _ in range(iters):
        cw = farneback_warp(cs, ua, va)
        du, dv = farneback_mine(ps, cw, kernels, avg_k, lam)
        ua += du
        va += dv


    ub = cv.resize(ua,(W,H),interpolation=cv.INTER_LINEAR) * scale
    vb = cv.resize(va,(W,H),interpolation=cv.INTER_LINEAR) * scale

    return ub, vb



def epipolar_mine_init(cfg):

    kernels = farn_build_kernels(cfg["mine_win_r"], cfg["mine_sigma"])
    avg_k = farn_build_avg_kernel(cfg["mine_avg_r"], cfg["mine_avg_sigma"])

    return dict(kernels=kernels, avg_k=avg_k, lam=cfg["mine_lambda"], iters=cfg["mine_iters"],
                scale=cfg["mine_scale"], n_samples=cfg["fund_orb_features"], minm=cfg["fund_min_matches"])



def norm_points(pts):

    c = pts.mean(axis=0); d=np.sqrt(((pts-c)**2).sum(axis=1)).mean()
    s  = math.sqrt(2) / max(d, 1e-8)
    T = np.array([[s, 0, -s*c[0]], [0, s, -s*c[1]], [0, 0, 1]],dtype=np.float64)
    ph = np.hstack([pts,np.ones((len(pts),1))])

    return (T @ ph.T).T[:,:2], T



def _eight_pt_F(p1, p2):

    if len(p1) < 8:
        return None

    pn1, T1 = norm_points(p1)
    pn2, T2 = norm_points(p2)


    x1, y1 = pn1[:,0], pn1[:,1]
    x2, y2 = pn2[:,0], pn2[:,1]
    A = np.column_stack([
        x2*x1, x2*y1, x2,
        y2*x1, y2*y1, y2,
        x1,    y1,    np.ones(len(p1))
    ])

    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)


    U, S, Vt2 = np.linalg.svd(F)
    S[2] = 0.0
    F = U @ np.diag(S) @ Vt2

    F = T2.T @ F @ T1
    n = F[2, 2]
    if abs(n) > 1e-12:
        F /= n

    return F



def d_s(F, p1, p2):

    ph1 = np.hstack([p1, np.ones((len(p1), 1))])
    ph2 = np.hstack([p2, np.ones((len(p2), 1))])
    Fp1 = (F @ ph1.T).T
    Ftp2 = (F.T @ ph2.T).T
    num = (ph2 * Fp1).sum(axis=1) ** 2
    den = Fp1[:, 0] **2 + Fp1[:, 1] ** 2 + Ftp2[:, 0] ** 2 + Ftp2[:, 1] ** 2

    return num/(den+1e-12)



def find_F(p1, p2, thr=1.5, iters=500):

    if len(p1) < 8: return None
    N = len(p1)
    bF, bm, bc = None, np.zeros(N,bool),0
    thr2 = thr ** 2

    for _ in range(iters):
        idx = np.random.choice(N,8, replace=False)
        F = _eight_pt_F(p1[idx],p2[idx])

        if F is None: continue
        mask = d_s(F, p1, p2) < thr2
        cnt = int(mask.sum())

        if cnt > bc:
            bc = cnt
            bF = F
            bm = mask

        if bc > 0.9 * N:
            break

    if bF is not None and bc >= 8:
        Fr = _eight_pt_F(p1[bm], p2[bm])

        if Fr is not None:
            bF = Fr

    return bF



def epipolar_mine_process(e, prev, curr, shape):

    h, w = shape
    u_fl,v_fl = compute_farneback_mine(
        prev.astype(np.float32), curr.astype(np.float32),
        e["kernels"], e["avg_k"], e["lam"], e["iters"], e["scale"])
    fl = np.stack([u_fl, v_fl], axis=-1)

    if np.median(np.hypot(u_fl, v_fl)) < CFG["min_flow_median"]:
        return np.ones((h, w), np.float32), fl

    #вибір N пар (m_L, m_R)
    rng = np.random.default_rng(0)
    ys_all, xs_all = np.mgrid[0:h, 0:w]
    idxs = rng.choice(h*w, size=min(e["n_samples"], h * w), replace=False)

    xs_s = xs_all.ravel()[idxs].astype(np.float64)
    ys_s = ys_all.ravel()[idxs].astype(np.float64)
    us_s = u_fl.ravel()[idxs].astype(np.float64)
    vs_s = v_fl.ravel()[idxs].astype(np.float64)

    p1m = np.column_stack([xs_s, ys_s])
    p2m = np.column_stack([xs_s + us_s, ys_s + vs_s])

    F = find_F(p1m, p2m, thr=CFG["fund_ransac_thresh"])
    if F is None:
        return np.ones((h,w), np.float32), fl

    return w_map(F.astype(np.float32), fl, h, w), fl



def threshold_init(cfg):

    scale = max(0.01, min(float(cfg.get("processing_scale", 1.0)), 1.0))
    kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, (cfg["morph_ksize"],)*2)

    return dict(k = cfg["threshold_k"], T_floor = cfg["threshold_floor"], fill = cfg["fill_holes"],
                min_a = int(cfg["min_blob_area"] * scale**2), max_a = int(cfg["max_blob_area"] * scale**2),
                asp = cfg["aspect_ratio_max"], sol = cfg["solidity_min"], sig = cfg["k_blur_sigma"], kern=kern)



def threshold_process(t, K):

    if K is None:
        return None

    if t["sig"]>0:
        K = cv.GaussianBlur(K,(0,0), t["sig"])

    T = max(t["T_floor"], float(K.mean()) + t["k"]*float(K.std()))

    binary=(K>=T).astype(np.uint8)*255
    binary=cv.morphologyEx(binary,cv.MORPH_OPEN, t["kern"])
    binary=cv.morphologyEx(binary,cv.MORPH_CLOSE,t["kern"])

    n,labels,stats,_=cv.connectedComponentsWithStats(binary)
    clean=np.zeros_like(binary)

    for lbl in range(1,n):

        a = int(stats[lbl,cv.CC_STAT_AREA])
        bw=int(stats[lbl,cv.CC_STAT_WIDTH])
        bh=int(stats[lbl,cv.CC_STAT_HEIGHT])

        if not(t["min_a"]<=a<=t["max_a"]) or bw<1 or bh<1:
            continue

        if max(bw,bh) / max(min(bw,bh),1) > t["asp"]:
            continue

        cnts, _ = cv.findContours((labels==lbl).astype(np.uint8),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

        if not cnts:
            continue

        hull_a = cv.contourArea(cv.convexHull(cnts[0]))

        if hull_a > 0 and a / hull_a < t["sol"]:
            continue

        cv.drawContours(clean, cnts, -1,255, cv.FILLED if t["fill"] else 1)

    return clean



def update_track_farneback(px, py, u, v, shape):

    H, W = shape[:2]

    cx = float(np.clip(px, 0, W-1))
    cy = float(np.clip(py, 0, H-1))

    x0, y0 = int(cx), int(cy)
    x1, y1 = min(x0+1, W-1), min(y0+1, H-1)

    fx = cx - x0
    fy = cy - y0

    du = ((1 - fx) * (1 - fy) * u[y0, x0] + fx * (1 - fy) * u[y0, x1] +
          (1 - fx) *    fy    * u[y1, x0] + fx *    fy    * u[y1, x1])
    dv = ((1 - fx) * (1 - fy) * v[y0, x0] + fx * (1 - fy) * v[y0, x1] +
          (1 - fx) *    fy    * v[y1, x0] + fx *    fy    * v[y1,x1])

    return float(np.clip(cx + du, 0, W  -1)), float(np.clip(cy + dv, 0, H - 1))



def correct_point_by_mask(px, py, mask):

    if mask is None or not np.any(mask):
        return px, py

    H, W = mask.shape[:2]
    x = int(np.clip(round(px), 0, W - 1))
    y = int(np.clip(round(py), 0, H - 1))

    if mask[y, x] > 0:
        return px, py

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return px, py

    distances = np.sqrt((xs - px) ** 2 + (ys - py) ** 2)
    min_dist = distances.min()

    if min_dist > 25:
        return px, py

    nearest_id = np.argmin(distances)

    return float(xs[nearest_id]), float(ys[nearest_id])



def draw_track(frame, px, py, history):

    out=frame.copy()
    H, W = out.shape[:2]
    tail = history[-CFG["tail_len"]:] + [(px,py)]

    for i in range(1,len(tail)):
        alpha = i / max(len(tail), 1)
        cv.line(out,(int(round(tail[i - 1][0])), int(round(tail[i - 1][1]))),(int(round(tail[i][0])),
                int(round(tail[i][1]))),(0, int(80 + 175 * alpha), 0),1, cv.LINE_AA)

    ix, iy = int(round(px)), int(round(py))

    hs = CFG["square_size"] // 2

    cv.rectangle(out,(max(0, ix - hs), max(0, iy - hs)), (min(W - 1, ix + hs), min(H - 1, iy + hs)), CFG["track_color"], CFG["track_thick"])

    lx = min(ix + hs + 3, W - 90)
    ly = max(iy, 14)

    cv.putText(out,f"({ix},{iy})",(lx,ly),cv.FONT_HERSHEY_SIMPLEX,0.45,CFG["track_color"],1,cv.LINE_AA)

    return out



def draw_flow(frame, mask, u, v, fill_col, arrow_col):

    out = frame.copy()
    if CFG["overlay_alpha"] > 0 and np.any(mask):
        ov = out.copy()
        ov[mask > 0] = fill_col
        out = cv.addWeighted(ov,CFG["overlay_alpha"], out, 1 - CFG["overlay_alpha"],0)

    if np.any(mask):
        h, w = mask.shape
        for yy in range(0, h, CFG["arrow_step"]):
            for xx in range(0, w, CFG["arrow_step"]):
                if mask[yy, xx] == 0:
                    continue

                du, dv = float(u[yy,xx]), float(v[yy,xx])
                mg = math.hypot(du,dv)

                if mg < CFG["arrow_min"]:
                    continue

                x2 = int(round(xx + du * CFG["arrow_coef"]))
                y2 = int(round(yy + dv * CFG["arrow_coef"]))
                if 0 <= x2 < w and 0 <= y2 < h:
                    cv.arrowedLine(out,(xx, yy),(x2, y2), arrow_col,1,tipLength=0.35)

    return out



def print_comparison_table(track_cv, track_mine, video_name, out_dir, x0, y0, H, W):

    n = min(len(track_cv),len(track_mine))

    if n==0:
        print("Немає даних!")
        return

    sep = "+"+"-"*7+"+"+"-"*13+"+"+"-"*13+"+"+"-"*13+"+"+"-"*13+"+"+"-"*11+"+"+"-"*11+"+"
    hdr = (f"| {'Кадр':^5} | {'x_o':^11} | {'y_o':^11} "
           f"| {'x_m':^11} | {'y_m':^11} "
           f"| {'Δx^k':^9} | {'Δy^k':^9} |")
    lines=["","="*90, f"  Початкова точка: ({x0:.0f}, {y0:.0f})","="*90,sep,hdr,sep]

    dxs, dys = [], []
    for i in range(n):
        fi, ox, oy = track_cv[i]
        _, mx, my = track_mine[i]

        dx = ox - mx
        dy = oy - my
        dxs.append(dx)
        dys.append(dy)
        lines.append(f"| {fi:^5d} | {ox:^11.2f} | {oy:^11.2f} "
                     f"| {mx:^11.2f} | {my:^11.2f} | {dx:^9.2f} | {dy:^9.2f} |")


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

    output="\n".join(lines)
    print(output)

    os.makedirs(out_dir,exist_ok=True)
    table_path = os.path.join(out_dir,f"{video_name}_table_combined.txt")
    with open(table_path,"w",encoding="utf-8") as f:
        f.write(output+"\n")
    print(f"\nТаблицю збережено: {table_path}")



def process_frame(frame, d, gabor_fn, epipolar_fn):

    H, W = frame.shape[:2]
    scale = d["scale"]
    sw, sh = (int(W * scale), int(H * scale)) if scale != 1.0 else (W, H)
    small = cv.resize(frame,(sw, sh), interpolation=cv.INTER_AREA) if scale!=1.0 else frame
    gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY)

    E_prime = gabor_fn(d["gabor"],gray)
    if E_prime is None:
        d["prev_gray"] = gray
        return None,None

    W_map, flow = (epipolar_fn(d["epipolar"],d["prev_gray"],gray,(sh,sw))
                if d["prev_gray"] is not None
                else (np.ones((sh,sw),np.float32),np.zeros((sh,sw,2),np.float32)))
    d["prev_gray"]=gray

    p = float(np.percentile(E_prime,d["e_perc"]))

    E_norm = np.clip(E_prime / p,0,1) if p>1e-9 else np.zeros_like(E_prime)

    K = E_norm * W_map

    mask_s = threshold_process(d["threshold"], K)
    if mask_s is None:
        mask_s = np.zeros((sh,sw),np.uint8)

    if scale != 1.0:
        mask = cv.resize(mask_s,(W,H),interpolation=cv.INTER_NEAREST)
        u = cv.resize(flow[...,0],(W,H),interpolation=cv.INTER_LINEAR) * (1 / scale)
        v = cv.resize(flow[...,1],(W,H),interpolation=cv.INTER_LINEAR) * (1 / scale)
    else:
        mask = mask_s
        u,v = flow[...,0],flow[...,1]

    return mask,(u,v)



def _reader(cap,rq,stop):

    while not stop.is_set():
        ok,f=cap.read()
        if not ok: break
        rq.put(f)
    rq.put(_STOP)



def _writer(vw,wq):

    while True:
        item=wq.get()
        if item is _STOP: return
        vw.write(item)



def run_combined(video_path, tx0, ty0, label, out_suffix, gabor_fn, epipolar_fn, fill_col, arrow_col,
                  gabor_init_fn, epipolar_init_fn, thr_k=None, flow_fn=None):

    print(f"\n[{label}] Обробка відео: {video_path}")
    cap = cv.VideoCapture(str(video_path))
    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        return []

    H, W = frame0.shape[:2]
    fps = cap.get(cv.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or None

    os.makedirs(output_dir, exist_ok=True)
    name = Path(video_path).stem
    out_path = os.path.join(output_dir, f"{name}_{out_suffix}.mp4")
    writer = cv.VideoWriter(out_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    scale = max(0.01, min(float(CFG.get("processing_scale", 1.0)), 1.0))
    cfg_thr = dict(CFG)

    if thr_k is not None:
        cfg_thr["threshold_k"] = thr_k

    d = dict(gabor=gabor_init_fn(CFG), epipolar=epipolar_init_fn(CFG), threshold=threshold_init(cfg_thr),
             prev_gray=None, scale=scale, e_perc=CFG["e_norm_percentile"])

    px, py = tx0, ty0
    track_log = []
    history = []
    win = label

    t0 = time.time()

    fi = 0
    prev_gray_track = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
    cur_mask = np.zeros((H, W),dtype = np.uint8)

    cur_u = np.zeros((H, W),dtype=np.float32)
    cur_v = np.zeros((H, W),dtype=np.float32)

    rq = queue.Queue(maxsize=CFG["read_buf"])
    wq = queue.Queue(maxsize=CFG["write_buf"])
    stop = threading.Event()
    threading.Thread(target=_reader,args=(cap,rq,stop),daemon=True).start()
    t_w = threading.Thread(target=_writer,args=(writer,wq),daemon=True); t_w.start()

    process_frame(frame0,d,gabor_fn,epipolar_fn)

    try:
        while True:
            frame = rq.get()

            if frame is _STOP:
                break

            fi+=1

            cur_gray_track = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            if flow_fn is not None:
                u_t, v_t = flow_fn(prev_gray_track, cur_gray_track)
                ft = np.stack([u_t, v_t], axis=-1)
            else:
                ft = cv.calcOpticalFlowFarneback(prev_gray_track, cur_gray_track,None, **TRACK_FARN)

            history.append((px, py))

            result = process_frame(frame, d, gabor_fn, epipolar_fn)

            if result[0] is not None:
                cur_mask,(cur_u, cur_v) = result

            px, py = update_track_farneback(px, py, ft[..., 0], ft[..., 1], frame.shape)
            px, py = correct_point_by_mask(px, py, cur_mask)

            track_log.append((fi, px, py))
            prev_gray_track = cur_gray_track

            vis = draw_flow(frame, cur_mask, cur_u, cur_v, fill_col, arrow_col)
            vis = draw_track(vis, px, py, history)
            wq.put(vis)

            try:
                cv.namedWindow(win, cv.WINDOW_NORMAL)
                prog = f"{fi}/{total}" if total else str(fi)
                cv.setWindowTitle(win,f"{win} [{prog}]")
            except Exception:
                pass
            cv.imshow(win, vis)

            prog = f"{fi}/{total}" if total else str(fi)
            sys.stdout.write(f"\r[{label}] {prog}  pt=({px:.1f},{py:.1f})  "
                             f"Elapsed {time_format(time.time()-t0)}")
            sys.stdout.flush()

            if cv.waitKey(1) & 0xFF == 27:
                stop.set()
                break

    except KeyboardInterrupt:
        stop.set()
    finally:
        wq.put(_STOP)
        t_w.join()
        cap.release()
        writer.release()
        cv.destroyWindow(win)
        gabor_shutdown(d["gabor"])

    sys.stdout.write("\n")
    print(f"[{label}] Готово. Результат: {out_path}")
    return track_log



def process_video_opencv(video_path, tx0, ty0):

    return run_combined(
        video_path, tx0, ty0,
        label="OpenCV", out_suffix="opencv",
        gabor_fn=gabor_opencv_process,
        epipolar_fn=epipolar_opencv_process,
        fill_col=CFG["color_opencv"],
        arrow_col=CFG["arrow_opencv"],
        gabor_init_fn=gabor_opencv_init,
        epipolar_init_fn=epipolar_opencv_init)



def process_video_mine(video_path, tx0, ty0):

    e = epipolar_mine_init(CFG)

    def farn_mine(prev, curr):
        return compute_farneback_mine(prev, curr, e["kernels"], e["avg_k"], e["lam"], e["iters"], e["scale"])

    return run_combined(
        video_path, tx0, ty0,
        label="Custom", out_suffix="custom",
        gabor_fn=gabor_mine_process,
        epipolar_fn=epipolar_mine_process,
        fill_col=CFG["color_mine"],
        arrow_col=CFG["arrow_mine"],
        gabor_init_fn=gabor_mine_init,
        epipolar_init_fn=epipolar_mine_init,
        thr_k=CFG.get("threshold_k_mine"),
        flow_fn=farn_mine)


def main():
    video_path = choose_video()

    print("\nОберіть точку для відстеження на першому кадрі")
    print("Клікніть мишею -> натисніть Enter")
    tx0, ty0 = choose_track_point(video_path)

    print("\n---- OpenCV реалізація [СИНІЙ] ----")
    t0 = time.time()
    track_cv = process_video_opencv(video_path, tx0, ty0)
    t1 = time.time()

    print("\n---- Власна реалізація [ЧЕРВОНИЙ] ----")
    track_mine = process_video_mine(video_path, tx0, ty0)
    t2 = time.time()

    print(f"\nOpenCV комбінований : {t1 - t0:.2f} с  ({time_format(t1 - t0)})")
    print(f"Власний комбінований : {t2 - t1:.2f} с  ({time_format(t2 - t1)})")

    cap = cv.VideoCapture(video_path)
    H = str(int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    W = str(int(cap.get(cv.CAP_PROP_FRAME_WIDTH)))

    video_name = Path(video_path).stem
    print_comparison_table(track_cv, track_mine, video_name, output_dir, tx0, ty0, H, W)

    print(f"Вихід у папці : {output_dir}")

if __name__=="__main__":
    main()