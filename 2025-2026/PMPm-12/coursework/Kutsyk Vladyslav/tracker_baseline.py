"""
tracker_baseline.py
Базовий ByteTrack зі стандартним фільтром Калмана (CV-модель).
Це конфігурація B1 для ablation study.

Використовується виключно для порівняння з ByteTrackIMM
у скрипті ablation.py.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum

from config import (TAU_HIGH, TAU_LOW, TAU_NEW,
                    MAX_AGE, N_INIT, DT, MEAS_NOISE)


# ─────────────────────────────────────────────────────────────
#  Стандартний KF — лише CV (стала швидкість)
#  Стан: [x, y, vx, vy, w, h]  (6-dim, як у DeepSORT/SORT)
# ─────────────────────────────────────────────────────────────

class SimpleKalmanCV:
    """
    Лінійний фільтр Калмана з моделлю сталої швидкості.
    Ідентична реалізація до тієї, що використовується
    у стандартному ByteTrack / SORT / DeepSORT.
    """

    def __init__(self, z_init: np.ndarray, dt: float = DT):
        """
        z_init: (cx, cy, w, h) — перше вимірювання
        """
        # Матриця переходу: position += velocity * dt
        self.F = np.array([
            [1, 0, dt, 0,  0, 0],
            [0, 1,  0, dt, 0, 0],
            [0, 0,  1,  0, 0, 0],
            [0, 0,  0,  1, 0, 0],
            [0, 0,  0,  0, 1, 0],
            [0, 0,  0,  0, 0, 1],
        ], dtype=float)

        # Матриця спостереження: вимірюємо cx, cy, w, h
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)

        self.Q = np.diag([1., 1., 5., 5., 0.5, 0.5]) * dt
        self.R = np.diag(MEAS_NOISE).astype(float)

        # Ініціалізація стану
        self.x = np.array([z_init[0], z_init[1],
                           0., 0.,
                           z_init[2], z_init[3]])
        self.P = np.diag([100., 100., 25., 25., 50., 50.])

        # Зберігаємо апріорний стан для gate-перевірки
        self.x_pred: Optional[np.ndarray] = None
        self.P_pred: Optional[np.ndarray] = None

    def predict(self) -> np.ndarray:
        self.x_pred = self.F @ self.x
        self.P_pred = self.F @ self.P @ self.F.T + self.Q
        return self.x_pred[[0, 1, 4, 5]]  # cx, cy, w, h

    def update(self, z: np.ndarray) -> None:
        """z: (cx, cy, w, h)"""
        if self.x_pred is None:
            self.predict()
        S = self.H @ self.P_pred @ self.H.T + self.R
        K = self.P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = self.x_pred + K @ (z - self.H @ self.x_pred)
        self.P = (np.eye(6) - K @ self.H) @ self.P_pred

    def get_bbox(self) -> np.ndarray:
        """Повертає (cx, cy, w, h)."""
        x = self.x_pred if self.x_pred is not None else self.x
        return x[[0, 1, 4, 5]]


# ─────────────────────────────────────────────────────────────
#  Стани та клас треку
# ─────────────────────────────────────────────────────────────

class TrackState(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    LOST      = 3


@dataclass
class TrackCV:
    id:    int
    kf:    SimpleKalmanCV
    state: TrackState    = TrackState.TENTATIVE
    hits:  int           = 1
    time_since_update: int = 0

    @property
    def bbox(self) -> np.ndarray:
        return self.kf.get_bbox()

    def predict(self) -> np.ndarray:
        self.time_since_update += 1
        return self.kf.predict()

    def update(self, z: np.ndarray) -> None:
        self.kf.update(z)
        self.hits += 1
        self.time_since_update = 0
        if (self.state == TrackState.TENTATIVE
                and self.hits >= N_INIT):
            self.state = TrackState.CONFIRMED

    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    def mark_missed(self) -> None:
        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.LOST


# ─────────────────────────────────────────────────────────────
#  Допоміжні функції (аналогічні tracker.py)
# ─────────────────────────────────────────────────────────────

def _iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    def to_xyxy(boxes):
        return np.stack([boxes[:,0]-boxes[:,2]/2,
                         boxes[:,1]-boxes[:,3]/2,
                         boxes[:,0]+boxes[:,2]/2,
                         boxes[:,1]+boxes[:,3]/2], axis=1)
    a_ = to_xyxy(a); b_ = to_xyxy(b)
    ix1 = np.maximum(a_[:,None,0], b_[None,:,0])
    iy1 = np.maximum(a_[:,None,1], b_[None,:,1])
    ix2 = np.minimum(a_[:,None,2], b_[None,:,2])
    iy2 = np.minimum(a_[:,None,3], b_[None,:,3])
    inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
    area_a = (a_[:,2]-a_[:,0]) * (a_[:,3]-a_[:,1])
    area_b = (b_[:,2]-b_[:,0]) * (b_[:,3]-b_[:,1])
    union  = area_a[:,None] + area_b[None,:] - inter
    return np.where(union > 0, inter/union, 0.0)


def _hungarian(cost: np.ndarray, threshold=0.8):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    ri, ci = linear_sum_assignment(cost)
    matched, unr, unc = [], [], []
    for r, c in zip(ri, ci):
        if cost[r, c] < threshold:
            matched.append((r, c))
        else:
            unr.append(r); unc.append(c)
    unr += list(set(range(cost.shape[0])) - set(ri))
    unc += list(set(range(cost.shape[1])) - set(ci))
    return matched, unr, unc


# ─────────────────────────────────────────────────────────────
#  Базовий трекер
# ─────────────────────────────────────────────────────────────

class ByteTrackCV:
    """
    ByteTrack із стандартним CV-фільтром Калмана.
    Конфігурація B1 — baseline для порівняння.
    Логіка двопрохідної асоціації ідентична ByteTrackIMM,
    відрізняється лише модуль прогнозування стану.
    """

    def __init__(self,
                 tau_high = TAU_HIGH, tau_low = TAU_LOW,
                 tau_new  = TAU_NEW,  max_age = MAX_AGE,
                 n_init   = N_INIT,   dt      = DT):
        self.tau_high = tau_high
        self.tau_low  = tau_low
        self.tau_new  = tau_new
        self.max_age  = max_age
        self.n_init   = n_init
        self.dt       = dt
        self.tracks:   List[TrackCV] = []
        self._next_id: int = 1

    def update(self, detections: np.ndarray) -> List[TrackCV]:
        """
        detections: (N, 5) — [cx, cy, w, h, score]
        """
        if detections is None or len(detections) == 0:
            detections = np.empty((0, 5))

        dets_high = detections[detections[:,4] >= self.tau_high]
        dets_low  = detections[(detections[:,4] >= self.tau_low) &
                               (detections[:,4] <  self.tau_high)]

        # Прогноз
        for t in self.tracks:
            t.predict()

        active    = [t for t in self.tracks
                     if t.time_since_update <= self.max_age]
        confirmed = [t for t in active if t.is_confirmed()]
        tentative = [t for t in active
                     if t.state == TrackState.TENTATIVE]
        lost      = [t for t in active
                     if t.state == TrackState.LOST]

        # Перший прохід
        pool1 = confirmed + tentative
        m1, untr1, undet1 = self._associate(pool1, dets_high)
        for ti, di in m1:
            pool1[ti].update(dets_high[di, :4])

        unmatched_tracks = [pool1[i] for i in untr1]
        unmatched_dets_h = dets_high[undet1]

        # Другий прохід
        pool2 = unmatched_tracks + lost
        m2, untr2, undet2 = self._associate(pool2, dets_low)
        for ti, di in m2:
            pool2[ti].update(dets_low[di, :4])

        # Нові треки
        remaining = (list(unmatched_dets_h) +
                     [dets_low[i] for i in undet2])
        for det in remaining:
            if (hasattr(det, '__len__') and len(det) >= 5
                    and det[4] >= self.tau_new):
                self._init_track(det[:4])

        # Позначення missed
        still_unmatched_ids = {id(pool2[i]) for i in untr2}
        for t in unmatched_tracks:
            if id(t) in still_unmatched_ids:
                t.mark_missed()

        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.max_age]
        return [t for t in self.tracks if t.is_confirmed()]

    def _associate(self, tracks, dets):
        if not tracks or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        tb = np.array([t.bbox for t in tracks])
        db = dets[:, :4]
        cost = 1.0 - _iou_batch(tb, db)
        return _hungarian(cost, threshold=0.8)

    def _init_track(self, bbox: np.ndarray) -> None:
        kf = SimpleKalmanCV(bbox, dt=self.dt)
        self.tracks.append(TrackCV(id=self._next_id, kf=kf))
        self._next_id += 1

    def reset(self) -> None:
        self.tracks   = []
        self._next_id = 1
