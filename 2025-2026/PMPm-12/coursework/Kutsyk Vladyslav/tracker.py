"""
tracker.py
ByteTrack із заміненим модулем прогнозування стану на IMM-KF.

Публічний інтерфейс:
    tracker = ByteTrackIMM()
    tracks  = tracker.update(detections, frame)
    # tracks: список Track об'єктів з полями id, bbox, state
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

from imm_kalman import IMMKalmanFilter
from config import (TAU_HIGH, TAU_LOW, TAU_NEW, MAX_AGE,
                    N_INIT, COST_ALPHA, GATE_CHI2, DT)


# ═══════════════════════════════════════════════════════════════
#  Допоміжні функції
# ═══════════════════════════════════════════════════════════════

def iou_batch(bboxes_a: np.ndarray,
              bboxes_b: np.ndarray) -> np.ndarray:
    """
    Обчислює матрицю IoU між двома наборами bbox.
    Формат: [u_center, v_center, w, h] у пікселях.
    """
    # Перетворення у xyxy
    def to_xyxy(b):
        x1 = b[:, 0] - b[:, 2] / 2
        y1 = b[:, 1] - b[:, 3] / 2
        x2 = b[:, 0] + b[:, 2] / 2
        y2 = b[:, 1] + b[:, 3] / 2
        return np.stack([x1, y1, x2, y2], axis=1)

    a = to_xyxy(bboxes_a)
    b = to_xyxy(bboxes_b)

    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union  = area_a[:, None] + area_b[None, :] - inter

    return np.where(union > 0, inter / union, 0.0)


def hungarian(cost_matrix: np.ndarray,
              threshold: float = 0.8
              ) -> Tuple[List[Tuple[int, int]],
                         List[int], List[int]]:
    """
    Угорський алгоритм для задачі зіставлення.
    Повертає: (matched_pairs, unmatched_rows, unmatched_cols)
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), \
                   list(range(cost_matrix.shape[1]))

    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    matched, unmatched_r, unmatched_c = [], [], []

    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] < threshold:
            matched.append((r, c))
        else:
            unmatched_r.append(r)
            unmatched_c.append(c)

    all_r = set(range(cost_matrix.shape[0]))
    all_c = set(range(cost_matrix.shape[1]))
    unmatched_r += list(all_r - set(row_idx))
    unmatched_c += list(all_c - set(col_idx))

    return matched, unmatched_r, unmatched_c


# ═══════════════════════════════════════════════════════════════
#  Клас треку
# ═══════════════════════════════════════════════════════════════

class TrackState(Enum):
    TENTATIVE   = 1   # ще не підтверджений (< N_INIT кадрів)
    CONFIRMED   = 2   # активний трек
    LOST        = 3   # тимчасово відсутній


@dataclass
class Track:
    id:          int
    imm:         IMMKalmanFilter
    state:       TrackState = TrackState.TENTATIVE
    hits:        int = 1      # кількість підтверджень
    age:         int = 1      # загальна кількість кадрів
    time_since_update: int = 0
    model_probs: np.ndarray = field(
        default_factory=lambda: np.array([0.7, 0.2, 0.1])
    )

    # Кешовані bbox із останнього predict/update
    _bbox: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def bbox(self) -> np.ndarray:
        """Поточний bbox [u, v, w, h]."""
        return self.imm.get_bbox()

    def predict(self) -> np.ndarray:
        x_pred, _ = self.imm.predict()
        self._bbox = x_pred[[0, 1, 6, 7]]
        self.age  += 1
        self.time_since_update += 1
        self.model_probs = self.imm.model_probs
        return self._bbox

    def update(self, z: np.ndarray) -> None:
        """z: [u_center, v_center, w, h] у пікселях."""
        self.imm.update(z)
        self.model_probs = self.imm.model_probs
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


# ═══════════════════════════════════════════════════════════════
#  Трекер
# ═══════════════════════════════════════════════════════════════

class ByteTrackIMM:
    """
    ByteTrack із IMM-KF замість стандартного CV-КФ.

    Використання:
        tracker = ByteTrackIMM()
        for frame in video:
            dets = detector.detect(frame)   # [[u,v,w,h,score], ...]
            tracks = tracker.update(dets)
    """

    def __init__(self,
                 tau_high: float = TAU_HIGH,
                 tau_low:  float = TAU_LOW,
                 tau_new:  float = TAU_NEW,
                 max_age:  int   = MAX_AGE,
                 n_init:   int   = N_INIT,
                 alpha:    float = COST_ALPHA,
                 dt:       float = DT):

        self.tau_high = tau_high
        self.tau_low  = tau_low
        self.tau_new  = tau_new
        self.max_age  = max_age
        self.n_init   = n_init
        self.alpha    = alpha
        self.dt       = dt

        self.tracks:    List[Track] = []
        self._next_id:  int = 1

    # ── Головний метод ────────────────────────────────────────

    def update(self, detections: np.ndarray) -> List[Track]:
        """
        Args:
            detections: np.ndarray shape (N, 5)
                        columns: [u_center, v_center, w, h, score]
                        координати у пікселях

        Returns:
            Список підтверджених активних Track об'єктів.
        """
        if detections is None or len(detections) == 0:
            detections = np.empty((0, 5))

        # ── Розбиття детекцій за score ────────────────────────
        dets_high = detections[detections[:, 4] >= self.tau_high]
        dets_low  = detections[(detections[:, 4] >= self.tau_low) &
                               (detections[:, 4] <  self.tau_high)]

        # ── Крок прогнозування IMM-KF для всіх треків ─────────
        for t in self.tracks:
            t.predict()

        active = [t for t in self.tracks
                  if t.state != TrackState.LOST
                  or t.time_since_update <= self.max_age]

        confirmed  = [t for t in active if t.is_confirmed()]
        tentative  = [t for t in active
                      if t.state == TrackState.TENTATIVE]
        lost_tracks = [t for t in active
                       if t.state == TrackState.LOST]

        # ── Перший прохід: confirmed + tentative ↔ dets_high ──
        matched1, unmatched_tr1, unmatched_det1 = \
            self._associate(confirmed + tentative, dets_high,
                            use_mahal=True)

        for ti, di in matched1:
            (confirmed + tentative)[ti].update(dets_high[di, :4])

        unmatched_tracks_after1 = [
            (confirmed + tentative)[i]
            for i in unmatched_tr1
        ]
        unmatched_dets_high = dets_high[unmatched_det1]

        # ── Другий прохід: lost/unmatched ↔ dets_low ──────────
        lost_pool = unmatched_tracks_after1 + lost_tracks
        matched2, unmatched_tr2, unmatched_det2 = \
            self._associate(lost_pool, dets_low, use_mahal=False)

        for ti, di in matched2:
            lost_pool[ti].update(dets_low[di, :4])

        # ── Ініціація нових треків ────────────────────────────
        remaining_dets = np.vstack([
            unmatched_dets_high,
            dets_low[unmatched_det2] if len(unmatched_det2) > 0
            else np.empty((0, 5))
        ]) if len(unmatched_dets_high) > 0 else (
            dets_low[unmatched_det2] if len(unmatched_det2) > 0
            else np.empty((0, 5))
        )

        for det in remaining_dets:
            if det[4] >= self.tau_new:
                self._init_track(det[:4])

        # ── Позначення пропущених підтверджених треків ────────
        still_unmatched = set(id(t) for t in
                              [lost_pool[i] for i in unmatched_tr2])
        for t in unmatched_tracks_after1:
            if id(t) in still_unmatched:
                t.mark_missed()

        # ── Видалення застарілих треків ───────────────────────
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.max_age
        ]

        return [t for t in self.tracks if t.is_confirmed()]

    # ── Асоціація ─────────────────────────────────────────────

    def _associate(self,
                   tracks: List[Track],
                   dets:   np.ndarray,
                   use_mahal: bool
                   ) -> Tuple[List[Tuple[int, int]],
                              List[int], List[int]]:
        """
        Будує матрицю вартостей і запускає угорський алгоритм.
        Формула вартості (розд. 2):
            d_cost = alpha * d_mah / chi2 + (1-alpha) * (1 - IoU)
        """
        if len(tracks) == 0 or len(dets) == 0:
            return ([], list(range(len(tracks))),
                    list(range(len(dets))))

        track_bboxes = np.array([t.bbox for t in tracks])
        det_bboxes   = dets[:, :4]

        iou_mat  = iou_batch(track_bboxes, det_bboxes)
        cost_iou = 1.0 - iou_mat

        if use_mahal:
            cost_mah = self._mahalanobis_matrix(tracks, det_bboxes)
            # Гейтування: нескінченна вартість поза воротами
            gate_mask = cost_mah > GATE_CHI2
            cost = (self.alpha * cost_mah / GATE_CHI2
                    + (1.0 - self.alpha) * cost_iou)
            cost[gate_mask] = 1e6
        else:
            cost = cost_iou

        return hungarian(cost, threshold=0.8)

    def _mahalanobis_matrix(self,
                            tracks: List[Track],
                            det_bboxes: np.ndarray) -> np.ndarray:
        """Матриця відстаней Махаланобіса: tracks × dets."""
        H = tracks[0].imm.H
        mat = np.zeros((len(tracks), len(det_bboxes)))

        for i, t in enumerate(tracks):
            if t.imm.P_pred is None:
                mat[i, :] = 1e6
                continue
            S_inv = np.linalg.inv(
                H @ t.imm.P_pred @ H.T + t.imm.R
            )
            mu_z = H @ (t.imm.x_pred
                        if t.imm.x_pred is not None
                        else t.imm.x[0])
            for j, z in enumerate(det_bboxes):
                d = z - mu_z
                mat[i, j] = float(d.T @ S_inv @ d)

        return mat

    def _init_track(self, bbox: np.ndarray) -> None:
        """Ініціює новий трек з першим вимірюванням bbox."""
        imm = IMMKalmanFilter(bbox, dt=self.dt)
        track = Track(
            id=self._next_id,
            imm=imm,
            state=TrackState.TENTATIVE,
        )
        self._next_id += 1
        self.tracks.append(track)

    def reset(self) -> None:
        """Скидає трекер між послідовностями."""
        self.tracks   = []
        self._next_id = 1
